import numpy as np
import math
import time

import torch
import torch.nn as nn
from torch.distributions import Normal
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from dataset.dataset_utils import reshape_to_graph, reshape_to_tensor
from model.mpgcn import MPGCN
from model.encoder_decoder import EncoderModel, DecoderModel
from utils.graph_utils import normalize_graph


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

class STGNP(nn.Module):
    def __init__(self, 
                 in_node_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 out_dim: int,
                 num_regions: int,
                 space_planes: int = 1,
                 time_planes: int = 1,
                 edge_inter_dim: int = 0, 
                 edge_intra_dim: int = 0, 
                 external_dim: int = 0,
                 hidden_dim_ext: int = 5,
                 dropout: float = 0.2,
                 encode_time: bool = True,
                 use_attention=False,
                 use_constant_edges=False,
                 use_hdyn=True,
                 cond_on_time=True,
                 use_regions=True,
                 decode_just_latent=False,
                 dt_step=0.1,
                 only_spatial=False,
                 use_norm=True,
                 use_edges=True,
                 use_mse=False,
                 classify=False,
                 predict_external=False,
                 use_diffusion=False,
                 use_einsum=False,
                 agg_type='sum',
                 compute_derivative=False,
                 use_norm_stmgcn=True,
                 use_bias_stmgcn=False,
                 depth_nodes = 2,
                 depth_edges = 2,
                 ):
        """
        Parameters
        ----------
        in_node_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layers in the GCN.
        """
        super(STGNP, self).__init__()
                
        self.in_dim = in_node_dim # Node feature dimensions
        self.out_dim = out_dim  # Output node feaure dimensions
        self.hidden_dim = hidden_dim  # Hidden dimensions of the node features
        self.latent_dim = latent_dim  # Dimensions of the latent dynamical space
        assert self.latent_dim > 0, "The latent dimension must be greater than 0"
        assert self.latent_dim < self.hidden_dim, "The latent dimension must be smaller than the hidden dimension"
        self.dyn_params = self.hidden_dim - latent_dim  # Number of fixed dimensions; control params of the dynamical system
        self.use_regions = use_regions
        self.decode_just_latent = decode_just_latent
        self.use_time = encode_time

        self.external_dim = external_dim  # External dimensions -- for the global data, not graph-based
        hidden_dim_ext = hidden_dim_ext if self.external_dim > 0 else 0  # Hidden dimensions for non-graph data
        self.hidden_dim_ext = hidden_dim_ext  # Hidden dimensions for non-graph data
        self.intra_edges_dim = edge_intra_dim  # Intra-plane edges feature dimensions
        self.inter_edges_dim = edge_inter_dim  # Inter-plane edges feature dimensions

        # ODE options
        self.dt_step = dt_step  # Time step for the ODE solver
        self.only_spatial = only_spatial

        # General options
        self.dropout = nn.Dropout(dropout)
        self.bias = True  # Use bias
        self.depth_nodes = depth_nodes  # Number of layers in the encoder and decoder
        self.depth_edges = depth_edges # Number of layers in the encoder

        self.use_edges = use_edges  # Use the edges in the graph
        self.use_mse = use_mse  # Be variational or not
        self.use_norm = use_norm
        # self.use_norm = False
        self.classify = classify  # Classify based on the latent space
        self.predict_external = predict_external  # Predict external data based on the latent space
        
        # STGCN
        self.space_planes = space_planes  # For the multiplex in space
        self.time_planes = time_planes  # For the multiplex in time
        self.num_regions = num_regions # Number of regions in the graph
        self.use_attention = use_attention
        self.use_constant_edges = use_constant_edges
        self.use_diffusion = use_diffusion  # Use the diffusion model
        self.use_einsum = use_einsum
        self.agg_type = agg_type
        self.compute_derivative = compute_derivative
        self.use_norm_stmgcn = use_norm_stmgcn
        self.use_bias_stmgcn = use_bias_stmgcn
        
        # =====================================================
        # ==================== ARCHITECURE ====================

        # 1: Define the data encoder that will generate the 'latent' graph
        self.encoder = EncoderModel(self.in_dim, self.external_dim, self.intra_edges_dim, self.inter_edges_dim, self.latent_dim, 
                                    self.dyn_params, self.hidden_dim_ext, self.space_planes, self.time_planes, self.num_regions, 
                                    depth_nodes=self.depth_nodes, depth_edges=self.depth_edges, dropout=0., 
                                    bias=self.bias, use_norm=self.use_norm, shared_mlp=False)

        # External data embedding
        # External data [batch_size, external_dim]
        if self.external_dim > 0:
            self.embedding_ext = nn.Sequential(
                nn.Linear(self.external_dim, self.hidden_dim_ext, bias=self.bias),
                nn.BatchNorm1d(self.hidden_dim_ext),
                nn.SiLU()
                )
        else:
            self.embedding_ext = lambda x: x


        # 2: Define the layer that will sove the latent dynamical process using a graph neural network

        #  STGCN_Diff / STGCN
        self.stgcn = MPGCN(
            self.latent_dim, 
            self.latent_dim, 
            self.dyn_params,
            edges_intra_nf=self.space_planes, 
            edges_inter_nf=self.time_planes,
            name="MPGCN",
            bias=self.use_bias_stmgcn,
            dropout=0.,
            use_attention=use_attention, 
            use_constant_edges=use_constant_edges,
            use_time=cond_on_time,
            use_hdyn=use_hdyn,
            only_spatial=self.only_spatial,
            add_source=False,
            summarise_state=classify,
            use_norm=self.use_norm_stmgcn,
            use_diffusion=self.use_diffusion,                             
            use_einsum=self.use_einsum,
            agg_type=self.agg_type,
            compute_derivative=self.compute_derivative,
            )
    
        # 3: Decoder of the graph embeddings [batch_size, h_dim, num_regions, frames]
        if self.decode_just_latent:            
            self.decoder_dim = self.latent_dim + self.hidden_dim_ext
        else:
            self.decoder_dim = self.dyn_params + self.latent_dim + self.hidden_dim_ext

        self.decoder_dim = self.decoder_dim + 1 if self.use_time else self.decoder_dim
        self.decoder_dim = self.decoder_dim + self.num_regions if self.use_regions else self.decoder_dim

        use_decoder_norm = self.use_norm
        self.decoder = DecoderModel(self.decoder_dim, out_dim, depth=self.depth_nodes, 
                                    dropout=0., bias=self.bias, use_norm=use_decoder_norm, 
                                    shared_mlp=True)


        # Init the weights
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """ Initialize the weights of the linear layers """
        if isinstance(m, nn.Linear):
            stdv = math.sqrt(6.0 / (m.weight.data.size(-2) + m.weight.data.size(-1)))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def set_edges_norm(self, g, plane, ft_name, use_ft=False, normalize=True):        
        """ Get the normalization coefficient for each edge """
        interaction = plane[1]
        if normalize:            
            norm_graph = normalize_graph(g[f"{interaction}"], ord='sym')
            g[f"{interaction}"].edata[f'norm_{interaction}'] = norm_graph.edata['w'].float()
        else:
            device = g[interaction].device
            g[f"{interaction}"].edata[f'norm_{interaction}'] = torch.ones(g[interaction].num_edges(), device=device).float()

    
    def set_tol(self):
        # Options
        self.method = 'rk4'
        self.use_adjoint = False

        if self.use_adjoint:
            self.integrator = odeint_adjoint
        else:
            self.integrator = odeint

        # Tolerances - for dopri5
        if self.method == 'rk4':
            self.options_solver = {'step_size': self.dt_step}
            self.atol = 1e-6
            self.rtol = 1e-8

        elif self.method == 'dopri5' or self.method == 'adaptive_heun' or self.method == 'bosh3':
            self.atol = 1e-6
            self.rtol = 1e-8
            self.options_solver = {'step_t': np.arange(0, 1, self.dt_step),
                                   'max_num_steps': 500,
                                   'first_step': self.dt_step*0.5,
                                   }
        else:
            raise ValueError(f"Method {self.method} not implemented")


    def integrate_ode(self, graph, ctx_t, t, v, space_edge_data, time_edge_data, rids=None):  # v = (L(x), z_)
        # Solver options        
        self.set_tol()  # Get the options for the integrator

        # Get the latent space and the control parameters
        L = v[:, :self.latent_dim]
        D = v[:, self.latent_dim:]

        # Assign node data to the graph
        space_edge_data = reshape_to_graph(space_edge_data.unsqueeze(-1))
        time_edge_data = reshape_to_graph(time_edge_data.unsqueeze(-1))
        graph['space'].edata['weight'] = space_edge_data.float()
        graph['time'].edata['weight'] = time_edge_data.float()

        # Set x0 initial condition and time
        x0 = reshape_to_graph(L.unsqueeze(-1)).squeeze().float()  # Initial latent space

        graph.ndata['time'] = reshape_to_graph(ctx_t).float() # Time
        graph.ndata['h_dyn'] = reshape_to_graph(D.unsqueeze(-1)).float()  # # Dynamical parameters
        graph.ndata['h_dyn_init'] = reshape_to_graph(D.unsqueeze(-1)).float()  # # Dynamical parameters
        if rids is not None:
            graph.ndata['rids'] = reshape_to_graph(rids.unsqueeze(-1)).float()
        self.stgcn.set_graph(graph, x0)
    
        # Time of integration
        t_int = t[0, 0, 0].type_as(x0)
        t_int = t_int / t_int.max()
        
        device_type = x0.device
        self.stgcn.dt = torch.FloatTensor([self.options_solver['step_size'] if 'step_size' in self.options_solver else self.dt_step])
        self.stgcn.dt = self.stgcn.dt.to(device_type)
        z = self.integrator(
            self.stgcn, x0, t_int,
            method=self.method,
            options=self.options_solver,
            atol=self.atol,
            rtol=self.rtol
            )

        return z
    

    def xy_to_mu_sigma(self, tgt_pts, in_time, in_node_data, in_edge_space, in_edge_time, global_data=None, rids=None, pos_data=None):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """

        # Get dimensions information
        batch_size, num_features, num_regions, num_times = in_node_data.shape

        # Sort the target points
        tgt_pts = torch.sort(tgt_pts)[0]
        in_edge_space = in_edge_space[..., tgt_pts]
        in_edge_time = in_edge_time[..., tgt_pts]

        if global_data is not None:
            # === Embedding of the external data
            h_ext = self.embedding_ext(global_data)
            
            g_data = h_ext.unsqueeze(-1).unsqueeze(-1)
            ge_sp_data = g_data.repeat(1, 1, in_edge_space.shape[2], num_times)[..., tgt_pts]
            ge_t_data = g_data.repeat(1, 1, in_edge_time.shape[2], num_times)[..., tgt_pts]
            g_ndata = g_data.repeat(1, 1, num_regions, in_node_data.shape[-1])# [..., tgt_pts]

            # Assign the edge features to the graph
            in_edge_space = torch.cat((in_edge_space, ge_sp_data), dim=1)
            in_edge_time = torch.cat((in_edge_time, ge_t_data), dim=1)
        else:
            g_ndata = None
        
        # === Node embedding
        node_data = in_node_data[..., tgt_pts]  # Select the target points
        if self.use_time:
            node_data = torch.cat((node_data, in_time[..., tgt_pts]), dim=1).float()

        if global_data is not None:
            node_data = torch.cat((node_data, g_ndata[..., tgt_pts]), dim=1).float()  # Concat along feature dimensions        
            
        if rids is not None:
            # Concatenate the region ids; repeat the region ids for each time-frame
            rids = rids.unsqueeze(-1).repeat(1, 1, 1, len(tgt_pts))
            node_data = torch.cat((node_data, rids), dim=1)

        if pos_data is not None:
            # Concatenate the position data
            pos_data = pos_data[..., tgt_pts]
            node_data = torch.cat((node_data, pos_data), dim=1)
        
        # self.graph_norm_test(node_data)
        mu, sigma, mu_space, sigma_space, mu_time, sigma_time = self.encoder(node_data, in_edge_space, in_edge_time)

        return mu, sigma, mu_space, sigma_space, mu_time, sigma_time, g_ndata


    def xz_to_y(self, time, z, latent, g_ndata=None, rids=None, pos_data=None):
        # ============================== Reconstructed features
        batch_size = time.shape[0]        
        h_rec = latent  # Latent trajectory

        # Initial control variables (D_0)
        h_dyn = z[:, self.latent_dim:]
        h_dyn = h_dyn.unsqueeze(-1).repeat(1, 1, 1, h_rec.shape[-1])  # Number of time-frames

        # Just latent trajectory (L) or also initial control dynamics (D_0)
        if self.decode_just_latent:
            h = h_rec            
        else:
            h = torch.cat((h_rec, h_dyn), dim=1)

        # External data in decoder
        if g_ndata is not None:
            h = torch.cat((h, g_ndata), dim=1)
        
        # Use time to decode
        if self.use_time:
            h = torch.cat((h, time), dim=1)

        # Use region ID to decode
        if rids is not None:
            # Concatenate the region ids
            rids = rids.unsqueeze(-1).repeat(1, 1, 1, h_rec.shape[-1])
            h = torch.cat((h, rids), dim=1)

        # Decode it
        mu, sigma = self.decoder(h.float())
        
        # Return the distribution
        distr_y = Normal(mu, sigma)

        return distr_y


    def predict_global(self, h):
        L = h[:, :self.latent_dim]
        D = h[:, self.latent_dim:]
        h_to_pred = torch.cat((L, D), dim=1)
        # h_to_pred = D
        # Aggregation
        h_agg = self.aggregate_to_pred(h_to_pred)

        # Prediction --- maybe force it to be just the RV, and LV EF. 
        pred_g = self.predict(h_agg)

        return pred_g
    

    def predict_trajectories(self, 
                graph,
                time_to_predict,
                latent_state,
                new_options=None,
                rids=None,
                global_data=None,):
        """ Forward pass."""

        # Load previous latent state into the graph
        self.stgcn.set_latent_state(latent_state)
        
        if new_options is not None:
            self.options_solver = new_options

        # Get the initial condition
        x0 = latent_state['l_end'].clone().squeeze().float()

        # Sort the target points
        tgt_pts = torch.sort(time_to_predict.clone().detach().float().requires_grad_(True))[0]
        num_tps = len(tgt_pts)

        # Get the external data if available
        if global_data is not None:
            # === Embedding of the external data
            h_ext = self.embedding_ext(global_data.float())
            g_data = h_ext.unsqueeze(-1).unsqueeze(-1)
            g_ndata = g_data.repeat(1, 1, self.num_regions, num_tps).float()
        else:
            g_ndata = None

        # Integrate the ODE
        t_int = tgt_pts.type_as(x0)
        t_int = t_int / t_int.max()
        t_int = 1 + t_int  # A bit hard-coded but assume that the previous one was until 1 already

        device_type = x0.device
        self.stgcn.dt = torch.FloatTensor([self.options_solver['step_size'] if 'step_size' in self.options_solver else self.dt_step])
        self.stgcn.dt = self.stgcn.dt.to(device_type)
        z = self.integrator(
            self.stgcn, x0, t_int,
            method=self.method,
            options=self.options_solver,
            atol=self.atol,
            rtol=self.rtol
            )
        # latent = reshape_to_tensor(latent.permute(1, 2, 0), graph.batch_size)
        latent = reshape_to_tensor(z.permute(1, 2, 0), graph.batch_size).float()

        # Extrapolate trajectory beyond the last context point (training window)
        h_dyn = reshape_to_tensor(latent_state['h_dyn_init'], graph.batch_size).squeeze().float()
        z_sample = torch.cat((latent[...,0], h_dyn), dim=1).float()
        p_y_pred = self.xz_to_y(time_to_predict, z_sample, latent, g_ndata=g_ndata, rids=rids)

        return p_y_pred, latent


    def forward(self, 
                graph,
                context_pts,
                target_pts,
                in_time,
                in_node_data,
                in_edge_space,
                in_edge_time,
                rids=None,
                pos_data=None,
                global_data=None,
                ):
        """ Forward pass."""
        #NOTE: Remember the expected data format: [batch_size, num_features, num_nodes, num_times]
        # Same for the edges: [batch_size, num_edge_fts, num_edges, num_times]        
        with graph.local_scope():            
            # Use the encoder internally to get the representation of both control and initial points            
            mu_context, sigma_context, mu_space, sigma_space, mu_time, sigma_time, g_data = self.xy_to_mu_sigma(context_pts, in_time, in_node_data, in_edge_space, in_edge_time, global_data, rids, pos_data)            

            # Get the distribution for the space and time edges
            q_space_ctx = Normal(mu_space, sigma_space)
            q_time_ctx = Normal(mu_time, sigma_time)
            q_context = Normal(mu_context, sigma_context)
            # NOTE: For the moment assume each node has it's own distribution            

            # Get edge normalization coefficients for each canonical edge type            
            for plane in graph.canonical_etypes:
                self.set_edges_norm(graph, plane, 'weight', use_ft=False, normalize=True)

            if self.training:
                # Encode target 
                # Context needs to be encoded to calculate KL term
                mu_target, sigma_target, mu_space_tgt, sigma_space_tgt, mu_time_tgt, sigma_time_tgt, g_data = self.xy_to_mu_sigma(target_pts, in_time, in_node_data, in_edge_space, in_edge_time, global_data, rids, pos_data)

                q_space_tgt = Normal(mu_space_tgt, sigma_space_tgt)
                q_time_tgt = Normal(mu_time_tgt, sigma_time_tgt)
                q_target = Normal(mu_target, sigma_target)
                
                if self.use_mse:
                    # NOT VARIATIONAL, HERE WE ARE DETERMINISTIC
                    z_sample = q_target.mean
                    space_edge_sample = q_space_tgt.mean
                    time_edge_sample = q_time_tgt.mean
                else:
                    # Sample from encoded distribution using reparameterization trick
                    z_sample = q_target.rsample()
                    space_edge_sample = q_space_tgt.rsample()
                    time_edge_sample = q_time_tgt.rsample()
                
                # Get z                
                tgt_time = in_time[..., target_pts]
                ctx_time = in_time[..., context_pts]
                latent = self.integrate_ode(graph, ctx_time, tgt_time, z_sample, space_edge_sample, time_edge_sample, rids)
                latent = reshape_to_tensor(latent.permute(1, 2, 0), graph.batch_size).float()

                # Graph norm data
                spatial_norm = self.stgcn.norm_spatial.float()
                temporal_norm = self.stgcn.norm_temporal.float()

                # Get parameters of output distribution                
                p_y_pred = self.xz_to_y(tgt_time, z_sample, latent, g_ndata=g_data, rids=rids, pos_data=pos_data)

                # Get the last latent state - for future prediction
                latent_state = self.stgcn.get_latent_state(self.stgcn.g, self.use_attention)
                latent_state['l_end'] = reshape_to_graph(latent[...,[-1]].clone().detach()).float()
                
                return p_y_pred, latent, (spatial_norm, temporal_norm), q_target, q_context,(q_space_tgt, q_time_tgt), (q_space_ctx, q_time_ctx), latent_state
            else:
                # Sample from distribution based on context
                # z_sample = q_context.rsample()
                # space_edge_sample = q_space_ctx.rsample()
                # time_edge_sample = q_time_ctx.rsample()

                # Get z and force
                tgt_time = in_time[..., target_pts]
                ctx_time = in_time[..., context_pts]

                # latent = self.integrate_ode(graph, ctx_time, tgt_time, z_sample, space_edge_data, time_edge_data)
                latent = self.integrate_ode(graph, ctx_time, tgt_time, mu_context, mu_space, mu_time, rids)
                latent = reshape_to_tensor(latent.permute(1, 2, 0), graph.batch_size).float()

                # Predict target points based on context
                p_y_pred = self.xz_to_y(tgt_time, mu_context, latent, g_ndata=g_data, rids=rids, pos_data=pos_data)

                # Graph norm data
                spatial_norm = self.stgcn.norm_spatial.float()
                temporal_norm = self.stgcn.norm_temporal.float()                

                # Get the last latent state - for future prediction
                latent_state = self.stgcn.get_latent_state(self.stgcn.g, self.use_attention)
                latent_state['l_end'] = reshape_to_graph(latent[...,[-1]].clone().detach()).float()

                return p_y_pred, latent, (spatial_norm, temporal_norm), None, q_context, (None, None, None), (q_space_ctx, q_time_ctx), latent_state