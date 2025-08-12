import numpy as np
import math 
from functools import partial
import time

import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

import torch.nn.functional as F
from torch.distributions import Normal

from dataset.dataset_utils import reshape_to_graph, reshape_to_tensor
from model.spatial_batch_norm import BatchNorm
from model.stmgcn import STGCN
from model.encoder_decoder import EncoderModel, DecoderModel
from model.classifier import Classifier
from model.graph_norm import GraphNorm
from utils.graph_utils import normalize_graph


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

# class STMultiplexODE(AbstractODEDecoder):
class STMultiplexODE(nn.Module):
    def __init__(self, 
                 in_node_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 out_dim: int,
                 class_dim: int,
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
        super(STMultiplexODE, self).__init__()
                
        self.in_dim = in_node_dim # Node feature dimensions
        self.out_dim = out_dim  # Output node feaure dimensions
        self.hidden_dim = hidden_dim  # Hidden dimensions of the node features
        self.latent_dim = latent_dim  # Dimensions of the latent dynamical space
        self.class_dim = class_dim  # Number of classes into which classify the data
        assert self.latent_dim > 0, "The latent dimension must be greater than 0"
        assert self.latent_dim < self.hidden_dim, "The latent dimension must be smaller than the hidden dimension"
        self.dyn_params = self.hidden_dim - latent_dim  # Number of fixed dimensions; control params of the dynamical system
        self.use_regions = use_regions
        self.decode_just_latent = decode_just_latent
        self.use_time = encode_time

        self.external_dim = external_dim  # External dimensions -- for the global data, not graph-based
        hidden_dim_ext = hidden_dim_ext if self.external_dim > 0 else 0  # Hidden dimensions for non-graph data
        self.hidden_dim_ext = hidden_dim_ext  # Hidden dimensions for non-graph data
        self.class_dim = class_dim  # Number of classes into which classify the data
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
        self.variational  = True
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
        
        # Number of prototypes
        # self.n_prototypes = 20
        # self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, self.dyn_params), requires_grad=True)  # Shape [K, D]
        self.prototypes = None

        # =====================================================
        # ==================== ARCHITECURE ====================

        # 1: Define the data encoder that will generate the 'latent' graph
        self.encoder = EncoderModel(self.in_dim, self.external_dim, self.intra_edges_dim, self.inter_edges_dim, self.latent_dim, 
                                    self.dyn_params, self.hidden_dim_ext, self.space_planes, self.time_planes, self.num_regions, 
                                    depth_nodes=self.depth_nodes, depth_edges=self.depth_edges, dropout=0., bias=self.bias, use_norm=self.use_norm, 
                                    variational=self.variational, shared_mlp=False, prototypes=self.prototypes)

        # External data embedding
        # External data [batch_size, external_dim]
        if self.external_dim > 0:
            self.embedding_ext = nn.Sequential(
                nn.Linear(self.external_dim, self.hidden_dim_ext, bias=self.bias),
                nn.BatchNorm1d(self.hidden_dim_ext),
                nn.SiLU()
                # nn.Linear(self.hidden_dim_ext, self.hidden_dim_ext, bias=self.bias),
                # nn.SiLU(),
                # nn.BatchNorm1d(self.hidden_dim_ext),
                )
        else:
            self.embedding_ext = lambda x: x


        # 2: Define the layer that will sove the latent dynamical process using a graph neural network

        #  STGCN_Diff / STGCN
        self.stgcn = STGCN(
            self.latent_dim, 
            self.latent_dim, 
            self.dyn_params,
            edges_intra_nf=self.space_planes, 
            edges_inter_nf=self.time_planes,
            name="STGCN",
            bias=self.use_bias_stmgcn,
            # bias=False,
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
    
        
        # 3: Classifier of the graph embeddings
        if class_dim > 0 and classify:
            # New approach; get the summary system state and just compute the logits
            # self.dyn_classifier = nn.Sequential(
            #     nn.Dropout(0.2),
            #     nn.Linear(self.hidden_dim, self.class_dim, bias=False),
            #     )

            # Classic approach 
            input_class_dim = self.hidden_dim
            # dropout_classifier = dropout * 3
            dropout_classifier = 0.4 #if dropout_classifier > 0.4 else dropout_classifier
            num_agg = 1
            # num_agg = self.class_dim
            self.dyn_classifier = Classifier(input_class_dim, self.hidden_dim, self.class_dim, num_regions=self.num_regions, 
                                             num_aggregations=num_agg, drop=dropout_classifier, use_region_id=use_regions, bias=True)

        # Predictor of global data
        if self.predict_external:
            self.aggregate_to_pred = nn.Sequential(
                nn.Linear(self.num_regions, 1, bias=self.bias),
                nn.Flatten(start_dim=1),
            )

            self.predict = nn.Sequential(
                nn.BatchNorm1d(self.latent_dim + self.dyn_params),
                nn.SiLU(),
                nn.Linear(self.latent_dim + self.dyn_params, 4, bias=self.bias),
            )
        
        # 4: Decoder of the graph embeddings [batch_size, h_dim, num_regions, frames]
        if self.decode_just_latent:            
            # self.decoder_dim = self.latent_dim # + self.num_regions
            self.decoder_dim = self.latent_dim + self.hidden_dim_ext
        else:
            self.decoder_dim = self.dyn_params + self.latent_dim + self.hidden_dim_ext
            # self.decoder_dim = self.dyn_params + self.latent_dim + 1 # + self.hidden_dim_ext

        self.decoder_dim = self.decoder_dim + 1 if self.use_time else self.decoder_dim
        self.decoder_dim = self.decoder_dim + self.num_regions if self.use_regions else self.decoder_dim

        use_decoder_norm = self.use_norm
        # use_decoder_norm = False
        self.decoder = DecoderModel(self.decoder_dim, out_dim, depth=self.depth_nodes, dropout=0., bias=self.bias, use_norm=use_decoder_norm, 
                                    variational=self.variational, shared_mlp=True)


        # Init the weights
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """ Initialize the weights of the linear layers """
        # gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            # torch.nn.init.constant_(m.weight, 1e-4)
            # torch.nn.init.xavier_uniform_(m.weight, gain=gain)

            # torch.nn.init.xavier_normal_(m.weight)

            # k = math.sqrt(1 / (m.weight.data.size(-1)))
            # torch.nn.init.uniform_(m.weight, -k, k)

            # torch.nn.init.normal_(m.weight)
            # torch.nn.init.constant_(m.weight, 1e-4)

            stdv = math.sqrt(6.0 / (m.weight.data.size(-2) + m.weight.data.size(-1)))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.xavier_normal_(m.weight, gain=gain)
            # torch.nn.init.constant_(m.weight, 1e-4)
            # stdv = math.sqrt(6.0 / (m.weight.data.size(-2) + m.weight.data.size(-1)))
            # m.weight.data.uniform_(-stdv, stdv)
            # torch.nn.init.uniform_(m.weight, -0.5, 0.5)
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
        # self.method = 'dopri5'  # 'rk4', 'dopri5', 'adaptive_heun'
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
            # self.options_solver = {'max_num_steps': 40}
            # self.options_solver = { }
            self.options_solver = {'step_t': np.arange(0, 1, self.dt_step),
                                   'max_num_steps': 500,
                                   'first_step': self.dt_step*0.5,
                                   }
        else:
            raise ValueError(f"Method {self.method} not implemented")


    def integrate_ode(self, graph, ctx_t, t, v, space_edge_data, time_edge_data, rids=None, ctx_edges=None, control_ctx=None):  # v = (L(x), z_)
        # Solver options        
        self.set_tol()  # Get the options for the integrator

        # Get the latent space and the control parameters
        L = v[:, :self.latent_dim]
        D = v[:, self.latent_dim:]
        # L = self.norm_h(L)  # Batch norm
        # a = self.prototye_classifier(D)
        # Project D into the sphere
        # The dimensions should be in the last axis
        # D = self.sphere_embed(D.permute(0, 2, 1)).permute(0, 2, 1)  

        # Assign node data to the graph
        space_edge_data = reshape_to_graph(space_edge_data.unsqueeze(-1))
        time_edge_data = reshape_to_graph(time_edge_data.unsqueeze(-1))
        graph['space'].edata['weight'] = space_edge_data.float() #.unsqueeze(-1)
        graph['time'].edata['weight'] = time_edge_data.float() #.unsqueeze(-1)

        # Set x0 initial condition and time
        # graph.ndata['h'] = reshape_to_graph(L.unsqueeze(-1))  # Latent space
        # graph.ndata['h_prev'] = h_graph[..., -1]
        x0 = reshape_to_graph(L.unsqueeze(-1)).squeeze().float()  # Latent space
        # x0 = self.norm_node(x0.unsqueeze(1)).squeeze()
        # graph.ndata['time'] = reshape_to_graph(t.unsqueeze(1)) # Time
        # graph.ndata['pseudo_labels'] = pseudo_labels.unsqueeze(-1)
        graph.ndata['time'] = reshape_to_graph(ctx_t).float() # Time
        graph.ndata['h_dyn'] = reshape_to_graph(D.unsqueeze(-1)).float()  # # Dynamical parameters
        graph.ndata['h_dyn_init'] = reshape_to_graph(D.unsqueeze(-1)).float()  # # Dynamical parameters
        if rids is not None:
            graph.ndata['rids'] = reshape_to_graph(rids.unsqueeze(-1)).float()
        self.stgcn.set_graph(graph, x0)
        
        # Add fine-grained control variables
        # ctrl_ctx = reshape_to_graph(control_ctx.permute(0, 3, 2, 1))
        # self.stgcn.set_control(ctx_t[0,0,0].type_as(x0), ctrl_ctx)

        # t = torch.tensor([0, 1])
        # t = torch.range(0, self.time_frames-1)
        # t = all_in_data[0, -1, 0, :]
        # t = t.type_as(x0)
        t_int = t[0, 0, 0].type_as(x0)
        t_int = t_int / t_int.max()
        # graph.ndata['sim_time'] = t_int[0]
        # t_int = torch.linspace(0, 1, steps=10).type_as(x0)
        # self.options_solver['step_size'] = 0.1
        # import time
        # start = time.time()
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
        # end = time.time()
        # print(f"Integration time: {end-start}")
        # z = torch.clamp(z, -10, 10)

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
        mu, sigma, mu_space, sigma_space, mu_time, sigma_time, mu_de, sigma_de, control_per_time = self.encoder(node_data, in_edge_space, in_edge_time, prototypes=self.prototypes)

        return mu, sigma, mu_space, sigma_space, mu_time, sigma_time, mu_de, sigma_de, g_ndata, control_per_time


    def xz_to_y(self, time, z, latent, g_ndata=None, rids=None, pos_data=None):
        # ============================== Reconstructed features
        batch_size = time.shape[0]
        # h_rec = reshape_to_tensor(latent.permute(1, 2, 0), batch_size)
        h_rec = latent

        # Normalize the latent space
        # h_rec = self.norm_hrec(h_rec) 
        # B x H x R x T -> B x T x H x R
        # self.norm_hrec = nn.LayerNorm([self.latent_dim, self.num_regions])
        # h_rec = self.norm_hrec(h_rec.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)        
        # h_rec = self.scale_out(h_rec)

        # Control variables / h_dyn
        h_dyn = z[:, self.latent_dim:]
        h_dyn = h_dyn.unsqueeze(-1).repeat(1, 1, 1, h_rec.shape[-1])  # Number of time-frames

        # # Decoder 
        # # h_rec - B x H x R x T
        # # hg - B x O
        # a = F.softmax(hg.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h_s.shape[2], self.time_frames), dim=1)

        # b = torch.cat((h_rec_dec, g_ndata), dim=1)
        # b = torch.cat((h_rec_dec, a), dim=1)
        # b = torch.cat((h_rec_dec, g_ndata), dim=1)
        # b = torch.cat((h_rec, g_ndata, h_times, h_dyn.repeat(1, 1, 1, self.time_frames)), dim=1)

        if self.decode_just_latent:
            h = h_rec            
        else:
            h = torch.cat((h_rec, h_dyn), dim=1)

        if g_ndata is not None:
            h = torch.cat((h, g_ndata), dim=1)
        
        if self.use_time:
            h = torch.cat((h, time), dim=1)

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

    
    def ld_to_class(self, h, latent, g_ndata=None, mu_space_node=None, mu_time_node=None, rids=None):
        # Get the class probabilities from the initial condition and the control variables
        # if g_ndata is not None:
        # #     # g_ndata = g_data.repeat(1, 1, num_regions, in_node_data.shape[-1])# [..., tgt_pts]
        #     h = torch.cat([h, g_ndata[..., 0]], dim=1)        
        # h_dyn = z[:, self.latent_dim:]
        # hg = self.dyn_classifier(h[:, self.latent_dim:])

        # # Reshape the latent to tensor and compute frequencies
        # batch_size = h.shape[0]
        # lat_rshp = reshape_to_tensor(latent.permute(1, 2, 0), batch_size)
        # freq_components = torch.fft.fft(lat_rshp, dim=-1)
        # freq_magnitudes = torch.abs(freq_components)[..., :10]  # Keep top-10 frequencies
        
        # # Now, reshape such that we have [batch_size, num_features*top_freqs, num_regions]
        # freq_magnitudes = freq_magnitudes.permute(0, 2, 1, 3)
        # freq_magnitudes = freq_magnitudes.reshape(batch_size, self.num_regions, -1)
        # freq_magnitudes = freq_magnitudes.permute(0, 2, 1)


        # Get the latent space and the control parameters
        is_summary = False
        if is_summary:
            h_to_class = h
        else:
            # It it is a sample or the mean initial state and context
            L = h[:, :self.latent_dim]
            D = h[:, self.latent_dim:]
            # Lnorm = self.norm_h(L.permute(0, 2, 1)).permute(0, 2, 1)   # Layer norm
            # Lnorm = self.norm_h(L)  # Batch norm
            Lnorm = L

            h_to_class = torch.cat((Lnorm, D), dim=1)

        # Add the frequencies
        # h_to_class = torch.cat((h_to_class, freq_magnitudes), dim=1)

        # Just take the mean of the latent space
        # hg = self.dyn_classifier(h_to_class.mean(dim=-1))        
        hg = self.dyn_classifier(h_to_class, region_id=rids)        

        return hg
    
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

        # Predict target points based on context
        h_dyn = reshape_to_tensor(latent_state['h_dyn_init'], graph.batch_size).squeeze().float()
        # h_dyn = reshape_to_tensor(self.stgcn.g.ndata['h_dyn_init'], graph.batch_size).squeeze()
        # h0 = reshape_to_tensor(x0.unsqueeze(-1), graph.batch_size).squeeze()
        # h0 = reshape_to_graph(L.unsqueeze(-1)).squeeze()  # Latent space
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
            # Define target and context points [ to train the KL ]

            # # Use the encoder internally to get the representation of both control and initial points
            # time_start = time.time()
            mu_context, sigma_context, mu_space, sigma_space, mu_time, sigma_time, mu_de, sigma_de, g_data, control_ctx = self.xy_to_mu_sigma(context_pts, in_time, in_node_data, in_edge_space, in_edge_time, global_data, rids, pos_data)
            # time_end = time.time()
            # encoder_time = time_end - time_start

            # Get the distribution for the space and time edges
            q_space_ctx = Normal(mu_space, sigma_space)
            q_time_ctx = Normal(mu_time, sigma_time)
            q_context = Normal(mu_context, sigma_context)
            # q_de = Normal(mu_de, sigma_de)
            # I might need the multivariate normal distribution
            # NOTE: For the moment assume distrbution is the same for all nodes -- each node has it's own distribution
            # q_context = MultiVariateNormal(mu_context, sigma_context)

            # Get edge normalization coefficients for each canonical edge type
            # time_start = time.time()
            for plane in graph.canonical_etypes:
                self.set_edges_norm(graph, plane, 'weight', use_ft=False, normalize=True)
            # time_end = time.time()
            # time_normalization = time_end - time_start

            if self.training:
                # Encode target 
                # Context needs to be encoded to calculate KL term
                mu_target, sigma_target, mu_space_tgt, sigma_space_tgt, mu_time_tgt, sigma_time_tgt, mu_de_tgt, sigma_de_tgt, g_data, control_tgt = self.xy_to_mu_sigma(target_pts, in_time, in_node_data, in_edge_space, in_edge_time, global_data, rids, pos_data)

                q_space_tgt = Normal(mu_space_tgt, sigma_space_tgt)
                q_time_tgt = Normal(mu_time_tgt, sigma_time_tgt)
                q_target = Normal(mu_target, sigma_target) # [N, HIDDEN, NODES]
                # q_de_tgt = Normal(mu_de_tgt, sigma_de_tgt)

                self.use_mse = False
                if self.use_mse:
                    # NOT VARIATIONAL, HERE WE ARE DETERMINISTIC
                    z_sample = q_target.mean
                    space_edge_sample = q_space_tgt.mean
                    time_edge_sample = q_time_tgt.mean
                    # ctx_edges = q_de_tgt.mean
                    ctx_edges = None
                else:
                    # Sample from encoded distribution using reparameterization trick
                    z_sample = q_target.rsample()
                    space_edge_sample = q_space_tgt.rsample()
                    time_edge_sample = q_time_tgt.rsample()
                    # ctx_edges = q_de_tgt.rsample()
                    ctx_edges = None
                
                # Get z
                time_start = time.time()
                tgt_time = in_time[..., target_pts]
                ctx_time = in_time[..., context_pts]
                latent = self.integrate_ode(graph, ctx_time, tgt_time, z_sample, space_edge_sample, time_edge_sample, rids, ctx_edges, control_tgt)
                latent = reshape_to_tensor(latent.permute(1, 2, 0), graph.batch_size).float()
                time_end = time.time()
                ode_time = time_end - time_start

                # force = torch.stack(self.stgcn.force, axis=0)
                # force = torch.stack(self.stgcn.dyn_list, axis=0)
                # force = force.permute(1, 3, 2, 0)
                spatial_norm = self.stgcn.force.float()
                temporal_norm = self.stgcn.dyn_list.float()
                symm_penalty = self.stgcn.symm_penalty.float()
                eig_penalty = self.stgcn.eig_penalty.float()
                acyc_penalty = self.stgcn.acyc_penalty.float()
                # force = torch.zeros_like(latent)

                # Get the classification -- using the initial condition and the control variables                                
                if self.classify:
                    h_to_class = z_sample
                    # h_to_class = reshape_to_tensor(self.stgcn.state_summary.unsqueeze(-1), graph.batch_size).squeeze()
                    hg = self.ld_to_class(h_to_class, latent, g_ndata=g_data, rids=rids)
                    # hg = self.ld_to_class(h_to_class, latent, g_ndata=g_data, rids=None)
                else:
                    device = graph.device
                    hg = torch.zeros((graph.batch_size, 5), device=device).float()

                # Predict external features
                if self.predict_external:
                    h_to_class = z_sample
                    p_global = self.predict_global(h_to_class)
                else:
                    device = graph.device
                    p_global = torch.zeros((graph.batch_size, 4), device=device).float()

                # Get parameters of output distribution
                # time_start = time.time()
                p_y_pred = self.xz_to_y(tgt_time, z_sample, latent, g_ndata=g_data, rids=rids, pos_data=pos_data)
                # time_end = time.time()
                # pred_time = time_end - time_start

                # The norm of the message
                # m_space = torch.stack(self.stgcn.message_norms['m_space'])
                # m_time = torch.stack(self.stgcn.message_norms['m_time'])

                # Get the latent state
                latent_state = self.stgcn.get_latent_state(self.stgcn.g, self.use_attention)
                latent_state['l_end'] = reshape_to_graph(latent[...,[-1]].clone().detach()).float()

                # print("Times: ", encoder_time, time_normalization, ode_time, pred_time)

                return hg, p_y_pred, latent, (spatial_norm, temporal_norm, symm_penalty, eig_penalty, acyc_penalty), q_target, q_context,(q_space_tgt, q_time_tgt), (q_space_ctx, q_time_ctx), p_global, latent_state
            else:
                # Sample from distribution based on context
                z_sample = q_context.rsample()
                space_edge_sample = q_space_ctx.rsample()
                time_edge_sample = q_time_ctx.rsample()
                # ctx_edges = q_de.rsample()
                ctx_edges = None

                # Get z and force
                tgt_time = in_time[..., target_pts]
                ctx_time = in_time[..., context_pts]

                # latent = self.integrate_ode(graph, ctx_time, tgt_time, z_sample, space_edge_data, time_edge_data)
                latent = self.integrate_ode(graph, ctx_time, tgt_time, mu_context, mu_space, mu_time, rids, ctx_edges, control_ctx)
                latent = reshape_to_tensor(latent.permute(1, 2, 0), graph.batch_size).float()

                if self.classify:
                    h_to_class = mu_context
                    # h_to_class = reshape_to_tensor(self.stgcn.state_summary.unsqueeze(-1), graph.batch_size).squeeze()
                    hg = self.ld_to_class(h_to_class, latent, g_ndata=g_data, rids=rids)
                    # hg = self.ld_to_class(h_to_class, latent, g_ndata=g_data, rids=None)
                else:
                    device = graph.device
                    hg = torch.zeros((graph.batch_size, 5), device=device).float()

                # Predict external features
                if self.predict_external:
                    h_to_class = mu_context
                    p_global = self.predict_global(h_to_class)
                else:
                    device = graph.device
                    p_global = torch.zeros((graph.batch_size, 4), device=device).float()

                # Predict target points based on context
                # p_y_pred = self.xz_to_y(tgt_time, z_sample, latent, g_ndata=g_data, rids=rids, pos_data=pos_data)
                p_y_pred = self.xz_to_y(tgt_time, mu_context, latent, g_ndata=g_data, rids=rids, pos_data=pos_data)

                # force = torch.stack(self.stgcn.force, axis=0)
                # force = torch.stack(self.stgcn.dyn_list, axis=0)
                # force = force.permute(1, 3, 2, 0)
                spatial_norm = self.stgcn.force.float()
                temporal_norm = self.stgcn.dyn_list.float()
                symm_penalty = self.stgcn.symm_penalty.float()
                eig_penalty = self.stgcn.eig_penalty.float()
                acyc_penalty = self.stgcn.acyc_penalty.float()
                # force = torch.zeros_like(latent)

                # The norm of the message
                # m_space = torch.stack(self.stgcn.message_norms['m_space'])
                # m_time = torch.stack(self.stgcn.message_norms['m_time'])

                # Get the latent state
                # latent_state = self.stgcn.latent_state
                latent_state = self.stgcn.get_latent_state(self.stgcn.g, self.use_attention)
                latent_state['l_end'] = reshape_to_graph(latent[...,[-1]].clone().detach()).float()

                return hg, p_y_pred, latent, (spatial_norm, temporal_norm, symm_penalty, eig_penalty, acyc_penalty), None, q_context, (None, None, None), (q_space_ctx, q_time_ctx), p_global, latent_state

