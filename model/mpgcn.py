import torch
import torch.nn as nn
from torch_geometric.utils import softmax

import dgl.function as fn
from functools import partial
from torch.nn import functional as F


def symmetry_penalty(A):
    """ Penalty for non-symmetric matrices """
    return torch.norm(A - A.conj().T, p='fro')


def eigenvalue_penalty(A, threshold=1):
    """ Penalty for eigenvalues larger than a threshold """
    eigvals = torch.linalg.eigvals(A)
    magnitude = torch.abs(eigvals)
    penalty = torch.sum(torch.relu(magnitude - threshold))
    return penalty


def acyclic_penalty(A):
    """ Penalty for cycles in the graph """
    # D is the number of nodes
    d = A.shape[0]

    # Hadamard of A 
    A_squared = A * A

    # Element-wise exponential
    exp_squared = torch.exp(A_squared)

    # Trace of the resulting matrix
    trace_value = torch.trace(exp_squared)

    # Compute penalty (trace - d)
    penalty = trace_value - d
    
    return penalty



def featurewise_cosine_similarity(x, y):
    """
    Compute cosine similarity for each feature dimension.
    :param x: Tensor of shape [num_edges, num_features]
    :param y: Tensor of shape [num_edges, num_features]
    :return: Tensor of shape [num_edges, num_features]
    """
    # Normalize inputs along the feature dimension
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # [num_edges, num_features]
    y_norm = y / (y.norm(dim=-1, keepdim=True) + 1e-8)  # [num_edges, num_features]

    # Compute feature-wise cosine similarity
    similarity = x_norm * y_norm  # Element-wise multiplication per feature
    return similarity


class MPGCN(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim, 
                 dyn_params,
                 edges_intra_nf=1, 
                 edges_inter_nf=1,
                 name="MPGCN", 
                 bias=True, 
                 dropout=0.0, 
                 use_attention = False, 
                 use_constant_edges = True,
                 use_time=True,
                 use_hdyn=True,
                 only_spatial=False,
                 add_source=False,
                 summarise_state=False,
                 use_norm=True,
                 use_diffusion=False,
                 use_einsum=False,
                 agg_type='sum',
                 compute_derivative=False,
                 ):
        """
        This is our layer, in which we combine the multiple message passing
        :param aggr_type: aggregator function for the messages at each 'plane' and between planes
        """
        super(MPGCN, self).__init__()
        self.name = name
        
        # Dimensions
        self.in_dim = in_dim  # Number dimensions in the node embedding / features  
        self.out_dim = out_dim  # Number of output dimensions (latent dim.)
        self.dyn_params = dyn_params  # Number of control parameters/dimensions
        self.add_source = add_source  # In the end "add" the initial state (f0)
        self.only_spatial = only_spatial # Use only "spatial" edges
        self.edges_inter_nf = edges_inter_nf  # Number of features in the inter-layer edges
        self.edges_intra_nf = edges_intra_nf  # Number of features in the intra-layer edges

        # Hidden nodes and edge features
        self.edges_hidden = 32
        self.nodes_hidden = 32                
        use_affine = True
        self.update_control = True
        self.norm_node = nn.Identity() # No node normalization

        # Options
        self.use_constant_edges = use_constant_edges  # Change the edges or not
        self.condition_on_time = use_time
        self.use_hdyn = use_hdyn
        self.use_attention = use_attention
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.distance_fn = nn.CosineSimilarity(dim=1, eps=1e-6)        
        self.noise_scale = nn.Parameter(torch.tensor([0.05]))  # Noise scale
        self.noise_scale.requires_grad = True

        # Aggregators
        self.use_einsum = use_einsum
        self.use_diffusion = use_diffusion
        self.agg_type = agg_type  # sum, mean or flatten [edge values]
        self.compute_derivative = compute_derivative
        self._reduce_sum = partial(torch.sum, dim=1)
        self._reduce_max = partial(torch.max, dim=1)
        self._reduce_mean = partial(torch.mean, dim=1)

        # ============================================= MLP transform/update the edges    
        self.in_edge_fts_space = self.edges_intra_nf + 3 if self.condition_on_time else self.edges_intra_nf
        if self.use_hdyn:
            # Use control params in the edge update
            self.in_edge_fts_space = self.in_edge_fts_space + self.dyn_params * 2

        # Learnable score for each dimension
        scores_dim = self.in_dim + self.dyn_params + 3 if self.condition_on_time else self.in_dim + self.dyn_params
        self.scores_time = nn.Sequential(nn.Identity(), nn.Linear(scores_dim, self.edges_inter_nf, bias=False), nn.Sigmoid())
        self.scores_space = nn.Sequential(nn.Identity(), nn.Linear(scores_dim, self.edges_intra_nf, bias=False), nn.Sigmoid())        

        norm_fn = lambda: nn.LayerNorm(self.edges_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()
        self.fc_space_edge = nn.Sequential(
            nn.Linear(self.in_edge_fts_space, self.edges_hidden, bias=self.bias),
            nn.GELU(),
            norm_fn(),
            nn.Dropout(dropout),
            nn.Linear(self.edges_hidden, self.edges_intra_nf, bias=self.bias),
            )
        
        self.in_edge_fts_time = self.edges_inter_nf + 3 if self.condition_on_time else self.edges_inter_nf
        if self.use_hdyn:
            self.in_edge_fts_time = self.in_edge_fts_time + self.dyn_params

        norm_fn = lambda: nn.LayerNorm(self.edges_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()
        self.fc_time_edge = nn.Sequential(
            nn.Linear(self.in_edge_fts_time, self.edges_hidden, bias=self.bias),
            nn.GELU(),
            norm_fn(),
            nn.Dropout(dropout),
            nn.Linear(self.edges_hidden, self.edges_inter_nf, bias=self.bias),
            )

        # ============================================= Attention options
        if self.use_attention:
            self.in_att = self.dyn_params
            self.att_dim_space = self.edges_intra_nf  # Imagine that this is used to define the graph used
            self.Qx_space = nn.Linear(self.in_att, self.att_dim_space, bias=self.bias)
            self.Kx_space = nn.Linear(self.in_att, self.att_dim_space, bias=self.bias)
            self.Vx_space = nn.Linear(self.in_att, self.att_dim_space, bias=self.bias)

            self.att_dim_time = self.edges_inter_nf  # Imagine that this is used to define the graph used
            self.Qx_time = nn.Linear(self.in_att, self.att_dim_time, bias=self.bias)
            self.Kx_time = nn.Linear(self.in_att, self.att_dim_time, bias=self.bias)
            self.Vx_time = nn.Linear(self.in_att, self.att_dim_time, bias=self.bias)

        # ========= Node update function
        self.include_state = True
        if self.use_diffusion:
            num_in_dim = self.in_dim * 1 # [h]
        else:            
            # Here, instead of computing the mean stack the messages
            if self.agg_type == 'flatten':
                num_in_dim = self.in_dim + self.in_dim * self.edges_inter_nf + self.in_dim * self.edges_intra_nf
            else:
                num_in_dim = self.in_dim * 3 # [h, dh_s, dh_t] 

        # Get the dimensions
        update_dim = num_in_dim + self.dyn_params + 3 if self.condition_on_time else num_in_dim + self.dyn_params
        control_dim = self.in_dim + self.dyn_params + 3 if self.condition_on_time else self.in_dim + self.dyn_params

        # shape: [num_nodes, num_features]
        norm_fn = lambda: nn.LayerNorm(self.nodes_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()        
        
        self.control_update = nn.Sequential(            
            nn.Linear(control_dim, self.nodes_hidden, bias=self.bias),
            nn.SiLU(),            
            nn.Linear(self.nodes_hidden, self.nodes_hidden, bias=self.bias),
            norm_fn(),
            nn.SiLU(),            
            nn.Dropout(dropout),
            nn.Linear(self.nodes_hidden, self.dyn_params, bias=self.bias),
        )

        norm_fn = lambda: nn.LayerNorm(self.nodes_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()
        self.fc_node = nn.Sequential(
            nn.Linear(update_dim, self.nodes_hidden, bias=self.bias),
            nn.GELU(),            
            nn.Linear(self.nodes_hidden, self.nodes_hidden, bias=self.bias),
            norm_fn(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.nodes_hidden, self.out_dim, bias=self.bias),
            )
        
        # --- Initialize
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """ Initialization of linear layers """
        if isinstance(m, nn.Linear):            
            torch.nn.init.xavier_normal_(m.weight)

            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def message_func(self, edges):
        """ Message function for the edges """
        # The batch node, will collect messages received for each node
        etype = edges._etype[1]
        
        # Get the edge features
        f_fij = edges._edge_data['weight'][..., 0]

        if etype == 'space':
            incoming_msg = edges.src['h']
        elif etype == 'time':
            incoming_msg = edges.src['h_prev']
        else:
            raise KeyError(f"Edge type {etype} not expected")
                
        if self.use_attention:
            m = torch.einsum('eh,ef->eh', incoming_msg, edges._edge_data['attention'] * f_fij)
        else:
            m = torch.einsum('eh,ef->eh', incoming_msg, f_fij)
        
        dict_return = {
            f'm_{etype}': m,
            f"m_sum_{etype}": f_fij,
            }

        return dict_return

    @staticmethod
    def reduce_func(nodes,
                    att_factor=1.,
                    reduce_method=[partial(torch.mean, dim=1)]):
        """ Reduction function for the received messages """                
        m_key = list(nodes.mailbox.keys())
        res_dict = {}
        if 'm_space' in m_key:
            res_dict['w_space'] = nodes.mailbox['m_sum_space'].mean(axis=1)                            
            res_dict['dh_space'] = nodes.mailbox['m_space'].sum(axis=1)            
        elif 'm_time' in m_key:
            res_dict['w_time'] = nodes.mailbox['m_sum_time'].mean(axis=1)            
            res_dict['dh_time'] = nodes.mailbox['m_time'].sum(axis=1)            
        else:
            raise KeyError(f"Message type {m_key} not expected")

        return res_dict

    @staticmethod
    def update_fn(flist):
        """ This is what combines the temporal and spatial messages """
        # start_time = time.time()
        # [(planes x message), nodes, nodes_features] 
        # Sum messages that are already reduced to each node
        res = torch.stack(flist, 0).sum(dim=0) 
        # end_time = time.time()
        # update_time = end_time - start_time
        # print(f"Update time: {update_time}")
        # So, basically, we stack the messages from the different planes obtaining: planes x nodes x features
        # And then sum them along the first dimensiofn, so we get: nodes x features, and each node receives the contribution from the different planes in which it appears
        # Now, if we have 2 sets of messages with the same id, i.e: h_s coming from two different planes -- we sum them

        return res

    def apply_node_func(self, g):
        """Apply the node function to the graph."""

        # 1: Get the spatial message
        dh_s = g.ndata['dh_space'] 
                
        # 2: Get the temporal message
        dh_t = g.ndata['dh_time']

        return dh_s, dh_t
    
    def set_control(self, ctx_t, control_ctx):
        self.control_times = ctx_t
        self.control_params = control_ctx

    @staticmethod
    def get_latent_state(graph, use_attention=False):
        # Update the latent state [to have acces to it]
        latent_state = {}
        # Node states
        latent_state['h'] = graph.ndata['h'].clone().detach()
        latent_state['h_prev'] = graph.ndata['h_prev'].clone().detach()
        # Node dynamics
        latent_state['h_dyn'] = graph.ndata['h_dyn'].clone().detach()
        latent_state['h_dyn_init'] = graph.ndata['h_dyn_init'].clone().detach()
        # Edge states
        latent_state['space'] = graph['space'].edata['weight'].clone().detach()
        latent_state['time'] = graph['time'].edata['weight'].clone().detach()
        if use_attention:
            latent_state['space_att'] = graph['space'].edata['attention'].clone().detach()
            latent_state['time_att'] = graph['time'].edata['attention'].clone().detach()
        
        # Store the latent states
        return latent_state

    def set_latent_state(self, latent_states, t=0):
        self.g.ndata['h'] = latent_states['h'].clone().detach()
        self.g.ndata['h_prev'] = latent_states['h_prev'].clone().detach()
        self.g.ndata['h_dyn'] = latent_states['h_dyn'].clone().detach()
        self.g.ndata['h_dyn_init'] = latent_states['h_dyn_init'].clone().detach()
        self.g['space'].edata['weight'] = latent_states['space'].clone().detach()
        self.g['time'].edata['weight'] = latent_states['time'].clone().detach()
        if self.use_attention:
            self.g['space'].edata['attention'] = latent_states['space_att'].clone().detach()
            self.g['time'].edata['attention'] = latent_states['time_att'].clone().detach()
        
        self.t = t
        self.nfe = 1 # To avoid the initialisation of the edges

    def set_graph(self, g, x0, t=0):
        # x0 shape: [num_nodes * num_graphs, num_features]
        g = g.local_var()
        self.g = g
        self.nfe = 0
        self.t = t
        self.x0 = x0.detach().clone()

        # ================== In case that we want to use symmetry
        # ===> NOTE! Assume the graph is the same all the time for all subjects
        # Get the unidirectional edges
        src_edges, dst_edges, eid = g['space'].edges(form='all')

        # Convert edges to tuples and sort them to filter duplicates
        src_edges = src_edges.cpu().numpy()
        dst_edges = dst_edges.cpu().numpy()
        edge_tuples = list(zip(src_edges, dst_edges))
        unique_edges = set()
        selected_indices = []
        complementary_indices = []
        for idx, (u, v) in enumerate(edge_tuples):
            if (v, u) not in unique_edges and (u, v) not in unique_edges:
                unique_edges.add((u, v))
                selected_indices.append(idx)
            else:
                complementary_indices.append(idx)

        self.space_idx = torch.tensor(selected_indices).to(g.device)
        self.space_idx_comp = torch.tensor(complementary_indices).to(g.device)
        
    def compute_attention(self, g, t_idx=0):
        kx_space = self.Kx_space(g.ndata['h_dyn'].squeeze().float())
        qx_space = self.Qx_space(g.ndata['h_dyn'].squeeze().float())
        vx_space = self.Vx_space(g.ndata['h_dyn'].squeeze().float())
    
        # Space
        src_edges, dst_edges = g['space'].edges()

        # z_src = kx[src_edges[self.space_idx]]
        # z_dst = qx[dst_edges[self.space_idx]]
        z_src = kx_space[src_edges]
        z_dst = qx_space[dst_edges]

        # lat_dist_space = self.attn_fc(torch.cat([z_src, z_dst], axis=1)).squeeze(-1).unsqueeze(-1)
        # lat_dist_space = featurewise_cosine_similarity(z_src, z_dst)
        lat_dist_space = self.distance_fn(z_src, z_dst).unsqueeze(1)

        # Compute attention
        # attention_space = softmax(lat_dist_space, dst_edges[self.space_idx])  # Attention coefficient per edge
        attention_space = softmax(lat_dist_space, dst_edges)  # Attention coefficient per edge // considering all incoming edges
        # space_values = vx[src_edges[self.space_idx]] * attention_space

        # attention_space = F.softmax(lat_dist_space, dim=1)  # Here each edge can take a differen path
        space_values = vx_space[src_edges] * attention_space

        # Time        
        kx_time = self.Kx_time(g.ndata['h_dyn'].squeeze().float())
        qx_time = self.Qx_time(g.ndata['h_dyn'].squeeze().float())
        vx_time = self.Vx_time(g.ndata['h_dyn'].squeeze().float())

        src_edges_time, dst_edges_time = g['time'].edges()

        # Distance between the nodes in time
        z_src_time = kx_time[src_edges_time]
        z_dst_time = qx_time[dst_edges_time]
        
        # lat_dist_time = self.attn_fc(torch.cat([z_src_time, z_dst_time], axis=1)).squeeze(-1).unsqueeze(-1)
        lat_dist_time = self.distance_fn(z_src_time, z_dst_time).unsqueeze(1)  # Or could be a dot product
        # lat_dist_time = featurewise_cosine_similarity(z_src_time, z_dst_time)
        # lat_dist_time = self.distance_fn(node_data_src_time, node_data_dst_time).unsqueeze(1)

        attention_time = softmax(lat_dist_time, dst_edges_time)
        # time_values = vx[src_edges_time] * attention_time
        # attention_time = F.softmax(lat_dist_time, dim=1)  # Here each edge can take a differen path
        time_values = vx_time[src_edges_time] * attention_time        

        # return space_values, time_values
        return attention_space, attention_time, space_values, time_values


    def compute_penalties(self, g):
        symm_penalty = 0
        eig_penalty = 0
        acyc_penalty = 0

        # Adjacency matrix of spatial edges
        src, dst = g['space'].edges()
        a = g.adjacency_matrix('space').to_dense()

        # Adjacency matrix of temporal edges
        src_time, dst_time = g['time'].edges()
        b = g.adjacency_matrix('time').to_dense()

        num_edges = max(self.edges_inter_nf, self.edges_intra_nf)
        for i in range(0, num_edges):
            if i < self.edges_intra_nf:
                a[src,dst] = g['space'].edata['weight'][:, i, 0]
                symm_penalty = symm_penalty + symmetry_penalty(a)
                eig_penalty = eig_penalty + eigenvalue_penalty(a)
                acyc_penalty = acyc_penalty + acyclic_penalty(a)

            if i < self.edges_inter_nf:
                b[src_time,dst_time] = g['time'].edata['weight'][:, i, 0]
                symm_penalty = symm_penalty + symmetry_penalty(b)
                eig_penalty = eig_penalty + eigenvalue_penalty(b)
                acyc_penalty = acyc_penalty + acyclic_penalty(b)

        return symm_penalty, eig_penalty, acyc_penalty


    def forward(self, t, x, dt=None):
        if isinstance(self.norm_node, nn.InstanceNorm1d):
            x = self.norm_node(x.unsqueeze(1)).squeeze()  # Instance norm
        else:
            x = self.norm_node(x) # Layer norm or any other

        # Update the graph data
        g = self.g
        g.ndata['h'] = x.squeeze().float()
        if self.nfe == 0:
            g.ndata['h_prev'] = x.squeeze().clone().float()

        # Update the edges
        if not self.use_constant_edges and self.nfe > 0:
            t_idx = 0  # We always use 0 since we don't store them, so it's always 0 the idx.

            # =========== Space
            src_edges, dst_edges = g['space'].edges()
            t_emb_edges = [t, torch.sin(t), torch.cos(t)]

            # Get the dynamical params.
            # h_dyn_space_src = g.ndata['h_dyn'][...,0][src_edges[self.space_idx]]
            # h_dyn_space_dst = g.ndata['h_dyn'][...,0][dst_edges[self.space_idx]]
            h_dyn_space_src = g.ndata['h_dyn'][...,0][src_edges]
            h_dyn_space_dst = g.ndata['h_dyn'][...,0][dst_edges]
            h_dyn_space = torch.cat([h_dyn_space_src, h_dyn_space_dst], axis=1)

            # Edge latent features
            # edge_weight_space = g['space'].edata['weight'][..., t_idx][self.space_idx]
            edge_weight_space = g['space'].edata['weight'][..., t_idx]

            # Time
            # frame_time_space = torch.ones_like(g.ndata['time'][...,0][src_edges[self.space_idx]]) * t            
            frame_time_space = torch.tensor(t_emb_edges, device=g.device).unsqueeze(0).repeat(g['space'].num_edges(), 1)

            # ============ Time 
            # Get the dynamical params.
            h_dyn_time = g.ndata['h_dyn'][...,0][g['time'].edges()[1]]  

            # Edge latent features
            edge_weight_time = g['time'].edata['weight'][..., t_idx]
            
            # Time
            # frame_time_time = torch.ones_like(g.ndata['time'][...,0][g['time'].edges()[1]]) * t            
            frame_time_time = torch.tensor(t_emb_edges, device=g.device).unsqueeze(0).repeat(g['time'].num_edges(), 1)

            # Combine the input data
            input_space_data = edge_weight_space
            input_time_data = edge_weight_time
            if self.use_hdyn:
                input_space_data = torch.cat([input_space_data, h_dyn_space], axis=1).float()
                input_time_data = torch.cat([input_time_data, h_dyn_time], axis=1).float()

            if self.condition_on_time:
                input_space_data = torch.cat([input_space_data, frame_time_space], axis=1).float()
                input_time_data = torch.cat([input_time_data, frame_time_time], axis=1).float()

            space_edges = self.fc_space_edge(input_space_data).unsqueeze(-1)
            time_edges = self.fc_time_edge(input_time_data).unsqueeze(-1)

            # If you want antisymmetric edges
            # g['space'].edata['weight'][self.space_idx] = space_edges
            # g['space'].edata['weight'][self.space_idx_comp] = -space_edges        

            g['space'].edata['weight'] = space_edges.float()
            g['time'].edata['weight'] = time_edges.float()
                        
        if self.use_attention:
            # Compute attention
            attention_space, attention_time, space_values, time_values = self.compute_attention(g, t_idx=0)
            # g['space'].edata['attention'] = attention_space.float()
            # g['time'].edata['attention'] = attention_time.float()
            g['space'].edata['attention'] = space_values.float()
            g['time'].edata['attention'] = time_values.float()
        
            # Normalize the edges
            g['space'].edata['weight'] = (g['space'].edata['weight'] * g['space'].edata['attention'].unsqueeze(-1) * g['space'].edata['norm_space'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_intra_nf
            g['time'].edata['weight'] = (g['time'].edata['weight'] * g['time'].edata['attention'].unsqueeze(-1) * g['time'].edata['norm_time'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_inter_nf
        else:
            # Normalize the edges
            g['space'].edata['weight'] = (g['space'].edata['weight'] * g['space'].edata['norm_space'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_intra_nf
            g['time'].edata['weight'] = (g['time'].edata['weight'] * g['time'].edata['norm_time'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_inter_nf
        
        # Define the message passing functions
        funcs = {}  # Dictionary to store the message passing functions    
        for ix_p, p in enumerate(g.canonical_etypes):
            etype = p[1]
            if self.use_einsum:
                # Here we need to use our message function
                funcs[f"{etype}"] = (self.message_func, fn.sum(f'm_{etype}', f'dh_{etype}'))
            else:
                if etype == 'space':
                    funcs[f"{etype}"] = (fn.u_mul_e('h', 'weight', f'm_{etype}'), fn.sum(f'm_{etype}', f'dh_{etype}'))                    
                elif etype == 'time':
                    funcs[f"{etype}"] = (fn.u_mul_e('h_prev', 'weight', f'm_{etype}'), fn.sum(f'm_{etype}', f'dh_{etype}'))
                else:
                    raise KeyError(f"Edge type {etype} not expected")
                
        # Stack: num_nodes x num_msgs x num_hidden x n_features edge ;; it stacks along the second dimension
        # Get the spatial and temporal derivative of each node.        
        g.multi_update_all(funcs, self.update_fn)
        dh_s, dh_t = self.apply_node_func(g)
        
        # Update information in the graph, nodes and edges
        g.ndata['dh_s'] = dh_s.float()
        g.ndata['dh_t'] = dh_t.float()

        # Get the dt
        dt = self.dt
        self.t = t

        # Time 
        if self.condition_on_time:
            # If we condition on time we use sin/cosine to compute a time-embedding
            t_emb_nodes = [t, torch.sin(t), torch.cos(t)]
            t_emb_nodes = torch.tensor(t_emb_nodes, device=g.device).unsqueeze(0).repeat(g.num_nodes(), 1)        
        else:
            t_emb_nodes = torch.zeros((g.num_nodes(), 0), device=g.device) # This works?

        if dt < 0:
            dt = torch.abs(dt)
            print("Negative time step")
        
        # Get the scores for each layer/edge/plane
        if self.agg_type == 'score':
            input_scores = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1)
            space_sc = self.scores_space(input_scores)
            time_sc = self.scores_time(input_scores)

            # Softmax the scores [edges, planes]
            space_sc = F.softmax(space_sc, dim=1)
            time_sc = F.softmax(time_sc, dim=1)

        # Obtain the final derivative by combining the spatial and 'temporal' derivatives
        if self.compute_derivative:
            if self.use_einsum:
                dh_s = dh_s - g.ndata['h']
                dh_t = g.ndata['h'] - dh_t
            else:
                h_tmp = g.ndata['h'].unsqueeze(1)
                dh_s = dh_s - h_tmp
                dh_t = h_tmp - dh_t
        else:
            pass
        
        # How to aggreagate the derivatives of the different edges/planes
        if self.use_einsum:
            pass
        else:
            if self.agg_type == 'sum':
                dh_s = dh_s.sum(axis=1)
                dh_t = dh_t.sum(axis=1)
            elif self.agg_type == 'mean':
                dh_s = dh_s.mean(axis=1)
                dh_t = dh_t.mean(axis=1)
            elif self.agg_type == 'score':
                a = space_sc.unsqueeze(-1) * dh_s
                b = time_sc.unsqueeze(-1) * dh_t
                dh_s = a.sum(axis=1)
                dh_t = b.sum(axis=1)
            elif self.agg_type == 'flatten':
                dh_s = dh_s.reshape(-1, self.edges_intra_nf * self.out_dim)
                dh_t = dh_t.reshape(-1, self.edges_inter_nf * self.out_dim)
            else:
                raise KeyError(f"Aggregation type {self.agg_type} not expected")
                
        if self.use_diffusion:
            # Use graph diffusion to obtain the updated value
            if self.only_spatial or self.nfe == 0:
                f = dh_s
            else:
                f = (dh_s + dh_t) / 2
            
            # Reaction term
            if self.include_state:
                input_data = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1)
                f_r = self.fc_node(input_data)
            else:
                # pass -- First step, no reaction term
                f_r = 0
            
            # New value
            f = f + f_r 
        else:
            # Use an MLP (combining the message passing updates and the control variable)
            input_data = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], dh_s, dh_t, t_emb_nodes], dim=1)
            f = self.fc_node(input_data.float())
        
        if self.training:
            # Brownian motion Weiner process -- to simulate random diffusion
            dW = torch.sqrt(dt) * torch.randn_like(f)  # Wiener increment
            f = f + torch.clamp(self.noise_scale, 0.005, 0.2) * dW
        
        # Add (or not) initial state to the update
        if self.add_source:
            f = f + self.x0.squeeze()

        # Update the data        
        g.ndata['h_prev'] = x.clone().float()

        # Update the control of the system  
        if self.update_control:
            input_control = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1).float()
            new_control = self.control_update(input_control).float()
            g.ndata['h_dyn'] = new_control.unsqueeze(-1)
        else:
            input_control = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1).float()
        
        # Just to have one, and prevent crashing
        if self.nfe == 0:            
            # Compute the L1 norm of the spatial and temporal edges
            norm_spatial = g['space'].edata['weight'].abs().sum(dim=(1,2))
            norm_temporal = g['time'].edata['weight'].abs().sum(dim=(1,2))

            # Initialize storage tensors
            self.norm_spatial = norm_spatial.unsqueeze(0)
            self.norm_temporal = norm_temporal.unsqueeze(0)
        else:
            norm_spatial = g['space'].edata['weight'].abs().sum(dim=(1,2))
            norm_temporal = g['time'].edata['weight'].abs().sum(dim=(1,2))

            # Update tensors dynamically
            self.norm_spatial = torch.cat([self.norm_spatial, norm_spatial.unsqueeze(0)])
            self.norm_temporal = torch.cat([self.norm_temporal, norm_temporal.unsqueeze(0)])

        self.nfe = self.nfe + 1        
        assert f.device == next(self.parameters()).device

        return f.float()