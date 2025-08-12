import math
import time
import torch
import torch.nn as nn
from torch_geometric.utils import softmax

import dgl.function as fn
from functools import partial
from torch.nn import functional as F

from model.layers import WIRE


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


class STGCN(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim, 
                 dyn_params,
                 edges_intra_nf=1, 
                 edges_inter_nf=1,
                 name="STGCN", 
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
        super(STGCN, self).__init__()
        self.name = name
        
        # Dimensions
        self.in_dim = in_dim  # Number dimensions in the node embedding / features  
        self.out_dim = out_dim  # Number of output dimensions
        # self.att_dim = self.in_dim // 2  # Attention dimensions
        self.dyn_params = dyn_params
        self.add_source = add_source
        self.only_spatial = only_spatial
        # self.only_spatial = True
        self.edges_inter_nf = edges_inter_nf  # Number of features in the inter-layer edges
        self.edges_intra_nf = edges_intra_nf  # Number of features in the intra-layer edges
        self.edges_hidden = 32
        self.nodes_hidden = 32
        # self.memory_dim = 10

        # Normalisation
        use_affine = True
        last_edges_layer = lambda: nn.Identity()  # Nothing or Tanh
        last_control_layer = lambda: nn.Identity()  # Nothing
        last_nodes_layer = lambda: nn.Identity()  
        # use_norm = False
        # self.norm_node = nn.LayerNorm(normalized_shape=self.in_dim, elementwise_affine=False)
        # self.norm_node = nn.InstanceNorm1d(num_features=1)
        self.norm_node = nn.Identity() # No normalization

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
        # self.in_edge_fts_space = self.edges_intra_nf + self.in_dim*2 + 1 if self.condition_on_time else self.edges_intra_nf + self.in_dim*2
        # self.in_edge_fts_space = self.edges_intra_nf + self.in_dim + 1 if self.condition_on_time else self.edges_intra_nf + self.in_dim
        self.in_edge_fts_space = self.edges_intra_nf + 3 if self.condition_on_time else self.edges_intra_nf
        if self.use_hdyn:
            self.in_edge_fts_space = self.in_edge_fts_space + self.dyn_params * 2
        # self.in_edge_fts_space = self.in_edge_fts_space + 1  # The distance between the nodes
        # self.in_edge_fts_space = self.in_edge_fts_space + self.att_dim # The values

        # Learnable score for each dimension
        # Sigmoid
        scores_dim = self.in_dim + self.dyn_params + 3 if self.condition_on_time else self.in_dim + self.dyn_params
        self.scores_time = nn.Sequential(nn.Identity(), nn.Linear(scores_dim, self.edges_inter_nf, bias=False), nn.Sigmoid())
        self.scores_space = nn.Sequential(nn.Identity(), nn.Linear(scores_dim, self.edges_intra_nf, bias=False), nn.Sigmoid())

        # Memory for f_g and f_nn balance
        # self.memory_predict = nn.Sequential(nn.Linear(self.memory_dim + self.dyn_params, 1, bias=False), 
        #                                     nn.Tanh(),
        #                                     )
        
        # self.memory_update = nn.Sequential(nn.Linear(self.memory_dim + self.dyn_params, self.memory_dim, bias=False),
        #                                    )

        norm_fn = lambda: nn.LayerNorm(self.edges_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.LayerNorm(self.edges_intra_nf, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.InstanceNorm1d(self.edges_hidden, affine=use_affine) if use_norm else nn.Identity()
        self.fc_space_edge = nn.Sequential(#nn.Dropout(dropout),
            # WIRE(self.in_edge_fts_space, self.edges_hidden, bias=self.bias, dropout=0.),
            nn.Linear(self.in_edge_fts_space, self.edges_hidden, bias=self.bias),
            # nn.Linear(self.in_edge_fts_space, self.edges_intra_nf, bias=self.bias),
            # nn.SiLU(),
            # norm_fn(),
            nn.GELU(),
            norm_fn(),
            nn.Dropout(dropout),
            nn.Linear(self.edges_hidden, self.edges_intra_nf, bias=self.bias),            
            # nn.SiLU(),
            # nn.Tanh(),
            )
        
        # self.in_edge_fts_time = self.edges_inter_nf + self.in_dim + 1 if self.condition_on_time else self.edges_inter_nf + self.in_dim
        # self.in_edge_fts_time = self.edges_inter_nf + self.in_dim + 1 if self.condition_on_time else self.edges_inter_nf + self.in_dim
        self.in_edge_fts_time = self.edges_inter_nf + 3 if self.condition_on_time else self.edges_inter_nf
        if self.use_hdyn:
            self.in_edge_fts_time = self.in_edge_fts_time + self.dyn_params
        # self.in_edge_fts_time = self.in_edge_fts_time + 1 # The distance between the nodes
        # self.in_edge_fts_time = self.in_edge_fts_time + self.att_dim # The values

        norm_fn = lambda: nn.LayerNorm(self.edges_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.LayerNorm(self.edges_inter_nf, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.InstanceNorm1d(self.edges_hidden, affine=use_affine) if use_norm else nn.Identity()
        self.fc_time_edge = nn.Sequential(#nn.Dropout(dropout),
            # WIRE(self.in_edge_fts_time, self.edges_hidden, bias=self.bias, dropout=0.),
            nn.Linear(self.in_edge_fts_time, self.edges_hidden, bias=self.bias),
            # nn.Linear(self.in_edge_fts_time, self.edges_inter_nf, bias=self.bias),
            # nn.SiLU(),
            # norm_fn(),
            nn.GELU(),
            norm_fn(),
            nn.Dropout(dropout),
            nn.Linear(self.edges_hidden, self.edges_inter_nf, bias=self.bias),            
            # nn.SiLU(),
            # nn.Tanh(),
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

        # ========= Node function
        self.include_state = True
        # num_in_dim = self.in_dim if self.only_spatial else self.in_dim*2
        # num_in_dim = num_in_dim + self.in_dim if self.include_state else 0
        if self.use_diffusion:
            num_in_dim = self.in_dim * 1 # [h]
        else:            
            # Here, instead of computing the mean stack the messages
            if self.agg_type == 'flatten':
                num_in_dim = self.in_dim + self.in_dim * self.edges_inter_nf + self.in_dim * self.edges_intra_nf
            else:
                num_in_dim = self.in_dim * 3 # [h, dh_s, dh_t] 
        # Time encoding      
        update_dim = num_in_dim + self.dyn_params + 3 if self.condition_on_time else num_in_dim + self.dyn_params
        
        # shape: [num_nodes, num_features]
        norm_fn = lambda: nn.LayerNorm(self.nodes_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.LayerNorm(self.dyn_params, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.InstanceNorm1d(self.nodes_hidden, affine=use_affine) if use_norm else nn.Identity()

        # dt + time_encoding
        control_dim = self.in_dim + self.dyn_params + 3 if self.condition_on_time else self.in_dim + self.dyn_params
        self.control_update = nn.Sequential(
            # WIRE(self.in_dim + self.dyn_params, self.nodes_hidden, bias=self.bias, dropout=0.),
            nn.Linear(control_dim, self.nodes_hidden, bias=self.bias),  # classic            
            # nn.Linear(self.in_dim + self.dyn_params, self.dyn_params, bias=self.bias),  # classic            
            # nn.SiLU(),
            # norm_fn(),
            nn.SiLU(),
            # nn.Linear(self.nodes_hidden, self.dyn_params, bias=self.bias),
            nn.Linear(self.nodes_hidden, self.nodes_hidden, bias=self.bias),
            norm_fn(),
            nn.SiLU(),
            # norm_fn(),
            nn.Dropout(dropout),
            nn.Linear(self.nodes_hidden, self.dyn_params, bias=self.bias),
            # nn.Tanh(),
        )

        norm_fn = lambda: nn.LayerNorm(self.nodes_hidden, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.LayerNorm(self.out_dim, elementwise_affine=use_affine) if use_norm else nn.Identity()
        # norm_fn = lambda: nn.InstanceNorm1d(self.nodes_hidden, affine=use_affine) if use_norm else nn.Identity()
        self.fc_node = nn.Sequential(
            # WIRE(update_dim, self.nodes_hidden, bias=self.bias, dropout=0.),
            # nn.Linear(update_dim, self.nodes_hidden, bias=self.bias),
            nn.Linear(update_dim, self.nodes_hidden, bias=self.bias),
            # nn.GELU(),
            # norm_fn(),
            nn.GELU(),
            # nn.Linear(self.nodes_hidden, self.out_dim, bias=self.bias),
            nn.Linear(self.nodes_hidden, self.nodes_hidden, bias=self.bias),
            norm_fn(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.nodes_hidden, self.out_dim, bias=self.bias),
            # nn.Tanh(),
            )

        # ======= Summary state of the system
        # At each iteration, we summarise the state of the system
        self.summarise_state = summarise_state
        summary_dim = self.out_dim + self.dyn_params
        if self.summarise_state:
            self.f_summary = nn.Linear(summary_dim, summary_dim, bias=self.bias)
            self.f_previous = nn.Linear(summary_dim, summary_dim, bias=self.bias)

        # --- Initialize
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """ Initialization of linear layers """
        if isinstance(m, nn.Linear):
            # torch.nn.init.constant_(m.weight, 1e-4)
            # gain = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

            # k = math.sqrt(1 / (m.weight.data.size(-1)))
            # torch.nn.init.uniform_(m.weight, -k, k)
            # stdv = math.sqrt(6.0 / (m.weight.data.size(-2) + m.weight.data.size(-1)))
            # m.weight.data.uniform_(-stdv, stdv)

            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def edge_attention(self, edges):
        """ Compute the attention coefficients """
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        a_es = self.attn_fc_es(z2)
        a_et = self.attn_fc_et(z2)
        return F.relu(a), F.relu(a_es), F.relu(a_et)

    def message_func(self, edges):
        """ Message function for the edges """
        # start_time = time.time()
        # The batch node, will collect messages received for each node
        etype = edges._etype[1]
        
        # Get the edge features
        # start_acces = time.time()
        f_fij = edges._edge_data['weight'][..., 0]
        # num_edge_features = f_fij.shape[1]
        # norm = edges._edge_data[f'norm_{etype}']  # Normalization factor, based on the symmetric adjaceny matrix
        # end_acces = time.time() - start_acces


        # Normalize the edge features
        # f_fij = f_fij * norm[..., None]  #norm.unsqueeze(-1) 
        
        # Edge contribution normalization
        # f_fij = f_fij / f_fij.sum(dim=1, keepdim=True)  # Worse
        # f_fij = f_fij / num_edge_features
        # start_edge_norm = time.time()
        # f_fij = (f_fij * norm[..., None]) / num_edge_features
        # time_edge_norm = time.time() - start_edge_norm

        if etype == 'space':
            # planes_att = softmax_planes(f_fij)  # Plane attention coefficients
            incoming_msg = edges.src['h']
            # node_state = edges.dst['h']

            # Attraction - repulsion / force-directed networks -- we want to minimze the total force in the network f22
            # Now, the incoming_msg is also the force directional vector, so we have diffusion + forces
            # K = 0.5 # Natural spring length
            # C = 1 # Relative strenght of attraction and repulsion forces
            # p = 2 # P-controls long-range interactions, reduced by bigger p
            # attraction_force = ((edges.src['h'] - edges.dst['h']) ** 2) / K
            # repulsive_force = -(C * K ** (1 + p)) / ((edges.src['h'] - edges.dst['h']) ** p + 1e-3)
            # combined_force = repulsive_force * incoming_msg + attraction_force * incoming_msg
        elif etype == 'time':
            # planes_att = softmax_planes(f_fij)
            incoming_msg = edges.src['h_prev']
            # incoming_msg = torch.sin(edges.src['h'] - edges.src['h_prev'])
            # incoming_msg = edges.src['h'] - edges.src['h_prev']  # x(t) - x(t-1)

            # K = 0.5 # Natural spring length
            # C = 1 # Relative strenght of attraction and repulsion forces
            # p = 2 # P-controls long-range interactions, reduced by bigger p
            # attraction_force = ((edges.src['h_prev'] - edges.dst['h']) ** 2) / K
            # repulsive_force = -(C * K ** (1 + p)) / ((edges.src['h_prev'] - edges.dst['h']) ** p + 1e-3)
            # combined_force = repulsive_force * incoming_msg + attraction_force * incoming_msg
        else:
            raise KeyError(f"Edge type {etype} not expected")
        
        # start_ein = time.time()
        if self.use_attention:
            m = torch.einsum('eh,ef->eh', incoming_msg, edges._edge_data['attention'] * f_fij)  # Diffusion term - just the laplacian
        else:
            m = torch.einsum('eh,ef->eh', incoming_msg, f_fij)  # Diffusion term - just the laplacian        
        # einsum_time = time.time() - start_ein
        
        dict_return = {
            f'm_{etype}': m,
            f"m_sum_{etype}": f_fij,
            }
        # dict_return['e'] = e  # Attention / for the aggregation of the messages

        # if etype == 'space':
        #     pass
        #     # m_state = torch.einsum('eh,ef->eh', node_state.squeeze(), f_fij)
        #     # dict_return['m_state'] = m_state
        # elif etype == 'time':
        #     pass
        # else:
        #     raise KeyError(f"Edge type {etype} not expected")
        
        # end_time = time.time()
        # message_time = end_time - start_time
        # print(f"Message time: {message_time}, Einsum time: {einsum_time}, Edge norm time: {time_edge_norm}, Acces time: {end_acces}")

        return dict_return

    @staticmethod
    def reduce_func(nodes,
                    att_factor=1.,
                    reduce_method=[partial(torch.mean, dim=1)]):
        """ Reduction function for the received messages """
        # THe att_factor depends on the number of time-frames that are connected 
        # start_time = time.time()
        m_key = list(nodes.mailbox.keys())
        res_dict = {}
        # res_dict = {'dh_s': [], 
        #             'dh_t': [],
        #             'w_space': [],
        #             'w_time': [],
        #             # 'w_state': [],
        #             # 'force': [],
        #             # 'dyn': [],
        #             }
        
        # if 'e' in m_key:
        #     alpha = F.softmax(nodes.mailbox['e'], dim=1)
        #     # alpha = nodes.mailbox['e']
        #     m_key.remove('e')
        # else:
        #     alpha= None    

        if 'm_space' in m_key:
            # if 'm_sum_space' in m_key:
            # w_space = nodes.mailbox['m_sum_space']
            # m_key.remove('m_sum_space')
            # res_dict['w_space'] = [w_space.mean(axis=1)] # Sum of the messages that each node receives
            res_dict['w_space'] = nodes.mailbox['m_sum_space'].mean(axis=1)
                # Add the 'identity' term

            # L = nodes.mailbox['m_space'].sum(axis=1)
            res_dict['dh_space'] = nodes.mailbox['m_space'].sum(axis=1)
            # res_dict['dh_s'].append(L)
        elif 'm_time' in m_key:
            # if 'm_sum_time' in m_key:
            # w_time = nodes.mailbox['m_sum_time']
            # m_key.remove('m_sum_time')
            # res_dict['w_time'] = [w_time.mean(axis=1)]
            res_dict['w_time'] = nodes.mailbox['m_sum_time'].mean(axis=1)
            # T = nodes.mailbox['m_time'].sum(axis=1)
            res_dict['dh_time'] = nodes.mailbox['m_time'].sum(axis=1)
            # res_dict['dh_t'].append(T)
        else:
            raise KeyError(f"Message type {m_key} not expected")
        
        # for m in m_key:
        #     if m == 'm_state':
        #         continue
        #     # print(f"Reduction function {m}")            
        #     # HERE I APPLY THE REDUCE METHOD, TO AGGREGATE THE MESSAGES FROM THE NEIGHS.
        #     # So, we first aggreageat the spatial / temporal message before combining it
        #     # for method in reduce_method:
        #     #     # Dimensions of the mailbox are: Nodes x Messages x Feature_space
        #     #     # reduce_method[0]((alpha * nodes.mailbox['m_space']))
        #     #     # num_planes = nodes.mailbox[f'{m}'].shape[1]
        #     #     # if alpha is not None:
        #     #     #     # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        #     #     #     res = method((alpha * nodes.mailbox[f'{m}']))
        #     #     # else:
        #     #     #     res = method(nodes.mailbox[f'{m}'])  

        #     #     # if method.func == torch.max:
        #     #     #     # It returns the max values and the indices, only interested in the max values
        #     #     #     res = res[0]
                                
        #     if 'space' in m:
        #         # This is already normalized based on the connectivity
        #         # L = nodes.mailbox['m_space'].mean(axis=1)
        #         L = nodes.mailbox['m_space'].sum(axis=1)
        #         res_dict['dh_s'].append(L)
        #         # if alpha is not None:
        #         #     force = torch.mean((alpha * nodes.mailbox['m_force']), axis=1)
        #         # else:

        #         # Mean of energy force
        #         # force = nodes.mailbox['m_force'].mean(axis=1)
        #         # # force = nodes.mailbox['m_force'].sum(axis=1)
        #         # res_dict['force'].append(force)

        #         # Mean of neigh's dynamical parameters
        #         # Dyn = nodes.mailbox['m_dyn'].sum(axis=1)
        #         # res_dict['dyn'].append(Dyn)
                
        #     elif 'time' in m:
        #         # T = nodes.mailbox['m_time'].mean(axis=1)
        #         T = nodes.mailbox['m_time'].sum(axis=1)
        #         res_dict['dh_t'].append(T)
        #         # if alpha is not None:
        #         #     force = torch.mean((alpha * nodes.mailbox['m_force']), axis=1)
        #         # else:
        #         #     force = nodes.mailbox['m_force'].mean(axis=1)
        #         # res_dict['force'].append(force)

        #     elif 'state' in m:
        #         pass

        #     else:
        #         pass
        #         # raise KeyError(f"Message type {m} not expected")

        # Stack the messages
        # if len(res_dict['dh_s']) > 0:
        #     res_dict['dh_s'] = torch.hstack(res_dict['dh_s'])  # num_nodes x (num_reduce_methods * num_features)
        # else:
        #     _ = res_dict.pop('dh_s')
        
        # if len(res_dict['force']) > 0:
        #     res_dict['force'] = torch.hstack(res_dict['force'])
        # else:
        #     _ = res_dict.pop('force')

        # if len(res_dict['dyn']) > 0:
        #     res_dict['dyn'] = torch.hstack(res_dict['dyn'])
        # else:
        #     _ = res_dict.pop('dyn')

        # if len(res_dict['dh_t']) > 0:
        #     res_dict['dh_t'] = torch.hstack(res_dict['dh_t'])  # num_nodes x (num_reduce_methods * num_features)
        # else:
        #     _ = res_dict.pop('dh_t')
        
        # if len(res_dict['w_space']) > 0:            
        #     res_dict['w_space'] = torch.hstack(res_dict['w_space'])
        #     # res_dict['w_state'] = torch.hstack(res_dict['w_state'])
        # else:
        #     _ = res_dict.pop('w_space')
        #     # _ = res_dict.pop('w_state')
        
        # if len(res_dict['w_time']) > 0:
        #     res_dict['w_time'] = torch.hstack(res_dict['w_time'])
        # else:
        #     _ = res_dict.pop('w_time')

        # end_time = time.time()
        # reduce_time = end_time - start_time
        # print(f"Reduce time: {reduce_time}")

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

        # 1: Get the spatial derivative at this iteration [the laplacian]
        # As - h [~forward difference] -- diffusion, 'smoothing' of the signal
        # dh_s = g.ndata['dh_s'] - g.ndata['h']         
        dh_s = g.ndata['dh_space'] 
                
        # 2: Get the temporal derivative at this iteration [this is basically the derivative w.r.t to the previous time frame]      
        # h - At [backward difference]
        # dh_t = g.ndata['h'] - g.ndata['dh_t']
        dh_t = g.ndata['dh_time']

        # 3: Get the regularization terms
        # space_reg = g.ndata['w_space'].sum() / g.num_nodes()
        # time_reg = g.ndata['w_time'].sum() / g.num_nodes()
        space_reg = []
        time_reg = []

        # 4: Get the force of the system
        # force = g.ndata['force']
        force = []

        # 5: Get the dynamical parameters
        # dyn = g.ndata['dyn']
        dyn = []

        return dh_s, dh_t, space_reg, time_reg, force, dyn
    
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
        if 'memory' in graph.ndata:
            latent_state['memory'] = graph.ndata['memory'].clone().detach()
        if 'summary_state' in graph.ndata:
            latent_state['summary_state'] = graph.ndata['summary_state'].clone().detach()
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
        if 'memory' in self.g.ndata:
            self.g.ndata['memory'] = latent_states['memory'].clone().detach()            
        if 'summary_state' in latent_states:
            self.g.ndata['summary_state'] = latent_states['summary_state'].clone().detach()
        self.t = t
        self.nfe = 1 # To avoid the initialisation of the edges

    def set_graph(self, g, x0, t=0):
        # x0 shape: [num_nodes * num_graphs, num_features]

        g = g.local_var()

        self.g = g
        self.force = []
        self.dyn_list = []
        self.message_norms = {'m_space': [], 'm_time': []}
        self.nfe = 0
        self.t = t
        
        self.x0 = x0.detach().clone()
        # self.x0 = x0.clone()
        # self.x0 = self.norm_node(x0.unsqueeze(1).clone()).squeeze()  # Layer norm
        # self.x0 = self.norm_node(x0.unsqueeze(1)).squeeze()  # Layer norm

        # self.x0 = self.norm_node(x0)  # Batch norm
        # self.x0 = self.norm_node_data(x0.T).T

        # === Since this is the same all the time
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
        
        funcs = {}  # Dictionary to store the message passing functions
        
        if isinstance(self.norm_node, nn.InstanceNorm1d):
            # x_transposed = x.permute(0, 2, 1)  # [Batch, Features, Nodes]
            # x_normalized = instance_norm(x_transposed)
            # x_normalized = x_normalized.permute(0, 2, 1)
            x = self.norm_node(x.unsqueeze(1)).squeeze()  # Instance norm
        else:
            x = self.norm_node(x) # Layer norm or any other
        # x = self.norm_node(self.g, x).squeeze()  # Graph norm
        # x = self.norm_node(x).squeeze() # Batch Norm
        # x = torch.sin(x)

        # x = x.clone()        
        # x = self.norm_node(x)  # Batch norm
        # g.ndata['h'] = self.norm_node_data(x.T).T

        # Update the data
        g = self.g
        g.ndata['h'] = x.squeeze().float()
        if self.nfe == 0:
            # g.ndata['h_prev'] = x.squeeze().detach().clone().float()
            g.ndata['h_prev'] = x.squeeze().clone().float()
            # Initialize memory states
            # g.ndata['memory'] = torch.ones((g.num_nodes(), self.memory_dim)).to(g.device)
            # g.ndata['m_state'] = torch.zeros((g.num_nodes(), self.message_dim)).to(g.device)
            # g.ndata['m_prev'] = torch.zeros((g.num_nodes(), self.message_dim)).to(g.device)
            
        # if self.nfe == 0:
            # Be sure the spatial edges are consistent
            # space_edges = g['space'].edata['weight'][..., 0][self.space_idx][..., None]#.unsqueeze(-1)

            # g['space'].edata['weight'][self.space_idx] = space_edges
            # # g['space'].edata['weight'][self.space_idx_comp] = g['space'].edata['weight'][self.space_idx_comp].detach()

            # g['space'].edata['weight'][self.space_idx_comp] = space_edges
            # # g['space'].edata['weight'][self.space_idx_comp] = -space_edges

            # if self.use_attention:
            #     att_space, att_time = self.compute_attention(g, t_idx=0)
            #     # space_att = torch.zeros((g['space'].num_edges(), self.att_dim))
            #     g['space'].edata['attention'] = torch.zeros_like(g['space'].edata['norm_space'])
            #     g['space'].edata['attention'][self.space_idx] = att_space[..., 0]#.squeeze()
            #     g['space'].edata['attention'][self.space_idx_comp] = att_space[..., 0]#.squeeze()
            #     g['time'].edata['attention'] = torch.zeros_like(g['time'].edata['norm_time'])
            #     g['time'].edata['attention'] = att_time[..., 0]#.squeeze()

        t_idx = 0
        # Update the edges
        if not self.use_constant_edges and self.nfe > 0:
            # start_time = time.time()
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
            # frame_time_space = torch.ones_like(g.ndata['time'][...,0][src_edges]) * t
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
            # end_time = time.time()
            # update_edges_time = end_time - start_time

            # g['space'].edata['weight'][self.space_idx] = space_edges
            # # g['space'].edata['weight'][self.space_idx_comp] = -space_edges
            # g['space'].edata['weight'][self.space_idx_comp] = space_edges
            g['space'].edata['weight'] = space_edges.float()
            g['time'].edata['weight'] = time_edges.float()
                
        # if self.use_attention and self.nfe == 0:
        if self.use_attention:
            attention_space, attention_time, space_values, time_values = self.compute_attention(g, t_idx=0)
            # g['space'].edata['attention'] = attention_space.float()
            # g['time'].edata['attention'] = attention_time.float()
            g['space'].edata['attention'] = space_values.float()
            g['time'].edata['attention'] = time_values.float()

        if self.use_attention:
            # Normalize the edges
            g['space'].edata['weight'] = (g['space'].edata['weight'] * g['space'].edata['attention'].unsqueeze(-1) * g['space'].edata['norm_space'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_intra_nf
            g['time'].edata['weight'] = (g['time'].edata['weight'] * g['time'].edata['attention'].unsqueeze(-1) * g['time'].edata['norm_time'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_inter_nf
        else:
            # Normalize the edges
            g['space'].edata['weight'] = (g['space'].edata['weight'] * g['space'].edata['norm_space'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_intra_nf
            g['time'].edata['weight'] = (g['time'].edata['weight'] * g['time'].edata['norm_time'].unsqueeze(-1).unsqueeze(-1)).float() / self.edges_inter_nf
        
        # Define the message passing functions        
        for ix_p, p in enumerate(g.canonical_etypes):
            etype = p[1]
            # reduce = partial(self.reduce_func, att_factor=1, reduce_method=[self._reduce_mean])
            # funcs[f"{etype}"] = (self.message_func, reduce)
            if self.use_einsum:
                funcs[f"{etype}"] = (self.message_func, fn.sum(f'm_{etype}', f'dh_{etype}'))
            else:
                if etype == 'space':
                    funcs[f"{etype}"] = (fn.u_mul_e('h', 'weight', f'm_{etype}'), fn.sum(f'm_{etype}', f'dh_{etype}'))
                    # funcs[f"{etype}"] = (fn.u_mul_e('m_state', 'weight', f'm_{etype}'), fn.sum(f'm_{etype}', f'dh_{etype}'))
                elif etype == 'time':
                    funcs[f"{etype}"] = (fn.u_mul_e('h_prev', 'weight', f'm_{etype}'), fn.sum(f'm_{etype}', f'dh_{etype}'))
                    # funcs[f"{etype}"] = (fn.u_mul_e('m_prev', 'weight', f'm_{etype}'), fn.sum(f'm_{etype}', f'dh_{etype}'))
                else:
                    raise KeyError(f"Edge type {etype} not expected")
                
        # Stack: num_nodes x num_msgs x num_hidden x n_features edge ;; it stacks along the second dimension
        # Get the spatial and temporal derivative of each node. 
        # start_time = time.time()
        g.multi_update_all(funcs, self.update_fn)
        # end_time = time.time()
        # update_time = end_time - start_time

        # start_time = time.time()
        dh_s, dh_t, space_reg, time_reg, force, dyn = self.apply_node_func(g)
        # end_time = time.time()
        # apply_time = end_time - start_time
        
        # Update information in the graph, nodes and edges
        g.ndata['dh_s'] = dh_s.float()
        g.ndata['dh_t'] = dh_t.float()
        # g.ndata['m_prev'] = g.ndata['m_state']
        # g.ndata['m_state'] = dh_s        

        # Get the dt
        # dt = t - self.t
        dt = self.dt
        self.t = t

        # Time 
        if self.condition_on_time:
            t_emb_nodes = [t, torch.sin(t), torch.cos(t)]
            t_emb_nodes = torch.tensor(t_emb_nodes, device=g.device).unsqueeze(0).repeat(g.num_nodes(), 1)        
        else:
            t_emb_nodes = torch.zeros((g.num_nodes(), 0), device=g.device) # This works?

        if dt < 0:
            dt = torch.abs(dt)
            print("Negative time step")
        
        # Get the scores for each layer
        if self.agg_type == 'score':
            input_scores = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1)
            space_sc = self.scores_space(input_scores)
            time_sc = self.scores_time(input_scores)

            # Softmax the scores [edges, planes]
            space_sc = F.softmax(space_sc, dim=1)
            time_sc = F.softmax(time_sc, dim=1)

        # Update the memory
        # memory_input = torch.cat([g.ndata['memory'], g.ndata['h_dyn'][..., 0]], dim=1)
        # balance_update = self.memory_predict(memory_input)
        # # new_memory = g.ndata['memory'] + torch.sigmoid(g.ndata['memory']) * self.memory_update(memory_input)
        # new_memory = self.memory_update(memory_input)
        # g.ndata['memory'] = new_memory

        # Update times
        # start_time = time.time()

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
            if self.only_spatial or self.nfe == 0:
                f = dh_s
            else:
                f = (dh_s + dh_t) / 2
            
            # Reaction term
            if self.include_state:
                input_data = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1)
                f_r = self.fc_node(input_data)
                # f = self.fc_node(torch.cat([f, g.ndata['h_dyn'].squeeze(), g.ndata['h'], self.x0], dim=1))
                # f = self.fc_node(torch.cat([f, g.ndata['h_dyn'].squeeze(), g.ndata['h']], dim=1))
            else:
                # pass -- First step, no reaction term
                f_r = 0
            
            # New value
            # f = f + balance_update * f_r 
            f = f + f_r 
        else:
            # input_data = torch.cat([g.ndata['h_dyn'].squeeze(), g.ndata['h'], dh_s, dh_t, self.x0], dim=1)
            # # input_data = torch.cat([g.ndata['h'], dh_s, dh_t, self.x0], dim=1)        
            # input_data = torch.cat([g.ndata['h'], dh_s, dh_t], dim=1)
            # input_data = torch.cat([g.ndata['h_dyn'].squeeze(), g.ndata['h'], dh_s, dh_t], dim=1)        
            # input_data = torch.cat([g.ndata['h_dyn'].squeeze(), g.ndata['h'], g.ndata['dh_s'], g.ndata['dh_t']], dim=1)
            input_data = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], dh_s, dh_t, t_emb_nodes], dim=1)
            f = self.fc_node(input_data.float())

        # Sample noise, brownian motion Weiner process -- to simulate random diffusion
        # alpha = 0.05  # Noise factor
        
        if self.training:
            dW = torch.sqrt(dt) * torch.randn_like(f)  # Wiener increment
            f = f + torch.clamp(self.noise_scale, 0.005, 0.2) * dW #.detach()
            # f = f + (alpha * dW).detach()  # Add scaled noise
        
        if self.add_source:
            f = f + self.x0.squeeze()

        # if self.nfe > 0 and self.nfe % 10 == 0:
        #     self.x0 = x.squeeze().clone()

        # Update the data
        # g.ndata['h_prev'] = x.detach().clone().float()
        g.ndata['h_prev'] = x.clone().float()
                
        # Update the control
        # closest_ctrl = torch.argmin(self.control_times - self.t)
        # ext_ctrl = self.control_params[..., closest_ctrl]
        # distance_to_ctrl = torch.abs(self.control_times[closest_ctrl] - self.t)
        # distance_to_ctrl = torch.ones((g.num_nodes(), 1), device=distance_to_ctrl.device) * distance_to_ctrl
        # # input_control = torch.cat([g.ndata['h_dyn_init'].squeeze(), ext_ctrl, distance_to_ctrl, g.ndata['h']], dim=1)
        # input_control = torch.cat([g.ndata['h_dyn'].squeeze(), ext_ctrl, distance_to_ctrl, g.ndata['h']], dim=1)
        # new_control = self.control_update(input_control)
        # g.ndata['h_dyn'] = new_control.unsqueeze(-1)

        # dt_step = torch.ones((g.num_nodes(), 1), device=x.device) * dt
        # input_control = torch.cat([g.ndata['h_dyn'].squeeze(), g.ndata['h'], dt_step], dim=1)
        # input_control = torch.cat([g.ndata['h_dyn'].squeeze(), norm_x], dim=1)
        
        # Laplacian of D
        # dyn_L = g.ndata['h_dyn'].squeeze() - dyn

        update_control = True
        # dt_tensor = torch.ones((x.size(0), 1), device=x.device) * self.dt
        if update_control:
            input_control = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1).float()
            new_control = self.control_update(input_control).float()
            g.ndata['h_dyn'] = new_control.unsqueeze(-1)
        else:
            input_control = torch.cat([g.ndata['h_dyn'][..., 0], g.ndata['h'], t_emb_nodes], dim=1).float()
        # end_time = time.time()
        # update_control_time = end_time - start_time

        # print("Times: ", update_edges_time, update_time, apply_time, update_control_time)

        # Project D into the sphere
        # The dimensions should be in the last axis
        # D = self.sphere_embed(D.permute(0, 2, 1)).permute(0, 2, 1)  

        # Summarise the system state
        if self.summarise_state:
            if self.nfe == 0:
                g.ndata['summary_state'] = torch.zeros_like(input_control)
            previous_state = g.ndata['summary_state']
            new_state = torch.tanh(self.f_summary(input_control) + self.f_previous(previous_state)) # RNN like
            g.ndata['summary_state'] = new_state

        # Store the norm of the messages
        # self.message_norms['m_space'].append(g.ndata['dh_s'].norm())
        # self.message_norms['m_time'].append(g.ndata['dh_t'].norm())

        # Just to have one, and prevent crashing
        if self.nfe == 0:            
            # Compute the L1 norm of the spatial and temporal edges
            # symm_penalty, eig_penalty, acyc_penatly = self.compute_penalties(g)
            symm_penalty = torch.FloatTensor([0])
            eig_penalty = torch.FloatTensor([0])
            acyc_penatly = torch.FloatTensor([0])

            norm_spatial = g['space'].edata['weight'].abs().sum(dim=(1,2))
            norm_temporal = g['time'].edata['weight'].abs().sum(dim=(1,2))
            # norm_spatial = torch.FloatTensor([0])
            # norm_temporal = torch.FloatTensor([0])

            # Initialize storage tensors
            self.force = norm_spatial.unsqueeze(0)
            self.dyn_list = norm_temporal.unsqueeze(0)
            self.symm_penalty = symm_penalty.unsqueeze(0)
            self.eig_penalty = eig_penalty.unsqueeze(0)
            self.acyc_penalty = acyc_penatly.unsqueeze(0)
        else:
            # symm_penalty, eig_penalty, acyc_penatly = self.compute_penalties(g)
            symm_penalty = torch.FloatTensor([0])
            eig_penalty = torch.FloatTensor([0])
            acyc_penatly = torch.FloatTensor([0])

            norm_spatial = g['space'].edata['weight'].abs().sum(dim=(1,2))
            norm_temporal = g['time'].edata['weight'].abs().sum(dim=(1,2))
            # norm_spatial = torch.FloatTensor([0])
            # norm_temporal = torch.FloatTensor([0])

            # Update tensors dynamically
            self.force = torch.cat([self.force, norm_spatial.unsqueeze(0)])
            self.dyn_list = torch.cat([self.dyn_list, norm_temporal.unsqueeze(0)])
            self.symm_penalty = torch.cat([self.symm_penalty, symm_penalty.unsqueeze(0)])
            self.eig_penalty = torch.cat([self.eig_penalty, eig_penalty.unsqueeze(0)])
            self.acyc_penalty = torch.cat([self.acyc_penalty, acyc_penatly.unsqueeze(0)])

        self.nfe = self.nfe + 1
        if self.summarise_state:
            self.state_summary = g.ndata['summary_state']
        assert f.device == next(self.parameters()).device

        return f.float()