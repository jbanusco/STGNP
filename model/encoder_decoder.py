import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from model.spatial_batch_norm import BatchNorm


use_batch = False # False = LayerNorm
learn_affine = True

class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of input representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim, bias=False, activation=nn.LeakyReLU, shared_mlp=True, is_edges=False):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.activation = activation()
        self.shared_mlp = shared_mlp
        self.is_edges = is_edges
        if self.shared_mlp:
            # self.r_to_hidden = nn.Linear(r_dim, r_dim, bias=bias)
            self.hidden_to_mu = nn.Linear(r_dim, z_dim, bias=bias)
            self.hidden_to_sigma = nn.Linear(r_dim, z_dim, bias=bias)
        else:
            # Independent heads              
            self.hidden_to_mu = nn.ModuleList([nn.Linear(self.r_dim, 1) for _ in range(self.z_dim)])
            self.hidden_to_sigma = nn.ModuleList([nn.Linear(self.r_dim, 1) for _ in range(self.z_dim)])


    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        # hidden = self.activation(self.r_to_hidden(r))
        hidden = r
        if self.shared_mlp:
            mu = self.hidden_to_mu(hidden)            
            # Define sigma following convention in "Empirical Evaluation of Neural
            # Process Objectives" and "Attentive Neural Processes"
            sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(r))
        else:
            mu = torch.cat([layer(hidden) for layer in self.hidden_to_mu], dim=-1)
            sigma = torch.cat([layer(hidden) for layer in self.hidden_to_sigma], dim=-1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)

        if self.is_edges:
            mu = self.activation(mu)

        return mu, sigma


class EncoderModel(nn.Module):
    def __init__(self, in_dim, external_dim, space_dim, time_dim, latent_dim, dyn_params, hidden_dim_ext, space_planes, time_planes, num_regions, 
                 depth_nodes=2, depth_edges=2, dropout=0., bias=True, use_norm=False, shared_mlp=False):
        super().__init__()

        # === Define the model's parameters
        # Nodes
        self.in_dim = in_dim
        self.latent_dim = latent_dim  # Latent representation
        self.dyn_params = dyn_params  # Context
        self.num_regions = num_regions
        self.hidden_dim = self.latent_dim + self.dyn_params        

        # External
        self.external_dim = external_dim
        self.hidden_dim_ext = hidden_dim_ext

        # Edges
        self.intra_edges_dim = space_dim
        self.inter_edges_dim = time_dim
        self.space_planes = space_planes
        self.time_planes = time_planes

        # Options
        self.use_norm = use_norm
        self.depth_nodes = depth_nodes
        self.depth_edges = depth_edges
        self.bias = bias
        self.use_batch = use_batch
        self.shared_mlp = shared_mlp

        # 1. Embedding of the input data
        # ================================== Node embedding    
        self.norm_in_data = nn.Identity()
        in_dim = self.in_dim + self.hidden_dim_ext
        
        if self.use_batch:
            norm_fn = lambda: BatchNorm(4, self.hidden_dim, 3) if use_norm else nn.Identity()
        else:
            norm_fn = lambda: nn.LayerNorm(self.hidden_dim, elementwise_affine=learn_affine) if use_norm else nn.Identity()

        layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, self.hidden_dim, bias=self.bias),
                nn.SiLU(),
                ),
            ])

        for _ in range(self.depth_nodes - 1):
            layers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias),
                nn.SiLU(),
                ))
        self.embedding_nodes = nn.Sequential(*layers)

        # ================================== Context embedding
        if self.use_batch:
            norm_fn = lambda: BatchNorm(4, self.hidden_dim, 3) if use_norm else nn.Identity()
        else:
            norm_fn = lambda: nn.LayerNorm(self.hidden_dim, elementwise_affine=learn_affine) if use_norm else nn.Identity()
        ctx_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, self.hidden_dim, bias=self.bias),
                norm_fn(),
                nn.SiLU(),
                ),
            ])
        
        for _ in range(0, self.depth_nodes - 1):
            ctx_layers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias),
                norm_fn(),
                nn.SiLU(),
                ))                        
        self.context_encoder = nn.Sequential(*ctx_layers)  # Shared backbone

        # In case we want to use an aggregator for the 'r'
        self.aggregator = nn.Sequential(
            nn.Linear(4, 1),
        )

        # ================================= Edge embedding
        if self.use_batch:            
            norm_fn = lambda: BatchNorm(4, self.space_planes, 3) if use_norm else nn.Identity()            
        else:
            norm_fn = lambda: nn.LayerNorm(self.space_planes, elementwise_affine=learn_affine) if use_norm else nn.Identity()
        
        edge_dim_space = self.intra_edges_dim + self.hidden_dim_ext
        edge_sp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_dim_space, self.space_planes, bias=self.bias),                
                nn.SiLU(),
                ),
        ])

        for _ in range(self.depth_edges - 1):
            edge_sp_layers.append(nn.Sequential(
                nn.Linear(self.space_planes, self.space_planes, bias=self.bias),                
                nn.SiLU(),
                ))
        self.embedding_edges_space = nn.Sequential(*edge_sp_layers)

        if self.use_batch:            
            norm_fn = lambda: BatchNorm(4, self.time_planes, 3) if use_norm else nn.Identity()
        else:
            norm_fn = lambda: nn.LayerNorm(self.time_planes, elementwise_affine=learn_affine) if use_norm else nn.Identity()        

        edge_dim_time = self.inter_edges_dim + self.hidden_dim_ext
        edge_time_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_dim_time, self.time_planes, bias=self.bias),
                nn.SiLU(),
                ),
        ])

        for _ in range(self.depth_edges - 1):
            edge_time_layers.append(nn.Sequential(
                nn.Linear(self.time_planes, self.time_planes, bias=self.bias),
                nn.SiLU(),
                ))
        self.embedding_edges_time = nn.Sequential(*edge_time_layers)

        # ============================= Variational parameters                
        self.L_r_to_mu_sigma = MuSigmaEncoder(self.hidden_dim, self.latent_dim, bias=self.bias, shared_mlp=self.shared_mlp)
        self.D_r_to_mu_sigma = MuSigmaEncoder(self.hidden_dim, self.dyn_params, bias=self.bias, shared_mlp=self.shared_mlp)
        self.Espace_to_mu_sigma = MuSigmaEncoder(self.space_planes, self.space_planes, bias=self.bias, shared_mlp=self.shared_mlp, activation=nn.SiLU, is_edges=False)
        self.Etime_to_mu_sigma = MuSigmaEncoder(self.time_planes, self.time_planes, bias=self.bias, shared_mlp=self.shared_mlp, activation=nn.SiLU, is_edges=False)            


    def aggregate(self, r_i, time_dim=-1, use_aggregator=False):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_times, h_dim, num_regions)
        """
        mean_rep = torch.mean(r_i, dim=time_dim)
        if use_aggregator and r_i.shape[time_dim] > 1:            
            var_rep = torch.var(r_i, dim=time_dim)    
            max_rep = torch.max(r_i, dim=time_dim)
            min_rep = torch.min(r_i, dim=time_dim)
            combined_rep = torch.stack([mean_rep, var_rep, max_rep.values, min_rep.values], dim=-1)
            return self.aggregator(combined_rep).squeeze()
        else:
            return mean_rep


    def get_pseudo_labels(self, D, temperature=0.1):
        # Find closest prototype / pseudo-label
        embeddings = D.clone().permute(0, 2, 1).reshape(-1, D.shape[1])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        prototypes = F.normalize(self.prototypes, p=2, dim=1)

        # Contrastive loss (e.g., SwAV-like assignment)
        logits = torch.mm(embeddings, prototypes.T) / temperature # Shape [N, K]
        assignments = F.softmax(logits, dim=1)  # Probabilistic assignments / (N, K)
        pseudo_labels = assignments.argmax(dim=1)

        return pseudo_labels


    def forward(self, node_data, edge_data_space, edge_data_time):
        # Input data in shape [batch_size, num_features, num_nodes, num_times]
        node_data = self.norm_in_data(node_data)
        # Now, change it to [batch_size, num_times, num_nodes, num_features]
        node_data = node_data.permute(0, 3, 2, 1)
        # Same for the edges
        edge_data_space = edge_data_space.permute(0, 3, 2, 1)
        edge_data_time = edge_data_time.permute(0, 3, 2, 1)

        # --- NOTE: I could have an encoder for L_0 and another for the dynamics D
        h_l = self.embedding_nodes(node_data[:,[0],...])  # Embed the data to the hidden space -- representation of the nodes        
        L_r = self.aggregate(h_l, time_dim=1, use_aggregator=False)

        # If using the normal MLP
        h_d = self.context_encoder(node_data)  # Embed the data to the hidden space -- representation of the nodes
        D_r = self.aggregate(h_d, time_dim=1, use_aggregator=False)

        # Return parameters of distribution -- for each node
        L_mu, L_sigma = self.L_r_to_mu_sigma(L_r)        
        D_mu, D_sigma = self.D_r_to_mu_sigma(D_r)
        
        # Combine L and D in a single random vector
        mu = torch.cat([L_mu, D_mu], dim=-1).permute(0, 2, 1)
        sigma = torch.cat([L_sigma, D_sigma], dim=-1).permute(0, 2, 1)

        # === Embed the edges
        edge_embedding_space = self.embedding_edges_space(edge_data_space[:, [0]])
        edge_embedding_time = self.embedding_edges_time(edge_data_time[:, [0]])

        L_sp = self.aggregate(edge_embedding_space, time_dim=1, use_aggregator=False)
        L_time = self.aggregate(edge_embedding_time, time_dim=1, use_aggregator=False)        

        mu_space, sigma_space = self.Espace_to_mu_sigma(L_sp)
        mu_time, sigma_time = self.Etime_to_mu_sigma(L_time)        

        mu_space = mu_space.permute(0, 2, 1)
        sigma_space = sigma_space.permute(0, 2, 1)

        mu_time = mu_time.permute(0, 2, 1)
        sigma_time = sigma_time.permute(0, 2, 1)

        assert mu.device == next(self.parameters()).device
        assert sigma.device == next(self.parameters()).device
        assert mu_space.device == next(self.parameters()).device
        assert sigma_space.device == next(self.parameters()).device
        assert mu_time.device == next(self.parameters()).device
        assert sigma_time.device == next(self.parameters()).device        

        return mu, sigma, mu_space, sigma_space, mu_time, sigma_time
        

        


class DecoderModel(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, dropout=0, bias=True, use_norm=True, shared_mlp=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.bias = bias
        self.dropout = dropout
        self.use_batch = use_batch
        self.shared_mlp = shared_mlp


        self.norm_in_data = nn.Identity()        
        if self.use_batch:            
            norm_fn = lambda: BatchNorm(4, self.in_dim, 3) if use_norm else nn.Identity()
        else:
            norm_fn = lambda: nn.LayerNorm(self.in_dim) if use_norm else nn.Identity()  # Need the MLP format                
        
        layers = nn.ModuleList([])
        for _ in range(depth-1):                
            layers.append(nn.Sequential(
                nn.Linear(self.in_dim, self.in_dim, bias=self.bias),
                norm_fn(),
                nn.SiLU(),
                ))
        
        # Add always at least 1
        layers.append(nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=self.bias),
            norm_fn(),
            nn.SiLU(),
            ))
            
        if shared_mlp:
            self.hidden_to_mean = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.in_dim, self.out_dim, bias=self.bias))
            self.hidden_to_sigma = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.in_dim, self.out_dim, bias=self.bias))
        else:                
            self.hidden_to_mean = nn.ModuleList([
                nn.Sequential(nn.Dropout(dropout), nn.Linear(self.in_dim, 1, bias=self.bias)) for _ in range(out_dim)])
            self.hidden_to_sigma = nn.ModuleList([
                nn.Sequential(nn.Dropout(dropout), nn.Linear(self.in_dim, 1, bias=self.bias)) for _ in range(out_dim)])
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, hidden):
        # The hidden representation is in the shape [batch_size, hidden_dim, num_regions, num_times]        
        # Now, we need to reshape it to [batch_size, num_times, num_regions, hidden_dim]
        hidden = hidden.permute(0, 3, 2, 1)        
        hidden = self.decoder(hidden)

        if self.shared_mlp:
            # Shared MLP
            mu = self.hidden_to_mean(hidden).permute(0, 3, 2, 1)
            pre_sigma = self.hidden_to_sigma(hidden).permute(0, 3, 2, 1)
        else:
            # One MLP for each output dimension
            mu = torch.cat([layer(hidden) for layer in self.hidden_to_mean], dim=-1).permute(0, 3, 2, 1)
            pre_sigma = torch.cat([layer(hidden) for layer in self.hidden_to_sigma], dim=-1).permute(0, 3, 2, 1)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"        

        # NOTE: Now, the thing that we do is to treat each pair as an independent sample, so we need to reshape the data
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        assert mu.device == next(self.parameters()).device
        assert sigma.device == next(self.parameters()).device

        return mu, sigma