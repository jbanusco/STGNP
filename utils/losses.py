import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F


class LossODEProcess(nn.Module):
    def __init__(self, 
                 gamma_rec: int = 1, 
                 gamma_class: int = 1,
                 gamma_lat: int = 1,
                 gamma_bc: int = 0,                 
                 gamma_graph: int = 1,
                 weight_classes: list = None,
                 use_mse=False,
                 ):
        super(LossODEProcess, self).__init__()
        self.gamma_rec = gamma_rec  # Weight of the reconstruction loss
        self.gamma_lat = gamma_lat  # Weight of the latent space regularisation        
        self.gamma_graph = gamma_graph  # Weight of the graph regularisation

        # Use MSE instead og log-likelihood
        self.use_mse = use_mse
        
        # Error loss function used when we optimize a deterministic error instead of log-likelihood
        self.loss_rec_fn = nn.HuberLoss(reduction='none', delta=0.1)
        # self.loss_rec_fn = nn.L1Loss(reduction='none')
    
    def forward(self, p_y_pred, y_target, q_target=None, q_context=None, latent_values=None, 
                graph_reg=None, tgt_edges=(None, None), ctx_edges=(None, None), warmup=False, 
                weights=None, rampup_weight=1.0):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """

        # ======= Log-likelihood =======
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and feature dimension and sum over number of targets (time).
        if self.use_mse:
            nll = self.loss_rec_fn(p_y_pred.mean, y_target)
            nll = nll.mean(dim=0).sum()
        else:                
            nll = -p_y_pred.log_prob(y_target)
            var_term = torch.log(p_y_pred.variance + 1e-8)
            if weights is not None and warmup:
                # Apply weights to the loss [should promote the learning of the first timepoints over the last ones]
                # weights = torch.tensor([1.0 for t in range(y_target.shape[-1])], device=y_target.device)
                nll = nll * weights[None, None, None, :]
                var_term = var_term * weights[None, None, None, :]

            nll = nll.mean(dim=0).sum(dim=1).mean()
            
            # Add variance reg. term -- penalise a bit more the variance
            nll += 0.05 * torch.mean(var_term, dim=0).sum(dim=1).mean()

        # ======= Latent space regularization =======
        # Take mean over batch and feature dimension and sum over number of targets (time).
        reg_latent_loss = latent_values.square().mean(dim=0).sum(dim=1).mean()

        # ======= Graph regularization =======
        if graph_reg is not None:
            # For the spatial and temporal norm: shape data [NFE, Num_Edges]
            # For the penalties: shape data [NFE, 1]
            # NFE: Number of function evaluations.
            # Get the different terms that we can use to regularize the graph
            spatial_norm, temporal_norm = graph_reg

            # The shape is [NFE, Num_Edges]; Sum over NFE and mean over edges 
            spatial_norm = spatial_norm.mean(axis=0).mean()
            temporal_norm = temporal_norm.mean(axis=0).mean()
            
            # Same weight both of them
            graph_reg_loss = spatial_norm + temporal_norm

        # ======== KL Divergence ========
        # KL has shape (batch_size, r_dim). Take mean over batch and mean over dimensions
        if q_target is not None and q_context is not None:
            kl = kl_divergence(q_target, q_context).mean(dim=0).mean()
        else:
            kl = torch.FloatTensor([0]).to(y_target.device)

        # Edges
        if tgt_edges[0] is not None and ctx_edges[0] is not None:
            kl_space = kl_divergence(tgt_edges[0], ctx_edges[0]).mean(dim=0).mean()
            kl_time = kl_divergence(tgt_edges[1], ctx_edges[1]).mean(dim=0).mean()
            kl = kl + (kl_space + kl_time)

        # Ensure the KL is positive and scale it by the number of timepoints
        kl = torch.clamp(kl, min=1e-6) # * y_target.shape[-1]

        # =============================
        # Aggregate the loss components
        # =============================
        if warmup:
            total = (
                self.gamma_rec * nll.float()
            )
        else:
            total = (
                self.gamma_rec * nll.float()
                + self.gamma_rec * kl.float() * rampup_weight
                + self.gamma_lat * reg_latent_loss.float() * rampup_weight
                + self.gamma_graph * graph_reg_loss.float() * rampup_weight
            )

        loss_components = {
            "L_rec": nll.float().detach().item(),
            "L_kl": kl.float().detach().item(),
            "L_lat": reg_latent_loss.float().detach().item(),
            "L_graph": graph_reg_loss.float().detach().item(),            
        }

        return total, loss_components