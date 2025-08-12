import numpy as np
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Bernoulli, Normal
from torch.nn import functional as F
from dataset.dataset_utils import reshape_to_graph, reshape_to_tensor


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise ValueError(f"Unexpected reduction type {self.reduction}")        



class MixedLoss(nn.Module):
    def __init__(self):
        super(MixedLoss, self).__init__()
        self.gamma = 1

    def forward(self, pred_labels, tgt_labels, pred_ft, target_ft):
        cre = nn.CrossEntropyLoss(reduction='mean')
        mse = nn.MSELoss(reduction='mean')
        gamma = .1
        # MSE: mean across features and then subjects
        # ((pred_ft - target_ft)**2).mean(dim=1).mean(dim=0)
        # CrossEntropy: Compute 1 CE per subject, and get the mean across them
        loss = mse(pred_ft, target_ft) + self.gamma*cre(pred_labels, tgt_labels)
        return loss
    


class SimpleMultiLabelConstrativeLoss(nn.Module):
    def __init__(self):
        super(SimpleMultiLabelConstrativeLoss, self).__init__()
        self.negative_margin = 1
        self.positive_margin = 0.4
    
    def forward(self, x, tgt_labels):
        one_hot_tgt = nn.functional.one_hot(tgt_labels).float()
        # one_hot_pred = torch.tensor(nn.functional.one_hot(torch.argmax(nn.functional.softmax(pred_labels, dim=1), dim=1)), dtype=torch.float32)

        # This gives the mean dynamical and initial point per region [B, F, R]
        init_latent = x.reshape(len(tgt_labels), -1)  # Flatten to [B, F*R]
        # Compute all the distances [symmetric]
        # dist = torch.cdist(init_latent, init_latent)  # -- Distance
        dist = F.cosine_similarity(init_latent.unsqueeze(1), init_latent.unsqueeze(0), dim=2)   # Similarity measure! [-1, 1]
        dist = 1 - dist  # Distance -- [0, 2], 0 means the same point, 2 means opposite points
        dist_to_group = dist @ one_hot_tgt  # Distance to each target label [of the same group]
        # Normalize the distances ?
        # dist_normalized = dist / (dist.norm(p=2, dim=1, keepdim=True) + 1e-8)
        # dist_to_group = dist_normalized @ one_hot_tgt  # Distance to each target label [of the same group]

        # Distance to the other labels
        class_counts = one_hot_tgt.sum(axis=0, keepdim=True)
        dist_to_group_avg = dist_to_group / (class_counts + 1e-8)
        dist_to_others = dist_to_group_avg * (1 - one_hot_tgt)
        dist_to_same_group = dist_to_group_avg * one_hot_tgt

        # Loss for positive pairs (minimize distance)
        # positive_loss = dist_to_same_group**2
        positive_loss = torch.clamp(self.positive_margin - dist_to_same_group, min=0.0)**2
        # (torch.min(dist_to_group, 1)[1] == tgt_labels).sum() / len(tgt_labels)
        # Loss for negative pairs (maximize distance by margin) -- no penalty if the distance is already larger than the margin
        # margin = self.margin #* init_latent.shape[-1]
        negative_loss = torch.clamp(self.negative_margin - dist_to_others, min=0.0)**2

        # Final loss
        loss = positive_loss + negative_loss

        # return (loss.sum(axis=1) * dist.norm(p=2, dim=1, keepdim=True)).mean()
        return loss.sum(axis=1).mean()
        


def cusum_loss(y_true, y_pred, threshold=1.0):
    """
    Uses CUSUM (Cumulative Sum) to detect and penalize significant changes
    """
    # Calculate basic error
    error = (y_true - y_pred) ** 2
    
    # Calculate CUSUM for true values
    # cusum = torch.cumsum(torch.abs(torch.diff(y_true)))
    # cusum = torch.cumsum(torch.abs(y_true[..., 1:] - y_true[..., :-1]), dim=0)
    cusum = torch.abs(y_true[..., 1:] - y_true[..., :-1])
    cusum = torch.cat([cusum, cusum[..., -1:]], dim=-1)  # Pad with the last value to match dimensions
    
    # Identify points where significant changes occur
    # significant_changes = cusum > threshold
    significant_changes = cusum 
    
    # Apply higher weights to errors during significant changes
    # weights = 1 + 2 * significant_changes.float()
    # weights = 1 + 2*(significant_changes.float() * cusum)
    weights = 1 + significant_changes.float()
    
    return error, weights



def contrastive_loss_two(embeddings, temperature=0.5):
    # Normalize embeddings to unit vectors
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise cosine similarity
    similarity_matrix = torch.mm(embeddings, embeddings.T)  # Shape: (N, N)
    
    # Scale similarities by temperature
    logits = similarity_matrix / temperature
    
    # Create labels for the positives (assume diagonal is the positive pair for simplicity)
    batch_size = embeddings.size(0)
    labels = torch.arange(batch_size).to(embeddings.device)
    
    # Compute the cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss



def compute_prototype_loss(embeddings, prototypes, 
                           temperature=0.1, lambda_proto=0.1, delta=5, batch_size=None):
    """
    Compute contrastive loss with prototype sparsity regularization.
    
    Args:
        embeddings (torch.Tensor): Shape [N, D], embeddings of batch samples.
        prototypes (torch.Tensor): Shape [K, D], prototype vectors.
        temperature (float): Temperature for contrastive loss.
        lambda_proto (float): Weight for prototype sparsity penalty.
        delta (int): Threshold for "minimum usage" of a prototype.

    Returns:
        loss (torch.Tensor): Total loss (contrastive + sparsity penalty).
    """
    # Normalize embeddings and prototypes
    embeddings = F.normalize(embeddings, p=2, dim=1)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    # Contrastive loss (e.g., SwAV-like assignment)
    logits = torch.mm(embeddings, prototypes.T) / temperature  # Shape [N, K]
    assignments = F.softmax(logits, dim=1)  # Probabilistic assignments / (N, K)
    contrastive_loss = -torch.sum(assignments * torch.log(assignments + 1e-8), dim=1)
    if batch_size is not None:        
        contrastive_loss = contrastive_loss.reshape(batch_size, -1).mean(dim=0).sum()
    else:
        contrastive_loss = contrastive_loss.mean(dim=0)  # Shape: [K]

    # Prototype usage regularization
    # usage = assignments.sum(dim=0)  # Sum of assignments per prototype, shape [K]
    # sparsity_penalty = lambda_proto * torch.sum(F.relu(delta - usage))  # Penalize underused prototypes

    # Compute mean probability for each prototype
    if batch_size is not None:
        prototype_probs = assignments.reshape(batch_size, -1, prototypes.shape[0])  # Shape: [B, R, K]
        # prototype_probs = prototype_probs.mean(dim=0).sum(dim=0).mean()
        prototype_probs = prototype_probs.mean(dim=0).sum()
        # prototype_probs = assignments.reshape(batch_size, -1).mean(dim=0).sum()
    else:
        prototype_probs = assignments.mean(dim=0)  # Shape: [K]
    
    # Compute entropy
    entropy_loss = -lambda_proto * torch.sum(prototype_probs * torch.log(prototype_probs + 1e-8))

    # Total loss
    total_loss = contrastive_loss + entropy_loss

    return total_loss



class LossODEProcess(nn.Module):
    def __init__(self, 
                 gamma_rec: int = 1, 
                 gamma_class: int = 1,
                 gamma_lat: int = 1,
                 gamma_bc: int = 0,                 
                 gamma_graph: int = 1,
                 weight_classes: list = None,
                 gamma_focal=1,
                 use_focal=False,
                 use_mse=False,
                 ):
        super(LossODEProcess, self).__init__()
        self.gamma_rec = gamma_rec  # Weight of the reconstruction (MSE) loss
        self.gamma_lat = gamma_lat  # Weight of the latent space regularisation 
        self.gamma_bc = gamma_bc # Weight of the latent space BC (MSE) loss
        self.gamma_class = gamma_class  # Weight of the classification (CrossEntropy) loss
        self.gamma_graph = gamma_graph

        # Options to use MSE and Focal loss 
        self.use_mse = use_mse
        self.use_focal = use_focal

        if use_focal:        
            self.loss_class_fn = FocalLoss(gamma=gamma_focal, alpha=1, reduction='mean', weight=weight_classes)
        else:
            self.loss_class_fn = nn.CrossEntropyLoss(reduction='mean')
        
        # Contrastive loss
        self.constrastive_loss = SimpleMultiLabelConstrativeLoss()

        # Error loss alternative to the log-likelihood                    
        self.loss_rec_fn = nn.HuberLoss(reduction='none', delta=0.1)
        self.mae_fn = nn.L1Loss(reduction='none')
    
    def forward(self, p_y_pred, y_target, pred_labels, tgt_labels, q_target=None, q_context=None, latent_values=None, 
                graph_reg=None, tgt_edges=(None, None), ctx_edges=(None, None), only_rec=False, pred_external=None, 
                tgt_external=None, prototypes=None, warmup=False, weights=None, rampup_weight=1.0):
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
        # over batch and sum over number of targets and dimensions of y
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
            # else:
            #     # print("No weights applied to the loss")
            #     weights = torch.tensor([1.0 for t in range(y_target.shape[-1])], device=y_target.device)
            #     # weights = torch.tensor([0.99 for t in range(y_target.shape[-1])], device=y_target.device)
            #     nll = nll * weights[None, None, None, :]
                # pass
            # nll = nll.mean(dim=0).sum()
            nll = nll.mean(dim=0).sum(dim=1).mean()
            
            # Add variance reg. term
            # nll += 0.05 * torch.mean(var_term, dim=0).sum()
            nll += 0.05 * torch.mean(var_term, dim=0).sum(dim=1).mean()
            
            # Add MAE
            # mae_values = self.mae_fn(p_y_pred.mean, y_target) * weights[None, None, None, :]
            # nll += 0.1 * mae_values.mean(0).sum()
            # Add MSE over t0 [to match the initial point first]
            # nll += self.loss_rec_fn(p_y_pred.mean, y_target)[..., 0].mean(0).sum()

        # ======= External (non-graph) prediction =======
        if pred_external is not None and tgt_external is not None:
            # MSE for the external features / or have also a likelihood?
            # ext_loss = self.loss_rec_fn(pred_external, tgt_external).mean(dim=0).sum()
            ext_loss = torch.FloatTensor([0]).to(y_target.device)
        else:
            ext_loss = torch.FloatTensor([0]).to(y_target.device)
        
        # ======= Latent BC, init = end =======
        if latent_values is not None:
            # bc_latent_loss = ((latent_values[0] - latent_values[-1]).square() / y_target.shape[0]).sum()
            bc_latent_loss = torch.FloatTensor([0]).to(y_target.device)
        else:
            bc_latent_loss = torch.FloatTensor([0]).to(y_target.device)

        # ======= Latent space regularization =======
        if latent_values is not None:
            # D = q_context.mean[:, latent_values.shape[1]:]
            # L0 = q_context.mean[:, :latent_values.shape[1]]
            # if weights is not None and warmup:
            #     # Apply weights to the loss [should promote the learning of the first timepoints over the last ones]
            #     latent_values = latent_values * weights[None, None, None, :]
            # reg_latent_loss = latent_values.square().mean(dim=0).sum()  # All latent values
            reg_latent_loss = latent_values.square().mean(dim=0).sum(dim=1).mean()  # All latent values
        else:
            reg_latent_loss = torch.FloatTensor([0]).to(y_target.device)

        # ======= Graph regularization =======
        if graph_reg is not None:
            # For the spatial and temporal norm: shape data [NFE, Num_Edges]
            # For the penalties: shape data [NFE, 1]
            # NFE: Number of function evaluations.
            # Get the different terms that we can use to regularize the graph
            spatial_norm, temporal_norm, symm_penalty, eig_penalty, acyc_penalty = graph_reg

            # The shape is [NFE, Num_Edges]; Sum over NFE and mean over edges 
            spatial_norm = spatial_norm.mean(axis=0).mean()
            temporal_norm = temporal_norm.mean(axis=0).mean()
            # spatial_norm = spatial_norm.sum() / y_target.shape[0]
            # temporal_norm = temporal_norm.sum() / y_target.shape[0]
            
            # Same weight both of them
            graph_reg_loss = spatial_norm + temporal_norm + symm_penalty.mean() + eig_penalty.mean()  # 1            
            # graph_reg_loss = spatial_norm + temporal_norm + acyc_penalty.sum()            
            # graph_reg_loss = symm_penalty.sum() + eig_penalty.sum()

        # ======== Classification ========
        if self.gamma_class > 0:
            class_loss = self.loss_class_fn(pred_labels, tgt_labels) #* torch.log(torch.tensor(self.epsilon_dt)) * 4 
            class_loss = np.prod(y_target.shape[2:]) * class_loss
        else:
            class_loss = torch.FloatTensor([0]).to(y_target.device)

        # ======== KL Divergence ========
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        if q_target is not None and q_context is not None:
            # kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
            kl = kl_divergence(q_target, q_context).mean(dim=0).mean()
        else:
            kl = torch.FloatTensor([0]).to(y_target.device)

        # Edges
        if tgt_edges[0] is not None and ctx_edges[0] is not None:
            # kl_space = kl_divergence(tgt_edges[0], ctx_edges[0]).mean(dim=0).sum()
            # kl_time = kl_divergence(tgt_edges[1], ctx_edges[1]).mean(dim=0).sum()
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
                + self.gamma_class * class_loss.float()
                # + self.gamma_bc * bc_latent_loss.float()
                # + self.gamma_lat * reg_latent_loss.float()
                # + self.gamma_graph * graph_reg_loss.float()
            )
        else:
            total = (
                self.gamma_rec * nll.float()
                + self.gamma_class * class_loss.float()
                + self.gamma_rec * kl.float() * rampup_weight
                + self.gamma_bc * bc_latent_loss.float() * rampup_weight
                + self.gamma_lat * reg_latent_loss.float() * rampup_weight
                + self.gamma_graph * graph_reg_loss.float() * rampup_weight
            )

        loss_components = {
            "L_rec": nll.float().detach().item(),
            "L_class": class_loss.float().detach().item(),
            "L_kl": kl.float().detach().item(),
            "L_bc": bc_latent_loss.float().detach().item(),
            "L_lat": reg_latent_loss.float().detach().item(),
            "L_graph": graph_reg_loss.float().detach().item(),            
        }

        return total, loss_components