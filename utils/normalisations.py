import torch
import torch.nn as nn
import numpy as np
from utils.kernels import whiten_data_kronecker
from sklearn.covariance import LedoitWolf


class SpatialMultivariate_Normalization(nn.Module):
    """Normalize using a multivariate spatial covariance."""    
    def __init__(self, mean_value, spatial_cov, feature_cov, eps=1e-6):
        super().__init__()
        
        self.mean = mean_value.clone().detach().float()
        self.spatial_cov = spatial_cov.clone().detach().float()
        self.feature_cov = feature_cov.clone().detach().float()     
        self.eps = eps

        L1 = torch.linalg.cholesky(self.spatial_cov + eps*torch.eye(self.spatial_cov.shape[0]))
        self.Lk1 = torch.linalg.inv(L1).float()

        L2 = torch.linalg.cholesky(self.feature_cov + eps*torch.eye(self.feature_cov.shape[0]))
        self.Lk2 = torch.linalg.inv(L2).float()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor using the spatial covariance.
        """        
        # Whiten data with the kronecker trick
        shifted_data = tensor - self.mean

        a = torch.einsum('ij,njk->nik', self.Lk1, shifted_data)
        whitened_data = torch.einsum('nik,kl->nil', a, self.Lk2.T)        

        return whitened_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_value={self.max}, min_value={self.min})"


class MaxMin_Normalization(nn.Module):
    """Normalize an image between 0 and 1."""

    def __init__(self, max_value, min_value):
        super().__init__()
        self.max = torch.tensor(max_value.clone().detach()).float()
        self.min = torch.tensor(min_value.clone().detach()).float()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor between 0 and 1.        
        """
        return ((tensor - self.min) / (self.max - self.min)).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_value={self.max}, min_value={self.min})"


class Ratio_Normalization(nn.Module):
    """Normalize an image between 0 and 1."""

    def __init__(self, ratios, inplace=False):
        super().__init__()
        self.ratios = ratios.clone().detach().float()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor with respect to a given values
        """
        return (tensor / (self.ratios + 1e-8)).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_value={self.max}, min_value={self.min})"


class ClampTensor(nn.Module):
    """Normalize an image between 0 and 1."""

    def __init__(self, min_value, max_value):
        super().__init__()
        self.max = max_value.clone().detach().float()
        self.min = min_value.clone().detach().float()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Clamp the tensor between the max and min values.
        """
        return torch.clamp(tensor, min=self.min, max=self.max)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_value={self.max}, min_value={self.min})"



def get_mean_and_cov(input_data):
    """Get the mean and covariance of the data."""

    valid_data = input_data.clone().float()
    num_samples, num_regions, num_fts = valid_data.shape
    
    mean_data = valid_data.mean(axis=0)
    shifted_data = valid_data - mean_data

    data_fts = shifted_data.reshape(num_samples*num_regions, num_fts).data.numpy()
    data_regions = shifted_data.permute(0, 2, 1).reshape(num_samples*num_fts, num_regions).data.numpy()

    # Get the covariance matrices with Ledoit-Wolf
    lw_features = LedoitWolf(store_precision=False, block_size=100, assume_centered=True)
    lw_features = lw_features.fit(data_fts)
    lw_space = LedoitWolf(store_precision=False, block_size=100, assume_centered=True)
    lw_space = lw_space.fit(data_regions)  
    
    K1 = lw_space.covariance_
    K2 = lw_features.covariance_

    return mean_data, K1, K2


def compute_norm_info(df_data, norm_idx, clamp_idx=None, only_ed=False, avg_regions=True):
    """Compute the normalization information for the data at the given indices."""
    # Assume data is in: [B, V, F, T]
        
    # Compute the min-max range normalization based on the .95 and .05 quantiles
    if clamp_idx is None:
        clamp_idx = np.arange(df_data.shape[0])

    if only_ed:
        # -- Only ED
        valid_data = df_data[norm_idx, :, :, 0].clone()

        # Mean over samples i.e: one value per region and features (or position)
        if avg_regions:
            num_regions = df_data.shape[1]
            mean_fts = df_data[norm_idx, :, :, 0].mean(axis=(0, 1)).unsqueeze(0).repeat(num_regions, 1)
            std_fts = df_data[norm_idx, :, :, 0].std(axis=(0, 1)).unsqueeze(0).repeat(num_regions, 1)
            min_values = torch.tensor(np.quantile(valid_data, 0.05, axis=(0, 1))).float().unsqueeze(0).repeat(num_regions, 1)
            max_values = torch.tensor(np.quantile(valid_data, 0.95, axis=(0, 1))).float().unsqueeze(0).repeat(num_regions, 1)
        else:
            mean_fts = df_data[norm_idx, :, :, 0].mean(axis=(0))  
            std_fts = df_data[norm_idx, :, :, 0].std(axis=(0))
            min_values = torch.tensor(np.quantile(df_data[norm_idx], 0.05, axis=(0, -1))).float()
            max_values = torch.tensor(np.quantile(df_data[norm_idx], 0.95, axis=(0, -1))).float()

        mean_data, K1, K2 = get_mean_and_cov(valid_data)
    else:
        # -- All frames
        valid_data = df_data[norm_idx].clone()

        # Mean over samples and time, i.e: one value per region and features (or position)
        if avg_regions:
            num_regions = df_data.shape[1]
            mean_fts = df_data[norm_idx].float().mean(axis=(0, 1, -1)).unsqueeze(0).repeat(num_regions, 1)
            std_fts = df_data[norm_idx].float().std(axis=(0, 1, -1)).unsqueeze(0).repeat(num_regions, 1)
            min_values = torch.tensor(np.quantile(df_data[norm_idx], 0.05, axis=(0, 1, -1))).float().unsqueeze(0).repeat(num_regions, 1)
            max_values = torch.tensor(np.quantile(df_data[norm_idx], 0.95, axis=(0, 1, -1))).float().unsqueeze(0).repeat(num_regions, 1)
        else:
            mean_fts = df_data[norm_idx].float().mean(axis=(0, -1)) 
            std_fts = df_data[norm_idx].float().std(axis=(0, -1)) 
            min_values = torch.tensor(np.quantile(df_data[norm_idx], 0.05, axis=(0, -1))).float()
            max_values = torch.tensor(np.quantile(df_data[norm_idx], 0.95, axis=(0, -1))).float()

        mean_data, K1, K2 = get_mean_and_cov(valid_data[..., 0])  #NOTE: Computed only on the first frame

    # Avoid 0 std
    std_fts[std_fts == 0] = 1  # Avoid division by zero

    min_clamp = torch.tensor(np.quantile(df_data[clamp_idx], 0.05, axis=(0, -1))).float()
    max_clamp = torch.tensor(np.quantile(df_data[clamp_idx], 0.95, axis=(0, -1))).float()

    return {'mean': mean_fts, 'std': std_fts, 'min': min_values, 'max': max_values, 'min_clamp': min_clamp, 'max_clamp': max_clamp,
            'mean_data': mean_data, 'K1': torch.tensor(K1), 'K2': torch.tensor(K2)}
