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

    # Compute the covariance of the data - empirical
    # emp_cov_fts = np.cov((data_fts - data_fts.mean(axis=0)).T)
    # emp_cov_regions = np.cov((data_regions - data_regions.mean(axis=0)).T)

    # Get the covariance matrices with Ledoit-Wolf
    lw_features = LedoitWolf(store_precision=False, block_size=100, assume_centered=True)
    lw_features = lw_features.fit(data_fts)
    lw_space = LedoitWolf(store_precision=False, block_size=100, assume_centered=True)
    lw_space = lw_space.fit(data_regions)

    # Compare the covariance matrices of LW and empirical
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 2, figsize=(20, 5))
    # ax[0][0].imshow(emp_cov_fts); ax[0][0].set_title('Features')
    # ax[0][1].imshow(lw_features.covariance_); ax[0][1].set_title('Features')
    # ax[1][0].imshow(emp_cov_regions); ax[1][0].set_title('Regions')
    # ax[1][1].imshow(lw_space.covariance_); ax[1][1].set_title('Regions')
    # plt.show();        
    
    K1 = lw_space.covariance_
    K2 = lw_features.covariance_
    # whiten_data = whiten_data_kronecker(valid_data.numpy(), K1, K2, method='cholesky')
    
    # Compute the covariance of the whitened data and compare
    # whiten_data = torch.tensor(whiten_data)
    # data_fts = whiten_data.reshape(num_samples*num_regions, num_fts).data.numpy()
    # data_regions = whiten_data.permute(0, 2, 1).reshape(num_samples*num_fts, num_regions).data.numpy()

    # emp_cov_fts = np.cov((data_fts - data_fts.mean(axis=0)).T)
    # emp_cov_regions = np.cov((data_regions - data_regions.mean(axis=0)).T)

    # # Plot average of data before and after whitening
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(valid_data.mean(axis=0)); ax[0].set_title('Before')
    # ax[1].imshow(whiten_data.mean(axis=0)); ax[1].set_title('After')

    # fig, ax = plt.subplots(2, 2, figsize=(20, 5))
    # ax[0][0].imshow(emp_cov_fts); ax[0][0].set_title('Features')
    # ax[0][1].imshow(lw_features.covariance_); ax[0][1].set_title('Features')
    # ax[1][0].imshow(emp_cov_regions); ax[1][0].set_title('Regions')
    # ax[1][1].imshow(lw_space.covariance_); ax[1][1].set_title('Regions')
    # plt.show()

    return mean_data, K1, K2


def compute_norm_info(df_data, norm_idx, clamp_idx=None, only_ed=False, avg_regions=True):
    """Compute the normalization information for the data at the given indices."""
    # Assume data is in: [B, V, F, T]
    #TODO: Make it more general
    
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


# if __name__ == "main":
#     #TODO: Explore this
#     # Normalize the data based on the range
#     norm_node_fts = ((node_fts.permute(0, 3, 1, 2) - min_values) / range_values).permute(0, 2, 3, 1)

#     # Compute the covariance matrix per feature across time
#     # =============================================
#     # 1: Get a covariance matrix and a mean vector
#     centered_data = (node_fts.permute(0, 3, 1, 2) - mean_fts)
#     a = torch.where(centered_data > 20)
#     node_fts.permute(0, 3, 1, 2)[a]

#     a = centered_data.permute(0, 1, 3, 2).reshape(-1, 26)
#     cov_regions = torch.cov(centered_data.mean(axis=(1, 3)).T)  # 26 by 26; average time and features
#     cov_features = torch.cov(centered_data.mean(axis=(1, 2)).T)  # 13 by 13; average time and regions
#     multidim_cov = torch.kron(cov_features, cov_regions)  # 26*13 by 26*13; average time and regions
#     # multidim_cov = torch.kron(cov_regions, cov_features)  # 26*13 by 26*13; average time and regions // it is different

#     u, s, v = torch.svd(multidim_cov)

#     # Regularize singular values if necessary
#     epsilon = 1e-6  # small positive constant for regularization
#     s_reg = torch.clamp(s, min=epsilon)  # ensure positive values

#     # Reconstruct covariance matrix
#     reconstructed_covariance_matrix = torch.mm(torch.mm(u, torch.diag(s_reg)), v.t())
    
#     fig, ax = plt.subplots(2, 2, figsize=(10, 5))
#     ax[0][0].imshow(multidim_cov); ax[0][0].set_title('Multidimensional covariance matrix')
#     ax[0][1].imshow(reconstructed_covariance_matrix); ax[0][1].set_title('Reconstructed covariance matrix')
#     ax[1][0].imshow(cov_regions); ax[1][0].set_title('Covariance regions')
#     ax[1][1].imshow(cov_features); ax[1][1].set_title('Covariance features')

#     # =============================================
#     # 2: use it to normalize data
#     # Compute inverse square root of covariance matrix
#     inv_sqrt_covariance_matrix = torch.mm(torch.mm(u, torch.diag(torch.rsqrt(s_reg))), v.t())
            
#     # Normalize data ??
#     a = centered_data.reshape(100, 50, -1) # ?
#     b = torch.einsum('ijk,kk->ijk', a, inv_sqrt_covariance_matrix)
#     normalized_data = b.reshape(100, 50, 26, 13)
    
#     # Compare the normalized data with the original data
#     fig, ax = plt.subplots(2, 2, figsize=(10, 5))
#     ax[0][0].imshow(a[0]); ax[0][0].set_title('Centered data')
#     ax[0][1].imshow(b[0]); ax[0][1].set_title('Normalized data')
#     ax[1][0].imshow(centered_data[0][0]); ax[1][0].set_title('Centered data')
#     ax[1][1].imshow(normalized_data[0][0]); ax[1][1].set_title('Normalized data')

#     # reg_factor = torch.diag(multidim_cov)* 1e-6
#     # L = torch.linalg.cholesky(multidim_cov+reg_factor)  # Cholesky decomposition

#     ft_dim = node_fts.shape[2]
#     joint_mean = torch.hstack([node_fts[:, :, 0].mean(axis=(-1, 0))  for f in range(ft_dim)])  # 2 by 26 -> 52
