import os
import torch
import logging
import numpy as np

from dataset.GraphDataset import load_dataset


def norm_data(X: torch.Tensor, 
              train_idx: np.array, 
              y: torch.Tensor = None, 
              norm_by_group: bool = False, 
              normalization: str = 'ZNorm',
              norm_only_ed: bool = False,
              ):
    """Normalize the data using the training set"""
    assert torch.is_tensor(X), "X must be a tensor"

    if norm_by_group:
        # Use only the healthy subjects to compute the mean
        valid_train = train_idx[y[train_idx] == 3]
    else:
        valid_train = train_idx

    mean_train = X[valid_train].mean(axis=0)
    std_train = X[valid_train].std(axis=0)    
    min_train, _ = X[valid_train].min(axis=0)
    max_train, _ = X[valid_train].max(axis=0)

    # Check if the std is close to 0
    ind_std_not_valid = np.where((std_train == 0))[0]
    if len(ind_std_not_valid) > 0:
        # Some edges are very similar or even equal in all subjects, specially when we normalize by group (e.g. healthy subjects only)
        logging.warning(f"Features with std close to 0: {ind_std_not_valid}")
        std_train[ind_std_not_valid] = 1.0  # Set to 1.0 to avoid division by 0

    #all_train_idxs = np.concatenate((train_idx, valid_idx))
    if normalization == "ZNorm":
        Xtrain = (X - mean_train) / (std_train + 1e-8)
    elif normalization == "MaxMin":
        Xtrain = (X - min_train) / (max_train - min_train)        

    norm_info = {'mean': mean_train, 'std': std_train, 'min': min_train, 'max': max_train}

    return Xtrain, norm_info