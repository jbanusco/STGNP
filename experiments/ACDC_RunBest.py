import os
import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
import logging
import xgboost as xgb
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split

from model.train_stmgcn_ode import Objective_Multiplex
from model.testing_model import get_accuracies, save_training_convergence, save_mean_positions, plot_subject_trajectory, plot_subject_position, plot_state
from utils.plot_utils import plot_confusion_matrx, plot_latent_space, to_long_format, plot_with_error

from dataset.dataset_utils import get_data
from dataset.GraphDataset import GraphDataset
from utils.utils import seed_everything, str2bool, get_best_params

from utils.model_selection_optuna import hypertune_optuna, optuna_cv, optuna_nested_cv
from utils.model_selection_sklearn import cv_stratified_split

from sklearn.ensemble import RandomForestClassifier
from utils.model_selection_sklearn import cv_model_scaler
from utils.plot_utils import save_mean_trajectories

from experiments.ACDC_CV import save_training_convergence, plot_results, export_latent_data, plot_combined_trajectories, get_data_in_original_scale
from model.testing_model import get_latex_table, wrap_latex_table

import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')



def plot_predicted_trajectories(objective, pred_latent, pred_trajectory, save_folder,
                                normalization='ZNorm', plot_individual=True, traj_dim=None, 
                                fts_to_predict=None, true_trajectory=None, plot_spatial=False,
                                transform = lambda x: x):
    """ Plot the predicted trajectories and latent space. 
    param: objective: Objective_Multiplex object
    param: pred_latent: torch.Tensor with the latent space [num_subjects, num_latent, num_nodes, time_frames]
    param: pred_trajectory: torch.Normal distribution with the predicted trajectories [num_subjects, num_features, num_nodes, time_frames]
    param: save_folder: str with the folder to save the results
    param: normalization: str with the normalization strategy
    param: plot_individual: bool to plot some individual trajectories
    param: traj_dim: int with the number of dimensions of the trajectories [before the positions]
    param: fts_to_predict: np.array with the features to predict (if not autoencoder-like, i.e: all)
    """
    # Indices
    train_idx = objective.train_idx
    valid_idx = objective.valid_idx
    test_idx = objective.test_idx

    # Trajectories
    if objective.use_position:   
        # num_pos = objective.dataset.nodes_data[0]['pos'].shape[1] 
        positions_dim = 3 # x, y, z [TODO: need to add it in the objective]
        trans_pos = objective.dataset._transform['pos']  # num_regions * num_features
        traj_dim = pred_trajectory.mean.shape[1] - positions_dim if traj_dim is None else traj_dim

        rec_trajectories = pred_trajectory.mean[:, :traj_dim, :, 1:].detach().cpu()
        rec_std_trajectories = pred_trajectory.scale[:, :traj_dim, :, 1:].detach().cpu()
        if true_trajectory is not None:
            true_trajectory = true_trajectory[:, :traj_dim].detach().cpu()

        rec_pos = pred_trajectory.mean[:, traj_dim:, :, 1:].detach().cpu()
        rec_pos_std = pred_trajectory.scale[:, traj_dim:, :, 1:].detach().cpu()
    else:
        traj_dim = pred_trajectory.mean.shape[1] if traj_dim is None else traj_dim

        rec_trajectories = pred_trajectory.mean.detach().cpu()[..., 1:]
        rec_std_trajectories = pred_trajectory.scale.detach().cpu()[..., 1:]
        if true_trajectory is not None:
            true_trajectory = true_trajectory.detach().cpu()

    if fts_to_predict is None:
        fts_to_predict = np.arange(0, traj_dim)
                                   
    # Transform it
    num_subjects, _, _, time_frames = rec_trajectories.shape
    trans_fts_predict = objective.dataset._transform['nfeatures']
    if normalization == 'ZNorm':
        if isinstance(objective.dataset, GraphDataset):
            trans_fts_mean = trans_fts_predict.mean[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_std = trans_fts_predict.std[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
        else:
            trans_fts_mean = trans_fts_predict.mean.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_std = trans_fts_predict.std.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)

        rec_trajectories = (rec_trajectories * trans_fts_std) + trans_fts_mean
        rec_std_trajectories = (rec_std_trajectories * trans_fts_std)
        if true_trajectory is not None:
            time_frames_true = true_trajectory.shape[-1]
            trans_fts_mean = trans_fts_predict.mean[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames_true).permute(0, 2, 1, 3)
            trans_fts_std = trans_fts_predict.std[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames_true).permute(0, 2, 1, 3)
            true_trajectory = (true_trajectory * trans_fts_std) + trans_fts_mean
        
    elif normalization == 'MaxMin':
        # Need to implement the MaxMin normalization
        if isinstance(objective.dataset, GraphDataset):
            trans_fts_max = trans_fts_predict.max[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_min = trans_fts_predict.min[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
        else:
            trans_fts_max = trans_fts_predict.max.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_min = trans_fts_predict.min.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)

        range_fts = trans_fts_max - trans_fts_min
        rec_trajectories = (rec_trajectories * range_fts) + trans_fts_min
        rec_std_trajectories = (rec_std_trajectories * range_fts)
        if true_trajectory is not None:
            true_trajectory = (true_trajectory * range_fts) + trans_fts_min
        
    else:
        # Need to implement the spatial normalization
        pass

    # Positions
    if normalization == 'ZNorm' and objective.use_position:
        trans_pos_std = trans_pos.std.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]
        trans_pos_mean = trans_pos.mean.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]

        rec_pos = (rec_pos * trans_pos_std) + trans_pos_mean
        rec_pos_std = (rec_pos_std * trans_pos_std)
        
    elif normalization == 'MaxMin' and objective.use_position:
        trans_pos_max = trans_pos.max.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]
        trans_pos_min = trans_pos.min.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]
        
        range_pos = trans_pos_max - trans_pos_min
        rec_pos = (rec_pos * range_pos) + trans_pos_min
        rec_pos_std = (rec_pos_std * range_pos)
        
    else:
        # Need to implement the spatial normalization
        pass

    # Latent
    latent_trajectories = pred_latent[..., 1:].detach().cpu().numpy()

    # ==== Plot latent space trajectories
    num_latent = latent_trajectories.shape[1]
    time_axis = np.arange(0, time_frames, 1)
    fig, ax = plt.subplots(1, num_latent, figsize=(20, 5), squeeze=False)
    for idx_l in range(num_latent):
        ax[0][idx_l].plot(time_axis, latent_trajectories[:, idx_l].mean(axis=0).T) 
        ax[0][idx_l].set_xlabel('Time')
        # ax[0][idx_l].set_ylabel(f'Latent {idx_l}')
        ax[0][idx_l].set_title(f'Latent {idx_l}')
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(save_folder, 'latent_trajectories_predicted.png'))
    plt.close('all')

    # Apply transform to the reconstruction
    rec_trajectories = transform(rec_trajectories)
    true_trajectory = transform(true_trajectory) if true_trajectory is not None else None

    # ==== Plot trajectories and positions
    save_mean_trajectories(rec_trajectories, true_trajectory, save_folder, 
                           append_name='mean_predicted',
                           fts_names=objective.dataset.features)
    
    list_subjects = objective.dataset.sub_id
    if plot_individual:
        plot_subject_trajectory(rec_trajectories, rec_std_trajectories, true_trajectory, save_folder, sub_id=train_idx[0], 
                                identifier=f"{list_subjects[train_idx[0]]}_train", append='_predicted', fts_names=objective.dataset.features)
        plot_subject_trajectory(rec_trajectories, rec_std_trajectories, true_trajectory, save_folder, sub_id=test_idx[0], 
                                identifier=f"{list_subjects[test_idx[0]]}_test", append='_predicted', fts_names=objective.dataset.features)

        # Plot the spatial layout
        if plot_spatial:
            # Train
            append_id = f"{list_subjects[train_idx[0]]}_train_predicted"
            save_path = os.path.join(save_folder, f'subject-{list_subjects[train_idx[0]]}_train', f'surface_plot_{append_id}.png')
            # Plot three time frames [init, mid, end]
            plotted_frames = [0, time_frames//2, time_frames-1]
            A = np.ones((int(objective.dataset.num_nodes**0.5), int(objective.dataset.num_nodes**0.5)))
            plot_state(rec_trajectories, A, save_path, dim_state=0, x_tgt=true_trajectory, frames_idx=plotted_frames, subj_idx=0)

            append_id = f"{list_subjects[test_idx[0]]}_test_predicted"
            save_path = os.path.join(save_folder, f'subject-{list_subjects[test_idx[0]]}_test', f'surface_plot_{append_id}.png')
            # Plot three time frames [init, mid, end]
            plotted_frames = [0, time_frames//2, time_frames-1]
            A = np.ones((int(objective.dataset.num_nodes**0.5), int(objective.dataset.num_nodes**0.5)))
            plot_state(rec_trajectories, A, save_path, dim_state=0, x_tgt=true_trajectory, frames_idx=plotted_frames, subj_idx=0)


def extract_trajectory_features(data):
    """
    Extract features from trajectory data with detailed summary statistics including
    velocity, acceleration, and curvature.
    
    Args:
        data: Tensor of shape [num_subjects, num_latent_dim, num_nodes, time_dim].
    
    Returns:
        A tensor of shape [num_subjects, num_latent_dim * num_features] containing summarized features.
    """
    # Compute mean over nodes and time for each subject and latent dimension
    mean_feature = data.mean(dim=(2, 3))  # Shape: [num_subjects, num_latent_dim]

    # Compute standard deviation over nodes and time
    std_feature = data.std(dim=(2, 3))  # Shape: [num_subjects, num_latent_dim]

    # Compute min and max over nodes and time
    min_feature = data.min(dim=3).values.min(dim=2).values  # Shape: [num_subjects, num_latent_dim]
    max_feature = data.max(dim=3).values.max(dim=2).values  # Shape: [num_subjects, num_latent_dim]

    # Compute the range (max - min)
    range_feature = max_feature - min_feature  # Shape: [num_subjects, num_latent_dim]

    # Compute skewness (third standardized moment)
    mean_nodes_time = data.mean(dim=(2, 3), keepdim=True)  # Broadcast mean
    std_nodes_time = data.std(dim=(2, 3), keepdim=True)  # Broadcast std
    skewness_feature = (
        ((data - mean_nodes_time) ** 3).mean(dim=(2, 3)) / (std_nodes_time.squeeze() ** 3 + 1e-8)
    )  # Shape: [num_subjects, num_latent_dim]

    # Compute kurtosis (fourth standardized moment, excess kurtosis)
    kurtosis_feature = (
        ((data - mean_nodes_time) ** 4).mean(dim=(2, 3)) / (std_nodes_time.squeeze() ** 4 + 1e-8) - 3
    )  # Shape: [num_subjects, num_latent_dim]

    # Compute energy (sum of squares)
    energy_feature = (data ** 2).sum(dim=(2, 3))  # Shape: [num_subjects, num_latent_dim]

    # # Compute entropy over nodes and time for each latent dimension
    # # Normalize the data to [0, 1] to calculate entropy (requires positive values)
    # data_min = data.min(dim=3).values.min(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    # data_max = data.max(dim=3).values.max(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    # normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
    # histogram_bins = 10  # Number of bins for entropy computation
    # histograms = torch.histc(normalized_data.flatten(start_dim=2), bins=histogram_bins, min=0.0, max=1.0)
    # probabilities = histograms / histograms.sum(dim=-1, keepdim=True)
    # entropy_feature = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)  # Shape: [num_subjects, num_latent_dim]

    # **Velocity** (First derivative of the state with respect to time)
    # velocity_data =  data[:, :, :, 1:] - data[:, :, :, :-1]  # Shape: [num_subjects, num_latent_dim, num_nodes, time_dim-1]    
    velocity_data = torch.diff(data, n=1, dim=-1)
    velocity_feature = velocity_data.mean(dim=(2, 3))  # Shape: [num_subjects, num_latent_dim]

    # **Acceleration** (Second derivative of the state with respect to time)
    acceleration_data = torch.diff(velocity_data, n=1, dim=-1)  # Shape: [num_subjects, num_latent_dim, time_dim-2]
    acceleration_feature = acceleration_data.mean(dim=(2, 3))  # Shape: [num_subjects, num_latent_dim]

    # **Curvature** (Change of velocity in the direction of the trajectory)
    # For simplicity, assume that the curvature is calculated as:
    # (velocity x acceleration) / |velocity|^2 for 2D/3D data, we simplify to just magnitude here.
    # Let's compute the magnitude of the acceleration vector
    curvature_feature = acceleration_feature.norm(p=2, dim=-1)  # L2 norm of acceleration vector, Shape: [num_subjects, num_latent_dim]

    # Combine all features
    combined_features = torch.cat(
        [
            mean_feature,
            std_feature,
            min_feature,
            max_feature,
            range_feature,
            skewness_feature,
            kurtosis_feature,
            energy_feature,
            # entropy_feature.unsqueeze(-1),  # Match dimensionality
            velocity_feature,
            acceleration_feature,
            curvature_feature.unsqueeze(-1),  # Match dimensionality
        ],
        dim=1,
    )  # Final shape: [num_subjects, num_latent_dim * num_features]

    return combined_features



def get_parser() -> argparse.ArgumentParser:
    data_path = "/media/jaume/DATA/Data/New_ACDC/MIDS/mixed/derivatives"
    experiment_name = 'GraphClassification'
    # study_name = 'Multiplex_HPT_ACDC_Pred_ADAM'
    # study_name = 'Multiplex_HPT_ACDC_ADAM'
    # study_name = 'Multiplex_HPT_ACDC_ADAM_FINAL_MAE_wACC'
    # study_name = "Multiplex_HPT_ACDC_ADAM_FINAL_MAE_FINAL_NOCLASS"
    # study_name = "Multiplex_HPT_ACDC_ADAM_FINAL_MAE_FINAL_NOCLASS_NOID"
    # study_name = "Multiplex_HPT_ACDC_ADAM_FINAL_MAE_FINAL_NOCLASS_JAC_END"
    # study_name = "Multiplex_HPT_ACDC_VERY_LAST_ONE" # GOOD
    # study_name = "Multiplex_HPT_ACDC_DIMENSIONS"
    # study_name = "Multiplex_HPT_ACDC_DIMENSIONS_ALL"
    study_name = "Multiplex_HPT_ACDC_DIMENSIONS_SUM"
    # study_name = "Multiplex_HPT_ACDC_DIMENSIONS_ALL_SUM"

    # study_name = 'Multiplex_HPT_ACDC_Pred'
    # study_name = 'Multiplex_HPT_ACDC'
    normalisation = 'ZNorm' # NoNorm, ZNorm, MaxMin, Spatial
    splits_file = os.path.join(data_path, experiment_name, 'nested_cv_indices.pkl')

    # Add an argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=False, default=f'{data_path}', help='Path to the data.')
    parser.add_argument('--experiment_name', type=str, required=False, default=f'{experiment_name}', help='Name of the experiment.')
    parser.add_argument('--study_name', type=str, required=False, default=f'{study_name}', help='Name of the experiment.')
    parser.add_argument('--normalization', type=str, default=f'{normalisation}', help='Normalization strategy')  
    parser.add_argument('--use_global_data', type=str2bool, default=True, help='Use global data or not.')
    parser.add_argument('--use_edges', type=str2bool, default=True, help='Use edge information or not.')
    parser.add_argument('--use_all', type=str2bool, default=False, help='Use all edges, or only AHA.')
    parser.add_argument('--use_similarity', type=str2bool, default=False, help='Use similarity instead of distance.')    
    parser.add_argument('--drop_blood_pools', type=str2bool, default=True, help='Drop the blood pools.')
    parser.add_argument('--reprocess_dataset', type=str2bool, default=False, help='Reprocess the dataset.')
    parser.add_argument('--load', type=str2bool, default=True, help='Load previous results or not.')
    parser.add_argument('--in_folds', type=int, default=5, help='Number of inner folds.')
    parser.add_argument('--out_folds', type=int, default=5, help='Number of outer folds.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set.')
    parser.add_argument('--splits_file', type=str, default=f'{splits_file}', help='File with the splits.')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=20, help='Hidden dimension')
    parser.add_argument('--space_planes', type=int, default=6, help='Number of space planes')
    parser.add_argument('--time_planes', type=int, default=6, help='Number of time planes')

    parser.add_argument('--use_norm', type=str2bool, default=True, help='Use normalization in the Encoder-Decoder')
    parser.add_argument('--use_diffusion', type=str2bool, default=False, help='Use diffusion in the model')
    parser.add_argument('--use_einsum', type=str2bool, default=False, help='Use einsum in the model')
    parser.add_argument('--agg_type', type=str, default='score', help='Aggregation type')  # sum, mean, flatten
    parser.add_argument('--use_attention', type=str2bool, default=True, help='Use attention in the model')
    parser.add_argument('--compute_derivative', type=str2bool, default=True, help='Compute the derivative')
    parser.add_argument('--use_norm_stmgcn', type=str2bool, default=True, help='Use normalization in the ST-MGCN')
    parser.add_argument('--use_bias_stmgcn', type=str2bool, default=False , help='Use bias in the ST-MGCN')
    parser.add_argument('--decode_just_latent', type=str2bool, default=False, help='Decoder just uses the latent space')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')

    parser.add_argument('--gamma_rec', type=float, default=1., help='Weight for the regression loss')
    parser.add_argument('--gamma_class', type=float, default=0., help='Weight for the classification loss')
    parser.add_argument('--gamma_bc', type=float, default=0., help='Weight for the boundary condition in the latent space')
    parser.add_argument('--gamma_lat', type=float, default=0.1, help='L2 weight for the latent space')
    parser.add_argument('--gamma_graph', type=float, default=0., help='Weight for the graph regularization')

    return parser


if __name__ == "__main__":    
    seed_everything()

    # Get the parameters
    parser = get_parser()

    # Parse the argument
    args = parser.parse_args()
    derivatives_folder = args.data_folder  
    experiment_name = args.experiment_name
    study_name = args.study_name
    normalization = args.normalization
    use_edges = args.use_edges
    use_global_data = args.use_global_data
    use_all = args.use_all
    use_similarity = args.use_similarity
    drop_blood_pools = args.drop_blood_pools
    load_previous = args.load
    reprocess_datasets = args.reprocess_dataset

    in_folds = args.in_folds
    out_folds = args.out_folds
    test_size = args.test_size
    splits_file = args.splits_file
    num_trials = 10

    print(args)
    
    #  ==================== Device setup ====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'

    #  ==================== Problem setup ====================    
    # Load the datasets
    track_experiment = False

    # Use of which data
    use_position = True
    use_region_id = True
    use_time = False

    # Normalization options
    norm_by_group = False # Normalize by group, i.e: healthy controls
    norm_only_ed = False  # Normalize only by the ED frame

    # Save folder
    use_blood_pools =  not drop_blood_pools
    wkspc_name = f"Edges-{use_edges}_Norm-{normalization}_Global-{use_global_data}_All-{use_all}_Sim-{use_similarity}_BP-{use_blood_pools}"
    save_folder = os.path.join(derivatives_folder, experiment_name, f'{wkspc_name}', f"{study_name}")
    os.makedirs(save_folder, exist_ok=True)

    # Configure the logger
    log_filename = os.path.join(save_folder, 'optimization.log')
    if os.path.isfile(log_filename) and not load_previous:
        os.remove(log_filename)
    logging.basicConfig(filename=f'{log_filename}', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the datasets (graph format)
    graph_train_dataset = get_data(derivatives_folder, 
                                   wkspc_name=f'{experiment_name}', 
                                   is_test=False, 
                                   reprocess_datasets=reprocess_datasets, 
                                   drop_blood_pools=drop_blood_pools, 
                                   use_similarity=use_similarity,
                                   use_all=use_all)
            
    # Default model parameters
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    hidden_dim_ext = 0
    dropout = 0. # Not used
    init_lr = args.init_lr
    weight_decay = args.weight_decay
    l1_weight = 0.
    l2_weight = 0.
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    use_attention = args.use_attention
    use_edges = True # Use the edges in the model / predict edges or just use ones
    use_constant_edges = False   # If False take the average of the edges initially and update them? 
    use_hdyn = True
    cond_on_time = False
    decode_just_latent = args.decode_just_latent

    # Loss weights
    gamma_rec = args.gamma_rec  # Regression
    gamma_class = args.gamma_class  # Classification
    gamma_lat = args.gamma_lat  # Latent space
    gamma_bc = args.gamma_bc  # Boundary condition
    gamma_graph = args.gamma_graph  # Graph regularization

    # dt_step_size = 1/((duration/dt)*0.5)
    # dt_step_size = 0.05 # ?
    dt_step_size = 0.1
    hidden_dim_ext = 6
    
    default_config = {'hidden_dim': hidden_dim, 
                      'hidden_dim_ext': hidden_dim_ext, 
                      'latent_dim': latent_dim,
                      'dropout': dropout,
                      'init_lr': init_lr, 
                      'weight_decay': weight_decay, 
                      'l1_weight': l1_weight, 
                      'l2_weight': l2_weight, 
                      'batch_size': batch_size, 
                      'num_epochs': num_epochs,
                      'use_attention': use_attention,
                      'use_constant_edges': use_constant_edges,
                      'gamma_rec': gamma_rec,
                      'gamma_class': gamma_class,
                      'gamma_lat': gamma_lat,
                      'gamma_bc': gamma_bc,
                      'gamma_graph': gamma_graph,
                      'use_region': use_region_id,
                      'decode_just_latent': decode_just_latent,
                      'cond_on_time': cond_on_time,
                      'use_hdyn': use_hdyn,
                      'use_position': use_position,
                      'encode_time': use_time,
                      'only_space': False,
                      'dt_step': dt_step_size,
                      'use_diffusion': args.use_diffusion,
                      'use_einsum': args.use_einsum,
                      'agg_type': args.agg_type,
                      'compute_derivative': args.compute_derivative,
                      'use_norm_stmgcn': args.use_norm_stmgcn,
                      'use_bias_stmgcn': args.use_bias_stmgcn,
                      'use_norm': args.use_norm,
                      }

    objective_optuna = Objective_Multiplex(study_name,
                                           graph_train_dataset,
                                           normalization=normalization,
                                           norm_by_group=norm_by_group,
                                           norm_only_ed=norm_only_ed,
                                           save_dir=save_folder,
                                           direction="maximize",
                                           device = device,
                                           track_experiment=False,
                                           use_global_data=use_global_data,
                                           use_position=use_position,
                                           use_region_id=use_region_id,
                                           use_time=use_time,
                                           use_weighted_sampler=False,
                                           use_focal_loss=False,
                                           class_dim=5,
                                           classify=False,
                                           only_spatial=False,                                          
                                           space_planes=args.space_planes,
                                           time_planes=args.time_planes,
                                           depth_nodes=2,
                                           depth_edges=2,
                                           use_edges=use_edges,
                                           use_norm=args.use_norm,
                                           use_mse=False,
                                           )
    objective_optuna.set_default_params(default_config)

    # ============================================================================================================================
    # ============================================================================================================================
    # ======================================================== Single run ========================================================
    # Indices for the train / valid split
    train_idx = graph_train_dataset.idx_train
    valid_idx = graph_train_dataset.idx_valid
    test_idx = graph_train_dataset.idx_test
    objective_optuna.set_indices(train_idx, valid_idx, test_idx=test_idx)

    # Just one run with the default parameters
    tmp_save = os.path.join(save_folder, 'FinalModel')
    if os.path.isdir(tmp_save) and not load_previous:
        os.system(f"rm {tmp_save}/model.pt")
        os.system(f"rm {tmp_save}/checkpoint.pt")
    os.makedirs(tmp_save, exist_ok=True)

    results_hp_folder = os.path.join(derivatives_folder, experiment_name, 'results')
    df_params_path = os.path.join(results_hp_folder, f'{study_name}_trials.csv')
    df_params = pd.read_csv(df_params_path)
    df_params.dropna(how='any', inplace=True)
    df_params = df_params.sort_values(by='value', ascending=False)
    # params_names = [key for key in df_params.columns if key.startswith('params_')]
    # best_params = df_params.iloc[0].to_dict()
    # # best_params = df_params.iloc[0:5][params_names].mean().to_dict()
    # best_params = {key.replace('params_', ''): value for key, value in best_params.items() if key.startswith('params_')}
    best_params = get_best_params(df_params.iloc[0:5], use_median=True)

    model_params = objective_optuna.default_params.copy()
    model_params.update(best_params)
    print(best_params)

    # Store the objective parameters
    json_filename = os.path.join(save_folder, 'objective_params.json')
    with open(json_filename, 'w') as f:
        json.dump(model_params, f, indent=4)

    # Model
    model = objective_optuna.build_model(model_params)
    # res_training = objective_optuna._train(model, objective_optuna.default_params, tmp_save, final_model=False)
    res_training = objective_optuna._train(model, model_params, tmp_save, final_model=True)  # Reload

    final_model = f"{tmp_save}/model.pt"    
    append_att = "_Att" if use_attention else ""
    dt_string = str(dt_step_size).replace('.', '')
    study_model_copy = os.path.join(tmp_save, f'model_{dt_string}{append_att}.pt')
    os.system(f"cp {final_model} {study_model_copy}")

    # Get the features to predict
    idx_thickness = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Thickness_Median']))[0]
    idx_volume = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Volume_Index']))[0]
    idx_median = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Intensity_Median']))[0]
    idx_J = np.where(np.isin(objective_optuna.dataset.list_node_features, ['J_Median']))[0]
    fts_to_predict = np.concatenate([idx_thickness, idx_volume])
    # fts_to_predict = np.concatenate([idx_thickness, idx_volume, idx_median])
    # fts_to_predict = np.concatenate([idx_thickness, idx_volume, idx_J])
    
    time_to_predict = torch.arange(0, 50, 1)
    pred_trajectory, pred_latent, tgt_trajectory = objective_optuna.predict_from_latent(model, objective_optuna.dataset, time_to_predict, model_params, device=device)    

    df_errors = get_data_in_original_scale(model, objective_optuna, model_params, save_folder, pred_trajectory, 
                                           fts_to_predict=fts_to_predict, true_trajectory=tgt_trajectory, normalization=normalization)
    df_errors.to_csv(os.path.join(save_folder, 'errors_per_feature.csv'))
    
    latex_table = get_latex_table(df_errors, objective_optuna)
    latex_table = latex_table.replace('train', 'train window')
    latex_table= latex_table.replace('predict', 'extrapolation')
    wrapped_table = wrap_latex_table(latex_table, caption="Errors per feature in the test set.", label="tab:feature_errors_ACDC")        
    print(wrapped_table)

    latent_filename = os.path.join(save_folder, 'latent_data.csv')
    plot_combined_trajectories(model, objective_optuna, model_params, save_folder, pred_latent, pred_trajectory, plot_individual=False, plot_spatial=False, 
                            fts_to_predict=fts_to_predict, true_trajectory=tgt_trajectory, normalization=normalization)
    
    plot_predicted_trajectories(objective_optuna, pred_latent, pred_trajectory, save_folder, fts_to_predict=fts_to_predict,
                                normalization=normalization, plot_individual=False, true_trajectory=tgt_trajectory, plot_spatial=False)
    save_training_convergence(res_training, save_folder)
    plot_results(model, objective_optuna, model_params, save_folder, plot_individual=False, plot_spatial=False, fts_to_predict=fts_to_predict)
    export_latent_data(model, objective_optuna, model_params, save_folder)

    # ==================================================== Classification using random forest
    # Load the data . pkl
    data_filename = os.path.join(save_folder, 'data.pkl')
    data_model = torch.load(data_filename)

    # Get the summary statistics
    # latent_traj_fts = extract_trajectory_features(torch.tensor(data_model['latent_trajectories']).float()).data.cpu().numpy()

    # Load the latent data    
    latent_data = pd.read_csv(latent_filename, index_col=0)
    ids = latent_data[['Subject']].copy()
    y = latent_data[['labels']].copy().values.squeeze()
    latent_info = latent_data.drop(columns=['Subject', 'labels']).copy()

    latent_edges_filename = os.path.join(save_folder, 'latent_edges.csv')
    latent_edges = pd.read_csv(latent_edges_filename, index_col=0)
    latent_edges.drop(columns=['Subject'], inplace=True)

    latent_data_std_filename = os.path.join(save_folder, 'latent_data_std.csv')
    latent_data_std = pd.read_csv(latent_data_std_filename, index_col=0)
    latent_data_std.drop(columns=['Subject'], inplace=True)

    results_filename = os.path.join(save_folder, 'model_results.csv')
    params_filename = os.path.join(save_folder, 'params.csv')
    search_filename = os.path.join(save_folder, 'search.pkl')

    # X = latent_traj_fts
    # X = pd.concat([latent_info, latent_edges], axis=1).to_numpy()
    # X = pd.concat([latent_info, latent_data_std], axis=1).to_numpy()
    X = latent_info.to_numpy()  # Initial state and control 
    # X = data_model['latent_trajectories'].reshape(y.shape[0], -1)  # The whole trajectory

    if load_previous and os.path.isfile(results_filename):
        df_results = pd.read_csv(results_filename, index_col=0)
        mean_acc = df_results.iloc[0].mean()
        std_acc = df_results.iloc[0].std()
        print(f"Acc. {mean_acc}, Std. {std_acc}")
    else:
        # Now, let's set-up a nested CV for the classification    
        params = [
            # {'n_estimators': [601, 1001], 'criterion': ['gini'], 'class_weight': ['balanced']}, #'max_depth': [10, 20]},
            # {'n_estimators': [601, 1001], 'criterion': ['gini', 'entropy', 'log_loss'], 'class_weight': ['balanced', 'balanced_subsample']},
            {'n_estimators': [601, 1001], 'criterion': ['gini', 'entropy', 'log_loss'], 'class_weight': ['balanced', 'balanced_subsample'], 'max_depth': [10, 20]},
            # {'n_estimators': [601, 1001], 'reg_alpha': np.linspace(0.1, 0.9, 3), 'reg_lambda': np.linspace(0.1, 0.9, 3), 'subsample': [0.5, 0.75]},
            # {'n_estimators': [601, 1001], 'reg_alpha': np.linspace(0.1, 0.9, 3), 'reg_lambda': np.linspace(0.1, 0.9, 3), 'subsample': [0.5, 0.75], 'max_depth': [10, 20]},
            # {'C': np.asarray([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), 'penalty': ['l2'], 'class_weight': ['balanced']},
        ]

        names = [
            "Random_Forest",
            # "XGBoost",
            # "Linear_SVM",
        ]

        classifiers = [
            RandomForestClassifier(),
            # xgb.XGBClassifier(eval_metric='mlogloss', num_class=5, learning_rate=0.05, n_jobs=2, objective='multi:softprob'),
            # svm.LinearSVC(dual=True, fit_intercept=True),
        ]
        
        out_folds = 1
        in_folds = 3
        acc_res = np.zeros((out_folds, in_folds))
        params_dict = dict()
        search_dict = dict()
            
        train_indices = np.concatenate([train_idx, valid_idx])
        y_train = y[train_indices]
        for ix_out in range(0, out_folds):        
            cv_indices = list()
            skfold = StratifiedKFold(n_splits=in_folds, shuffle=False,)
            for idx, (fold_train_index, fold_test_index) in enumerate(skfold.split(train_indices, y_train)):
                cv_indices.append(
                    {'X_train': train_indices[fold_train_index], 'X_valid': train_indices[fold_test_index],
                    'y_train': y_train[fold_train_index], 'y_valid': y_train[fold_test_index]})
                            
            acc_cv, params_cv, search_cv = cv_model_scaler(X, y, classifiers[0], params[0], in_folds, cv_indices, test_idx, rerun_best=False, 
                                                        in_params=None, use_scaler=False, n_jobs=2)
            print(acc_cv)
            # Save the results
            search_dict[f"{ix_out}"] = search_cv
            params_dict[f"{ix_out}"] = params_cv
            # list_params = list_params + params_cv
            acc_res[ix_out] = acc_cv.reshape(-1)

        acc_res
        # Train a final one with the best parameters
        print(f"{acc_res.mean():.3f}")
        df_params = pd.concat([pd.DataFrame(params_dict[p]) for p in params_dict], keys=params_dict.keys())
        df_params.reset_index(inplace=True)
        df_params.rename(columns={'level_1':'Inner_Fold', 'level_0': 'Outer_Fold'}, inplace=True)

        best_params = df_params[list(params[0].keys())].iloc[np.argmax(acc_res.reshape(-1))]
        best_params = best_params.to_dict()
        if 'n_estimators' in best_params:
            best_params['n_estimators'] = int(best_params['n_estimators'])

        # Store the results
        df_results = pd.DataFrame(data=acc_res, columns=[f'Fold_{x}' for x in range(0, in_folds)])
        df_results.to_csv(results_filename)

        # Store the parameters        
        df_params.to_csv(params_filename)

        # Store the search object
        with open(search_filename, 'wb') as file:
            pickle.dump(search_dict, file)