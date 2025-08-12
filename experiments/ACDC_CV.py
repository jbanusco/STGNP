import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import logging
import pickle
import json
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from model.train_stmgcn_ode import Objective_Multiplex
from model.testing_model import get_accuracies, save_training_convergence, save_mean_positions, plot_subject_trajectory, plot_subject_position, plot_state
from utils.plot_utils import plot_confusion_matrx, plot_latent_space, to_long_format, plot_with_error

from dataset.dataset_utils import get_data
from dataset.GraphDataset import GraphDataset
from utils.utils import seed_everything, str2bool

from utils.model_selection_optuna import hypertune_optuna, optuna_cv, optuna_nested_cv
from utils.model_selection_sklearn import cv_stratified_split
from utils.plot_utils import plot_latent_trajectories, save_mean_trajectories

import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')


def reshape_array_to_plot(input_array):
    # Reshape the array to a long format
    reshaped_arr = input_array.reshape(-1)

    # Create indices for sample, feature, and node
    num_samples, num_features, num_nodes = input_array.shape
    sample_indices = np.repeat(np.arange(num_samples), num_features * num_nodes)
    feature_indices = np.tile(np.repeat(np.arange(num_features), num_nodes), num_samples)
    node_indices = np.tile(np.arange(num_nodes), num_samples * num_features)

    # Create the DataFrame
    df = pd.DataFrame({
        'Sample': sample_indices,
        'Feature': feature_indices,
        'Node': node_indices,
        'Value': reshaped_arr.flatten()
    })

    assert np.all(np.isclose(df.query(f"Node==0 and Feature==0")['Value'].values, input_array[:, 0, 0], atol=1e-5))
    assert np.all(np.isclose(df.query(f"Node==1 and Feature==1")['Value'].values, input_array[:, 1, 1], atol=1e-5))
    # assert np.all(np.isclose(df.query(f"Node==10 and Feature==1")['Value'].values, input_array[:, 1, 10], atol=1e-5))
    # assert np.all(np.isclose(df.query(f"Node==10 and Feature==5")['Value'].values, input_array[:, 5, 10], atol=1e-5))

    return df


def export_latent_data(trained_model, objective, params_model, save_path):
    n_samples = 1
    normalization = objective.normalization
    output = objective.get_output_of_model(trained_model, objective.dataset, None, None, n_samples=n_samples, params=params_model)
    # labels_tmp = objective.dataset.label

    class_data = output[0]  # Classification ouptut    
    rec_ft = output[1]  # Reconstruction output - mean
    rec_std = output[2]  # Reconstruction output - std
    latent_rec_plot = output[3] # Latent space trajectorys
    q_context = output[4]
    force = output[5]  # Force of the system
    q_space_ctx = output[6]  # Initial edges
    q_time_ctx = output[7]
    pred_g = output[8]
    if objective.class_dim > 0:
        tgt_data = output[-3]
        labels = output[-2].to('cpu')
    else:
        tgt_data = output[-3]
        labels = torch.ones((rec_ft.shape[0], 1))
    
    # Transform back target and 3ions
    rec_data = rec_ft.detach().cpu()
    rec_data_std = rec_std.detach().cpu()
    target_data = tgt_data.detach().cpu()

    # Edges
    init_space_edges = q_space_ctx.mean.float().cpu().detach().numpy()
    init_time_edges = q_time_ctx.mean.float().cpu().detach().numpy()
    df_space_edges = reshape_array_to_plot(init_space_edges)
    df_time_edges = reshape_array_to_plot(init_time_edges)

    df_space = df_space_edges.pivot(index='Sample', columns=['Feature', 'Node'], values='Value')
    df_space.columns = [f'Node_{node}_Space_{feature}' for feature, node in df_space.columns]
    df_time = df_time_edges.pivot(index='Sample', columns=['Feature', 'Node'], values='Value')
    df_time.columns = [f'Node_{node}_Time_{feature}' for feature, node in df_time.columns]
    df_edges = pd.concat([df_space, df_time], axis=1)
    df_edges['Subject'] = objective.dataset.sub_id
    df_edges.to_csv(os.path.join(save_path, 'latent_edges.csv'))

    # Create dataframe with the latent data and the labels        
    init_params = q_context.mean.float().cpu().detach().numpy()
    std_init_params = q_context.scale.float().cpu().detach().numpy()
    latent_numpy = latent_rec_plot.cpu().detach().numpy()
    num_latent = latent_numpy.shape[1]

    # # Save data in pkl
    # output_data = {'latent_trajectories': latent_numpy,
    #                'reconstructed_trajectories': rec_data,
    #                'reconstructed_std_trajectories': rec_data_std,
    #                'target_trajectories': target_data,
    #                'labels': labels,}
    # torch.save(output_data, os.path.join(save_path, 'latent_data.pkl'))

    # Plot the initial and control parameters
    init_points = init_params[:, :trained_model.latent_dim]
    std_init_points = std_init_params[:, :trained_model.latent_dim]
    control_params = init_params[:, trained_model.latent_dim:]
    std_control_params = std_init_params[:, trained_model.latent_dim:]
    assert trained_model.dyn_params == control_params.shape[1]

    # Example plot
    df_control = reshape_array_to_plot(control_params)
    df_init = reshape_array_to_plot(init_points)

    # Pivot df_control to create an array of sample value using the feature and node as columns, and rename the
    # columns to be more descriptive based on the node and feture combination
    df_control = df_control.pivot(index='Sample', columns=['Feature', 'Node'], values='Value')
    df_control.columns = [f'Node_{node}_Control_{feature}' for feature, node in df_control.columns]
    df_init = df_init.pivot(index='Sample', columns=['Feature', 'Node'], values='Value')
    df_init.columns = [f'Node_{node}_Init_{feature}' for feature, node in df_init.columns]
    df_joined = pd.concat([df_control, df_init], axis=1)

    df_control_std = reshape_array_to_plot(std_control_params)
    df_init_std = reshape_array_to_plot(std_init_points)
    df_control_std = df_control_std.pivot(index='Sample', columns=['Feature', 'Node'], values='Value')
    df_control_std.columns = [f'Node_{node}_Control_{feature}_std' for feature, node in df_control_std.columns]
    df_init_std = df_init_std.pivot(index='Sample', columns=['Feature', 'Node'], values='Value')
    df_init_std.columns = [f'Node_{node}_Init_{feature}_std' for feature, node in df_init_std.columns]
    df_joined_std = pd.concat([df_control_std, df_init_std], axis=1)
    df_joined_std['Subject'] = objective.dataset.sub_id
    df_joined_std.to_csv(os.path.join(save_path, 'latent_data_std.csv'))

    # Get the labels    
    df_joined['labels'] = labels.cpu().unsqueeze(-1).numpy()
    df_joined['Subject'] = objective.dataset.sub_id
    df_joined.to_csv(os.path.join(save_path, 'latent_data.csv'))


def get_model_results(trained_model, objective, params_model, save_path, fts_to_predict=None, traj_dim=None):
    n_samples = 1
    normalization = objective.normalization
    output = objective.get_output_of_model(trained_model, objective.dataset, None, None, n_samples=n_samples, params=params_model)
    # labels_tmp = objective.dataset.label

    class_data = output[0]  # Classification ouptut    
    rec_ft = output[1]  # Reconstruction output - mean
    rec_std = output[2]  # Reconstruction output - std
    latent_rec_plot = output[3] # Latent space trajectorys
    q_context = output[4]
    force = output[5]  # Force of the system    
    # if objective.class_dim > 0:
    #     tgt_data = output[-2]
    #     labels = output[-1].to('cpu')
    # else:
    tgt_data = output[-3]
    labels = output[-2]
    context_pts = output[-1]
    if objective.class_dim == 0 or labels is None:
        labels = torch.ones((rec_ft.shape[0], 1))
    else:
        labels = labels.to('cpu')
    
    # Transform back target and predictions
    rec_data = rec_ft.detach().cpu()
    rec_data_std = rec_std.detach().cpu()
    target_data = tgt_data.detach().cpu()

    if objective.use_position:
        positions_dim = 3 # x, y, z [TODO: need to add it in the objective]
        traj_dim = rec_data.shape[1] - positions_dim if traj_dim is None else traj_dim
    else:
        traj_dim = rec_data.shape[1] if traj_dim is None else traj_dim

    if isinstance(objective.dataset, GraphDataset):
        # Trajectories
        rec_trajectories = rec_data[:, :traj_dim]
        rec_std_trajectories = rec_data_std[:, :traj_dim]
        tgt_trajectories = target_data[:, :traj_dim]

        # Positions
        rec_pos = rec_data[:, traj_dim:]
        rec_pos_std = rec_data_std[:, traj_dim:]
        tgt_pos = target_data[:, traj_dim:]
    else:
        num_fts = len(objective.dataset.list_node_features)
        rec_trajectories = rec_data[:, :num_fts]
        rec_std_trajectories = rec_data_std[:, :num_fts]
        tgt_trajectories = target_data[:, :num_fts]

        if objective.use_position:
            num_pos = objective_optuna.dataset.nodes_data[0]['pos'].shape[1]
            # num_pos = len(objective.dataset.list_pos_features)
            rec_pos = rec_data[:, num_fts:num_fts+num_pos]
            rec_pos_std = rec_data_std[:, num_fts:num_fts+num_pos]
            tgt_pos = target_data[:, num_fts:num_fts+num_pos]
    
    if fts_to_predict is None:
        fts_to_predict = np.arange(0, traj_dim)

    if normalization == 'ZNorm':
        num_subjects, _, _, time_frames = rec_trajectories.shape
        trans_fts_predict = objective.dataset._transform['nfeatures']
        if isinstance(objective.dataset, GraphDataset):
            trans_fts_mean = trans_fts_predict.mean[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_std = trans_fts_predict.std[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
        else:
            trans_fts_mean = trans_fts_predict.mean.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_std = trans_fts_predict.std.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)

        rec_trajectories = (rec_trajectories * trans_fts_std) + trans_fts_mean
        rec_std_trajectories = (rec_std_trajectories * trans_fts_std) #+ trans_fts_mean
        tgt_trajectories = (tgt_trajectories * trans_fts_std) + trans_fts_mean
    elif normalization == 'MaxMin':
        # Need to implement the MaxMin normalization
        num_subjects, _, _, time_frames = rec_trajectories.shape
        trans_fts_predict = objective.dataset._transform['nfeatures']
        if isinstance(objective.dataset, GraphDataset):
            trans_fts_max = trans_fts_predict.max[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_min = trans_fts_predict.min[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
        else:
            trans_fts_max = trans_fts_predict.max.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
            trans_fts_min = trans_fts_predict.min.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3)
        range_fts = trans_fts_max - trans_fts_min
        rec_trajectories = (rec_trajectories * range_fts) + trans_fts_min
        rec_std_trajectories = (rec_std_trajectories * range_fts)
        tgt_trajectories = (tgt_trajectories * range_fts) + trans_fts_min
    else:
        # Need to implement the spatial normalization
        pass

    # Positions
    if normalization == 'ZNorm' and objective.use_position:
        num_subjects, _, _, time_frames = rec_pos.shape
        trans_pos = objective.dataset._transform['pos']  # num_regions * num_features
        num_pos = objective.dataset.nodes_data[0]['pos'].shape[1]
        trans_pos_std = trans_pos.std.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]
        trans_pos_mean = trans_pos.mean.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]

        rec_pos = (rec_pos * trans_pos_std) + trans_pos_mean
        rec_pos_std = (rec_pos_std * trans_pos_std) #+ trans_pos_mean
        tgt_pos = (tgt_pos * trans_pos_std) + trans_pos_mean
    elif normalization == 'MaxMin' and objective.use_position:
        num_subjects, _, _, time_frames = rec_pos.shape
        trans_pos = objective.dataset._transform['pos']  # num_regions * num_features
        num_pos = objective.dataset.nodes_data[0]['pos'].shape[1]
        trans_pos_max = trans_pos.max.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]
        trans_pos_min = trans_pos.min.unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames).permute(0, 2, 1, 3) #[:, :2]
        
        range_pos = trans_pos_max - trans_pos_min
        rec_pos = (rec_pos * range_pos) + trans_pos_min
        rec_pos_std = (rec_pos_std * range_pos)
        tgt_pos = (tgt_pos * range_pos) + trans_pos_min
    else:
        # Need to implement the spatial normalization
        pass

    train_idx = objective.train_idx
    valid_idx = objective.valid_idx
    test_idx = objective.test_idx

    if objective.class_dim > 0:
        # ==== Plot the confusion matrix ==== 
        # Rows: true classes
        # Columns: predicted classes
        # -- Get accuracy
        dict_labels = objective.dataset.global_data[['Group', 'Group_Cat']].dropna().drop_duplicates().set_index('Group_Cat').to_dict()['Group']    
        acc_train, acc_valid, acc_test = get_accuracies(class_data, labels, train_idx, valid_idx, test_idx,
                                                        store_confusion=False, dict_labels=dict_labels, save_path=save_path)
        # print(f"Train accuracy: {acc_train}")
        # print(f"Valid accuracy: {acc_valid}")
        # print(f"Test accuracy: {acc_test}")

        predictions = class_data.argmax(dim=1).cpu().numpy()
    else:
        predictions = labels.clone().cpu().numpy()
        dict_labels = None

    init_params = q_context.mean.float().cpu().detach().numpy()
    latent_numpy = latent_rec_plot.float().cpu().detach().numpy()
    
    # Plot the initial and control parameters
    init_points = init_params[:, :trained_model.latent_dim]
    control_params = init_params[:, trained_model.latent_dim:]
    assert trained_model.dyn_params == control_params.shape[1]

    # ------ Prepare the data for the plots
    final_data = {}

    # 1 == Reconstructed and target features
    final_data['rec_trajectories'] = rec_trajectories  # [B, F, N, T]
    final_data['rec_std_trajectories'] = rec_std_trajectories  # [B, F, N, T]
    final_data['tgt_trajectories'] = tgt_trajectories  # [B, F, N, T]

    # 2 == Reconstructed and target positions
    if objective.use_position:
        final_data['rec_pos'] = rec_pos  # [B, F, N, T]
        final_data['rec_pos_std'] = rec_pos_std  # [B, F, N, T]
        final_data['tgt_pos'] = tgt_pos  # [B, F, N, T]

    # 3 == Latent space trajectories
    final_data['latent_trajectories'] = latent_numpy  # [B, L, N, T]

    # 4 == Initial and control parameters
    final_data['init_params'] = init_points  # [B, L]
    final_data['control_params'] = control_params  # [B, D]

    # 5 == Predictions and labels
    final_data['labels'] = labels  # [B,]
    final_data['predictions'] = predictions  # [B,]
    final_data['subject_ids'] = np.array(objective.dataset.sub_id)  # [B,]
    final_data['dict_labels'] = dict_labels

    # 6 == Sotore the parameters
    final_data['params'] = params_model

    # 7 == Store the context points
    final_data['context_pts'] = context_pts  # [B, L]

    # Save it!
    save_path = os.path.join(save_path, 'data.pkl')
    torch.save(final_data, save_path)

    return final_data



def get_combined_trajectories(trained_model, objective, params_model, save_path, pred_latent, pred_trajectory, plot_individual=False, plot_spatial=False, 
                               fts_to_predict=None, transform=lambda x: x, true_trajectory=None, traj_dim=None, 
                               normalization="ZNorm", save_format='png'):

    assert save_format in ['png', 'pdf', 'svg', 'eps'], "Save format not supported. Use png, pdf, svg or eps"
    
    # Get the data of the training time
    model_data = get_model_results(trained_model, objective, params_model, save_path, fts_to_predict)
    
    # Trajectories
    rec_trajectories_train = model_data['rec_trajectories']
    rec_std_trajectories_train = model_data['rec_std_trajectories']
    tgt_trajectories_train = model_data['tgt_trajectories']

    # Positions
    if objective.use_position:
        rec_pos_train = model_data['rec_pos']
        rec_pos_std_train = model_data['rec_pos_std']
        tgt_pos = model_data['tgt_pos']

    # Latent
    latent_trajectories = model_data['latent_trajectories']
    control_params = model_data['control_params']
    init_points = model_data['init_params']

    # Labels
    labels = model_data['labels']
    predictions = model_data['predictions']
    dict_labels = model_data['dict_labels']

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
    
    # Combine the trajectories during training and at prediction time
    combined_trajectories = torch.cat([rec_trajectories_train.detach().cpu(), rec_trajectories], dim=-1)
    combined_std_trajectories = torch.cat([rec_std_trajectories_train.detach().cpu(), rec_std_trajectories], dim=-1)
    combined_tgt_trajectories = torch.cat([tgt_trajectories_train.detach().cpu(), true_trajectory], dim=-1) if true_trajectory is not None else tgt_trajectories_train.detach().cpu()

    return combined_trajectories, combined_std_trajectories, combined_tgt_trajectories



def plot_combined_trajectories(trained_model, objective, params_model, save_path, pred_latent, pred_trajectory, plot_individual=False, plot_spatial=False, 
                               fts_to_predict=None, transform=lambda x: x, true_trajectory=None, traj_dim=None, 
                               normalization="ZNorm", save_format='png', dot_size=4, vertical_layout=False):

    assert save_format in ['png', 'pdf', 'svg', 'eps'], "Save format not supported. Use png, pdf, svg or eps"
    
    # Get the data of the training time
    model_data = get_model_results(trained_model, objective, params_model, save_path, fts_to_predict)
    
    # Trajectories
    rec_trajectories_train = model_data['rec_trajectories']
    rec_std_trajectories_train = model_data['rec_std_trajectories']
    tgt_trajectories_train = model_data['tgt_trajectories']

    # Positions
    if objective.use_position:
        rec_pos_train = model_data['rec_pos']
        rec_pos_std_train = model_data['rec_pos_std']
        tgt_pos = model_data['tgt_pos']

    # Latent
    latent_trajectories = model_data['latent_trajectories']
    control_params = model_data['control_params']
    init_points = model_data['init_params']

    # Labels
    labels = model_data['labels']
    predictions = model_data['predictions']
    dict_labels = model_data['dict_labels']

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
    
    # Combine the trajectories during training and at prediction time
    combined_trajectories = torch.cat([rec_trajectories_train.detach().cpu(), rec_trajectories], dim=-1)
    combined_std_trajectories = torch.cat([rec_std_trajectories_train.detach().cpu(), rec_std_trajectories], dim=-1)
    combined_tgt_trajectories = torch.cat([tgt_trajectories_train.detach().cpu(), true_trajectory], dim=-1) if true_trajectory is not None else tgt_trajectories_train.detach().cpu()

    # =======================================================================
    # =========================== Actual plotting ===========================
    # =======================================================================
    context_pts = model_data['context_pts'].detach().cpu().numpy()

    # Latent
    combined_latent_trajectories = np.concatenate([latent_trajectories, pred_latent[..., 1:].detach().cpu().numpy()], axis=-1)
    num_training_steps = latent_trajectories.shape[-1]
    time_frames = combined_trajectories.shape[-1]
    # ==== Plot latent space trajectories    
    num_latent = combined_latent_trajectories.shape[1]
    time_axis = np.arange(0, time_frames, 1)
    fig, ax = plt.subplots(1, num_latent, figsize=(30, 6), squeeze=False)
    for idx_l in range(num_latent):
        ax[0][idx_l].plot(time_axis, combined_latent_trajectories[:, idx_l].mean(axis=0).T) 
        ax[0][idx_l].set_xlabel('Time')
        # ax[0][idx_l].set_ylabel(f'Latent {idx_l}')
        ax[0][idx_l].set_title(f'Latent {idx_l+1}')
        ax[0][idx_l].axvline(num_training_steps, color='r', linestyle='--')
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(save_path, f'latent_trajectories_combined.{save_format}'), bbox_inches='tight', dpi=300)
    plt.close('all')


    # ==== Plot trajectories and positions

    # Apply transform to the reconstruction
    rec_trajectories = transform(combined_trajectories)
    true_trajectory = transform(combined_tgt_trajectories) if true_trajectory is not None else None
    rec_std_trajectories = combined_std_trajectories
    # ==== Plot trajectories and positions
    save_mean_trajectories(rec_trajectories, true_trajectory, 
                           append_name='mean_combined', num_training_steps=num_training_steps, same_axis=True, 
                           true_in_training=combined_tgt_trajectories,
                           save_path=save_path,
                           fts_names=objective.dataset.features,
                           context_pts=context_pts,
                           save_format=save_format,
                           dot_size=dot_size, vertical_layout=vertical_layout)
    
    list_subjects = objective.dataset.sub_id
    if plot_individual:
        errors = (combined_trajectories[..., :combined_tgt_trajectories.shape[-1]] - combined_tgt_trajectories).abs().sum(axis=(1,2,3))
        train_idx_subject = errors[train_idx].argmin() # train_idx[0]
        test_idx_subject = errors[test_idx].argmin() # test_idx[0]
        plot_subject_trajectory(rec_trajectories, rec_std_trajectories, true_trajectory, save_path, sub_id=train_idx_subject, num_training_steps=num_training_steps,
                                identifier=f"{list_subjects[train_idx_subject]}_train", append='_combined', fts_names=objective.dataset.features, same_axis=True, true_in_training=combined_tgt_trajectories,
                                context_pts=context_pts, save_format=save_format)
        plot_subject_trajectory(rec_trajectories, rec_std_trajectories, true_trajectory, save_path, sub_id=test_idx_subject, num_training_steps=num_training_steps,
                                identifier=f"{list_subjects[test_idx_subject]}_test", append='_combined', fts_names=objective.dataset.features, same_axis=True, true_in_training=combined_tgt_trajectories,
                                context_pts=context_pts, save_format=save_format)

        # Plot the spatial layout
        if plot_spatial:
            # Train
            append_id = f"{list_subjects[train_idx[0]]}_train_predicted"
            save_path = os.path.join(save_path, f'subject-{list_subjects[train_idx[0]]}_train', f'surface_plot_{append_id}.png')
            # Plot three time frames [init, mid, end]
            plotted_frames = [0, time_frames//2, time_frames-1]
            A = np.ones((int(objective.dataset.num_nodes**0.5), int(objective.dataset.num_nodes**0.5)))
            plot_state(rec_trajectories, A, save_path, dim_state=0, x_tgt=true_trajectory, frames_idx=plotted_frames, subj_idx=0)

            append_id = f"{list_subjects[test_idx[0]]}_test_predicted"
            save_path = os.path.join(save_path, f'subject-{list_subjects[test_idx[0]]}_test', f'surface_plot_{append_id}.png')
            # Plot three time frames [init, mid, end]
            plotted_frames = [0, time_frames//2, time_frames-1]
            A = np.ones((int(objective.dataset.num_nodes**0.5), int(objective.dataset.num_nodes**0.5)))
            plot_state(rec_trajectories, A, save_path, dim_state=0, x_tgt=true_trajectory, frames_idx=plotted_frames, subj_idx=0)



def get_data_in_original_scale(trained_model, objective, params_model, save_path, pred_trajectory, 
                               fts_to_predict=None, true_trajectory=None, traj_dim=None, normalization="ZNorm", transform=lambda x: x):
        
    # Get the data of the training time
    model_data = get_model_results(trained_model, objective, params_model, save_path, fts_to_predict)
    
    # Trajectories
    rec_trajectories_train = model_data['rec_trajectories']
    rec_std_trajectories_train = model_data['rec_std_trajectories']
    tgt_trajectories_train = model_data['tgt_trajectories']

    # Positions
    if objective.use_position:
        rec_pos_train = model_data['rec_pos']
        rec_pos_std_train = model_data['rec_pos_std']
        tgt_pos = model_data['tgt_pos']

    # Latent
    latent_trajectories = model_data['latent_trajectories']
    control_params = model_data['control_params']
    init_points = model_data['init_params']

    # Labels
    labels = model_data['labels']
    predictions = model_data['predictions']
    dict_labels = model_data['dict_labels']

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
    
    # Combine the trajectories during training and at prediction time
    combined_trajectories = torch.cat([rec_trajectories_train.detach().cpu(), rec_trajectories], dim=-1)
    combined_std_trajectories = torch.cat([rec_std_trajectories_train.detach().cpu(), rec_std_trajectories], dim=-1)
    combined_tgt_trajectories = torch.cat([tgt_trajectories_train.detach().cpu(), true_trajectory], dim=-1) if true_trajectory is not None else tgt_trajectories_train.detach().cpu()

    combined_trajectories = transform(combined_trajectories)
    combined_std_trajectories = transform(combined_std_trajectories)
    combined_tgt_trajectories = transform(combined_tgt_trajectories)

    # =======================================================================
    # ===================== Now, compute errors =============================
    lenght_train_window = rec_trajectories_train.shape[-1]
    errors_windows = ['train'] if true_trajectory is None else ['train', 'predict']
    use_idx = test_idx
    # [Batch, Features, Nodes, Time]
    # Compute the errors - per feature across nodes, time and batch
    df_errors = pd.DataFrame(data=None)
    for window in errors_windows:
        window_tgt = combined_tgt_trajectories[use_idx, :, :, lenght_train_window:] if window == 'predict' else combined_tgt_trajectories[use_idx, :, :, :lenght_train_window]
        window_rec = combined_trajectories[use_idx, :, :, lenght_train_window:] if window == 'predict' else combined_trajectories[use_idx, :, :, :lenght_train_window]
        window_rec_std = combined_std_trajectories[use_idx, :, :, lenght_train_window:] if window == 'predict' else combined_std_trajectories[use_idx, :, :, :lenght_train_window]

        # Errors: [B, F, N, T]
        errors = torch.FloatTensor(window_tgt - window_rec)

        # Median Absolute Error
        # mse = errors.square().mean(dim=(0,2,3))
        # mse_std = errors.square().std(dim=(0,2,3))
        mae = errors.abs()
        med_ae = mae.quantile(0.5, dim=0).quantile(0.5, dim=1).quantile(0.5, dim=1)
        q1_ae = mae.permute(1, 0, 2, 3).flatten(start_dim=1).quantile(0.25, dim=1)
        q3_ae = mae.permute(1, 0, 2, 3).flatten(start_dim=1).quantile(0.75, dim=1)
        iqr_ae = q3_ae - q1_ae        

        # Median Squared Error
        # mae = errors.abs().mean(dim=(0,2,3))
        # mae_std = errors.abs().std(dim=(0,2,3))
        mse = errors.square()
        med_se = mse.quantile(0.5, dim=0).quantile(0.5, dim=1).quantile(0.5, dim=1)
        q1_se = mse.permute(1, 0, 2, 3).flatten(start_dim=1).quantile(0.25, dim=1)
        q3_se = mse.permute(1, 0, 2, 3).flatten(start_dim=1).quantile(0.75, dim=1)
        iqr_se = q3_se - q1_se

        # Relative errors
        tmp_mse = (errors / (window_tgt + 1e-8)).square()
        mse_percentage = (tmp_mse.quantile(0.5, dim=0).quantile(0.5, dim=1).quantile(0.5, dim=1)) * 100
        
        tmp_mae = (errors / (window_tgt + 1e-8)).abs()
        mae_percentage = tmp_mae.quantile(0.5, dim=0).quantile(0.5, dim=1).quantile(0.5, dim=1) * 100
            
        errors_arrays = torch.cat([med_se.unsqueeze(0), iqr_se.unsqueeze(0), mse_percentage.unsqueeze(0),
                                   med_ae.unsqueeze(0), iqr_ae.unsqueeze(0), mae_percentage.unsqueeze(0)], dim=0)
        errors_names = ['MSE', 'MSE_IQR', 'MSE [%]', 'MAE', 'MAE_IQR', 'MAE [%]']
                
        df_errors_window = pd.DataFrame(errors_arrays.numpy(), index=errors_names, columns=objective.dataset.features)                        
        df_errors_window['Set'] = window
        df_errors = pd.concat([df_errors, df_errors_window], axis=0)
    
    df_errors = df_errors.reset_index().rename(columns={'index': 'Metric'})
    
    return df_errors
    


def plot_results(trained_model, objective, params_model, save_path, plot_individual=False, plot_spatial=False, fts_to_predict=None, 
                 transform=lambda x: x, save_format='png', dot_size=4, vertical_layout=False):

    assert save_format in ['png', 'pdf', 'svg', 'eps'], "Save format not supported. Use png, pdf, svg or eps"

    # Get the data
    model_data = get_model_results(trained_model, objective, params_model, save_path, fts_to_predict)
    train_idx = objective.train_idx
    valid_idx = objective.valid_idx
    test_idx = objective.test_idx

    # Trajectories
    rec_trajectories = model_data['rec_trajectories']
    rec_std_trajectories = model_data['rec_std_trajectories']
    tgt_trajectories = model_data['tgt_trajectories']

    # Positions
    if objective.use_position:
        rec_pos = model_data['rec_pos']
        rec_pos_std = model_data['rec_pos_std']
        tgt_pos = model_data['tgt_pos']

    # Latent
    latent_trajectories = model_data['latent_trajectories']
    control_params = model_data['control_params']
    init_points = model_data['init_params']

    # Labels
    labels = model_data['labels']
    predictions = model_data['predictions']
    dict_labels = model_data['dict_labels']

    # ==== Plot latent space trajectories
    time_frames = latent_trajectories.shape[-1]
    plot_latent_trajectories(latent_trajectories, save_path=save_path, save_format=save_format)

    # Example plot
    df_control = reshape_array_to_plot(control_params)
    df_init = reshape_array_to_plot(init_points)
    # sns.lineplot(data=df, x='Feature', y='Value', hue='Node')    
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].boxplot(control_params); ax[0].set_title('Control')
    sns.boxenplot(data=df_control, x='Feature', y='Value', hue='Node', ax=ax[0])
    # ax[1].boxplot(init_points); ax[1].set_title('Initial')
    sns.boxenplot(data=df_init, x='Feature', y='Value', hue='Node', ax=ax[1])
    # ax[0].plot(control_params.mean(axis=0)); ax[0].set_title('Control')
    # ax[1].plot(init_points.mean(axis=0)); ax[1].set_title('Initial')
    fig.savefig(os.path.join(save_path, f'initial_values_and_params.{save_format}'), bbox_inches='tight', dpi=300)

    # ==== Plot the latent space    
    predictions_test = predictions[test_idx]
    labels_test = labels[test_idx]
    try:
        plot_latent_space(latent_trajectories, labels, predictions, os.path.join(save_path, f'latent_space.{save_format}'))        
        plot_latent_space(latent_trajectories[test_idx, :], labels_test, predictions_test, os.path.join(save_path, f'latent_space_test.{save_format}'))
    except Exception as e:
        print(f"Error in the latent space: {e}")
        pass

    # ==== Plot trajectories and positions
    context_pts = model_data['context_pts'].detach().cpu().numpy()
    rec_trajectories = transform(rec_trajectories)
    tgt_trajectories = transform(tgt_trajectories)

    trajectories_limits = save_mean_trajectories(rec_trajectories, tgt_trajectories, fts_names=objective.dataset.features, save_path=save_path, context_pts=context_pts, 
                                                 save_format=save_format, dot_size=dot_size, vertical_layout=vertical_layout, return_limits=True)
    trajectories_limits = {}
    for idx_ft, ft_name in enumerate(objective.dataset.features):
        max_value = np.array([rec_trajectories[:, idx_ft].quantile(0.99), tgt_trajectories[:, idx_ft].quantile(0.99)]).max()
        min_value = np.array([rec_trajectories[:, idx_ft].quantile(0.01), tgt_trajectories[:, idx_ft].quantile(0.01)]).min()
        trajectories_limits[ft_name] = [(min_value, max_value), (min_value, max_value)]

    if objective.use_position:
        save_mean_positions(rec_pos, tgt_pos, save_path, time_frame=10, transparent_plots=True, animated=True)
    list_subjects = objective.dataset.sub_id
    if plot_individual:
        errors = np.abs(tgt_trajectories - rec_trajectories).sum(axis=(1,2,3))
        train_idx_subject = errors[train_idx].argmin() # train_idx[0]
        test_idx_subject = errors[test_idx].argmin() # test_idx[0]
    
        plot_subject_trajectory(rec_trajectories, rec_std_trajectories, tgt_trajectories, save_path, sub_id=train_idx_subject,
                                identifier=f"{list_subjects[train_idx_subject]}_train", fts_names=objective.dataset.features, context_pts=context_pts, save_format=save_format)
        plot_subject_trajectory(rec_trajectories, rec_std_trajectories, tgt_trajectories, save_path, sub_id=test_idx_subject,
                                identifier=f"{list_subjects[test_idx_subject]}_test", fts_names=objective.dataset.features, context_pts=context_pts, save_format=save_format)

        if plot_spatial:
            # Train
            append_id = f"{list_subjects[train_idx[0]]}_train"
            save_path_fig = os.path.join(save_path, f'subject-{list_subjects[train_idx[0]]}_train', f'surface_plot_{append_id}.png')
            # Plot three time frames [init, mid, end]
            plotted_frames = [0, time_frames//2, time_frames-1]
            A = np.ones((int(objective.dataset.num_nodes**0.5), int(objective.dataset.num_nodes**0.5)))
            plot_state(rec_trajectories, A, save_path_fig, dim_state=0, x_tgt=tgt_trajectories, frames_idx=plotted_frames, subj_idx=0)

            append_id = f"{list_subjects[test_idx[0]]}_test"
            save_path_fig = os.path.join(save_path, f'subject-{list_subjects[test_idx[0]]}_test', f'surface_plot_{append_id}.png')
            # Plot three time frames [init, mid, end]
            plotted_frames = [0, time_frames//2, time_frames-1]
            A = np.ones((int(objective.dataset.num_nodes**0.5), int(objective.dataset.num_nodes**0.5)))
            plot_state(rec_trajectories, A, save_path_fig, dim_state=0, x_tgt=tgt_trajectories, frames_idx=plotted_frames, subj_idx=0)

    if objective.class_dim > 0:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            save_folder = os.path.join(save_path, f'{dict_labels[label]}')
            os.makedirs(save_folder, exist_ok=True)

            if plot_individual:
                # ==== One per group
                train_example = idx[np.isin(idx, train_idx)][0]
                test_example = idx[np.isin(idx, test_idx)][0]
                if objective.use_position:
                    plot_subject_position(rec_pos, rec_pos_std, tgt_pos, save_folder, sub_id=train_example, time_frame=10, transparent_plots=True, 
                                          identifier=f"{list_subjects[train_example]}_train")
                    plot_subject_position(rec_pos, rec_pos_std, tgt_pos, save_folder, sub_id=test_example, time_frame=10, transparent_plots=True, 
                                          identifier=f"{list_subjects[test_example]}_test")
                    
                plot_subject_trajectory(rec_trajectories, rec_std_trajectories, tgt_trajectories, save_folder, sub_id=train_example,
                                        identifier=f"{list_subjects[train_example]}_train", fts_names=objective.dataset.features, context_pts=context_pts, save_format=save_format)
                plot_subject_trajectory(rec_trajectories, rec_std_trajectories, tgt_trajectories, save_folder, sub_id=test_example, 
                                        identifier=f"{list_subjects[test_example]}_test", fts_names=objective.dataset.features, context_pts=context_pts, save_format=save_format)

            # ==== Average per group        
            save_mean_trajectories(rec_trajectories[idx], tgt_trajectories[idx], fts_names=objective.dataset.features, save_path=save_folder, 
                                   context_pts=context_pts, dot_size=dot_size, vertical_layout=vertical_layout, limits=trajectories_limits, save_format=save_format)
            if objective.use_position:
                save_mean_positions(rec_pos[idx], tgt_pos[idx], save_folder, time_frame=10, transparent_plots=True, animated=True)


def get_parser() -> argparse.ArgumentParser:
    data_path = "/media/jaume/DATA/Data/New_ACDC/MIDS/mixed/derivatives"
    # paper_folder = "/usr/data/Multiplex_Synthetic_FINAL"

    experiment_name = 'GraphClassification'
    study_name = 'Multiplex_HPT_ACDC'
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
    parser.add_argument('--space_planes', type=int, default=3, help='Number of space planes')
    parser.add_argument('--time_planes', type=int, default=3, help='Number of time planes')

    parser.add_argument('--use_norm', type=str2bool, default=True, help='Use normalization in the Encoder-Decoder')
    parser.add_argument('--use_diffusion', type=str2bool, default=False, help='Use diffusion in the model')
    parser.add_argument('--use_einsum', type=str2bool, default=False, help='Use einsum in the model')
    parser.add_argument('--agg_type', type=str, default='score', help='Aggregation type')  # sum, mean, flatten
    parser.add_argument('--use_attention', type=str2bool, default=False, help='Use attention in the model')
    parser.add_argument('--compute_derivative', type=str2bool, default=True, help='Compute the derivative')
    parser.add_argument('--use_norm_stmgcn', type=str2bool, default=True, help='Use normalization in the ST-MGCN')
    parser.add_argument('--use_bias_stmgcn', type=str2bool, default=False , help='Use bias in the ST-MGCN')
    parser.add_argument('--decode_just_latent', type=str2bool, default=False, help='Decoder just uses the latent space')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials for the hyper-parameter tuning')
    parser.add_argument('--num_jobs', type=int, default=0, help='Number of jobs in parallel')

    parser.add_argument('--gamma_rec', type=float, default=1., help='Weight for the regression loss')
    parser.add_argument('--gamma_class', type=float, default=0., help='Weight for the classification loss')
    parser.add_argument('--gamma_bc', type=float, default=0., help='Weight for the boundary condition in the latent space')
    parser.add_argument('--gamma_lat', type=float, default=0.05, help='L2 weight for the latent space')
    parser.add_argument('--gamma_graph', type=float, default=0.1, help='Weight for the graph regularization')

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
    num_trials = args.num_trials

    print(args)
    
    #  ==================== Device setup ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    #  ==================== Problem setup ====================    
    # Load the datasets
    track_experiment = False

    # Use of which data
    use_position = True
    use_region_id = True
    # use_region_id = False
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
    
    # dt_step_size = 0.05
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
    
    # Store the objective parameters
    json_filename = os.path.join(save_folder, 'objective_params.json')
    with open(json_filename, 'w') as f:
        json.dump(objective_optuna.default_params, f, indent=4)

    # ============================================================================================================================
    # ============================================================================================================================
    # ======================================================== Single run ========================================================
    # Indices for the train / valid split
    train_idx = graph_train_dataset.idx_train
    valid_idx = graph_train_dataset.idx_valid
    test_idx = graph_train_dataset.idx_test
    objective_optuna.set_indices(train_idx, valid_idx, test_idx=test_idx)

    # Just one run with the default parameters
    tmp_save = os.path.join(save_folder, 'Test_Run')
    if os.path.isdir(tmp_save) and not load_previous:
        os.system(f"rm {tmp_save}/model.pt")
        os.system(f"rm {tmp_save}/checkpoint.pt")
    os.makedirs(tmp_save, exist_ok=True)

    # Model
    model = objective_optuna.build_model(objective_optuna.default_params)
    # res_training = objective_optuna._train(model, objective_optuna.default_params, tmp_save, final_model=False)
    # res_training = objective_optuna._train(model, objective_optuna.default_params, tmp_save, final_model=True)  # Reload
    
    # ============================================================================================================================
    # ============================================================================================================================
    # ====================================== Hyper-parameter tuning on fix train-test split ======================================
    final_model = os.path.join(save_folder, 'model.pt')
    if os.path.isfile(final_model):
        os.system(f"rm {final_model}")
        checkpoint_name = os.path.join(save_folder, 'checkpoint.pt')
        if os.path.isfile(checkpoint_name):
            os.system(f"rm {checkpoint_name}")
    
    classes, count_per_class_train = np.unique(graph_train_dataset.label[graph_train_dataset.idx_train].squeeze().data.numpy(), return_counts=True)
    _, count_per_class_valid = np.unique(graph_train_dataset.label[graph_train_dataset.idx_valid].squeeze().data.numpy(), return_counts=True)
    _, count_per_class_test = np.unique(graph_train_dataset.label[graph_train_dataset.idx_test].squeeze().data.numpy(), return_counts=True)
    print("=== Counts per class:\n"
          f"Classes: {classes}\n"
          f"Train: {count_per_class_train}, {count_per_class_train.sum()}\n"
          f"Test: {count_per_class_test}, {count_per_class_test.sum()}\n"
          f"Valid: {count_per_class_valid}, {count_per_class_valid.sum()}\n")
    
    if args.num_jobs > 0:
        num_cpus = args.num_jobs
    else:
        num_cpus = multiprocessing.cpu_count() // 2
    objective_optuna.num_jobs = num_cpus
    sq_database = False  # sqlite or postgresql
    load_previous = True
    model, res_training, best_params = hypertune_optuna(objective_optuna,
                                                        save_folder,
                                                        study_name,
                                                        num_trials=num_trials,
                                                        load_previous=load_previous,
                                                        output_probs=True, max_cpus=num_cpus, 
                                                        sq_database=sq_database)

    # ============================================================================================================================
    # ============================================================================================================================
    # ============================================ Cross-validation =======================================================    
    # cv_folder = os.path.join(save_folder, "CV")
    # if os.path.isfile(splits_file):
    #     with open(splits_file, 'rb') as f:
    #         splits = pickle.load(f)
    #         cv_indices = splits['cv_indices']
    #         test_indices = splits['test_indices']
    #     cv_indices = cv_indices[0]
    #     test_idx = test_indices[0]['X_test']
    # else:
    #     all_indices = np.arange(len(graph_train_dataset))
    #     y = graph_train_dataset.label
    #     cv_indices, test_indices = cv_stratified_split(all_indices, y, k_folds=in_folds, test_size=test_size)
    #     test_idx = test_indices['X_test']

    # acc_graph, list_params, list_probs, list_decisions = optuna_cv(objective_optuna, 
    #                                                                cv_folder, 
    #                                                                k_folds=in_folds, 
    #                                                                cv_indices=cv_indices, 
    #                                                                test_idx=test_idx, 
    #                                                                num_trials=num_trials, 
    #                                                                append="", 
    #                                                                load_previous=load_previous,
    #                                                                )

    # ============================================================================================================================
    # ============================================================================================================================
    # ============================================ Nested cross-validation =======================================================    
    # df_results, df_params, save_cv_data = optuna_nested_cv(objective_optuna,
    #                                                        save_folder,
    #                                                        out_folds=out_folds,
    #                                                        in_folds=in_folds,
    #                                                        test_size=test_size,
    #                                                        num_trials=num_trials,
    #                                                        load_previous=load_previous,
    #                                                        splits_file=splits_file,
    #                                                        redo=True,
    #                                                        )

    # =======================================================================
    # ==================== Visualization of latent space ====================
    # n_samples = 1    
    # all_params = default_config
    # all_params.update(best_params)
    # output = objective_optuna.get_output_of_model(model, objective_optuna.dataset, None, None, n_samples=n_samples, params=all_params)    
    # labels_tmp = objective_optuna.dataset.label