import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
import seaborn as sns
import torch
from matplotlib.animation import PillowWriter

from dataset.GraphDataset import GraphDataset
from utils.plot_utils import plot_confusion_matrx, plot_latent_space, to_long_format, plot_with_error, plot_latent_trajectories, save_mean_trajectories



def plot_predicted_trajectories(objective, pred_latent, pred_trajectory, save_folder,
                                normalization='ZNorm', plot_individual=True, traj_dim=None, 
                                fts_to_predict=None, true_trajectory=None, plot_spatial=False,
                                save_format='png',
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
    assert save_format in ['png', 'pdf', 'eps', 'svg'], "Save format not supported. Use png, pdf, svg or eps."

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
    # fig.savefig(os.path.join(save_folder, 'latent_trajectories_predicted.png'))
    fig.savefig(os.path.join(save_folder, f'latent_trajectories_predicted.{save_format}'), bbox_inches='tight', dpi=300)
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
                                identifier=f"{list_subjects[train_idx[0]]}_train", append='_predicted', 
                                fts_names=objective.dataset.features, save_format=save_format)
        plot_subject_trajectory(rec_trajectories, rec_std_trajectories, true_trajectory, save_folder, sub_id=test_idx[0], 
                                identifier=f"{list_subjects[test_idx[0]]}_test", append='_predicted', fts_names=objective.dataset.features,
                                save_format=save_format)

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


def compute_accuracy(pred_probs, tgt_labels, indices,
                     store_confusion=False, dict_labels=None, 
                     save_path=None, name='train'):
    # Get the predictions per set
    predictions = pred_probs.argmax(dim=1).cpu().numpy()
    predictions = predictions[indices]
    tgt_labels = tgt_labels[indices]

    if store_confusion:
        plot_confusion_matrx(tgt_labels, predictions, dict_labels, os.path.join(save_path, f'confusion_matrix_{name}.csv'), name=f"{name}")

    return np.sum(predictions == tgt_labels.squeeze().cpu().numpy()) / len(tgt_labels)        


def get_accuracies(pred_probs, tgt_labels, train_idx, valid_idx, test_idx, 
                   store_confusion=False, dict_labels=None, save_path=None):
    acc_train = compute_accuracy(pred_probs, tgt_labels, train_idx, store_confusion, dict_labels, save_path=save_path, name='train')
    acc_valid = compute_accuracy(pred_probs, tgt_labels, valid_idx, store_confusion, dict_labels, save_path=save_path, name='valid')
    acc_test = compute_accuracy(pred_probs, tgt_labels, test_idx, store_confusion, dict_labels, save_path=save_path, name='test')

    return acc_train, acc_valid, acc_test


def plot_subject_trajectory(pred_data, pred_data_std, tgt_data, save_path, 
                            sub_id=0,
                            num_training_steps=None,
                            identifier=None, 
                            append='',
                            same_axis=True,
                            true_in_training=None,
                            fts_names=None,
                            context_pts=None,
                            save_format='png',
                            ):

    assert save_format in ['png', 'pdf', 'svg', 'eps'], "save_format must be either 'png', 'pdf', 'svg', or 'eps'"

    if identifier is None:
        identifier = sub_id
    
    save_folder = os.path.join(save_path, f'subject-{identifier}')
    os.makedirs(save_folder, exist_ok=True)

    if not isinstance(pred_data, np.ndarray):
        pred_data = pred_data.detach().cpu().numpy()
    
    if not isinstance(pred_data_std, np.ndarray):
        pred_data_std = pred_data_std.detach().cpu().numpy()

    if not isinstance(tgt_data, np.ndarray) and tgt_data is not None:
        tgt_data = tgt_data.detach().cpu().numpy()

    num_rows = 2 if (tgt_data is not None or true_in_training is not None) else 1
    rec_data_lower = pred_data[sub_id] - pred_data_std[sub_id]
    rec_data_upper = pred_data[sub_id] + pred_data_std[sub_id]

    rec_data_long = to_long_format(pred_data[sub_id], rec_data_lower, rec_data_upper, 'rec_data')
    if tgt_data is not None:
        target_data_long = to_long_format(tgt_data[sub_id], tgt_data[sub_id], tgt_data[sub_id], 'target_data')  # No std data for target_data
    elif true_in_training is not None:
        target_data_long = to_long_format(true_in_training[sub_id], true_in_training[sub_id], true_in_training[sub_id], 'target_data')  # No std data for target_data

    # Plotting using seaborn
    n_features = pred_data.shape[1]
    if fts_names is None:
        fts_names = [f'Feature {i+1}' for i in range(n_features)]
    
    fig, ax = plt.subplots(num_rows, n_features, figsize=(20, 6), squeeze=False)
    for i in range(n_features):
        plot_with_error(rec_data_long, f'rec_data {i+1}', ax[0][i], add_error=False)
        if context_pts is not None:
            plot_rec_data = rec_data_long[rec_data_long['Channel'] == f'rec_data {i+1}']
            context_pts = np.array(context_pts)                            

            # Subset the target data for context time points
            context_df = plot_rec_data[plot_rec_data['Time'].isin(context_pts)].copy()

            # Add a column to mark them (optional)
            context_df['Context'] = True

            # Plot scatter points
            sns.scatterplot(data=context_df, x='Time', y='Mean', hue='Trajectory', ax=ax[0][i], 
                            legend=False, marker='o', s=100)
    
        ax[0][i].set_title(f'Predicted {fts_names[i]}')
        if tgt_data is not None or true_in_training is not None:
            plot_tgt_data = target_data_long[target_data_long['Channel'] == f'target_data {i+1}']
            sns.lineplot(data=plot_tgt_data, x='Time', y='Mean', hue='Trajectory', ax=ax[1][i], legend=False)
            ax[1][i].set_title(f'Target {fts_names[i]}')
            ax[0][i].set_xlabel('Time')
            ax[1][i].set_ylabel(f'{fts_names[i]}')
            if same_axis:
                # ax[0][i].set_ylim(ax[1][i].get_ylim())
                y_min = min(ax[0][i].get_ylim()[0], ax[1][i].get_ylim()[0])
                y_max = max(ax[0][i].get_ylim()[1], ax[1][i].get_ylim()[1])
                ax[0][i].set_ylim(y_min, y_max)
                ax[1][i].set_ylim(y_min, y_max)
            ax[1][i].set_xlim((0, pred_data.shape[-1]))

            if context_pts is not None:
                context_pts = np.array(context_pts)                            

                # Subset the target data for context time points
                context_df = plot_tgt_data[plot_tgt_data['Time'].isin(context_pts)].copy()

                # Add a column to mark them (optional)
                context_df['Context'] = True

                # Plot scatter points
                sns.scatterplot(data=context_df, x='Time', y='Mean', hue='Trajectory', ax=ax[1][i], 
                                legend=False, marker='o', s=100)
            
        ax[0][i].set_xlabel('Time')
        ax[0][i].set_ylabel(f'{fts_names[i]}')
        ax[0][i].set_xlim((0, pred_data.shape[-1]))
                
        if num_training_steps is not None:
            ax[0][i].axvline(num_training_steps, color='r', linestyle='--', label='Training steps')
            if tgt_data is not None:
                ax[1][i].axvline(num_training_steps, color='r', linestyle='--', label='Training steps')

    fig.tight_layout(pad=0.5)
    # fig.savefig(os.path.join(save_folder, f'trajectories-subject_{identifier}{append}.png'))
    fig.savefig(os.path.join(save_folder, f'trajectories-subject_{identifier}{append}.{save_format}'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def position_plot(in_pred_pos, in_tgt_pos, time_frame=0, animated=False, type='2D'):
    if isinstance(in_pred_pos, pd.DataFrame):
        pred_pos = in_pred_pos.copy()
        tgt_pos = in_tgt_pos.copy()

        x_pred = pred_pos.query("Channel=='rec_data 1'")[['Mean', 'Time', 'Trajectory']]
        x_pred = x_pred.pivot(index='Trajectory', columns='Time', values='Mean').values

        y_pred = pred_pos.query("Channel=='rec_data 2'")[['Mean', 'Time', 'Trajectory']]
        y_pred = y_pred.pivot(index='Trajectory', columns='Time', values='Mean').values

        z_pred = pred_pos.query("Channel=='rec_data 3'")[['Mean', 'Time', 'Trajectory']]
        z_pred = z_pred.pivot(index='Trajectory', columns='Time', values='Mean').values
        
        x_target = tgt_pos.query("Channel=='target_data 1'")[['Mean', 'Time', 'Trajectory']]
        x_target = x_target.pivot(index='Trajectory', columns='Time', values='Mean').values

        y_target = tgt_pos.query("Channel=='target_data 2'")[['Mean', 'Time', 'Trajectory']]
        y_target = y_target.pivot(index='Trajectory', columns='Time', values='Mean').values

        z_target = tgt_pos.query("Channel=='target_data 3'")[['Mean', 'Time', 'Trajectory']]
        z_target = z_target.pivot(index='Trajectory', columns='Time', values='Mean').values
    else:
        # Get the components
        x_pred = in_pred_pos[0]
        y_pred = in_pred_pos[1]

        x_target = in_tgt_pos[0]
        y_target = in_tgt_pos[1]        

        if type == '3D':
            z_pred = in_pred_pos[2]
            z_target = in_tgt_pos[2]
        
    if type == '2D':
        if animated:
            import matplotlib.animation as animation
            fig, ax = plt.subplots(1, 1, squeeze=False)
            scatter_pred = ax[0][0].scatter(x_pred[..., time_frame], y_pred[..., time_frame], c='r', label='Predicted')
            scatter_tgt = ax[0][0].scatter(x_target[..., time_frame], y_target[..., time_frame], c='b', label='Target')
            ax[0][0].set_xlabel("X")
            ax[0][0].set_ylabel("Y")
            ax[0][0].set_title("XY region coordinates")
            def update(frame):
                scatter_pred.set_offsets(np.c_[x_pred[..., frame], y_pred[..., frame]])
                scatter_tgt.set_offsets(np.c_[x_target[..., frame], y_target[..., frame]])
                ax[0][0].set_title(f"XY region coordinates - frame {frame}")

            ani = animation.FuncAnimation(fig, update, frames=y_pred.shape[-1], interval=50)
            return ani
        else:
            # 2D projection
            fig, ax = plt.subplots(1, 1, squeeze=False)
            ax[0][0].scatter(x_pred, y_pred, c='r', label='Predicted')
            ax[0][0].scatter(x_target, y_target, c='b', label='Target')
            ax[0][0].set_xlabel("X")
            ax[0][0].set_ylabel("Y")
            ax[0][0].set_title("XY region coordinates")
            return fig
    
    elif type == '3D':        
        if animated:
            import matplotlib.animation as animation
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            scatter_pred = ax.scatter(x_pred[..., time_frame], y_pred[..., time_frame], z_pred[..., time_frame], c='r', label='Predicted')
            scatter_tgt = ax.scatter(x_target[..., time_frame], y_target[..., time_frame], z_target[..., time_frame], c='b', label='Target')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            # Get axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()

            def update(frame):
                # Keep the axes fixed                        
                ax.clear()
                ax.scatter(x_pred[..., frame], y_pred[..., frame], z_pred[..., frame], c='r', marker='.')
                ax.scatter(x_target[..., frame], y_target[..., frame], z_target[..., frame], c='b', marker='.')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)

                # scatter_pred.set_offsets(np.c_[x_pred[..., frame], y_pred[..., frame], z_pred[..., frame]])
                # scatter_tgt.set_offsets(np.c_[x_target[..., frame], y_target[..., frame], z_target[..., frame]])
                # scatter_pred._offsets3d = (x_pred[..., frame], y_pred[..., frame], z_pred[..., frame])
                # scatter_tgt._offsets3d = (x_target[..., frame], y_target[..., frame], z_target[..., frame])
                # ax.set_title(f"XYZ region coordinates - frame {frame}")
            
            ani = animation.FuncAnimation(fig, update, frames=y_pred.shape[-1], interval=50)
            return ani
        else:
            # 3D projection
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(x_pred[..., time_frame], y_pred[..., time_frame], z_pred[..., time_frame], c='r', marker='.')
            ax.scatter(x_target[..., time_frame], y_target[..., time_frame], z_target[..., time_frame], c='b', marker='.')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            return fig
        
    else:
        raise ValueError("Type not supported")
    

def plot_state(x, A, save_path, dim_state=0, x_tgt=None, frames_idx=[0], subj_idx=0):
    # Get spatial dimensions
    nx = A.shape[0]
    ny = A.shape[1]

    # Create grid for plotting
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Create figure
    num_frames = len(frames_idx)
    fig, axes = plt.subplots(2, num_frames, figsize=(20, 10), squeeze=False, subplot_kw={'projection': '3d'})

    for ix, frame_ix in enumerate(frames_idx):
        x_tmp = (x[..., frame_ix][subj_idx]).permute(1, 0).cpu().detach().numpy()
        x_tmp_tgt = (x_tgt[..., frame_ix][subj_idx]).permute(1, 0).cpu().detach().numpy()

        # Reshape the state
        z = x_tmp[..., dim_state].reshape(nx, ny)
        
        # Get the target state
        z_tgt = None
        if x_tgt is not None:
            z_tgt = x_tmp_tgt[..., dim_state].reshape(nx, ny)

        # Plot prediction as a 3D surface    
        surf_pred = axes[0, ix].plot_surface(X, Y, z.T, cmap='viridis', edgecolor='k', alpha=0.8)
        axes[0, ix].set_title('Prediction')
        axes[0, ix].set_xlabel('Node X')
        axes[0, ix].set_ylabel('Node Y')
        axes[0, ix].set_zlabel('Value')
        fig.colorbar(surf_pred, ax=axes[0, ix], shrink=0.5, aspect=10)

        if z_tgt is not None:
            # Plot ground truth as a 3D surface        
            surf_tgt = axes[1, ix].plot_surface(X, Y, z_tgt.T, cmap='viridis', edgecolor='k', alpha=0.8)
            axes[1, ix].set_title('Ground Truth')
            axes[1, ix].set_xlabel('Node X')
            axes[1, ix].set_ylabel('Node Y')
            axes[1, ix].set_zlabel('Value')
            fig.colorbar(surf_tgt, ax=axes[1, ix], shrink=0.5, aspect=10)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)




def save_mean_positions(pred_pos, tgt_pos, save_path, 
                        time_frame=10, transparent_plots=True, animated=True):
    # Get the mean value
    pred_pos = pred_pos.mean(axis=0)
    tgt_pos = tgt_pos.mean(axis=0)

    fig_2d = position_plot(pred_pos, tgt_pos, time_frame=time_frame, animated=False, type='2D')
    fig_2d.savefig(os.path.join(save_path, f'aha_coordinates2D-frame_{time_frame}.png'), dpi=300, transparent=transparent_plots)
    plt.close('all')

    # fig_3d = position_plot(pred_pos, tgt_pos, time_frame=time_frame, animated=False, type='3D')
    # fig_3d.savefig(os.path.join(save_path, f'aha_coordinates3D-frame_{time_frame}.png'), dpi=300, transparent=transparent_plots)
    # plt.close('all')

    ani_2d = position_plot(pred_pos, tgt_pos, time_frame=time_frame, animated=True, type='2D')
    # ani_2d.save(os.path.join(save_path, 'aha_coordinates2D.gif'), writer='imagemagick', dpi=300)
    ani_2d.save(os.path.join(save_path, 'aha_coordinates2D.gif'), writer=PillowWriter(fps=30), dpi=300)
    plt.close('all')

    # ani_3d = position_plot(pred_pos, tgt_pos, time_frame=time_frame, animated=True, type='3D')
    # ani_3d.save(os.path.join(save_path, 'aha_coordinates3D.gif'), writer='imagemagick', dpi=300)
    # plt.close('all')


def plot_subject_position(pred_pos, pred_pos_std, tgt_pos, save_path, sub_id=0, time_frame=10, transparent_plots=True, identifier=None):
    if identifier is None:
        identifier = sub_id

    rec_data_lower = pred_pos[sub_id] - pred_pos_std[sub_id]
    rec_data_upper = pred_pos[sub_id] + pred_pos_std[sub_id]
    rec_data_long = to_long_format(pred_pos[sub_id], rec_data_lower, rec_data_upper, 'rec_data')
    target_data_long = to_long_format(tgt_pos[sub_id], tgt_pos[sub_id], tgt_pos[sub_id], 'target_data')  # No std data for target_data

    save_folder = os.path.join(save_path, f'subject-{identifier}')
    os.makedirs(save_folder, exist_ok=True)

    fig_2d = position_plot(rec_data_long, target_data_long, time_frame=time_frame, animated=False, type='2D')
    fig_2d.savefig(os.path.join(save_folder, f'aha_coordinates2D-subject_{identifier}.png'), dpi=300, transparent=transparent_plots)
    plt.close('all')

    # fig_3d = position_plot(rec_data_long, target_data_long, time_frame=time_frame, animated=False, type='3D')
    # fig_3d.savefig(os.path.join(save_folder, f'aha_coordinates3D-subject_{identifier}.png'), dpi=300, transparent=transparent_plots)
    # plt.close('all')

    ani_2d = position_plot(rec_data_long, target_data_long, time_frame=time_frame, animated=True, type='2D')    
    # ani_2d.save(os.path.join(save_folder, f'aha_coordinates2D-subject_{identifier}.gif'), writer='imagemagick', dpi=300)
    ani_2d.save(os.path.join(save_folder, f'aha_coordinates2D-subject_{identifier}.gif'), writer=PillowWriter(fps=30), dpi=300)
    plt.close('all')

    # ani_3d = position_plot(rec_data_long, target_data_long, time_frame=time_frame, animated=True, type='3D')
    # ani_3d.save(os.path.join(save_folder, f'aha_coordinates3D-subject_{identifier}.gif'), writer='imagemagick', dpi=300)
    # plt.close('all')


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
            num_pos = objective.dataset.nodes_data[0]['pos'].shape[1]
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



def get_latex_table(df_errors, objective):
    # Pivot and merge errors into MSE ± std, MAE ± std format
    formatted_rows = []
    feature_names = objective.dataset.features

    for set_name in df_errors['Set'].unique():
        for feat in feature_names:
            mse = df_errors.query("Set == @set_name and Metric == 'MSE'")[feat].values[0]
            mse_std = df_errors.query("Set == @set_name and Metric == 'MSE_IQR'")[feat].values[0]
            mae = df_errors.query("Set == @set_name and Metric == 'MAE'")[feat].values[0]
            mae_std = df_errors.query("Set == @set_name and Metric == 'MAE_IQR'")[feat].values[0]
            mse_pct = df_errors.query("Set == @set_name and Metric == 'MSE [%]'")[feat].values[0]
            mae_pct = df_errors.query("Set == @set_name and Metric == 'MAE [%]'")[feat].values[0]

            formatted_rows.append({
                "Set": set_name,
                "Feature": feat,
                "MSE ± IQR": f"{mse:.4f} ± {mse_std:.4f}",
                "MAE ± IQR": f"{mae:.4f} ± {mae_std:.4f}",
                "MSE [%]": f"{mse_pct:.2f}%",
                "MAE [%]": f"{mae_pct:.2f}%"
            })

    df_latex = pd.DataFrame(formatted_rows)

    # Reorder
    df_latex = df_latex[['Set', 'Feature', 'MSE ± IQR', 'MAE ± IQR', 'MSE [%]', 'MAE [%]']]

    # Convert to LaTeX
    latex_table = df_latex.to_latex(index=False, escape=False, column_format='llcccc')

    return latex_table


def format_latex_math(value_str):
    value_str = value_str.replace("±", r"$\pm$").replace("%", r"$\%$")
    return f"${value_str}$"


def wrap_latex_table(latex_body: str,
                        caption: str = "Prediction and reconstruction errors.",
                        label: str = "tab:prediction_errors",
                        position: str = "!htb",
                        columnwidth: str = "\\columnwidth",
                        tabcolsep: int = 3) -> str:
    """
    Wrap a LaTeX table body with resizing, caption, and formatting options.

    Parameters:
        latex_body (str): The LaTeX string from DataFrame.to_latex(...).
        caption (str): Table caption.
        label (str): Table label for referencing.
        position (str): Placement specifier.
        columnwidth (str): Width for \resizebox.
        tabcolsep (int): Column padding in pt (default 3).

    Returns:
        str: Full LaTeX table string.
    """
    
    latex_body = format_latex_math(latex_body)
    wrapped = f"""\\begin{{table}}[{position}]
    \\centering
    \\setlength\\tabcolsep{{{tabcolsep}pt}} % default value: 6pt
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\resizebox{{{columnwidth}}}{{!}}{{%
    {latex_body}
    }} % end resizebox
    \\end{{table}}"""
    
    return wrapped



def save_training_convergence(training_results, save_path, save_format='png'):
    assert save_format in ['png', 'pdf', 'svg', 'eps'], "save_format must be either 'png', 'pdf', 'svg', or 'eps'"
    
    # ==== Plot convergence results
    logging.info("Plotting the convergence results...")
    epochs = np.arange(0, len(training_results['losses_train']), 1)

    # Put together all the losses values
    all_losses = np.stack((training_results['losses_train'], training_results['losses_valid'], training_results['losses_test']), axis=1).flatten()
    all_losses = all_losses[~np.isnan(all_losses)]
    max_limit = np.quantile(all_losses, 0.95)
    min_limit = np.min(all_losses) - np.min(all_losses)*0.1

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    # Losses
    ax[0].plot(epochs, training_results['losses_train'], label='Train loss')
    ax[0].plot(epochs, training_results['losses_valid'], label='Val loss')
    ax[0].plot(epochs, training_results['losses_test'], label='Test loss')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Losses')
    ax[0].set_ylim(min_limit, max_limit)

    # Accuracy
    acc_train_track = training_results['metrics_train']['accuracy']
    acc_valid_track = training_results['metrics_valid']['accuracy']
    acc_test_track = training_results['metrics_test']['accuracy']
    ax[1].plot(epochs, acc_train_track, label='Train acc')
    ax[1].plot(epochs, acc_valid_track, label='Val acc')
    ax[1].plot(epochs, acc_test_track, label='Test acc')    
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy')
    
    # fig.savefig(os.path.join(save_path, 'training_results.png'))
    fig.savefig(os.path.join(save_path, f'training_results.{save_format}'), bbox_inches='tight', dpi=300)
    logging.info("Convergence results plotted!")

    logging.info(f"ALL DONE. Results available at: {save_path}")        
    plt.close('all')