import logging
import os
import pandas as pd
import numpy as np
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


# Reshape data to long format for seaborn
def to_long_format(mean_data, lower_data, upper_data, label):
    n_channels, n_trajectories, n_timepoints = mean_data.shape
    df_list = []
    for ch in range(n_channels):
        for tr in range(n_trajectories):
            df = pd.DataFrame({
                'Time': np.arange(n_timepoints),
                'Mean': mean_data[ch, tr, :],
                'Lower': lower_data[ch, tr, :],
                'Upper': upper_data[ch, tr, :],
                'Trajectory': tr,
                'Channel': f'{label} {ch + 1}'
            })
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Plotting with seaborn and custom standard deviation
def plot_with_error(data, channel_label, ax, add_error=True):
    channel_data = data[data['Channel'] == channel_label]
    sns.lineplot(data=channel_data, x='Time', y='Mean', hue='Trajectory', ax=ax, legend=False)
    if add_error:
        for key, grp in channel_data.groupby('Trajectory'):
            ax.fill_between(grp['Time'], grp['Lower'], grp['Upper'], alpha=0.3)


def plot_confusion_matrx(labels, predictions, dict_labels, save_filename, name=''):    
    cm_train = confusion_matrix(labels.squeeze().cpu().numpy(), predictions)
    df_conf_train = pd.DataFrame(data=cm_train, columns=np.unique(labels), index=np.unique(labels))
    df_conf_train.rename(columns=dict_labels, index=dict_labels, inplace=True)
    print(f"Confusion matrix: {name}")
    print(df_conf_train)
    logging.info(f"Confusion matrix: {name}")
    logging.info(f"{df_conf_train}")
    df_conf_train.to_csv(save_filename)


def plot_latent_space(latent_space, labels, save_filename, predictions=None):
    num_samples = latent_space.shape[0]

    logging.info("Computing the latent space...")
    latent_vectors_train = latent_space #.detach().cpu().numpy()
    latent_vectors = latent_vectors_train.reshape(num_samples, -1)
    
    if not isinstance(latent_vectors, np.ndarray):
        latent_vectors = latent_vectors.cpu().numpy()

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300, init='pca', n_jobs=1)
    latent_tsne = tsne.fit_transform(latent_vectors)
    logging.info("Latent space computed!")

    # Labels for colors       
    all_labels = labels.squeeze().cpu().numpy()

    # Identify misclassified points
    # misclassified_indices = np.where(all_labels != predictions)[0]
    # correct_indices = np.where(all_labels == predictions)[0]

    # assuming you have a numpy array of labels called `labels`
    logging.info("Plotting the latent space...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter plot for all classified data with marker 'o'
    ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=all_labels, cmap='jet', marker='o')
    # ax.scatter(latent_tsne[correct_indices, 0], latent_tsne[correct_indices, 1], c=all_labels[correct_indices], cmap='jet', marker='o', label='Correct')

    # # Scatter plot for misclassified data with marker 'x'
    # ax.scatter(latent_tsne[misclassified_indices, 0], latent_tsne[misclassified_indices, 1], c=all_labels[misclassified_indices], cmap='jet', marker='x', label='Missclassified')

    # Create a ScalarMappable for color mapping
    sm = ScalarMappable(cmap='jet')
    sm.set_array(all_labels)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Labels')

    plt.legend()
    fig.savefig(save_filename, dpi=300, bbox_inches='tight')
    logging.info("Latent space plotted!")


def plot_latent_trajectories(latent_trajectories, save_path=None, save_format='png'):
    assert save_format in ['png', 'pdf', 'svg', 'eps'], "save_format must be either 'png', 'pdf', 'svg', or 'eps'"
    # latent_rec_plot = output[3] # Latent space trajectorys
    # latent_numpy = latent_rec_plot.cpu().detach().numpy()

    # latent_numpy  # [B, L, N, T]
    # ==== Plot latent space trajectories
    num_latent = latent_trajectories.shape[1]
    time_frames = latent_trajectories.shape[-1]
    time_axis = np.arange(0, time_frames, 1)

    fig, ax = plt.subplots(1, num_latent, figsize=(20, 5), squeeze=False)
    for idx_l in range(num_latent):
        ax[0][idx_l].plot(time_axis, latent_trajectories[:, idx_l].mean(axis=0).T)
        ax[0][idx_l].set_xlabel('Time')
        # ax[0][idx_l].set_ylabel(f'Latent {idx_l}')
        ax[0][idx_l].set_title(f'Latent {idx_l+1}')
    fig.tight_layout(pad=0.5)

    if save_path is not None:
        fig.savefig(os.path.join(save_path, f'latent_trajectories.{save_format}'), bbox_inches='tight', dpi=300)
        plt.close('all')
    else:
        return fig
    

def save_mean_trajectories(pred_data, 
                           tgt_data,
                           num_training_steps=None,
                           same_axis=True,
                           append_name='mean_trajectories',
                           true_in_training=None,
                           fts_names=['Thickness', 'Volume', 'Intensity'],
                           context_pts=None,
                           save_format='png',
                           save_path=None,
                           dot_size=2,
                           vertical_layout=False,
                           return_limits=False,
                           limits=None):

    assert save_format in ['png', 'pdf', 'svg', 'eps'], "save_format must be either 'png', 'pdf', 'svg', or 'eps'"

    if not isinstance(pred_data, np.ndarray):
        pred_data = pred_data.detach().cpu().numpy()

    if tgt_data is not None and not isinstance(tgt_data, np.ndarray):
        tgt_data = tgt_data.detach().cpu().numpy()
    
    # Take the mean of the trajectories
    mean_pred_data = pred_data.mean(axis=0)
    if tgt_data is not None:
        mean_tgt_data = tgt_data.mean(axis=0)
        num_rows = 2
        time_axis_tgt = np.arange(0, tgt_data.shape[-1], 1)
    elif true_in_training is not None:
        mean_tgt_data = true_in_training.mean(axis=0)
        num_rows = 2
        time_axis_tgt = np.arange(0, true_in_training.shape[-1], 1)

        # Extend true_in_training with NaNs if it's shorter than pred_data
        if true_in_training.shape[-1] < pred_data.shape[-1]:
            padding = pred_data.shape[-1] - true_in_training.shape[-1]
            # Correct padding dimensions: ((Features, 0), (Regions, 0), (Time, padding))
            mean_tgt_data = np.pad(mean_tgt_data, ((0, 0), (0, 0), (0, padding)),  mode='constant', constant_values=np.nan)
            # Ensure the time axis matches pred_data
            time_axis_tgt = np.arange(0, pred_data.shape[-1], 1)
    else:
        num_rows = 1

    time_axis_pred = np.arange(0, pred_data.shape[-1], 1)
    # Get the number of features
    n_features = mean_pred_data.shape[0]
    if len(fts_names) != n_features:  # just in case
        fts_names = [f'Feature {i}' for i in range(n_features)]

    # fig, ax = plt.subplots(num_rows, n_features, figsize=(20, 5), squeeze=False)
    if vertical_layout:
        fig, ax = plt.subplots(n_features, num_rows, figsize=(20, 5 * n_features), squeeze=False)
        # ax = ax.T  # Transpose to keep [row][col] = [data row][feature]
        print("Using vertical layout for plotting.")
    else:
        fig, ax = plt.subplots(num_rows, n_features, figsize=(20, 5), squeeze=False)


    for i in range(n_features):
        ax_pred = ax[0][i] if not vertical_layout else ax[i][0]
        ax_tgt = ax[1][i] if not vertical_layout and num_rows > 1 else (ax[i][1] if vertical_layout and num_rows > 1 else None)
        # ax[0][i].plot(time_axis_pred, mean_pred_data[i].T)
        # ax[0][i].set_title(f'Predicted {fts_names[i]}')
        ax_pred.plot(time_axis_pred, mean_pred_data[i].T)
        ax_pred.set_title(f'Predicted {fts_names[i]}')
        if tgt_data is not None:
            # ax[1][i].plot(time_axis_tgt, mean_tgt_data[i].T)
            # ax[1][i].set_title(f'Target {fts_names[i]}')            
            # ax[1][i].set_xlabel('Time')
            # ax[1][i].set_ylabel(f'{fts_names[i]}')
            ax_tgt.plot(time_axis_tgt, mean_tgt_data[i].T)
            ax_tgt.set_title(f'Target {fts_names[i]}')            
            ax_tgt.set_xlabel('Time')
            ax_tgt.set_ylabel(f'{fts_names[i]}')

            # Set the axis limits to be the same
            if same_axis:
                # ax[0][i].set_ylim(ax[1][i].get_ylim())
                # y_min = min(ax[0][i].get_ylim()[0], ax[1][i].get_ylim()[0])
                # y_max = max(ax[0][i].get_ylim()[1], ax[1][i].get_ylim()[1])
                y_min = min(ax_pred.get_ylim()[0], ax_tgt.get_ylim()[0])
                y_max = max(ax_pred.get_ylim()[1], ax_tgt.get_ylim()[1])
                # ax[0][i].set_ylim(y_min, y_max)
                # ax[1][i].set_ylim(y_min, y_max)
                ax_pred.set_ylim(y_min, y_max)
                ax_tgt.set_ylim(y_min, y_max)
            # ax[1][i].set_xlim((0, time_axis_tgt[-1]))
            # ax[0][i].set_xlim((0, time_axis_tgt[-1]))
            ax_tgt.set_xlim((0, time_axis_tgt[-1]))
            ax_pred.set_xlim((0, time_axis_tgt[-1]))
        elif true_in_training is not None:
            # ax[1][i].plot(time_axis_tgt, mean_tgt_data[i].T)
            # ax[1][i].set_title(f'Target {fts_names[i]}')            
            # ax[1][i].set_xlabel('Time')
            # ax[1][i].set_ylabel(f'{fts_names[i]}')
            ax_tgt.plot(time_axis_tgt, mean_tgt_data[i].T)
            ax_tgt.set_title(f'Target {fts_names[i]}')            
            ax_tgt.set_xlabel('Time')
            ax_tgt.set_ylabel(f'{fts_names[i]}')
            # Set the axis limits to be the same
            if same_axis:
                # ax[0][i].set_ylim(ax[1][i].get_ylim())
                # y_min = min(ax[0][i].get_ylim()[0], ax_tgt.get_ylim()[0])
                # y_max = max(ax[0][i].get_ylim()[1], ax_tgt.get_ylim()[1])
                # ax[0][i].set_ylim(y_min, y_max)
                # ax[1][i].set_ylim(y_min, y_max)
                y_min = min(ax_pred.get_ylim()[0], ax_tgt.get_ylim()[0])
                y_max = max(ax_pred.get_ylim()[1], ax_tgt.get_ylim()[1])
                ax_pred.set_ylim(y_min, y_max)
                ax_tgt.set_ylim(y_min, y_max)

            # ax[1][i].set_xlim((0, time_axis_tgt[-1]))
            # ax[0][i].set_xlim((0, time_axis_tgt[-1]))            
            ax_pred.set_xlim((0, time_axis_tgt[-1]))
            ax_tgt.set_xlim((0, time_axis_tgt[-1]))
            
        # Set the labels
        # ax[0][i].set_xlabel('Time')
        # ax[0][i].set_ylabel(f'{fts_names[i]}')
        ax_pred.set_xlabel('Time')
        ax_pred.set_ylabel(f'{fts_names[i]}')
        
        # Add context points
        if context_pts is not None:
            context_pts = np.array(context_pts)                            
            x_vals = np.tile(context_pts, (mean_pred_data.shape[1], 1))  # shape: (R, T)
            # ax[0][i].scatter(x_vals, mean_pred_data[i, :, context_pts].T, label='Context Point', s=dot_size)
            ax_pred.scatter(x_vals, mean_pred_data[i, :, context_pts].T, label='Context Point', s=dot_size)
            if num_rows > 1:
                # ax[1][i].scatter(x_vals, mean_tgt_data[i, :, context_pts].T, s=dot_size)
                ax_tgt.scatter(x_vals, mean_tgt_data[i, :, context_pts].T, s=dot_size)

        # Vertical lines
        if num_training_steps is not None:
            # ax[0][i].axvline(num_training_steps, color='r', linestyle='--', label='Training steps')
            ax_pred.axvline(num_training_steps, color='r', linestyle='--', label='Training steps')
            if tgt_data is not None:
                # ax[1][i].axvline(num_training_steps, color='r', linestyle='--', label='Training steps')
                ax_tgt.axvline(num_training_steps, color='r', linestyle='--', label='Training steps')
        
        # Specific limits for each feature
        if limits is not None:
            if isinstance(limits, dict) and fts_names[i] in limits:
                ax_pred.set_ylim(limits[fts_names[i]][0])
                if ax_tgt is not None:
                    ax_tgt.set_ylim(limits[fts_names[i]][1])
            else:
                logging.warning(f"Limits for {fts_names[i]} not found in provided limits dictionary.")

    fig.tight_layout()
    
    if save_path is not None:
        # fig.savefig(os.path.join(save_path, f'{append_name}.png'))
        fig.savefig(os.path.join(save_path, f'{append_name}.{save_format}'), bbox_inches='tight', dpi=300)
        plt.close('all')

        if return_limits:
            # Return the limits of the y-axis for each feature
            limits = {fts_names[i]: (ax[0][i].get_ylim(), ax[1][i].get_ylim() if num_rows > 1 else None) for i in range(n_features)}
            return limits
    else:
        return fig
    

def fig_to_tensorboard(fig):
    """Convert a Matplotlib figure to a TensorBoard image format."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)  # Convert to NumPy array for TensorBoard
