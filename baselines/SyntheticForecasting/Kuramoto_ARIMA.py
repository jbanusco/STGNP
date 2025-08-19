import os
import pandas as pd
import numpy as np
import json
import torch
import argparse

from baselines.SyntheticForecasting.ARIMA_functions import arima_baseline
from synthetic_data.train_synthetic_models import ObjectiveSynthetic, batch_loop
from utils.model_selection_sklearn import stratified_split
from utils.utils import seed_everything, str2bool, get_best_params
from synthetic_data.kuramoto import KuramotoDataset


def get_parser() -> argparse.ArgumentParser:
    paper_folder = "/media/jaume/DATA/Data/Multiplex_Synthetic_FINAL"
    study_name = "Multiplex_Kuramoto_DIMENSIONS_NEW_LOSS"
    
    parser = argparse.ArgumentParser(description='Kuramoto Oscillators')

    # Folder
    parser.add_argument('--save_folder', type=str, default=f"{paper_folder}", help='Folder to save the data')
    parser.add_argument('--name', type=str, default='KuramotoOscillator', help='Name of the dataset')
    parser.add_argument('--reprocess', type=str2bool, default=False, help='Reprocess the data')
    parser.add_argument('--reload_model', type=str2bool, default=True, help='Reload the model')
    parser.add_argument('--experiment_name', type=str, required=False, default=f'{study_name}', help='Name of the experiment.')
    parser.add_argument('--run_best', type=str2bool, default=True, help='Run the best model')
    parser.add_argument('--is_optuna', type=str2bool, default=False, help='Run the best model')
    
    # Simulation
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes. Should be a square number. 16=4x4 grid')
    parser.add_argument('--k', type=float, default=0.33, help='Oscillator coupling')
    parser.add_argument('--space_coupling', type=float, default=1, help='Space coupling')
    parser.add_argument('--time_coupling', type=float, default=1, help='Time coupling')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--duration', type=float, default=10, help='Duration of the simulation')
    parser.add_argument('--spatial_graph_type', type=str, default='barabasi', help='Type of spatial graph') # 'fully_connected', 'identity', 'small_world', 'barabasi', 'erdos'
    parser.add_argument('--temporal_graph_type', type=str, default='identity', help='Type of temporal graph')

    # Model parameters
    parser.add_argument('--normalization', type=str, default='ZNorm', help='Normalization method')  # ZNorm, NoNorm
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
    parser.add_argument('--decode_just_latent', type=str2bool, default=True, help='Decoder just uses the latent space')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials for the hyper-parameter tuning')
    parser.add_argument('--num_jobs', type=int, default=0, help='Number of jobs in parallel')
    
    parser.add_argument('--gamma_rec', type=float, default=1., help='Weight for the regression loss')
    parser.add_argument('--gamma_lat', type=float, default=0.1, help='L2 weight for the latent space')
    parser.add_argument('--gamma_graph', type=float, default=0.1, help='Weight for the graph regularization')
    
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    seed_everything()
    study_name = args.experiment_name
    duration = float(args.duration)
    dt = float(args.dt)
    k = float(args.k)
    num_nodes = int(args.num_nodes)
    spatial_graph_type = args.spatial_graph_type
    temporal_graph_type = args.temporal_graph_type
    # dataset_name = f'duration-{duration}_dt-{dt}_k-{k}_nodes-{num_nodes}'
    dataset_name = f'duration-{duration}_dt-{dt}_k-{k}_nodes-{num_nodes}_space-{spatial_graph_type}_time-{temporal_graph_type}'

    # Save folder
    save_folder = os.path.join(args.save_folder, "Kuramoto_Graphs", f"{dataset_name}")

    # Load previous data
    model_filename = os.path.join(save_folder, 'FinalModel', 'model.pt')
    params_filename = os.path.join(save_folder, f'objective_params_{study_name}.json')

    # Create the dataset
    dataset = KuramotoDataset(name='KuramotoOscillator',
                              save_dir=save_folder,
                              reprocess=args.reprocess,
                              has_labels=False,
                              save_info=True,
                              spatial_graph_type=spatial_graph_type, 
                              temporal_graph_type=temporal_graph_type,
                              g=1,
                              spatial_coupling=args.space_coupling, 
                              temporal_coupling=args.time_coupling,
                              k=k,
                              num_samples=args.num_samples, 
                              num_nodes=num_nodes, 
                              duration=duration, dt=dt
                              )

    # Read the params filename from the JSON
    with open(params_filename, 'r') as f:
        params_dict = json.load(f)

    # Load the study parameters
    results_hp_folder = os.path.join(args.save_folder, "Kuramoto_Graphs", "results")
    df_params_path = os.path.join(results_hp_folder, f'{study_name}_trials.csv')
    df_params = pd.read_csv(df_params_path)
    df_params.dropna(how='any', inplace=True)
    df_params = df_params.sort_values(by='value', ascending=False)
    best_params = get_best_params(df_params.iloc[0:5], use_median=True)
    best_params['hidden_dim'] = 17
    best_params['latent_dim'] = 6


    # Update the param_dict
    params_dict.update(best_params)

    # Create the objective
    normalization = args.normalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_edges = True # Use the edges in the model / predict edges or just use ones
    use_region_id = False
    use_position = False
    use_time = False
    

    objective_optuna = ObjectiveSynthetic(study_name,
                                          dataset,
                                          normalization=normalization,
                                          norm_only_ed=False,
                                          save_dir=save_folder,
                                          direction="maximize",
                                          device = device,
                                          track_experiment=False,
                                          use_position=use_position,
                                          use_region_id=use_region_id,
                                          use_time=use_time,
                                          fn_batch_loop=batch_loop,
                                          space_planes=args.space_planes,
                                          time_planes=args.time_planes,
                                          depth_nodes=2,
                                          depth_edges=2,
                                          use_edges=use_edges,
                                          only_spatial=False,
                                          use_norm=args.use_norm,
                                          use_mse=False,
                                          )
    objective_optuna.set_default_params(params_dict)

    indices = np.arange(len(dataset))
    labels = np.ones(len(dataset))  # Dummy labels
    splits = stratified_split(indices, labels, test_size=0.2, valid_size=0.2)
    train_idx = splits['X_train']
    valid_idx = splits['X_valid']
    test_idx = splits['X_test']
    objective_optuna.set_indices(train_idx, valid_idx, test_idx=test_idx)

    # Model
    model = objective_optuna.build_model(params_dict)
    tmp_save = os.path.join(save_folder, 'FinalModel')
    res_training = objective_optuna._train(model, params_dict, tmp_save, final_model=True)  # Reload

    steps_to_predict = int(duration/dt)
    time_to_predict = torch.arange(0, steps_to_predict, 1)
    pred_trajectory, pred_latent, tgt_trajectory = objective_optuna.predict_from_latent(model, objective_optuna.dataset, time_to_predict, params_dict, device=device)
    # The shape of the results is [num_samples, num_features, num_nodes, num_time_steps]

    # Convert to the true scale
    trans_fts_predict = objective_optuna.dataset._transform['nfeatures']
    fts_to_predict = np.arange(0, 1)
    num_subjects, _, _, time_frames = tgt_trajectory.shape
    time_frames_true = tgt_trajectory.shape[-1]
    trans_fts_mean = trans_fts_predict.mean[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames_true).permute(0, 2, 1, 3)
    trans_fts_std = trans_fts_predict.std[..., fts_to_predict].unsqueeze(0).unsqueeze(-1).repeat(num_subjects, 1, 1, time_frames_true).permute(0, 2, 1, 3)
    tgt_trajectory = tgt_trajectory.to('cpu')
    trans_fts_std = trans_fts_std.to('cpu')
    trans_fts_mean = trans_fts_mean.to('cpu')
    true_trajectory = (tgt_trajectory * trans_fts_std) + trans_fts_mean

    # Load data
    data_filename = os.path.join(save_folder, 'data.pkl')
    data = torch.load(data_filename, map_location=device)
    tgt_trajectories = data['tgt_trajectories'] #.numpy() # --- This is my ground truth-data to predict true_trajectory
    tgt_trajectories = tgt_trajectories.to('cpu')

    # Normalize them
    norm_tgt_trajectory = (tgt_trajectories - trans_fts_mean) / trans_fts_std

    forecast_steps = tgt_trajectories.shape[-1]

    metrics_filename = os.path.join(args.save_folder, "Kuramoto_Graphs", 'arima_metrics.csv')
    if not os.path.isfile(metrics_filename):

        # arima_preds = arima_baseline(tgt_trajectories, forecast_steps=forecast_steps)
        arima_preds = arima_baseline(norm_tgt_trajectory.numpy(), forecast_steps=forecast_steps)

        arima_preds.shape
        arima_preds = (arima_preds * trans_fts_std.numpy()) + trans_fts_mean.numpy()


        # Initialize storage
        metrics = {
            'Dimension': [],
            'MAE': [],
            'MSE': [],
            'MAPE (%)': [],
            'MSPE (%)': [],
            'IQR (Abs Error)': []
        }

        N, D, R, T = tgt_trajectory.shape
        dimension_names = ['$x$']
        for d in range(D):  # For each dimension
            rec_d = arima_preds[:, d, :, :] #.flatten()
            tgt_d = true_trajectory[:, d, :, :].numpy() #.flatten()        

            abs_err = np.abs(rec_d - tgt_d)
            sq_err = (rec_d - tgt_d) ** 2

            # Handle small values in denominator to avoid division by zero
            tgt_d_safe = np.clip(np.abs(tgt_d), 1e-8, None)

            mae = np.mean(abs_err)
            mse = np.mean(sq_err)
            mape = np.mean(abs_err / tgt_d_safe) * 100
            mspe = np.mean(sq_err / tgt_d_safe) * 100
            iqr = np.percentile(abs_err, 75) - np.percentile(abs_err, 25)

            # Store metrics
            metrics['Dimension'].append(dimension_names[d])
            metrics['MAE'].append(mae)
            metrics['MSE'].append(mse)
            metrics['MAPE (%)'].append(mape)
            metrics['MSPE (%)'].append(mspe)
            metrics['IQR (Abs Error)'].append(iqr)

        # print(abs_error.shape)
        # print(data.keys())

        # Create and display DataFrame
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(metrics_filename)
    else:
        df_metrics = pd.read_csv(metrics_filename, index_col=0)
    print(df_metrics)

    for i, row in df_metrics.iterrows():
        print(f"extrapolation & {row['Dimension']} & "
            f"{row['MSE']:.4f} $\\pm$ {row['IQR (Abs Error)']:.4f} & "
            f"{row['MAE']:.4f} $\\pm$ {row['IQR (Abs Error)']:.4f} & "
            f"{row['MSPE (%)']:.2f}\\% & {row['MAPE (%)']:.2f}\\% \\\\")


if __name__ == "__main__":
    main()