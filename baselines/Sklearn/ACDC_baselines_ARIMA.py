import os
import numpy as np
import pickle
import pandas as pd
import argparse
import logging
import torch

# For the sampling
from baselines.MLP.train_mlp import Objective_MLP

# Classification
from utils.utils import seed_everything, str2bool
from utils.model_selection_sklearn import nested_cv_model
from dataset.dataset_utils import get_data, get_data_in_tensor, data_per_subject
from utils.model_selection_sklearn import cv_model_scaler
from sklearn.model_selection import StratifiedKFold

# ARIMA
import pmdarima as pm
import matplotlib.pyplot as plt


def averaged_arima_forecast_with_static(tgt_node_data: torch.Tensor, 
                                        X_static: np.ndarray,
                                        forecast_steps=5,
                                        use_all_series=False):
    """
    Averaged ARIMA: one model per feature, uses mean of static features as exogenous input.
    """
    tgt_np = tgt_node_data.numpy()  # [subjects, features, regions, time]
    T_total = tgt_np.shape[-1]
    T_train = T_total - forecast_steps

    # Mean time series per feature: [features, T_train]
    if use_all_series:
        avg_series = tgt_np.mean(axis=(0, 2))
        T_train = T_total
    else:
        avg_series = tgt_np[:, :, :, :T_train].mean(axis=(0, 2))

    # Mean static feature across all subjects: [static_dim]
    static_mean = X_static.mean(axis=0)  # [d]
    static_exog_train = np.tile(static_mean, (T_train, 1))  # [T_train, d]
    static_exog_forecast = np.tile(static_mean, (forecast_steps, 1))  # [forecast_steps, d]

    n_features = avg_series.shape[0]
    predictions = np.zeros((forecast_steps, n_features))

    for feat in range(n_features):
        series = avg_series[feat]
        try:
            model = pm.auto_arima(series,
                                  exogenous=static_exog_train,
                                  seasonal=False,
                                  ensure_all_finite=True,
                                  suppress_warnings=True)
            forecast = model.predict(n_periods=forecast_steps, exogenous=static_exog_forecast)
            predictions[:, feat] = forecast
        except Exception as e:
            print(f"ARIMAX failed for feature {feat}: {e}")
            predictions[:, feat] = np.nan

    return predictions  # [forecast_steps, features]



def per_subject_arima_forecast_with_static(tgt_node_data: torch.Tensor, 
                                           X_static: np.ndarray,
                                           forecast_steps=5,
                                           use_all_series=False):
    """
    ARIMAX: One model per subject per feature. Each subject uses their own static features.
    """
    tgt_np = tgt_node_data.numpy()  # [subjects, features, regions, time]
    T_total = tgt_np.shape[-1]
    T_train = T_total - forecast_steps

    if use_all_series:
        T_train = T_total

    subj_series = tgt_np[:, :, :, :T_train].mean(axis=2)  # [subjects, features, T_train]

    n_subjects, n_features, _ = subj_series.shape
    predictions = np.zeros((n_subjects, forecast_steps, n_features))

    for subj in range(n_subjects):
        static_exog_train = np.tile(X_static[subj], (T_train, 1))  # [T_train, d]
        static_exog_forecast = np.tile(X_static[subj], (forecast_steps, 1))  # [forecast_steps, d]

        for feat in range(n_features):
            series = subj_series[subj, feat]
            try:
                model = pm.auto_arima(series,
                                      exogenous=static_exog_train,
                                      seasonal=False,
                                      suppress_warnings=True)
                forecast = model.predict(n_periods=forecast_steps, exogenous=static_exog_forecast)
                predictions[subj, :, feat] = forecast
            except Exception as e:
                print(f"ARIMAX failed for subject {subj}, feature {feat}: {e}")
                predictions[subj, :, feat] = np.nan

    return predictions  # [subjects, forecast_steps, features]





def get_parser() -> argparse.ArgumentParser:
    # data_path = '/home/jaume/Desktop/Data/New_ACDC/MIDS/mixed/derivatives'
    data_path = "/media/jaume/DATA/Data/New_ACDC/MIDS/mixed/derivatives"
    experiment_name = 'GraphClassification'
    normalisation = 'ZNorm' # NoNorm, ZNorm, MaxMin, Spatial
    splits_file = os.path.join(data_path, experiment_name, 'nested_cv_indices.pkl')

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    parser.add_argument('--data_folder', type=str, required=False, default=f'{data_path}', help='Path to the data.')
    parser.add_argument('--experiment_name', type=str, required=False, default=f'{experiment_name}', help='Name of the experiment.')    
    parser.add_argument('--normalization', type=str, default=f'{normalisation}', help='Normalization strategy')  
    parser.add_argument('--use_global_data', type=str2bool, default=True, help='Use global data or not.')
    parser.add_argument('--use_edges', type=str2bool, default=True, help='Use edge information or not.')
    parser.add_argument('--use_all', type=str2bool, default=True, help='Use all edges, or only AHA.')
    parser.add_argument('--use_similarity', type=str2bool, default=False, help='Use similarity instead of distance.')    
    parser.add_argument('--drop_blood_pools', type=str2bool, default=True, help='Drop the blood pools.')
    parser.add_argument('--reprocess_dataset', type=str2bool, default=False, help='Reprocess the dataset.')
    parser.add_argument('--load', type=str2bool, default=True, help='Load previous results or not.')
    parser.add_argument('--in_folds', type=int, default=5, help='Number of inner folds.')
    parser.add_argument('--out_folds', type=int, default=5, help='Number of outer folds.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set.')
    parser.add_argument('--splits_file', type=str, default=f'{splits_file}', help='File with the splits.')

    return parser


def main():
    seed_everything()
    parser = get_parser()

    # Parse the argument
    args = parser.parse_args()
    derivatives_folder = args.data_folder  
    study_name = args.experiment_name
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

    print(args)
    
    #  ==================== Problem setup ====================    
    # Load the datasets
    track_experiment = False

    # Use of which data
    use_position = True
    use_region_id = False
    use_time = False    

    # Normalization options    
    norm_by_group = False  # Normalize by group, i.e: healthy controls
    norm_only_ed = False  # Normalize only by the ED frame    
        
    # Save folder
    use_blood_pools = not drop_blood_pools
    wkspc_name = f"Edges-{use_edges}_Norm-{normalization}_Global-{use_global_data}_All-{use_all}_Sim-{use_similarity}_BP-{use_blood_pools}"
    save_folder_dataset = os.path.join(derivatives_folder, study_name, f'{wkspc_name}')
    os.makedirs(save_folder_dataset, exist_ok=True)
    just_t0 = False
    if just_t0:
        df_results_filename = os.path.join(save_folder_dataset, 'sklearn_results_t0.csv')
    else:
        df_results_filename = os.path.join(save_folder_dataset, 'sklearn_results.csv')

    # Configure the logger
    log_filename = os.path.join(save_folder_dataset, 'optimization_sklearn.log')
    if os.path.isfile(log_filename) and not load_previous:
        os.remove(log_filename)
    logging.basicConfig(filename=f'{log_filename}', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the datasets (graph format)
    graph_train_dataset = get_data(derivatives_folder, 
                                   wkspc_name=f'{study_name}', 
                                   is_test=False, 
                                   reprocess_datasets=reprocess_datasets, 
                                   drop_blood_pools=drop_blood_pools, 
                                   use_similarity=use_similarity,
                                   use_all=use_all)

    default_config = {'use_edges': use_edges,
                      'use_region': use_region_id,
                      'use_position': use_position,
                      }
    
    objective_optuna = Objective_MLP(study_name,graph_train_dataset,
                                     normalization=normalization,
                                     norm_by_group=norm_by_group,
                                     norm_only_ed=norm_only_ed,
                                     save_dir=save_folder_dataset,
                                     direction="maximize",
                                     device = 'cpu',
                                     track_experiment=False,
                                     use_global_data=use_global_data,
                                     use_position=use_position,
                                     use_region_id=use_region_id,
                                     use_time=use_time,
                                     use_weighted_sampler=False,
                                     use_focal_loss=False,
                                     class_dim=5,
                                     )
    objective_optuna.set_default_params(default_config)

    train_idx = graph_train_dataset.idx_train
    valid_idx = graph_train_dataset.idx_valid
    test_idx = graph_train_dataset.idx_test

    node_data, edge_data, global_data, y = data_per_subject(graph_train_dataset, use_edges=use_edges, is_test=False)
    
    # Get the features to predict
    idx_thickness = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Thickness_Median']))[0]
    idx_volume = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Volume_Index']))[0]
    idx_median = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Intensity_Median']))[0]
    idx_J = np.where(np.isin(objective_optuna.dataset.list_node_features, ['J_Median']))[0]
    # fts_to_predict = np.concatenate([idx_thickness, idx_volume])
    # fts_to_predict = np.concatenate([idx_thickness, idx_volume, idx_median])
    fts_to_predict = np.concatenate([idx_thickness, idx_volume, idx_J])
    pos_idx = np.arange(node_data.shape[1]-3, node_data.shape[1])

    # Averaged ARIMAX    
    forecast_steps = node_data.shape[-1]
    avg_arimax = averaged_arima_forecast_with_static(node_data, global_data, forecast_steps, use_all_series=True)

    # Get the series in which i am interested    
    avg_arimax_fts = avg_arimax[:, fts_to_predict]    
    # Get the positions
    avg_arimax_pos = avg_arimax[:, pos_idx]

    # Plot the forecast for each series vs the target data
    avg_node_data = node_data.numpy().mean(axis=(0, 2))

    # Plot
    num_pred = len(fts_to_predict)
    fig, ax = plt.subplots(2, num_pred)
    for idx_f in range(num_pred):
        ax[0][idx_f].plot(avg_arimax_fts[:, idx_f])
        ax[1][idx_f].plot(avg_node_data[fts_to_predict[idx_f], :].T)
    plt.show()
    
    # Per-subject ARIMAX
    subj_arimax = per_subject_arima_forecast_with_static(node_data, global_data, forecast_steps, use_all_series=True)
    




if __name__ == '__main__':
    main()