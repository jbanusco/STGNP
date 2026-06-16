import os
import json
import argparse

import pickle
import numpy as np
import pandas as pd
import torch
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset.dataset_utils import get_data
from utils.utils import seed_everything, str2bool, get_best_params
from model.train_stgnp import Objective_Multiplex
from sklearn.ensemble import RandomForestClassifier
from utils.model_selection_sklearn import cv_model_scaler
from model.plot_and_print_utils import get_latex_table, wrap_latex_table, save_training_convergence, plot_results, export_latent_data, plot_combined_trajectories, get_data_in_original_scale, plot_predicted_trajectories

import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')


def get_parser() -> argparse.ArgumentParser:
    data_path = "/media/jaume/DATA/Data/Urblauna_SFTP/UKB_Cardiac_BIDS/derivatives"
    experiment_name = 'GraphClassification'
    # study_name = 'Multiplex_HPT_UKB_Pred'
    # study_name = "Multiplex_HPT_UKB_ADAM"
    # study_name = "Multiplex_HPT_UKB_ADAM_FINAL_MAE"
    # study_name = "Multiplex_HPT_UKB_ADAM_FINAL_MAE_wACC"
    # study_name = "Multiplex_HPT_UKB_ADAM_FINAL_MAE_FINAL_NOCLASS"
    # study_name = "Multiplex_HPT_UKB_ADAM_FINAL_MAE_FINAL_NOCLASS_JAC_END"
    # study_name = "Multiplex_HPT_UKB_VERY_LAST_ONE"
    # study_name = "Multiplex_HPT_UKB_DIMENSIONS"
    # study_name = "Multiplex_HPT_UKB_DIMENSIONS_ALL"
    # study_name = "Multiplex_HPT_UKB_DIMENSIONS_SUM"
    # study_name = "Multiplex_HPT_UKB_DIMENSIONS_ALL_SUM"
    study_name = "Multiplex_HPT_UKB_DIMENSIONS_NEW_LOSS"
    # study_name = "Multiplex_HPT_UKB_DIMENSIONS_NEW_LOSS_ALL"

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
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size')
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

    #  ==================== Problem setup ====================    
    # Load the datasets
    track_experiment = False

    # Use of which data
    use_position = True
    use_region_id = True
    use_time = False

    # Normalization options    
    norm_by_group = False  # Normalize by group, i.e: healthy controls
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
    best_params = get_best_params(df_params.iloc[0:5], use_median=True)
    best_params['hidden_dim'] = 18
    best_params['space_planes'] = 6
    best_params['time_planes'] = 5

    model_params = objective_optuna.default_params.copy()
    model_params.update(best_params)
    print(best_params)

    # Store the objective parameters
    json_filename = os.path.join(save_folder, 'objective_params.json')
    with open(json_filename, 'w') as f:
        json.dump(model_params, f, indent=4)

    # Model
    model = objective_optuna.build_model(model_params)
    res_training = objective_optuna._train(model, model_params, tmp_save, final_model=True)

    final_model = f"{tmp_save}/model.pt"    
    append_att = "_Att" if use_attention else ""
    dt_string = str(dt_step_size).replace('.', '')
    study_model_copy = os.path.join(tmp_save, f'model_{dt_string}{append_att}.pt')
    os.system(f"cp {final_model} {study_model_copy}")

    # Get the features to predict
    idx_thickness = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Thickness_Median']))[0]
    idx_volume = np.where(np.isin(objective_optuna.dataset.list_node_features, ['Volume_Index']))[0]
    fts_to_predict = np.concatenate([idx_thickness, idx_volume])

    time_to_predict = torch.arange(0, 50, 1)
    pred_trajectory, pred_latent, tgt_trajectory = objective_optuna.predict_from_latent(model, objective_optuna.dataset, time_to_predict, model_params, device=device)    

    df_errors = get_data_in_original_scale(model, objective_optuna, model_params, save_folder, pred_trajectory, 
                                            fts_to_predict=fts_to_predict, true_trajectory=tgt_trajectory, normalization=normalization)
    df_errors.to_csv(os.path.join(save_folder, 'errors_per_feature.csv'))
    
    latex_table = get_latex_table(df_errors, objective_optuna)
    latex_table = latex_table.replace('train', 'train window')
    latex_table= latex_table.replace('predict', 'extrapolation')
    wrapped_table = wrap_latex_table(latex_table, caption="Errors per feature in the test set.", label="tab:feature_errors_ukb")       
    print(wrapped_table)

    latent_filename = os.path.join(save_folder, 'latent_data.csv')
    # if (load_previous and not os.path.isfile(latent_filename)) or not load_previous:
    plot_combined_trajectories(model, objective_optuna, model_params, save_folder, pred_latent, pred_trajectory, plot_individual=False, plot_spatial=False, 
                               fts_to_predict=fts_to_predict, true_trajectory=tgt_trajectory, normalization=normalization)
    
    plot_predicted_trajectories(objective_optuna, pred_latent, pred_trajectory, save_folder, fts_to_predict=fts_to_predict,
                                normalization=normalization, plot_individual=False, true_trajectory=tgt_trajectory, plot_spatial=False)
    save_training_convergence(res_training, save_folder)
    plot_results(model, objective_optuna, model_params, save_folder, plot_individual=False, plot_spatial=False, fts_to_predict=fts_to_predict)
    export_latent_data(model, objective_optuna, model_params, save_folder)

    # ============================================================================================================================
    # Load the data . pkl
    data_filename = os.path.join(save_folder, 'data.pkl')
    data_model = torch.load(data_filename)

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

    X = latent_info.to_numpy()  # Initial state and control     

    results_filename = os.path.join(save_folder, 'model_results.csv')
    params_filename = os.path.join(save_folder, 'params.csv')
    search_filename = os.path.join(save_folder, 'search.pkl')

    # Now, let's set-up a nested CV for the classification    
    params = [
        {'n_estimators': [601, 1001], 'criterion': ['gini', 'entropy', 'log_loss'], 'class_weight': ['balanced', 'balanced_subsample'], 'max_depth': [10, 20]},        
    ]

    names = [
        "Random_Forest",
    ]

    classifiers = [
        RandomForestClassifier(),
    ]

    out_folds = 1
    in_folds = 3
    acc_res = np.zeros((out_folds, in_folds))
    params_dict = dict()
    search_dict = dict()
    
    train_indices = np.concatenate([train_idx, valid_idx])
    y_train = y[train_indices]
    # Nested CV with the pre-found parameters
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
    print(acc_res.mean())
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