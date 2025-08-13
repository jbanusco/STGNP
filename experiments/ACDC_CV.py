import os
import argparse

import numpy as np
import torch
import logging
import json
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

from model.train_stmgcn_ode import Objective_Multiplex

from dataset.dataset_utils import get_data
from utils.utils import seed_everything, str2bool
from utils.model_selection_optuna import hypertune_optuna


import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')


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