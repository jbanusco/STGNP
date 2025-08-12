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
import xgboost as xgb
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from utils.utils import seed_everything, str2bool
from utils.model_selection_sklearn import nested_cv_model
from dataset.dataset_utils import get_data
from dataset.dataset_utils import get_data_in_tensor
from utils.model_selection_sklearn import cv_model_scaler
from sklearn.model_selection import StratifiedKFold


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
    parser.add_argument('--use_all', type=str2bool, default=False, help='Use all edges, or only AHA.')
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
    use_region_id = True
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
    
    # Get the data for the latent projection
    # Get all the data of the dataset
    # y = objective_optuna.dataset.label.squeeze().data.numpy() 
    # all_indices = np.arange(0, len(y))
    # x, x_edges, label = get_data_in_tensor(objective_optuna.dataset, all_indices, device='cpu')
    # if objective_optuna.default_params['use_edges']:
    #     normX = torch.cat((x, x_edges), dim=1)
    # else:
    #     normX = x
    # df_raw = pd.DataFrame(data=normX.numpy(), columns=[f'X_{i}' for i in range(normX.shape[1])])
    # df_raw['Y'] = y
    # df_raw['Subject'] = objective_optuna.dataset.sub_id
    # df_raw.to_parquet(os.path.join(derivatives_folder, study_name, 'raw_data.parquet'))

    # kNN: NaN results with neighbors above 7
    params = [
        {'n_neighbors': np.arange(3, 6, 1), 'p': [1, 2], 'leaf_size': [20, 30, 40], 'metric': ['minkowski'], 'weights': ['uniform', 'distance']},        
        # {'C': np.asarray([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), 'penalty': ['l2'], 'class_weight': ['balanced']},
        # {'n_estimators': [601, 1001], 'criterion': ['gini', 'entropy'], 'class_weight': ['balanced', 'balanced_subsample']},
        {'n_estimators': [601, 1001], 'criterion': ['gini', 'entropy', 'log_loss'], 'class_weight': ['balanced', 'balanced_subsample'], 'max_depth': [10, 20]},
        # {'n_estimators': [601, 1001], 'reg_alpha': np.linspace(0.1, 0.9, 3), 'reg_lambda': np.linspace(0.1, 0.9, 3), 'subsample': [0.5, 0.75]},
        {'n_estimators': [601, 1001], 'reg_alpha': np.linspace(0.1, 0.9, 3), 'reg_lambda': np.linspace(0.1, 0.9, 3), 'subsample': [0.5, 0.75], 'max_depth': [10, 20]},
        # {'C': np.asarray([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]), 'l1_ratio': np.linspace(0.15, 0.95, 10), 'class_weight': [None, 'balanced']},
    ]

    names = [
        "Nearest_Neighbors",
        # "Linear_SVM",
        "Random_Forest",
        "XGBoost",
        # "LogisticRegression",
    ]

    classifiers = [
        KNeighborsClassifier(5),
        # svm.LinearSVC(dual=True, fit_intercept=False),
        RandomForestClassifier(),
        xgb.XGBClassifier(eval_metric='mlogloss', num_class=5, learning_rate=0.05, n_jobs=4, objective='multi:softprob'),
        # LogisticRegression(penalty = 'elasticnet', solver = 'saga'),
    ]

    acc = np.zeros((len(classifiers),2))  # Train, Train_Std
    out_folds = 1
    in_folds = 3

    train_idx = graph_train_dataset.idx_train
    valid_idx = graph_train_dataset.idx_valid
    test_idx = graph_train_dataset.idx_test

    y = graph_train_dataset.label.squeeze().data.numpy()
    all_indices = np.arange(0, len(y))
    x, x_edges, label = get_data_in_tensor(graph_train_dataset, all_indices, device='cpu', just_t0=just_t0)
    X = np.concatenate((x.numpy(), x_edges.numpy()), axis=1) if use_edges else x.numpy()

    for ix, (name, clf) in enumerate(zip(names, classifiers)):
        print(f"{name}")

        results_folder = os.path.join(save_folder_dataset, f'{name}')
        os.makedirs(results_folder, exist_ok=True)

        # Get the best model after nested CV            
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
            
            acc_cv, params_cv, search_cv = cv_model_scaler(X, y, classifiers[ix], params[ix], in_folds, cv_indices, test_idx, rerun_best=False, 
                                                        in_params=None, use_scaler=False, n_jobs=2)
            print(acc_cv)
            # Save the results
            search_dict[f"{ix_out}"] = search_cv
            params_dict[f"{ix_out}"] = params_cv
            # list_params = list_params + params_cv
            acc_res[ix_out] = acc_cv.reshape(-1)

        # best_est, acc_res, norm_info = nested_cv_model(in_folds, 
        #                                                out_folds, 
        #                                                results_folder, 
        #                                                params[ix], 
        #                                                clf, 
        #                                                test_size=test_size,
        #                                                load_previous=load_previous, 
        #                                                objective=objective_optuna,
        #                                                splits_file=splits_file,
        #                                                rerun_best=True,
        #                                                )
        
        # Final performance estimate
        acc[ix, 0] = acc_res.mean()
        acc[ix, 1] = acc_res.std()        
        
        print(f"Train: {acc[ix, 0]:.2f} +/- {acc[ix, 1]:.2f}")
    
    # df_results = pd.DataFrame(data=acc, columns=['Acc_Train', 'Acc_Train_Std', 'Acc_Test'], index=names)
    df_results = pd.DataFrame(data=acc, columns=['Acc_Train', 'Acc_Train_Std'], index=names)
    df_results['Edges'] = use_edges
    df_results['Normalization'] = normalization
    df_results['Global'] = use_global_data
    df_results['Conn'] = use_all
    df_results['Similarity'] = use_similarity
    df_results['Dataset'] = 'ACDC'
    df_results.to_csv(df_results_filename)
            
    print(df_results)        
    print(df_results.to_latex(float_format='%.2f'))



if __name__ == '__main__':
    main()