import os
import numpy as np
import pandas as pd
import json
import pickle
import argparse

from utils.model_selection_sklearn import cv_stratified_split
from dataset.dataset_utils import get_data
from utils.utils import str2bool, seed_everything


def get_parser():
    # dataset_path = "/media/jaume/DATA/Data/Urblauna_SFTP/UKB_Cardiac_BIDS"
    dataset_path = "/media/jaume/DATA/Data/New_ACDC/MIDS/mixed"

    dataset_name = 'GraphClassification'
    list_subjects = os.path.join(dataset_path, 'metadata_participants.tsv')

    parser = argparse.ArgumentParser(description='Generate the splits for nested CV.')
    parser.add_argument('--data_path', type=str, default=f"{dataset_path}", help='Path to the data.')
    parser.add_argument('--experiment_name', type=str, default=f"{dataset_name}", help='Name of the experiment.')
    parser.add_argument('--use_all', type=str2bool, default=True, help='Use all edges, or only AHA.')
    parser.add_argument('--use_similarity', type=str2bool, default=False, help='Use similarity instead of distance.')    
    parser.add_argument('--drop_blood_pools', type=str2bool, default=False, help='Drop the blood pools.')
    parser.add_argument('--out_folds', type=int, default=2, help='Number of outer folds.')
    parser.add_argument('--in_folds', type=int, default=3, help='Number of inner folds.')
    parser.add_argument('--test_size', type=float, default=0.15, help='Size of the test set.')
    parser.add_argument('--seed', type=int, default=5593, help='Random seed.')
    parser.add_argument('--list_subjects', type=str, default=list_subjects, help='List of subjects to use.')
    parser.add_argument('--reprocess_datasets', type=str2bool, default=False, help='Reprocess the dataset.')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    dataset_path = args.data_path
    experiment_name = args.experiment_name

    # Cross-validation parameters
    out_folds = args.out_folds
    in_folds = args.in_folds
    test_size = args.test_size

    # Options to select one pre-generated graph or the other
    use_all = args.use_all
    use_similarity = args.use_similarity
    drop_blood_pools = args.drop_blood_pools
    reprocess_datasets = args.reprocess_datasets  # Re-load
    list_subjects_path = args.list_subjects

    # Seed everything
    seed_everything(seed)

    if list_subjects_path is not None and os.path.isfile(list_subjects_path):
        df_subjects = pd.read_csv(list_subjects_path, sep='\t')  # Assume it is the meatadata.tsv
        list_subjects = list(df_subjects['Subject'].values)
    else:
        list_subjects = None
    
    if os.path.isfile(list_subjects_path):
        df_metadata = pd.read_csv(list_subjects_path, sep='\t')
    else:
        df_metadata = None
    
    # Load the dataset    
    graph_dataset = get_data(dataset_path, 
                             wkspc_name=experiment_name, 
                             is_test=False,
                             reprocess_datasets=reprocess_datasets, 
                             drop_blood_pools=drop_blood_pools, 
                             use_similarity=use_similarity,
                             use_all=use_all,
                             list_subjects=list_subjects,
                             df_metadata=df_metadata)

    # Get the whole indices
    save_path = os.path.join(dataset_path, experiment_name)
    data_indices = np.arange(0, len(graph_dataset))
    data_labels = graph_dataset.label

    nested_cv_indices = []
    nested_test_indices = []
    for ix_out in range(0, out_folds):
        # Get the cross-validation indices
        cv_indices, test_indices = cv_stratified_split(data_indices, data_labels, k_folds=in_folds, test_size=test_size)
        nested_cv_indices.append(cv_indices)
        nested_test_indices.append(test_indices)

    # Generate .json file with the configuration options
    config = {
        'seed': seed,
        'out_folds': out_folds,
        'in_folds': in_folds,
        'test_size': test_size,
    }
    with open(os.path.join(save_path, 'nested_cv_config.json'), 'w') as file:
        json.dump(config, file)

    # Store the indices
    data_to_store = {'cv_indices': nested_cv_indices, 'test_indices': nested_test_indices}
    with open(os.path.join(save_path, 'nested_cv_indices.pkl'), 'wb') as file:
        pickle.dump(data_to_store, file)

    print("Splits generated - Done!")

if __name__ == '__main__':
    main()