import os
import json
import argparse
import pandas as pd

from dataset.dataset_utils import get_data
from utils.utils import str2bool, seed_everything


def get_parser():
    dataset_path = '/home/jaume/Desktop/Data/New_ACDC/MIDS/mixed/derivatives'
    # dataset_path = '/home/jaume/Desktop/Data/Urblauna_SFTP/UKB_Cardiac_BIDS/derivatives'
    dataset_name = 'GraphClassification'
    list_subjects = os.path.join(dataset_path, 'metadata_participants.tsv')
    # list_subjects = os.path.join(dataset_path, 'metadata_participants_AFib.tsv')

    parser = argparse.ArgumentParser(description='Generate the splits for nested CV.')
    parser.add_argument('--data_path', type=str, default=f"{dataset_path}", help='Path to the data.')
    parser.add_argument('--experiment_name', type=str, default=f"{dataset_name}", help='Name of the experiment.')
    parser.add_argument('--use_all', type=str2bool, default=True, help='Use all edges, or only AHA.')
    parser.add_argument('--use_similarity', type=str2bool, default=False, help='Use similarity instead of distance.')
    parser.add_argument('--drop_blood_pools', type=str2bool, default=False, help='Drop the blood pools.')
    parser.add_argument('--seed', type=int, default=5593, help='Random seed.')
    parser.add_argument('--list_subjects', type=str, default=list_subjects, help='List of subjects to use.')
    parser.add_argument('--reprocess_datasets', type=str2bool, default=True, help='Reprocess the dataset.')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    dataset_path = args.data_path
    experiment_name = args.experiment_name
    list_subjects_path = args.list_subjects

    # Options to select one pre-generated graph or the other
    use_all = args.use_all
    use_similarity = args.use_similarity
    drop_blood_pools = args.drop_blood_pools
    reprocess_datasets = args.reprocess_datasets  # Re-load
    
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

    # Generate .json file with the configuration options
    save_path = os.path.join(dataset_path, experiment_name, graph_dataset.name)
    config = {
        'seed': seed,
        'use_all': use_all,
        'use_similarity': use_similarity,
        'drop_blood_pools': drop_blood_pools,
        'list_subjects': list_subjects_path,
    }
    with open(os.path.join(save_path, 'nested_cv_config.json'), 'w') as file:
        json.dump(config, file)    

    print("Dataset generated - Done!")

if __name__ == '__main__':
    main()