#!/bin/bash
#SBATCH --job-name=generate_best_models
#SBATCH --output=/cluster/home/ja1659/logs/generate_best_models.out
#SBATCH --error=/cluster/home/ja1659/logs/generate_best_models.err
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2

# Define paths
code_path='/cluster/home/ja1659/Code/stmgcn'
singularity_image='/data/bdip2/jbanusco/SingularityImages/multiplex-cpu_0.0.sif'
num_cpus=14

run_best_model() {
    is_synthetic=$1
    data_dir=$2
    dataset_config=$3
    script_path=$4

    config_filename=${code_path}/configs/${dataset_config}.json

    config_params=$(python3 -c "
import json
with open('${config_filename}', 'r') as f:
    config = json.load(f)        
params = ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['experiment'].items() if v is not None)
params += ' ' + ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['model'].items() if v is not None)
params += ' ' + ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['optimization'].items() if v is not None)
if 'simulation' in config:
    params += ' ' + ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['simulation'].items() if v is not None)
print(params)
    ")

    # SLURM Job Submission Loop    
    if [[ "${is_synthetic}" == "True" ]]; then
        sbatch --job-name=best_${dataset_config} \
               --output=/cluster/home/ja1659/logs/best_${dataset_config}.out \
               --error=/cluster/home/ja1659/logs/best_${dataset_config}.err \
               --ntasks=1 \
               --cpus-per-task=${num_cpus} \
               --mem=16G \
               --time=10:00:00 \
               --wrap="singularity exec --bind ${data_dir}:/usr/data --bind ${code_path}:/usr/src ${singularity_image} /bin/bash -c 'cd /usr/src && python3 -m ${script_path} --run_best True ${config_params}'"
    else
        sbatch --job-name=best_${dataset_config} \
               --output=/cluster/home/ja1659/logs/best_${dataset_config}.out \
               --error=/cluster/home/ja1659/logs/best_${dataset_config}.err \
               --ntasks=1 \
               --cpus-per-task=${num_cpus} \
               --mem=16G \
               --time=10:00:00 \
               --wrap="singularity exec --bind ${data_dir}:/usr/data --bind ${code_path}:/usr/src ${singularity_image} /bin/bash -c 'cd /usr/src && python3 -m ${script_path} ${config_params}'"
    fi
}

synthetic_data="/data/bdip2/jbanusco/Data/Multiplex_Synthetic_FINAL"
run_best_model True ${synthetic_data} "pendulum" "synthetic_data.pendulum"
run_best_model True ${synthetic_data} "lorenz" "synthetic_data.lorenz_model"
# run_best_model True ${synthetic_data} "kuramoto" "synthetic_data.kuramoto"
# run_best_model False "/data/bdip2/jbanusco/ACDC/MIDS/mixed/derivatives" "cardiac_acdc" "experiments.ACDC_RunBest"
# n_best_model False "/data/bdip2/jbanusco/UKB_Cardiac_BIDS/derivatives" "cardiac_ukb" "experiments.UKB_RunBest"