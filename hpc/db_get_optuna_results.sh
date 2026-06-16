#!/bin/bash
#SBATCH --job-name=get_optuna_results
#SBATCH --output=/cluster/home/ja1659/logs/get_optuna_results.out
#SBATCH --error=/cluster/home/ja1659/logs/get_optuna_results.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Define paths
code_path='/cluster/home/ja1659/Code/stmgcn'
datapath="/data/bdip2/jbanusco/Data/Multiplex_Synthetic_FINAL"
singularity_image='/data/bdip2/jbanusco/SingularityImages/multiplex-cpu_0.0.sif'

# Load parameters JSON
default_params_file="${code_path}/configs/default_params.json"

# Function to process studies
process_study() {
    study_name=$1
    data_dir=$2
    config_name=$3
    results_dir="${data_dir}/results"
    
    mkdir -p ${results_dir}

    echo "Processing study: ${study_name}"

    # Extract correct parameters for this study
    update_params=$(python3 -c "
import json
with open('${default_params_file}', 'r') as f:
    params = json.load(f)
print(json.dumps(params.get('${config_name}'.split('_')[-1].lower(), {})))  # Extract relevant params
")

    # Fetch and save results
    singularity exec --bind ${data_dir}:/usr/data --bind ${code_path}:/usr/src ${singularity_image} \
        /bin/bash -c "cd /usr/src && python3 -m utils.get_optuna_results --study_name ${study_name} --save_folder /usr/data/results --delete_study False"
}

# ==============================================================================

# For example
process_study "Multiplex_CoupledPendulum" "${datapath}/CoupledPendulum_Graphs" "pendulum"
process_study "Multiplex_Lorenz" "${datapath}/LorenzAttractor_Graphs" "lorenz"
process_study "Multiplex_Kuramoto" "${datapath}/Kuramoto_Graphs" "kuramoto"
process_study "Multiplex_HPT_ACDC" "/data/bdip2/jbanusco/ACDC/MIDS/mixed/derivatives/GraphClassification" "cardiac"
process_study "Multiplex_HPT_UKB" "/data/bdip2/jbanusco/UKB_Cardiac_BIDS/derivatives/GraphClassification" "cardiac"

echo "All studies processed successfully!"
