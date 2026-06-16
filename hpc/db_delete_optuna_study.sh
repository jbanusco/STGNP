#!/bin/bash
#SBATCH --job-name=delete_optuna_study
#SBATCH --output=/cluster/home/ja1659/logs/delete_optuna_study.out
#SBATCH --error=/cluster/home/ja1659/logs/delete_optuna_study.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Define paths
code_path='/cluster/home/ja1659/Code/stmgcn'
singularity_image='/data/bdip2/jbanusco/SingularityImages/multiplex-cpu_0.0.sif'

# Load parameters JSON
default_params_file="${code_path}/configs/default_params.json"

# Function to delete studies
delete_study() {
    study_name=$1
    echo "Processing study: ${study_name}"

    # Delete study
    singularity exec --bind ${code_path}:/usr/src ${singularity_image} \
        /bin/bash -c "cd /usr/src && python3 -m utils.get_optuna_results --study_name ${study_name} --delete_study True"
}

# ==============================================================================

# For example
delete_study "Multiplex_CoupledPendulum"
delete_study "Multiplex_Lorenz"
delete_study "Multiplex_Kuramoto"
delete_study "Multiplex_HPT_ACDC"
delete_study "Multiplex_HPT_UKB"

echo "All studies processed successfully!"
