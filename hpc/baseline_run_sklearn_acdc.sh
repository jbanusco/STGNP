#!/bin/bash
#SBATCH --job-name=baselines_sk
#SBATCH --output=/cluster/home/ja1659/logs/baselines_sk_%A_%a.out
#SBATCH --error=/cluster/home/ja1659/logs/baselines_sk_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=36:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=gpunode01,gpunode02

# Define the path to your configuration file
config_file="/cluster/home/ja1659/Code/stmgcn/hpc/configurations.txt"

# Print the chosen parameters
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Read the line corresponding to the array index
line=$(head -n $((SLURM_ARRAY_TASK_ID + 1)) "$config_file" | tail -n 1)

# Parse the line into variables
IFS=',' read -r normalization use_all use_similarity drop_bp use_global_data <<< "$line"

# Print the current configuration
echo "Normalization: $normalization"
echo "Use All Options: $use_all"
echo "Use Similarity Options: $use_similarity"
echo "Drop Blood Pools: $drop_bp"
echo "Use Global Data: $use_global_data"
use_edges=True
load=True
reprocess_dataset=False

# Script to generate the different datasets and the set of indices for the nested CV
singularity_image='/data/bdip2/jbanusco/SingularityImages/multiplex-cpu_0.0.sif'
datapath='/data/bdip2/jbanusco/ACDC/MIDS/mixed/derivatives'
code_path='/cluster/home/ja1659/Code/stmgcn'

# Binding paths
project_path='/usr/src'
data_folder='/usr/data'

# Main options
data_folder='/home/jaume/Desktop/Data/New_ACDC/MIDS/mixed/derivatives'
experiment_name='GraphClassification'

# Print the chosen parameters
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "use_all=${use_all}, use_similarity=${use_similarity}, drop_bp=${drop_bp}"

echo "Generating Sklearn for use_all=${use_all}, use_similarity=${use_similarity}, drop_bp=${drop_bp} and normalization=${normalization}"
singularity exec --bind ${datapath}:${data_folder} --bind ${code_path}:${project_path} ${singularity_image} /bin/bash -c "cd ${project_path} && python3 -m baselines.Sklearn.ACDC_baselines_sklearn --data_folder ${data_folder} --experiment_name ${experiment_name} --normalization ${normalization} --use_global_data ${use_global_data} --use_edges ${use_edges} --use_all ${use_all} --use_similarity ${use_similarity} --drop_blood_pools ${drop_bp} --reprocess_dataset ${reprocess_dataset} --load ${load}"


