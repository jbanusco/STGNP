#!/bin/bash
#SBATCH --output=/cluster/home/ja1659/logs/kuramoto_batch.out
#SBATCH --error=/cluster/home/ja1659/logs/kuramoto_batch.err

# Define paths
code_path='/cluster/home/ja1659/Code/stmgcn'
datapath="/data/bdip2/jbanusco/Data/Multiplex_Synthetic_FINAL"
singularity_image='/data/bdip2/jbanusco/SingularityImages/multiplex-cpu_0.0.sif'

# Load JSON Config
dataset_config=${code_path}"/configs/kuramoto.json"

# Parse JSON for job-specific parameters
num_jobs=10  # Number of SLURM jobs
trials_per_job=7  # Number of trials per job
num_cpus=14  # Number of CPUs per job

config_params=$(python3 -c "
import json
with open('${dataset_config}', 'r') as f:
    config = json.load(f)
params = ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['simulation'].items() if v is not None)
params += ' ' + ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['experiment'].items() if v is not None)
params += ' ' + ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['model'].items() if v is not None)
params += ' ' + ' '.join(f'--{k} {str(v).lower() if isinstance(v, bool) else v}' for k, v in config['optimization'].items() if v is not None)
print(params)
")

echo "Config params: ${config_params}"
tmp_folder="/cluster/home/ja1659/tmp"
mkdir -p ${tmp_folder}

# SLURM Job Submission Loop
for job_id in $(seq 1 $num_jobs); do
    job_file="${tmp_folder}/tmp_job_kuramoto_${job_id}.sh"
    cat <<EOL > ${job_file}
#!/bin/bash
#SBATCH --job-name=optuna_kuramoto_${job_id}
#SBATCH --output=/cluster/home/ja1659/logs/optuna_kuramoto_${job_id}.out
#SBATCH --error=/cluster/home/ja1659/logs/optuna_kuramoto_${job_id}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${num_cpus}
#SBATCH --mem=24G
#SBATCH --time=48:00:00

echo "Starting Optuna job ${job_id} with ${trials_per_job} trials"

singularity exec --bind ${datapath}:/usr/data --bind ${code_path}:/usr/src ${singularity_image} \
    /bin/bash -c "cd /usr/src && python3 -m synthetic_data.kuramoto --num_trials ${trials_per_job} --num_jobs ${num_cpus} --run_best False --is_optuna True ${config_params}"
EOL

    sbatch ${job_file}
    sleep 5  # Optional: Sleep for a second to avoid overwhelming the scheduler
    
done

echo "Submitted ${num_jobs} Optuna jobs to SLURM"
