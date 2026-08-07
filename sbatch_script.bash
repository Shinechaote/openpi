#!/bin/bash
#SBATCH --job-name=csil-for-vlas
#SBATCH --mem-per-cpu=10G
#SBATCH --time=48:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gn[01-11]
#SBATCH --nodes=1
#SBATCH -p main
# %A represents the array's master job ID, %a represents the specific array task ID
#SBATCH -o "/home/stud_scherer/slurm-output/slurm-%j.out"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('$HOME/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate pizero

# Define the array of configurations
CONFIGS=(
    # "demo_src_threading_task_D0.json"
    # "demo_src_coffee_task_D0.json"
    "demo_src_nut_assembly_task_D0.json"
    # "demo_src_hammer_cleanup_task_D0.json"
    # "demo_src_mug_cleanup_task_D0.json"
)

# Select the configuration for this specific array task
CURRENT_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CONFIG_PATH="/home/stud_scherer/mimicgen/core_configs/${CURRENT_CONFIG}"

echo "Running SLURM Array Master Job ${SLURM_ARRAY_JOB_ID}, Task ID ${SLURM_ARRAY_TASK_ID}"
echo "GPUs allocated on node: ${SLURM_GPUS_ON_NODE}"
echo "Selected config: ${CURRENT_CONFIG}"

cd /home/stud_scherer/openpi

# Execute the python script with the selected configuration
# echo "python mimicgen/scripts/generate_dataset.py --config ${CONFIG_PATH} --auto-remove-exp"
uv run convert_hdf5_dataset_to_lerobot_parallel.py

uv run scripts/compute_norm_stats.py --config-name pi05_2k_six_env_mimicgen


exit_code=$?
echo "Process exited with exit code ${exit_code}"
exit ${exit_code}
