#!/bin/bash

#SBATCH -J molhiv_train          # Job name (shows up in squeue)
#SBATCH -p gpudev                   # Partition - gpu1 for single node jobs
#SBATCH --gres=gpu:a100:1         # Request 1 A100 GPU
#SBATCH --cpus-per-task=36        # CPUs (gpu1 default is 36 per GPU)
#SBATCH --mem=125000              # Memory in MB (~125 GB, gpu1 default)
#SBATCH --time=00:15:00         # Max walltime - job gets killed after this
#SBATCH -o logs/%x_%j.out        # Stdout file: jobname_jobid.out
#SBATCH -e logs/%x_%j.err        # Stderr file: jobname_jobid.err

# Load modules - check what's available with 'module avail'
module load anaconda/3/2023.03
module load cuda/12.6

# Activate your conda/venv environment
source activate molhiv

# Run the training script
cd ~/projects/molhiv/scripts
python training.py