#!/bin/bash
#SBATCH --ntasks-per-node=10
#SBATCH --partition=htc
#SBATCH --gres=gpu:2
#SBATCH --job-name="unas"

# TODO: do a pipenv install to make sure environment exists and is up to date?
# TODO: load modules
srun pipenv install
srun module load gpu/cuda/10.1.243
srun module load gpu/cudnn/7.6.5__cuda-10.1
srun pipenv run python driver.py "$@"
