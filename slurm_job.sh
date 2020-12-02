#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --job-name="unas"

# TODO: do a pipenv install to make sure environment exists and is up to date?
srun pipenv run python driver.py "$@"
