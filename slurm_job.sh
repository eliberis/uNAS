#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --job-name="mu-nas"

# TODO: do a pipenv install to make sure environment exists and is up to date?
srun pipenv run python driver.py "$@"
