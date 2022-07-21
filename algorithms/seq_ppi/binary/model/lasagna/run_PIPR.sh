#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=run_PIPR
#SBATCH --output=PIPR.out
#SBATCH --error=PIPR.err
#SBATCH --mem=90G

python train_all_datasets.py