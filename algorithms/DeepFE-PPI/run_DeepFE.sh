#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=run_deepFE
#SBATCH --output=deepFE.out
#SBATCH --error=deepFE.err
#SBATCH --mem=90G

python train_all_datasets.py