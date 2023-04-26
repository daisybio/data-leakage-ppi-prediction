#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=runp_PIPR
#SBATCH --output=part_PIPR.out
#SBATCH --error=part_PIPR.err
#SBATCH --mem=150G

#python train_all_datasets.py original
#python train_all_datasets.py rewired
python train_all_datasets.py partition
