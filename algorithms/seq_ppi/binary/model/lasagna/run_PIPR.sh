#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --job-name=es_gold_PIPR
#SBATCH --output=es_gold_PIPR.out
#SBATCH --error=es_gold_PIPR.err
#SBATCH --mem=350G
#SBATCH --time=72:00:00
#SBATCH --partition=shared-gpu

#mkdir best_models
python train_all_datasets.py original split_train
python train_all_datasets.py rewired split_train
python train_all_datasets.py partition split_train
python train_all_datasets.py gold_standard split_train
