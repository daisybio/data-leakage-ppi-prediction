#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=es_gold_deepFE
#SBATCH --output=deepFE_es_gold.out
#SBATCH --error=deepFE_es_gold.err
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --partition=shared-gpu

#mkdir best_models
#python train_all_datasets.py original split_train
#python train_all_datasets.py rewired split_train
#python train_all_datasets.py partition split_train
python train_all_datasets.py gold split_train