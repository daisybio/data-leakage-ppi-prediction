#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=gold_all
#SBATCH --output=gold_all.out
#SBATCH --error=gold_all.err
#SBATCH --mem=350G

# activate pipr env
cd seq_ppi/binary/model/lasagna
python train_all_datasets.py gold_standard_unbalanced
