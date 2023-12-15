#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=gold_pipr
#SBATCH --output=gold_pipr.out
#SBATCH --error=gold_pipr.err
#SBATCH --mem=370G

# activate pipr env
cd seq_ppi/binary/model/lasagna
python train_all_datasets.py gold_standard
