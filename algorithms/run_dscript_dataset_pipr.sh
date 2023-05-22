#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=pipr_dscript
#SBATCH --output=pipr_dscript.out
#SBATCH --error=pipr_dscript.err
#SBATCH --mem=350G

# activate pipr env
cd seq_ppi/binary/model/lasagna
python train_all_datasets.py partition
