#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=train_tt_rew_human
#SBATCH --output=train_tt_rewired_human%j.out
#SBATCH --error=train_tt_rewired_human%j.err
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --partition=compms-gpu-a40
#SBATCH --array=0-3

declare -a datasets
datasets=(huang pan richoux_regular richoux_strict)

{ time dscript train --topsy-turvy --train data/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_train.txt --test data/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${datasets[${SLURM_ARRAY_TASK_ID}]}_tt_rewired -o ./results_topsyturvy/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_train.txt; } 2> results_topsyturvy/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_time.txt
