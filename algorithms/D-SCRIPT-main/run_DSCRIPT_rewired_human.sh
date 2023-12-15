#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=train_ds_rew_human
#SBATCH --output=train_ds_rewired_human%j.out
#SBATCH --error=train_ds_rewired_human%j.err
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --partition=compms-gpu-a40
#SBATCH --array=0-3

declare -a datasets=(huang pan richoux_regular richoux_strict)
{ time dscript train --train data/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_train.txt --test data/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${datasets[${SLURM_ARRAY_TASK_ID}]}_dscript_rewired -o ./results_dscript/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_train.txt; } 2> results_dscript/rewired/${datasets[${SLURM_ARRAY_TASK_ID}]}_time.txt

