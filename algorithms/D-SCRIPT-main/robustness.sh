#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=train_dscript_ds
#SBATCH --output=train_dscript_ds_%A_%a.out
#SBATCH --error=train_dscript_ds_%A_%a.err
#SBATCH --mem=300G
#SBATCH --time=30:00:00
#SBATCH --partition=shared-gpu
#SBATCH --array=0-2
#SBATCH --exclude=gpu01.exbio.wzw.tum.de,jlab-gpu01.exbio.wzw.tum.de

dataset=richoux_regular

dscript train --train data/original/${dataset}_train.txt --test data/original/${dataset}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/robustness_${dataset}_dscript_original_${SLURM_ARRAY_TASK_ID} -o ./robustness_tests/robustness_${dataset}_dscript_original_train_${SLURM_ARRAY_TASK_ID}.txt

dscript evaluate --test data/original/${dataset}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/robustness_${dataset}_dscript_original_${SLURM_ARRAY_TASK_ID}_final.sav  -o ./robustness_tests/robustness_${dataset}_dscript_original_${SLURM_ARRAY_TASK_ID}.txt
	
