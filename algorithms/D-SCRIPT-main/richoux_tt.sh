#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=train_tt_richoux
#SBATCH --output=train_tt_richoux.out
#SBATCH --error=train_tt_richoux.err
#SBATCH --mem=300G
#SBATCH --time=30:00:00
#SBATCH --partition=compms-gpu-a40



DATASET=richoux_regular

{ time dscript train --topsy-turvy --train data/original/${DATASET}_train.txt --test data/original/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${DATASET}_tt_original -o ./results_topsyturvy/original/${DATASET}_train.txt -d 1; } 2> results_topsyturvy/original/${DATASET}_time.txt
