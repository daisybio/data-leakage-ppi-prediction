#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=eval_gold
#SBATCH --output=evaluate_gold_es.out
#SBATCH --error=evaluate_gold_es.err
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --partition=compms-gpu-a40


dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/dscript_gold_epoch02.sav  -o ./results_dscript/original/gold_es.txt
dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/tt_gold_epoch10.sav  -o ./results_topsyturvy/original/gold_es.txt 
