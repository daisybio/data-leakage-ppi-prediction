#!/bin/bash
#SBATCH --partition=exbio-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --job-name=dscript_gold
#SBATCH --output=tt_gold.out
#SBATCH --error=tt_gold.err
#SBATCH --cpus-per-gpu=10
#SBATCH --nodelist=gpu02.exbio.wzw.tum.de

echo topsyturvy
{ time dscript train --topsy-turvy --train data/gold/Intra1.txt --test data/gold/Intra0.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/tt_gold -o ./results_topsyturvy/original/gold_train.txt -d 2; } 2> results_topsyturvy/original/gold_time.txt
#{ time dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/tt_gold_final.sav  -o ./results_topsyturvy/original/gold.txt -d 2; } 2> results_topsyturvy/original/gold_time_test.txt
