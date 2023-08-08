#!/bin/bash
#SBATCH --partition=exbio-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --job-name=dscript_gold
#SBATCH --output=dscript_gold.out
#SBATCH --error=dscript_gold.err
#SBATCH --cpus-per-gpu=10
#SBATCH --nodelist=gpu02.exbio.wzw.tum.de

echo dscript
{ time dscript train --train data/gold/Intra1.txt --test data/gold/Intra0.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/dscript_gold -o ./results_dscript/original/gold_train.txt -d 1; } 2> results_dscript/original/gold_time.txt
#{ time dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/dscript_gold_final.sav  -o ./results_dscript/original/gold.txt -d 1; } 2> results_dscript/original/gold_time_test.txt

