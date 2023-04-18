#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=dscript_train
#SBATCH --output=dscript_train.out
#SBATCH --error=dscript_train.err
#SBATCH --mem=50G


for DATASET in guo du
do
  echo dataset ${DATASET}
  { awk -F '\t' -v OFS='\t' '{$(NF+1) = 1; print }' ../SPRINT/data/original/${DATASET}_train_pos.txt && awk -F '\t' -v OFS='\t' '{$(NF+1) =  0; print }'  ../SPRINT/data/original/${DATASET}_train_neg.txt; } > tmp_train.txt
  { awk -F '\t' -v OFS='\t' '{$(NF+1) = 1; print }' ../SPRINT/data/original/${DATASET}_test_pos.txt && awk -F '\t' -v OFS='\t' '{$(NF+1) =  0; print }'  ../SPRINT/data/original/${DATASET}_test_neg.txt; } > tmp_test.txt
  { time dscript train --train tmp_train.txt --test tmp_test.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_dscript/original/${DATASET}_train.txt; } 2> results_dscript/original/${DATASET}_time.txt
  { time dscript train --topsy-turvy --train tmp_train.txt --test tmp_test.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/original/${DATASET}_train.txt; } 2> results_topsyturvy/original/${DATASET}_time.txt
done
for DATASET in huang pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
  { awk -F '\t' -v OFS='\t' '{$(NF+1) = 1; print }' ../SPRINT/data/original/${DATASET}_train_pos.txt && awk -F '\t' -v OFS='\t' '{$(NF+1) =  0; print }'  ../SPRINT/data/original/${DATASET}_train_neg.txt; } > tmp_train.txt
  { awk -F '\t' -v OFS='\t' '{$(NF+1) = 1; print }' ../SPRINT/data/original/${DATASET}_test_pos.txt && awk -F '\t' -v OFS='\t' '{$(NF+1) =  0; print }'  ../SPRINT/data/original/${DATASET}_test_neg.txt; } > tmp_test.txt
  { time dscript train --train tmp_train.txt --test tmp_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_dscript/original/${DATASET}_train.txt; } 2> results_dscript/original/${DATASET}_time.txt
  { time dscript train --topsy-turvy --train tmp_train.txt --test tmp_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/original/${DATASET}_train.txt; } 2> results_topsyturvy/original/${DATASET}_time.txt
done