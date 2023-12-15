#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=train_part_yeast
#SBATCH --output=train_part_yeast.out
#SBATCH --error=train_part_yeast.err
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --partition=compms-gpu-a40

for DATASET in guo du
do
  echo dataset ${DATASET}
  for TRAIN in "both" "0"
  do
    for TEST in "0" "1"
    do
      if [ "$TRAIN" = "0" ] && [ "$TEST" = "0" ]
      then
        continue
      fi
      echo dataset ${DATASET}, training on ${TRAIN}, testing on ${TEST}
      echo DSCRIPT
      { time dscript train --train data/partitions/${DATASET}_partition_${TRAIN}.txt --test data/partitions/${DATASET}_partition_${TEST}.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models/${DATASET}_${TRAIN}_${TEST}_dscript_partitions -o ./results_dscript/partitions/train_${DATASET}_${TRAIN}_${TEST}.txt -d 1; } 2> results_dscript/partitions/train_${DATASET}_${TRAIN}_${TEST}_time.txt
      #{ time dscript evaluate --test data/partitions/${DATASET}_partition_${TEST}.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_${TRAIN}_${TEST}_dscript_partitions_final.sav  -o ./results_dscript/partitions/${DATASET}_${TRAIN}_${TEST} -d 1; } 2> results_dscript/partitions/${DATASET}_${TRAIN}_${TEST}_time.txt

      echo TOPSYTURVY
      { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_${TRAIN}.txt --test data/partitions/${DATASET}_partition_${TEST}.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models/${DATASET}_${TRAIN}_${TEST}_tt_partitions -o ./results_topsyturvy/partitions/train_${DATASET}_${TRAIN}_${TEST}.txt -d 1; } 2> results_topsyturvy/partitions/train_${DATASET}_${TRAIN}_${TEST}_time.txt
      #{ time dscript evaluate --test data/partitions/${DATASET}_partition_${TEST}.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_${TRAIN}_${TEST}_tt_partitions_final.sav  -o ./results_topsyturvy/partitions/${DATASET}_${TRAIN}_${TEST} -d 1; } 2> results_topsyturvy/partitions/${DATASET}_${TRAIN}_${TEST}_time.txt
    done
  done
done
