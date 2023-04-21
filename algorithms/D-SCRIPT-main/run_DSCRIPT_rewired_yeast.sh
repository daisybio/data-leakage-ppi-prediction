#!/bin/bash

for DATASET in guo du
do
  echo dataset ${DATASET}
  { time dscript train --train data/rewired/${DATASET}_train.txt --test data/rewired/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_dscript/rewired/${DATASET}_train.txt -d 0; } 2> results_dscript/rewired/${DATASET}_time.txt
  { time dscript train --topsy-turvy --train data/rewired/${DATASET}_train.txt --test data/rewired/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/rewired/${DATASET}_train.txt -d 0; } 2> results_topsyturvy/rewired/${DATASET}_time.txt
done
