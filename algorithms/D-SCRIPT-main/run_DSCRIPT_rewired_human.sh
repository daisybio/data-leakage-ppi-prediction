#!/bin/bash

for DATASET in dscript huang pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
  { time dscript train --train data/rewired/${DATASET}_train.txt --test data/rewired/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${DATASET}_dscript_rewired -o ./results_dscript/rewired/${DATASET}_train.txt -d 1; } 2> results_dscript/rewired/${DATASET}_time.txt
  { time dscript evaluate --test data/rewired/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/${DATASET}_dscript_rewired_final.sav  -o ./results_dscript/rewired/${DATASET}.txt -d 1; } 2> results_dscript/rewired/${DATASET}_time_test.txt

  { time dscript train --topsy-turvy --train data/rewired/${DATASET}_train.txt --test data/rewired/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${DATASET}_tt_rewired -o ./results_topsyturvy/rewired/${DATASET}_train.txt -d 1; } 2> results_topsyturvy/rewired/${DATASET}_time.txt
  { time dscript evaluate --test data/rewired/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/${DATASET}_tt_rewired_final.sav  -o ./results_topsyturvy/rewired/${DATASET}.txt -d 1; } 2> results_topsyturvy/rewired/${DATASET}_time_test.txt
done