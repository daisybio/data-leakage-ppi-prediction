#!/bin/bash

for DATASET in huang pan richoux_regular richoux_strict dscript
do
  echo dataset ${DATASET}
  { time dscript train --train data/original/${DATASET}_train.txt --test data/original/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${DATASET}_dscript_original -o ./results_dscript/original/${DATASET}_train.txt -d 1; } 2> results_dscript/original/${DATASET}_time.txt
  { time dscript evaluate --test data/original/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/${DATASET}_dscript_original_final.sav  -o ./results_dscript/original/${DATASET}.txt -d 1; } 2> results_dscript/original/${DATASET}_time_test.txt

  { time dscript train --topsy-turvy --train data/original/${DATASET}_train.txt --test data/original/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${DATASET}_tt_original -o ./results_topsyturvy/original/${DATASET}_train.txt -d 1; } 2> results_topsyturvy/original/${DATASET}_time.txt
  { time dscript evaluate --test data/original/${DATASET}_test.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/${DATASET}_tt_original_final.sav  -o ./results_topsyturvy/original/${DATASET}.txt -d 1; } 2> results_topsyturvy/original/${DATASET}_time_test.txt
done
