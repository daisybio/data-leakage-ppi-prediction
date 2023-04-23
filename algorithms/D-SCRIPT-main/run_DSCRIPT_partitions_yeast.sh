#!/bin/bash

for DATASET in guo du
do
  echo dataset ${DATASET}
  echo train both test 0
  { time dscript train --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models/${DATASET}_both_0_dscript_partitions -o ./results_dscript/partitions/${DATASET}_partition_both_0.txt -d 0; } 2> results_dscript/partitions/${DATASET}_partition_both_0_time.txt
  { time dscript evaluate --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_both_0_dscript_partitions_final.sav  -o ./results_dscript/partitions/${DATASET}_both_0.txt -d 0; } 2> results_dscript/partitions/${DATASET}_test_both_0_time.txt

  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models/${DATASET}_both_0_tt_partitions -o ./results_topsyturvy/partitions/${DATASET}_partition_both_0.txt -d 0; } 2> results_topsyturvy/partitions/${DATASET}_partition_both_0_time.txt
  { time dscript evaluate --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_both_0_tt_partitions_final.sav  -o ./results_topsyturvy/partitions/${DATASET}_both_0.txt -d 0; } 2> results_topsyturvy/partitions/${DATASET}_both_0_time.txt

  echo train both test 1
  { time dscript train --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ../models/${DATASET}_both_1_dscript_partitions -o ./results_dscript/partitions/${DATASET}_partition_both_1.txt -d 0; } 2> results_dscript/partitions/${DATASET}_partition_both_1_time.txt
  { time dscript evaluate --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_both_1_dscript_partitions_final.sav  -o ./results_dscript/partitions/${DATASET}_both_1.txt -d 0; } 2> results_dscript/partitions/${DATASET}_both_1_time.txt

  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models/${DATASET}_both_1_tt_partitions -o ./results_topsyturvy/partitions/${DATASET}_partition_both_1.txt -d 0; } 2> results_topsyturvy/partitions/${DATASET}_partition_both_1_time.txt
  { time dscript evaluate --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_both_1_tt_partitions_final.sav  -o ./results_topsyturvy/partitions/${DATASET}_both_1.txt -d 0; } 2> results_topsyturvy/partitions/${DATASET}_both_1_time.txt

  echo train 0 test 1
  { time dscript train --train data/partitions/${DATASET}_partition_0.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models/${DATASET}_0_1_dscript_partitions -o ./results_dscript/partitions/${DATASET}_partition_0_1.txt -d 0; } 2> results_dscript/partitions/${DATASET}_partition_0_1_time.txt
  { time dscript evaluate --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_0_1_dscript_partitions_final.sav  -o ./results_dscript/partitions/${DATASET}_0_1.txt -d 0; } 2> results_dscript/partitions/${DATASET}_0_1_time.txt

  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_0.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models/${DATASET}_0_1_tt_partitions -o ./results_topsyturvy/partitions/${DATASET}_partition_0_1.txt -d 0; } 2> results_topsyturvy/partitions/${DATASET}_partition_0_1_time.txt
  { time dscript evaluate --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --model ./models/${DATASET}_0_1_tt_partitions_final.sav  -o ./results_topsyturvy/partitions/${DATASET}_0_1.txt -d 0; } 2> results_topsyturvy/partitions/${DATASET}_0_1_time.txt
done
