#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=dscript_partitions
#SBATCH --output=dscript_partitions.out
#SBATCH --error=dscript_partitions.err
#SBATCH --mem=50G


for DATASET in guo du
do
  echo dataset ${DATASET}
  echo train both test 0
  { time dscript train --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_dscript/partitions/${DATASET}_partition_both_0.txt; } 2> results_dscript/partitions/${DATASET}_partition_both_0_time.txt
  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/partitions/${DATASET}_partition_both_0.txt; } 2> results_topsyturvy/partitions/${DATASET}_partition_both_0_time.txt
  echo train both test 1
  { time dscript train --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_dscript/partitions/${DATASET}_partition_both_1.txt; } 2> results_dscript/partitions/${DATASET}_partition_both_1_time.txt
  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/partitions/${DATASET}_partition_both_1.txt; } 2> results_topsyturvy/partitions/${DATASET}_partition_both_1_time.txt
  echo train 0 test 1
  { time dscript train --train data/partitions/${DATASET}_partition_0.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_dscript/partitions/${DATASET}_partition_0_1.txt; } 2> results_dscript/partitions/${DATASET}_partition_0_1_time.txt
  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_0.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/yeast_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/partitions/${DATASET}_partition_0_1.txt; } 2> results_topsyturvy/partitions/${DATASET}_partition_0_1_time.txt
done
for DATASET in huang pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
    echo train both test 0
  { time dscript train --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_dscript/partitions/${DATASET}_partition_both_0.txt; } 2> results_dscript/partitions/${DATASET}_partition_both_0_time.txt
  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_0.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/partitions/${DATASET}_partition_both_0.txt; } 2> results_topsyturvy/partitions/${DATASET}_partition_both_0_time.txt
  echo train both test 1
  { time dscript train --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_dscript/partitions/${DATASET}_partition_both_1.txt; } 2> results_dscript/partitions/${DATASET}_partition_both_1_time.txt
  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_both.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/partitions/${DATASET}_partition_both_1.txt; } 2> results_topsyturvy/partitions/${DATASET}_partition_both_1_time.txt
  echo train 0 test 1
  { time dscript train --train data/partitions/${DATASET}_partition_0.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_dscript/partitions/${DATASET}_partition_0_1.txt; } 2> results_dscript/partitions/${DATASET}_partition_0_1_time.txt
  { time dscript train --topsy-turvy --train data/partitions/${DATASET}_partition_0.txt --test data/partitions/${DATASET}_partition_1.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models -o ./results_topsyturvy/partitions/${DATASET}_partition_0_1.txt; } 2> results_topsyturvy/partitions/${DATASET}_partition_0_1_time.txt
done
