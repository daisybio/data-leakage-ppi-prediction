#!/bin/bash
#Reproduce what I did
# first: download pre_computed_HSP and Uniprot_human_protein_sequences.fasta files from https://www.csd.uwo.ca/~ilie/SPRINT/
# then: try to reproduce results:

#Biogrid C1 dataset:
# bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr PPI_dataset/Biogrid/train.pos.1.txt -pos PPI_dataset/Biogrid/test.pos.c1.1.txt -neg PPI_dataset/Biogrid/test.neg.c1.1.txt -o results/result_Biogrid_C1_1.txt

#HPRD C3 dataset:
# bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr PPI_dataset/HPRD/train.pos.1.txt -pos PPI_dataset/HPRD/test.pos.c1.1.txt -neg PPI_dataset/HPRD/test.neg.c1.1.txt -o results/result_HPRD_C3_1.txt

# run other datasets huang, richoux, pan;
# Train on both, test on 0. Train on both, test on 1. Train on 0, test on 1:

for DATASET in huang richoux pan
do
  for TRAIN in "both" "0"
  do
    for TEST in "0" "1"
    do
      if [ "$TRAIN" = "0" ] && [ "$TEST" = "0" ]
      then
        continue
      fi
      echo dataset ${DATASET}, training on ${TRAIN}, testing on ${TEST}
      bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr data/${DATASET}_partition_${TRAIN}_pos.txt -pos data/${DATASET}_partition_${TEST}_pos.txt -neg data/${DATASET}_partition_${TEST}_neg.txt -o results/${DATASET}_train_${TRAIN}_test_${TEST}.txt
    done
  done
done