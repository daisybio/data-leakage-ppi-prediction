#!/bin/bash

# run on original yeast datasets: du, guo
for DATASET in du guo
do
  echo dataset ${DATASET}
  { time bin/predict_interactions -p ../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta -h HSP/pre_computed_yeast_HSP -tr data/rewired/${DATASET}_train_pos.txt -pos data/rewired/${DATASET}_test_pos.txt -neg data/rewired/${DATASET}_test_neg.txt -o results/rewired/${DATASET}_results.txt ; } 2> results/rewired/${DATASET}_time.txt
done
for DATASET in huang pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
  { time bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr data/rewired/${DATASET}_train_pos.txt -pos data/rewired/${DATASET}_test_pos.txt -neg data/rewired/${DATASET}_test_neg.txt -o results/rewired/${DATASET}_results.txt ; } 2> results/rewired/${DATASET}_time.txt
done