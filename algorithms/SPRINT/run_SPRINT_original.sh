#!/bin/bash

# run on original yeast datasets: du, guo
for DATASET in du guo
do
  echo dataset ${DATASET}
  { time bin/predict_interactions -p ../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta -h HSP/pre_computed_yeast_HSP -tr data/original/${DATASET}_train_pos.txt -pos data/original/${DATASET}_test_pos.txt -neg data/original/${DATASET}_test_neg.txt -o results/original/${DATASET}_results.txt ; } 2> results/original/${DATASET}_time.txt
done
for DATASET in huang pan richoux_regular richoux_strict dscript
do
  echo dataset ${DATASET}
  { time bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr data/original/${DATASET}_train_pos.txt -pos data/original/${DATASET}_test_pos.txt -neg data/original/${DATASET}_test_neg.txt -o results/original/${DATASET}_results.txt ; } 2> results/original/${DATASET}_time.txt
done