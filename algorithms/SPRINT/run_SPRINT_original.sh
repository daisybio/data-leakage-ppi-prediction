#!/bin/bash

# run on original human datasets: huang, pan, richoux regular, richoux strict

for DATASET in huang pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
  bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr data/original/${DATASET}_train_pos.txt -pos data/original/${DATASET}_test_pos.txt -neg data/original/${DATASET}_test_neg.txt -o results/original/${DATASET}_results.txt
done