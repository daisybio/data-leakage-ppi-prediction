#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=DeepPPI
#SBATCH --output=deepPPI.out
#SBATCH --error=deepPPI.err
#SBATCH --mem=15G


for DATASET in guo huang du pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
  echo FC
  python train_all_datasets.py -name FC_rewired_${DATASET} -train_pos ../../SPRINT/data/rewired/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/rewired/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/rewired/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/rewired/${DATASET}_test_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
  echo LSTM
  python train_all_datasets.py -name LSTM_rewired_${DATASET} -train_pos ../../SPRINT/data/rewired/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/rewired/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/rewired/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/rewired/${DATASET}_test_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
done
