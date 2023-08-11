#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=es_DeepPPI
#SBATCH --output=es_deepPPI.out
#SBATCH --error=es_deepPPI.err
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --partition=shared-gpu

mkdir best_models
echo original
for DATASET in guo huang du pan richoux_regular richoux_strict dscript
do
  echo dataset ${DATASET}
  echo FC
  python train_all_datasets.py -name FC_original_${DATASET}_es -train_pos ../../SPRINT/data/original/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/original/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/original/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/original/${DATASET}_test_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048 -patience 5 -split_train true
  echo LSTM
  python train_all_datasets.py -name LSTM_original_${DATASET}_es -train_pos ../../SPRINT/data/original/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/original/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/original/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/original/${DATASET}_test_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048 -patience 5 -split_train true
done

echo rewired
for DATASET in guo huang du pan richoux_regular richoux_strict dscript
do
  echo dataset ${DATASET}
  echo FC
  python train_all_datasets.py -name FC_rewired_${DATASET}_es -train_pos ../../SPRINT/data/rewired/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/rewired/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/rewired/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/rewired/${DATASET}_test_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048 -split_train true
  echo LSTM
  python train_all_datasets.py -name LSTM_rewired_${DATASET}_es -train_pos ../../SPRINT/data/rewired/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/rewired/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/rewired/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/rewired/${DATASET}_test_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048 -split_train true
done

echo partitions
for DATASET in guo huang du pan richoux dscript
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
      echo FC
      python train_all_datasets.py -name FC_partition_${DATASET}_tr${TRAIN}_te${TEST}_es -train_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_pos.txt -train_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_neg.txt -test_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_pos.txt -test_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048 -split_train true
      echo LSTM
      python train_all_datasets.py -name LSTM_partition_${DATASET}_tr${TRAIN}_te${TEST}_es -train_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_pos.txt -train_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_neg.txt -test_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_pos.txt -test_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048 -split_train true
    done
  done
done

#echo gold
#python train_all_datasets.py -name FC_gold_standard_es -train_pos ../../../Datasets_PPIs/Hippiev2.3/Intra1_pos_rr.txt -train_neg ../../../Datasets_PPIs/Hippiev2.3/Intra1_neg_rr.txt -val_pos ../../../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt -val_neg ../../../Datasets_PPIs/Hippiev2.3/Intra0_neg_rr.txt -test_pos ../../../Datasets_PPIs/Hippiev2.3/Intra2_pos_rr.txt -test_neg ../../../Datasets_PPIs/Hippiev2.3/Intra2_neg_rr.txt -model fc2_20_2dense -epochs 25 -batch 2048
#python train_all_datasets.py -name LSTM_gold_standard_es -train_pos ../../../Datasets_PPIs/Hippiev2.3/Intra1_pos_rr.txt -train_neg ../../../Datasets_PPIs/Hippiev2.3/Intra1_neg_rr.txt -val_pos ../../../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt -val_neg ../../../Datasets_PPIs/Hippiev2.3/Intra0_neg_rr.txt -test_pos ../../../Datasets_PPIs/Hippiev2.3/Intra2_pos_rr.txt -test_neg ../../../Datasets_PPIs/Hippiev2.3/Intra2_neg_rr.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048


