#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=DeepPPI
#SBATCH --output=deepPPI.out
#SBATCH --error=deepPPI.err
#SBATCH --mem=90G

echo original
for DATASET in guo huang du pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
  echo FC
  python train_all_datasets.py -name FC_original_${DATASET} -train_pos ../../SPRINT/data/original/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/original/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/original/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/original/${DATASET}_test_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
  echo LSTM
  python train_all_datasets.py -name LSTM_original_${DATASET} -train_pos ../../SPRINT/data/original/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/original/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/original/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/original/${DATASET}_test_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
done

echo rewired
for DATASET in guo huang du pan richoux_regular richoux_strict
do
  echo dataset ${DATASET}
  echo FC
  python train_all_datasets.py -name FC_rewired_${DATASET} -train_pos ../../SPRINT/data/rewired/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/rewired/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/rewired/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/rewired/${DATASET}_test_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
  echo LSTM
  python train_all_datasets.py -name LSTM_rewired_${DATASET} -train_pos ../../SPRINT/data/rewired/${DATASET}_train_pos.txt -train_neg ../../SPRINT/data/rewired/${DATASET}_train_neg.txt -test_pos ../../SPRINT/data/rewired/${DATASET}_test_pos.txt -test_neg ../../SPRINT/data/rewired/${DATASET}_test_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
done

echo partitions
for DATASET in guo huang du pan richoux
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
      python train_all_datasets.py -name FC_partition_${DATASET}_tr${TRAIN}_te${TEST} -train_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_pos.txt -train_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_neg.txt -test_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_pos.txt -test_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
      echo LSTM
      python train_all_datasets.py -name LSTM_partition_${DATASET}_tr${TRAIN}_te${TEST} -train_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_pos.txt -train_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TRAIN}_neg.txt -test_pos ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_pos.txt -test_neg ../../SPRINT/data/partitions/${DATASET}_partition_${TEST}_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
    done
  done
done

