#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=dscript_all
#SBATCH --output=dscript_all.out
#SBATCH --error=dscript_all.err
#SBATCH --mem=350G

# activate deep_PPIs env
# Custom
#cd Custom
#echo Custom
#python run.py partition
#cd ..
# DeepFE-PPI
#cd DeepFE-PPI
#echo DeepFE
#python train_all_datasets.py partition
#cd ..
# Richoux-FC
cd DeepPPI/keras
echo Richoux_FC
python train_all_datasets.py -name FC_partition_dscript_trboth_te0 -train_pos ../../SPRINT/data/partitions/dscript_partition_both_pos.txt -train_neg ../../SPRINT/data/partitions/dscript_partition_both_neg.txt -test_pos ../../SPRINT/data/partitions/dscript_partition_0_pos.txt -test_neg ../../SPRINT/data/partitions/dscript_partition_0_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
python train_all_datasets.py -name FC_partition_dscript_trboth_te1 -train_pos ../../SPRINT/data/partitions/dscript_partition_both_pos.txt -train_neg ../../SPRINT/data/partitions/dscript_partition_both_neg.txt -test_pos ../../SPRINT/data/partitions/dscript_partition_1_pos.txt -test_neg ../../SPRINT/data/partitions/dscript_partition_1_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
#Richoux-LSTM
echo Richoux_LSTM
python train_all_datasets.py -name LSTM_partition_dscript_trboth_te0 -train_pos ../../SPRINT/data/partitions/dscript_partition_both_pos.txt -train_neg ../../SPRINT/data/partitions/dscript_partition_both_neg.txt -test_pos ../../SPRINT/data/partitions/dscript_partition_0_pos.txt -test_neg ../../SPRINT/data/partitions/dscript_partition_0_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
python train_all_datasets.py -name LSTM_partition_dscript_trboth_te1 -train_pos ../../SPRINT/data/partitions/dscript_partition_both_pos.txt -train_neg ../../SPRINT/data/partitions/dscript_partition_both_neg.txt -test_pos ../../SPRINT/data/partitions/dscript_partition_1_pos.txt -test_neg ../../SPRINT/data/partitions/dscript_partition_1_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
cd ../..
# SPRINT
cd SPRINT
echo SPRINT
{ time bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr data/partitions/dscript_partition_both_pos.txt -pos data/partitions/dscript_partition_0_pos.txt -neg data/partitions/dscript_partition_0_neg.txt -o results/partitions/dscript_train_both_test_0.txt ; } 2> results/partitions/dscript_train_both_test_0_time.txt
{ time bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr data/partitions/dscript_partition_both_pos.txt -pos data/partitions/dscript_partition_1_pos.txt -neg data/partitions/dscript_partition_1_neg.txt -o results/partitions/dscript_train_both_test_1.txt ; } 2> results/partitions/dscript_train_both_test_1_time.txt
