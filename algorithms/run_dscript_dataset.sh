#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=gold_all
#SBATCH --output=gold_all.out
#SBATCH --error=gold_all.err
#SBATCH --mem=350G

# activate deep_PPIs env
# Custom
cd Custom
echo Custom
python run.py original
cd ..
#DeepFE-PPI
cd DeepFE-PPI
echo DeepFE
python train_all_datasets.py original
cd ..
#Richoux-FC
cd DeepPPI/keras
echo Richoux_FC
python train_all_datasets.py -name FC_gold_standard -train_pos ../../SPRINT/data/original/dscript_train_pos.txt -train_neg ../../SPRINT/data/original/dscript_train_neg.txt -test_pos ../../SPRINT/data/original/dscript_test_pos.txt -test_neg ../../SPRINT/data/original/dscript_test_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
#Richoux-LSTM
echo Richoux_LSTM
python train_all_datasets.py -name LSTM_gold_standard -train_pos ../../SPRINT/data/original/dscript_train_pos.txt -train_neg ../../SPRINT/data/original/dscript_train_neg.txt -test_pos ../../SPRINT/data/original/dscript_test_pos.txt -test_neg ../../SPRINT/data/original/dscript_test_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
cd ../..
#SPRINT
cd SPRINT
echo SPRINT
{ time bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr ../../SPRINT/data/original/dscript_train_pos.txt -pos ../../SPRINT/data/original/dscript_test_pos.txt -neg ../../SPRINT/data/original/dscript_test_neg.txt -o results/original/dscript_results.txt ; } 2> results/original/dscript_time.txt
