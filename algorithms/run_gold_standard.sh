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
python run.py gold_standard_unbalanced
cd ..
#DeepFE-PPI
cd DeepFE-PPI
echo DeepFE
python train_all_datasets.py gold_standard_unbalanced
cd ..
#Richoux-FC
cd DeepPPI/keras
echo Richoux_FC
python train_all_datasets.py -name FC_gold_standard_unbalanced -train_pos ../../../Datasets_PPIs/unbalanced_gold/Intra1_pos.txt -train_neg ../../../Datasets_PPIs/unbalanced_gold/Intra1_neg.txt -val_pos ../../../Datasets_PPIs/unbalanced_gold/Intra0_pos.txt -val_neg ../../../Datasets_PPIs/unbalanced_gold/Intra0_neg.txt -test_pos ../../../Datasets_PPIs/unbalanced_gold/Intra2_pos.txt -test_neg ../../../Datasets_PPIs/unbalanced_gold/Intra2_neg.txt -model fc2_20_2dense -epochs 25 -batch 2048
#Richoux-LSTM
echo Richoux_LSTM
python train_all_datasets.py -name LSTM_gold_standard_unbalanced -train_pos ../../../Datasets_PPIs/unbalanced_gold/Intra1_pos.txt -train_neg ../../../Datasets_PPIs/unbalanced_gold/Intra1_neg.txt -val_pos ../../../Datasets_PPIs/unbalanced_gold/Intra0_pos.txt -val_neg ../../../Datasets_PPIs/unbalanced_gold/Intra0_neg.txt -test_pos ../../../Datasets_PPIs/unbalanced_gold/Intra2_pos.txt -test_neg ../../../Datasets_PPIs/unbalanced_gold/Intra2_neg.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
cd ../..
#SPRINT
cat ../Datasets_PPIs/unbalanced_gold/Intra0_pos.txt ../Datasets_PPIs/unbalanced_gold/Intra1_pos.txt > ../Datasets_PPIs/unbalanced_gold/Intra0_Intra1_unbalanced_pos.txt
cd SPRINT
echo SPRINT
{ time bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr ../Datasets_PPIs/unbalanced_gold/Intra0_Intra1_unbalanced_pos.txt -pos ../Datasets_PPIs/unbalanced_gold/Intra2_pos.txt -neg ../Datasets_PPIs/unbalanced_gold/Intra2_neg.txt -o results/original/gold_standard_unbalanced_results.txt ; } 2> results/original/gold_standard_unbalanced_time.txt

