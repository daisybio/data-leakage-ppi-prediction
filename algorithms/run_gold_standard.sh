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
python run.py gold_standard
cd ..
#DeepFE-PPI
cd DeepFE-PPI
echo DeepFE
python train_all_datasets.py gold_standard
cd ..
#Richoux-FC
cd DeepPPI/keras
echo Richoux_FC
python train_all_datasets.py -name FC_gold_standard -train_pos ../../../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt -train_neg ../../../Datasets_PPIs/Hippiev2.3/Intra0_neg_rr.txt -val_pos ../../../Datasets_PPIs/Hippiev2.3/Intra1_pos_rr.txt -val_neg ../../../Datasets_PPIs/Hippiev2.3/Intra1_neg_rr.txt -test_pos ../../../Datasets_PPIs/Hippiev2.3/Intra2_pos_rr.txt -test_neg ../../../Datasets_PPIs/Hippiev2.3/Intra2_neg_rr.txt -model fc2_20_2dense -epochs 25 -batch 2048
#Richoux-LSTM
echo Richoux_LSTM
python train_all_datasets.py -name LSTM_gold_standard -train_pos ../../../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt -train_neg ../../../Datasets_PPIs/Hippiev2.3/Intra0_neg_rr.txt -val_pos ../../../Datasets_PPIs/Hippiev2.3/Intra1_pos_rr.txt -val_neg ../../../Datasets_PPIs/Hippiev2.3/Intra1_neg_rr.txt -test_pos ../../../Datasets_PPIs/Hippiev2.3/Intra2_pos_rr.txt -test_neg ../../../Datasets_PPIs/Hippiev2.3/Intra2_neg_rr.txt -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
cd ../..
#SPRINT
cat ../Datasets_PPIs/Hippiev2.3/Intra0_pos_rr.txt ../Datasets_PPIs/Hippiev2.3/Intra1_pos_rr.txt > ../Datasets_PPIs/Hippiev2.3/Intra0_Intra1_pos_rr.txt
cd SPRINT
echo SPRINT
{ time bin/predict_interactions -p data/Uniprot_human_protein_sequences.fasta -h HSP/pre_computed_HSP -tr ../../Datasets_PPIs/Hippiev2.3/Intra0_Intra1_pos_rr.txt -pos ../../Datasets_PPIs/Hippiev2.3/Intra2_pos_rr.txt -neg ../../Datasets_PPIs/Hippiev2.3/Intra2_neg_rr.txt -o results/original/gold_standard_results.txt ; } 2> results/original/gold_standard_time.txt

