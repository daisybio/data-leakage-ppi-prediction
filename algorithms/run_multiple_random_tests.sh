#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=multiple_random_tests
#SBATCH --output=multiple_random_tests_%A_%a.out
#SBATCH --error=multiple_random_tests_%A_%a.err
#SBATCH --partition=shared-gpu
#SBATCH --mem=100G
#SBATCH --array=0-119

declare -a combis
index=0
for SETTING in original rewired
do
  for MODEL in Custom Richoux_FC SPRINT
  do
    for DATASET in huang guo
    do
      for SEED in 7413 17612 29715 30940 31191 42446 50495 60688 75212 81645
      do
        combis[$index]="$SETTING $MODEL $DATASET $SEED"
        index=$((index+1))
    done
  done
done

parameters=${combis[$SLURM_ARRAY_TASK_ID]}
setting=${parameters[0]}
model=${parameters[1]}
dataset=${parameters[2]}
seed=${parameters[3]}
# echo all parameters
echo $setting $model $dataset $seed

# activate deep_PPIs env
if [ "$model" == "Custom" ]; then
  # Custom
  cd Custom
  python run.py $setting $dataset $seed
  cd ..
elif [ "$model" == "Richoux_FC" ]; then
  #Richoux-FC
  cd DeepPPI/keras
  train_pos=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_train_pos_${seed}.txt
  train_neg=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_train_neg_${seed}.txt
  test_pos=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_test_pos_${seed}.txt
  test_neg=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_test_neg_${seed}.txt
  python train_all_datasets.py -name FC_${setting}_${dataset}_${seed} -train_pos $train_pos -train_neg $train_neg -test_pos $test_pos -test_neg $test_neg -model fc2_20_2dense -epochs 25 -batch 2048
  cd ../..
else
  # SPRINT
  cd SPRINT
  if [[ "$dataset" == "guo" ||  "$dataset" == "du" ]] ; then
    PROTEINS='../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta'
    HSP='HSP/pre_computed_yeast_HSP'
  else
    PROTEINS='data/Uniprot_human_protein_sequences.fasta'
    HSP='HSP/pre_computed_HSP'
  fi
  { time bin/predict_interactions -p $PROTEINS -h $HSP -tr data/${setting}/multiple_random_splits/${dataset}_train_pos_${seed}.txt -pos data/${setting}/multiple_random_splits/${dataset}_test_pos_${seed}.txt -neg data/${setting}/multiple_random_splits/${dataset}_test_neg_${seed}.txt -o results/multiple_runs/${setting}_${dataset}_${seed}_results.txt ; } 2> results/multiple_runs/${setting}_${dataset}_${seed}_time.txt
  cd ..
fi