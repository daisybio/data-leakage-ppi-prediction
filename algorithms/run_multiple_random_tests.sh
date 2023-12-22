#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=multiple_random_tests
#SBATCH --output=multiple_random_tests_%A_%a.out
#SBATCH --error=multiple_random_tests_%A_%a.err
#SBATCH --partition=shared-gpu
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --array=0-39


declare -a combis
index=0
for SETTING in original rewired
do
	for MODEL in PIPR #DeepFE D-SCRIPT Topsy-Turvy Custom Richoux_FC Richoux_LSTM SPRINT
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
done
parameters=${combis[$SLURM_ARRAY_TASK_ID]}

# Split the parameters string into an array
IFS=' ' read -r -a params_array <<< "$parameters"

# Assign individual elements to variables
setting=${params_array[0]}
model=${params_array[1]}
dataset=${params_array[2]}
seed=${params_array[3]}

# echo all parameters
echo $setting $model $dataset $seed

# activate deep_PPIs env
echo model: $model
if [ "$model" == "Custom" ]; then
  # Custom
  cd Custom
  python run.py $setting $dataset $seed
  cd ..
elif [ "$model" == "Richoux_FC" ]; then
  #Richoux-FC
  export LD_LIBRARY_PATH=/nfs/home/students/jbernett/.conda/envs/deep_PPIs/lib
  cd DeepPPI/keras
  train_pos=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_train_pos_${seed}.txt
  train_neg=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_train_neg_${seed}.txt
  test_pos=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_test_pos_${seed}.txt
  test_neg=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_test_neg_${seed}.txt
  python train_all_datasets.py -name FC_${setting}_${dataset}_${seed} -train_pos $train_pos -train_neg $train_neg -test_pos $test_pos -test_neg $test_neg -model fc2_20_2dense -epochs 25 -batch 2048
  cd ../..
elif [ "$model" == "Richoux_LSTM" ]; then
  #Richoux-LSTM
  export LD_LIBRARY_PATH=/nfs/home/students/jbernett/.conda/envs/deep_PPIs/lib
  cd DeepPPI/keras
  train_pos=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_train_pos_${seed}.txt
  train_neg=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_train_neg_${seed}.txt
  test_pos=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_test_pos_${seed}.txt
  test_neg=../../SPRINT/data/${setting}/multiple_random_splits/${dataset}_test_neg_${seed}.txt
  python train_all_datasets.py -name LSTM_${setting}_${dataset}_${seed} -train_pos $train_pos -train_neg $train_neg -test_pos $test_pos -test_neg $test_neg -model lstm32_3conv3_2dense_shared -epochs 100 -batch 2048
  cd ../..
elif [ "$model" == "DeepFE" ]; then
  # DeepFE
  export LD_LIBRARY_PATH=/nfs/home/students/jbernett/.conda/envs/deep_PPIs/lib
  cd DeepFE-PPI
  python train_all_datasets.py $setting $dataset $seed
  cd ..
elif [ "$model" == "PIPR" ]; then
  # PIPR activate pipr
  export LD_LIBRARY_PATH=/nfs/home/students/jbernett/.conda/envs/pipr/lib
  cd seq_ppi/binary/model/lasagna
  python train_all_datasets.py $setting $dataset $seed
  cd ..
elif [ "$model" == "D-SCRIPT" ]; then
  # D-SCRIPT: activate dscript env
  if [[ "$dataset" == "guo" ||  "$dataset" == "du" ]] ; then
    EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
  else
    EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
  fi
  cd D-SCRIPT-main
  train_file=data/multiple_runs/${setting}_${dataset}_train_${seed}.txt
  test_file=data/multiple_runs/${setting}_${dataset}_test_${seed}.txt
  outfile="${setting}_${dataset}_${seed}_train.txt"
  timefile="${setting}_${dataset}_${seed}_time.txt"
  model_name="${dataset}_${seed}_dscript_${setting}"
  { time dscript train --train ${train_file} --test ${test_file} --embedding $EMBEDDING --save-prefix ./models/${model_name} -o results_dscript/multiple_runs/${outfile}; } 2> results_dscript/multiple_runs/${timefile}
  cd ..
elif [ "$model" == "Topsy-Turvy" ]; then
  # Topsy-Turvy: activate dscript env
  if [[ "$dataset" == "guo" ||  "$dataset" == "du" ]] ; then
    EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
  else
    EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
  fi
  cd D-SCRIPT-main
  train_file=data/multiple_runs/${setting}_${dataset}_train_${seed}.txt
  test_file=data/multiple_runs/${setting}_${dataset}_test_${seed}.txt
  outfile="${setting}_${dataset}_${seed}_train.txt"
  timefile="${setting}_${dataset}_${seed}_time.txt"
  model_name="${dataset}_${seed}_tt_${setting}"
  { time dscript train --topsy-turvy --train ${train_file} --test ${test_file} --embedding $EMBEDDING --save-prefix ./models/${model_name} -o results_topsyturvy/multiple_runs/${outfile}; } 2> results_topsyturvy/multiple_runs/${timefile}
  cd ..
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
  training_data=data/${setting}/multiple_random_splits/${dataset}_train_pos_${seed}.txt
  test_pos=data/${setting}/multiple_random_splits/${dataset}_test_pos_${seed}.txt
  test_neg=data/${setting}/multiple_random_splits/${dataset}_test_neg_${seed}.txt
  output=results/multiple_runs/${setting}_${dataset}_${seed}_results.txt
  time_file=results/multiple_runs/${setting}_${dataset}_${seed}_time.txt
  echo $training_data $test_pos $test_neg $output $time_file
  { time bin/predict_interactions -p $PROTEINS -h $HSP -tr $training_data -pos $test_pos -neg $test_neg -o $output ; } 2> $time_file
fi
