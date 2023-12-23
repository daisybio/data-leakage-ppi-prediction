#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=eval_ds_tt
#SBATCH --output=eval_ds_tt_%A_%a.out
#SBATCH --error=eval_ds_tt_%A_%a.err
#SBATCH --partition=shared-cpu
#SBATCH --mem=100G
#SBATCH --array=0-39


declare -a combis
index=0
for SETTING in original rewired
do
	for MODEL in D-SCRIPT #Topsy-Turvy #Custom Richoux_FC Richoux_LSTM SPRINT PIPR DeepFE
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

cd D-SCRIPT-main

test_file="data/multiple_runs/${setting}_${dataset}_test_${seed}.txt"

if [[ "$dataset" == "guo" ||  "$dataset" == "du" ]] ; then
    embedding='/nfs/scratch/jbernett/yeast_embedding.h5'
else
    embedding='/nfs/scratch/jbernett/human_embedding.h5'
fi

if [ "$model" == "D-SCRIPT" ]; then
  outfile="results_dscript/multiple_runs/${setting}_${dataset}_${seed}.txt"
  model_path="models/${dataset}_${seed}_dscript_${setting}_final.sav"
elif [ "$model" == "Topsy-Turvy" ]; then
  outfile="results_topsyturvy/multiple_runs/${setting}_${dataset}_${seed}.txt"
  model_path="models/${dataset}_${seed}_tt_${setting}_final.sav"

dscript evaluate --test $test_file  --embedding $embedding --model $model_path -o $outfile

