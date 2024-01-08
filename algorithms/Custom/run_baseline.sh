#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --job-name=baseML
#SBATCH --output=baselineML_%A_%a.out
#SBATCH --error=baselineML_%A_%a.err
#SBATCH --mem=100G
#SBATCH --array=3

declare -a settings
settings[0]="original dscript"
settings[1]="rewired dscript"
settings[2]="partition dscript"
settings[3]="gold_standard"

parameters=${settings[$SLURM_ARRAY_TASK_ID]}
IFS=' ' read -r -a params_array <<< "$parameters"
setting=${params_array[0]}
dataset=${params_array[1]}

echo $setting $dataset

python run.py $setting $dataset
