#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --job-name=baseML
#SBATCH --output=baselineML_%A_%a.out
#SBATCH --error=baselineML_%A_%a.err
#SBATCH --mem=40G
#SBATCH --array=0-3

declare -a settings
settings=(original rewired partition gold_standard)
echo ${settings[${SLURM_ARRAY_TASK_ID}]}

python run.py ${settings[${SLURM_ARRAY_TASK_ID}]}
