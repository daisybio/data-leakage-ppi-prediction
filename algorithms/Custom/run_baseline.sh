#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --job-name=baseML
#SBATCH --output=baselineML.out
#SBATCH --error=baselineML.err
#SBATCH --mem=40G

python run.py original
python run.py rewired
python run.py partition
python run.py gold
