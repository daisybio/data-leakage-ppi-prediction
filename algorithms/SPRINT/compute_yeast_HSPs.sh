#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=yeast_HSPs
#SBATCH --output=yeast_HSPs.out
#SBATCH --error=yeast_HSPs.err
#SBATCH --mem=20G

bin/compute_HSPs -p ../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta -h pre_computed_yeast_HSP