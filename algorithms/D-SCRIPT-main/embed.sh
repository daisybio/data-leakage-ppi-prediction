#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --job-name=embed_dscript
#SBATCH --output=embed_dscript.out
#SBATCH --error=embed_dscript.err
#SBATCH --mem=20G

#echo ###### YEAST #########
#mkdir embeddings
#dscript embed --seqs ../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta --outfile embeddings/yeast_embedding.h5
echo ###### HUMAN #########
dscript embed --seqs ../../Datasets_PPIs/SwissProt/human_swissprot_oneliner.fasta --outfile /nfs/scratch/jbernett/human_embedding.h5
