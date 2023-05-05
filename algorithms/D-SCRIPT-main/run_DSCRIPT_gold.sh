#!/bin/bash

echo dscript
#cat data/gold/Intra0.txt data/gold/Intra1.txt > data/gold/Intra0_Intra1.txt
#{ time dscript train --train data/gold/Intra0_Intra1.txt --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/dscript_gold -o ./results_dscript/original/gold_train.txt -d 2; } 2> results_dscript/original/gold_time.txt
#{ time dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/dscript_gold_final.sav  -o ./results_dscript/original/gold.txt -d 2; } 2> results_dscript/original/gold_time_test.txt

cat data/gold/Intra0_unbalanced.txt data/gold/Intra1_unbalanced.txt > data/gold/Intra0_Intra1_unbalanced.txt
{ time dscript train --train data/gold/Intra0_Intra1_unbalanced.txt --test data/gold/Intra2_unbalanced.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/dscript_gold_unbalanced -o ./results_dscript/original/gold_unbalanced_train.txt -d 2; } 2> results_dscript/original/gold_unbalanced_time.txt
{ time dscript evaluate --test data/gold/Intra2_unbalanced.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/dscript_gold_unbalanced_final.sav  -o ./results_dscript/original/gold_unbalanced.txt -d 2; } 2> results_dscript/original/gold_unbalanced_time_test.txt



