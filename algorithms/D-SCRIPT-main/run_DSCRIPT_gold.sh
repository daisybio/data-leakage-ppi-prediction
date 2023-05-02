#!/bin/bash

echo dscript
cat data/gold/Intra0.txt data/gold/Intra1.txt > Intra0_Intra1.txt
{ time dscript train --train data/gold/Intra0_Intra1.txt --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/dscript_gold -o ./results_dscript/original/gold_train.txt -d 2; } 2> results_dscript/original/gold_time.txt
{ time dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/dscript_gold_final.sav  -o ./results_dscript/original/gold.txt -d 2; } 2> results_dscript/original/gold_time_test.txt



