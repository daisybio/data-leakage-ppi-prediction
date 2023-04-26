#!/bin/bash

echo dscript
{ time dscript train --train data/gold/Intra1.txt --test data/gold/Intra0.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/dscript_gold -o ./results_dscript/original/gold_train.txt -d 1; } 2> results_dscript/original/gold_time.txt
{ time dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/dscript_gold_final.sav  -o ./results_dscript/original/gold.txt -d 1; } 2> results_dscript/original/gold_time_test.txt

echo topsyturvy
{ time dscript train --topsy-turvy --train data/gold/Intra1.txt --test data/gold/Intra0.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/tt_gold -o ./results_topsyturvy/original/gold_train.txt -d 1; } 2> results_topsyturvy/original/gold_time.txt
{ time dscript evaluate --test data/gold/Intra2.txt --embedding /nfs/scratch/jbernett/human_embedding.h5 --model ./models/tt_gold_final.sav  -o ./results_topsyturvy/original/gold.txt -d 1; } 2> results_topsyturvy/original/gold_time_test.txt