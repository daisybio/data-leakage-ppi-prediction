#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=train_dscript_ds
#SBATCH --output=train_dscript_ds_%A_%a.out
#SBATCH --error=train_dscript_ds_%A_%a.err
#SBATCH --mem=350G
#SBATCH --time=96:00:00
#SBATCH --partition=shared-gpu
#SBATCH --array=0-3
#SBATCH --exclude=gpu01.exbio.wzw.tum.de,jlab-gpu01.exbio.wzw.tum.de

declare -a combis
index=0
for FOLDER in "rewired" "partitions"
do
	if [ "$FOLDER" == "rewired" ]
	then
		combis[$index]="$FOLDER train test"
		index=$((index+1))
	else
		for TRAIN in "both" "0"
		do
			for TEST in "0" "1"
			do
				if [ "$TRAIN" = "0" ] && [ "$TEST" = "0" ]
				then
					continue
				fi
				combis[$index]="$FOLDER $TRAIN $TEST"
				index=$((index+1))
			done	
		done
	fi
done

parameters=(${combis[${SLURM_ARRAY_TASK_ID}]})
folder=${parameters[0]}

if [ $folder == "rewired" ]
then
	train_file=data/${folder}/dscript_${parameters[1]}.txt
	test_file=data/${folder}/dscript_${parameters[2]}.txt
	model_name="dscript_dscript_rewired"
	outfile="dscript_train.txt"
	timefile="dscript_time.txt"
else
	train_file=data/${folder}/dscript_partition_${parameters[1]}.txt
	test_file=data/${folder}/dscript_partition_${parameters[2]}.txt
	model_name="dscript_${parameters[1]}_${parameters[2]}_dscript_partitions"
	outfile="train_dscript_${parameters[1]}_${parameters[2]}.txt"
	timefile="train_dscript_${parameters[1]}_${parameters[2]}_time.txt"
fi
echo folder: ${folder}
echo trainfile: ${train_file}
echo testfile: ${test_file}
echo modelname: ${model_name}
echo outfile: ${outfile}
echo timefile: ${timefile}

{ time dscript train --train ${train_file} --test ${test_file} --embedding /nfs/scratch/jbernett/human_embedding.h5 --save-prefix ./models/${model_name} -o results_dscript/${folder}/${outfile}; } 2> results_dscript/${folder}/${timefile}
