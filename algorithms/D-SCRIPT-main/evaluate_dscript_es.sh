#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=eval_dscript
#SBATCH --output=evaluate_dscript_%A_%a.out
#SBATCH --error=evaluate_dscript_%A_%a.err
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --partition=shared-cpu
#SBATCH --array=0-31

echo evaluate d-script with manual early stopping. See best epochs in table
declare -a combis

declare -A bestepochs=( ["huang"]="05" ["guo"]="10" ["du"]="09" ["pan"]="09" ["richoux_regular"]="01" ["richoux_strict"]="01" ["dscript"]="07" )

index=0
for DATASET in huang guo du pan richoux_regular richoux_strict dscript
do
	if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi
        echo $EMBEDDING
	
	EPOCH=${bestepochs[$DATASET]}
	MODEL="./models/${DATASET}_dscript_original_epoch${EPOCH}.sav"
	TESTSET="data/original/${DATASET}_test.txt"
	OUTFILE="./results_dscript/original/${DATASET}_es.txt"
  combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
  index=$((index+1))
done

declare -A bestepochs=( ["huang"]="08" ["guo"]="09" ["du"]="09" ["pan"]="06" ["richoux_regular"]="01" ["richoux_strict"]="01" ["dscript"]="02" )
for DATASET in huang guo du pan richoux_regular richoux_strict dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_dscript_rewired_epoch${EPOCH}.sav"
        TESTSET="data/rewired/${DATASET}_test.txt"
        OUTFILE="./results_dscript/rewired/${DATASET}_es.txt"
        combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
        index=$((index+1))
done

declare -A bestepochs=( ["huang"]="08" ["guo"]="09" ["du"]="08" ["pan"]="06" ["richoux"]="03" ["dscript"]="02" )
for DATASET in huang guo du pan richoux dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_both_0_dscript_partitions_epoch${EPOCH}.sav"
        TESTSET="data/partitions/${DATASET}_partition_0.txt"
        OUTFILE="./results_dscript/partitions/${DATASET}_both_0_es"
        combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
        index=$((index+1))
done

declare -A bestepochs=( ["huang"]="10" ["guo"]="10" ["du"]="05" ["pan"]="08" ["richoux"]="02" ["dscript"]="01" )
for DATASET in huang guo du pan richoux dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_both_1_dscript_partitions_epoch${EPOCH}.sav"
        TESTSET="data/partitions/${DATASET}_partition_1.txt"
        OUTFILE="./results_dscript/partitions/${DATASET}_both_1_es"
        combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
        index=$((index+1))
done

declare -A bestepochs=( ["huang"]="06" ["guo"]="01" ["du"]="03" ["pan"]="10" ["richoux"]="04" ["dscript"]="01" )
for DATASET in huang guo du pan richoux dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_0_1_dscript_partitions_epoch${EPOCH}.sav"
        TESTSET="data/partitions/${DATASET}_partition_1.txt"
        OUTFILE="./results_dscript/partitions/${DATASET}_0_1_es"
        combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
        index=$((index+1))
done

parameters=(${combis[${SLURM_ARRAY_TASK_ID}]})
testset=${parameters[0]}
embedding=${parameters[1]}
model=${parameters[2]}
outfile=${parameters[3]}
echo testset $testset embedding $embedding model $model outfile $outfile

dscript evaluate --test $testset  --embedding $embedding --model $model -o $outfile