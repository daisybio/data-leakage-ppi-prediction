#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=eval_tt
#SBATCH --output=evaluate_tt_%A_%a.out
#SBATCH --error=evaluate_tt_%A_%a.err
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --partition=shared-cpu
#SBATCH --array=0-27

echo evaluate topsyturvy with manual early stopping. See best epochs in table
declare -a combis

declare -A bestepochs=( ["huang"]="07" ["guo"]="07" ["du"]="03" ["pan"]="02" ["richoux_regular"]="01" ["richoux_strict"]="05" ["dscript"]="04" )

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
	MODEL="./models/${DATASET}_tt_original_epoch${EPOCH}.sav"
  TESTSET="data/original/${DATASET}_test.txt"
  OUTFILE="./results_topsyturvy/original/${DATASET}_es.txt"
	combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
  index=$((index+1))
done

declare -A bestepochs=( ["huang"]="10" ["guo"]="10" ["du"]="01" ["pan"]="02" ["richoux_regular"]="02" ["richoux_strict"]="01" ["dscript"]="06" )
for DATASET in huang guo du pan richoux_regular richoux_strict #dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_tt_rewired_epoch${EPOCH}.sav"
        TESTSET="data/rewired/${DATASET}_test.txt"
        OUTFILE="./results_topsyturvy/rewired/${DATASET}_es.txt"
        combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
        index=$((index+1))
done

declare -A bestepochs=( ["huang"]="01" ["guo"]="09" ["du"]="07" ["pan"]="05" ["richoux"]="04" ["dscript"]="10" )
for DATASET in huang guo du pan richoux #dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi
        echo $EMBEDDING

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_both_0_tt_partitions_epoch${EPOCH}.sav"
        TESTSET="data/partitions/${DATASET}_partition_0.txt"
        OUTFILE="./results_topsyturvy/partitions/${DATASET}_both_0_es"
        combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
        index=$((index+1))
done

declare -A bestepochs=( ["huang"]="01" ["guo"]="03" ["du"]="07" ["pan"]="09" ["richoux"]="04" ["dscript"]="05" )
for DATASET in huang guo du pan richoux #dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_both_1_tt_partitions_epoch${EPOCH}.sav"
        TESTSET="data/partitions/${DATASET}_partition_1.txt"
        OUTFILE="./results_topsyturvy/partitions/${DATASET}_both_1_es"
        combis[$index]="$TESTSET $EMBEDDING $MODEL $OUTFILE"
        index=$((index+1))
done

declare -A bestepochs=( ["huang"]="08" ["guo"]="01" ["du"]="07" ["pan"]="09" ["richoux"]="01" ["dscript"]="01" )
for DATASET in huang guo du pan richoux #dscript
do
        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi

        EPOCH=${bestepochs[$DATASET]}
        MODEL="./models/${DATASET}_0_1_tt_partitions_epoch${EPOCH}.sav"
        TESTSET="data/partitions/${DATASET}_partition_1.txt"
        OUTFILE="./results_topsyturvy/partitions/${DATASET}_0_1_es"
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