#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=eval_dscript
#SBATCH --output=evaluate_dscript_original.out
#SBATCH --error=evaluate_dscript_original.err
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --partition=compms-gpu-a40

echo evaluate d-script with manual early stopping. See best epochs in table
echo ORIGINAL

declare -A bestepochs=( ["huang"]="05" ["guo"]="10" ["du"]="09" ["pan"]="09" ["richoux_regular"]="01" ["richoux_strict"]="01" ["dscript"]="04" ["gold"]="02" )

for DATASET in huang guo du pan richoux_regular richoux_strict dscript gold
do
	echo $DATASET
	if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi
        echo $EMBEDDING
	
	EPOCH=${bestepochs[$DATASET]}
	if [ "$DATASET" == "gold" ]; then
                MODEL="./models/dscript_${DATASET}_epoch${EPOCH}.sav"
        else
                MODEL="./models/${DATASET}_dscript_original_epoch${EPOCH}.sav"
        fi
	echo evaluating on model $MODEL

	dscript evaluate --test data/original/${DATASET}_test.txt --embedding $EMBEDDING --model $MODEL -o ./results_dscript/original/${DATASET}_es.txt
done

#echo REWIRED
#declare -A bestepochs=( ["huang"]="10" ["guo"]="10" ["du"]="01" ["pan"]="02" ["richoux_regular"]="02" ["richoux_strict"]="01" ["dscript"]="06" )

#for DATASET in huang guo du pan richoux_regular richoux_strict dscript
#do
#        echo $DATASET
#        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
#                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
#        else
#                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
#        fi
#        echo $EMBEDDING
#
#        EPOCH=${bestepochs[$DATASET]}
#        MODEL="./models/${DATASET}_dscript_rewired_epoch${EPOCH}.sav"
#        echo evaluating on model $MODEL
#
#        dscript evaluate --test data/rewired/${DATASET}_test.txt --embedding $EMBEDDING --model $MODEL -o ./results_dscript/rewired/${DATASET}_es.txt -d 1
#done

#echo PARTITION both_0
#declare -A bestepochs=( ["huang"]="01" ["guo"]="09" ["du"]="07" ["pan"]="05" ["richoux"]="04" ["dscript"]="10" )
#for DATASET in huang guo du pan richoux dscript 
#do
#        echo $DATASET
#        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
#                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
#        else
#                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
#        fi
#        echo $EMBEDDING

#        EPOCH=${bestepochs[$DATASET]}
#        MODEL="./models/${DATASET}_both_0_dscript_partitions_epoch${EPOCH}.sav"
#        echo evaluating on model $MODEL
#
#        dscript evaluate --test data/partitions/${DATASET}_partition_0.txt --embedding $EMBEDDING --model $MODEL -o ./results_dscript/partitions/${DATASET}_both_0_es.txt -d 1
#done

#echo PARTITION both_1
#declare -A bestepochs=( ["huang"]="01" ["guo"]="03" ["du"]="07" ["pan"]="09" ["richoux"]="04" ["dscript"]="05" )
#for DATASET in huang guo du pan richoux dscript                       
#do
#        echo $DATASET
#        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
#                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
#        else
#                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
#        fi
#        echo $EMBEDDING
#
#        EPOCH=${bestepochs[$DATASET]}
#        MODEL="./models/${DATASET}_both_1_dscript_partitions_epoch${EPOCH}.sav"
#        echo evaluating on model $MODEL
#
#        dscript evaluate --test data/partitions/${DATASET}_partition_1.txt --embedding $EMBEDDING --model $MODEL -o ./results_dscript/partitions/${DATASET}_both_1_es.txt -d 1
#done
#
#echo PARTITION 0_1
#declare -A bestepochs=( ["huang"]="08" ["guo"]="01" ["du"]="07" ["pan"]="09" ["richoux"]="01" ["dscript"]="01" )
#for DATASET in huang guo du pan richoux dscript
#do
#        echo $DATASET
#        if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
#                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
#        else
#                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
#        fi
#        echo $EMBEDDING
#
#        EPOCH=${bestepochs[$DATASET]}
#        MODEL="./models/${DATASET}_0_1_dscript_partitions_epoch${EPOCH}.sav"
#        echo evaluating on model $MODEL
#
#        dscript evaluate --test data/partitions/${DATASET}_partition_1.txt --embedding $EMBEDDING --model $MODEL -o ./results_dscript/partitions/${DATASET}_0_1_es.txt -d 1
#done
