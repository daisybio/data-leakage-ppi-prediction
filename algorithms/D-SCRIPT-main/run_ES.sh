#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --job-name=train_dscript_es
#SBATCH --output=train_dscript_es_%A_%a.out
#SBATCH --error=train_dscript_es_%A_%a.err
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=shared-gpu
#SBATCH --array=0-19
#SBATCH --exclude=gpu01.exbio.wzw.tum.de

declare -a combis
index=0
for FOLDER in "original" "rewired" "partitions"; do
  for DATASET in "huang" "guo" "du" "pan"; do
    if [[ "$DATASET" == "guo" ||  "$DATASET" == "du" ]] ; then
                EMBEDDING='/nfs/scratch/jbernett/yeast_embedding.h5'
        else
                EMBEDDING='/nfs/scratch/jbernett/human_embedding.h5'
        fi
    if [ "$FOLDER" == "rewired" ] || [ "$FOLDER" == "original" ]; then
      combis[$index]="$FOLDER $DATASET train val $EMBEDDING"
      index=$((index + 1))
    else
      for TRAIN in "both" "0"; do
        for TEST in "0" "1"; do
          if [ "$TRAIN" = "0" ] && [ "$TEST" = "0" ]; then
            continue
          fi
          combis[$index]="$FOLDER $DATASET $TRAIN $TEST $EMBEDDING"
          index=$((index + 1))
        done
      done
    fi
  done
done

echo $index

parameters=(${combis[${SLURM_ARRAY_TASK_ID}]})
folder=${parameters[0]}
dataset=${parameters[1]}
embedding=${parameters[4]}

if [ $folder == "rewired" ] || [ $folder == "original" ]; then
  train_file=data/${folder}/${dataset}_${parameters[2]}_es.txt
  val_file=data/${folder}/${dataset}_${parameters[3]}_es.txt
  model_name="es_${folder}_${dataset}"
  outfile="es_${folder}_${dataset}_train.txt"
else
  train_file=data/${folder}/${dataset}_partition_${parameters[2]}_train_es.txt
  val_file=data/${folder}/dscript_partition_${parameters[3]}_val_es.txt
  model_name="es_${folder}_${dataset}_${parameters[2]}_${parameters[3]}"
  outfile="es_${folder}_${dataset}_${parameters[2]}_${parameters[3]}_train.txt"
fi
echo folder: ${folder}
echo trainfile: ${train_file}
echo valfile: ${val_file}
echo embedding: ${embedding}
echo modelname: ${model_name}
echo outfile: ${outfile}

dscript train --train ${train_file} --test ${val_file} --embedding ${embedding} --save-prefix ./models/${model_name} -o results_dscript/${folder}/${outfile}