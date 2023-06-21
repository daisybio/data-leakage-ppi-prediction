#!/bin/bash

if [ "$#" -ne 16 ]; then
    echo "$0 [PREFIX] [NUM_GPUS] [OUTPUT_DIR] [DATA_DIR] [ENCODER_EMBED_DIM]" \
	"[ENCODER_LAYERS] [TOTAL_UPDATES]" \
	"[WARMUP_UPDATES] [PEAK_LR] [MAX_SENTENCES]" \
	"[UPDATE_FREQ] [NUM_CLASSES] [PATIENCE]" \
	"[PRETRAIN_CHECKPOINT] [RESUME] [USE_CLS]"
fi

PREFIX="$1"
NUM_GPUS=$2
OUTPUT_DIR="$3"
DATA_DIR="$4"

ENCODER_EMBED_DIM=$5
ENCODER_LAYERS=$6
TOTAL_UPDATES=$7
WARMUP_UPDATES=$8
PEAK_LR=$9
MAX_SENTENCES=${10}
UPDATE_FREQ=${11}

NUM_CLASSES=${12}
PATIENCE=${13}
ROBERTA_PATH="${14}"
RESUME=${15}

USE_CLS=${16}

TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512

BATCH_SIZE=$((MAX_SENTENCES*UPDATE_FREQ*NUM_GPUS))

PREFIX="$PREFIX.DIM_$ENCODER_EMBED_DIM.LAYERS_$ENCODER_LAYERS"
PREFIX="$PREFIX.UPDATES_$TOTAL_UPDATES.WARMUP_$WARMUP_UPDATES"
PREFIX="$PREFIX.LR_$PEAK_LR.BATCH_$BATCH_SIZE.PATIENCE_$PATIENCE"

CHECKPOINT_DIR="$OUTPUT_DIR/$PREFIX/checkpoints"
LOG_FILE="$OUTPUT_DIR/$PREFIX/$PREFIX.log"

mkdir -p "$CHECKPOINT_DIR"

if [ "$RESUME" = "no" ]; then
    fairseq-train --fp16 --fp16-no-flatten-grads $DATA_DIR \
        --max-positions $MAX_POSITIONS --max-sentences $MAX_SENTENCES \
        --arch roberta_base --task sentence_prediction \
        --truncate-sequence --use-cls-token $USE_CLS \
	--bpe sentencepiece \
        --classification-head-name protein_interaction_prediction \
        --restore-file "$ROBERTA_PATH" --reset-optimizer --reset-dataloader --reset-meters \
        --init-token 0 --separator-token 2 \
        --criterion sentence_prediction --num-classes $NUM_CLASSES \
        --optimizer lamb \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES \
        --encoder-embed-dim $ENCODER_EMBED_DIM --encoder-layers $ENCODER_LAYERS \
        --save-dir "$CHECKPOINT_DIR" --save-interval 1 --save-interval-updates 100 --keep-interval-updates 5 \
        --distributed-world-size $NUM_GPUS --ddp-backend=no_c10d \
        --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --patience $PATIENCE \
        --log-format simple --log-interval 1000 2>&1 | tee -a "$LOG_FILE"
else
    fairseq-train --fp16 --fp16-no-flatten-grads $DATA_DIR \
        --max-positions $MAX_POSITIONS --max-sentences $MAX_SENTENCES \
        --arch roberta_base --task sentence_prediction \
        --truncate-sequence --use-cls-token $USE_CLS \
        --bpe sentencepiece \
        --classification-head-name protein_interaction_prediction \
        --init-token 0 --separator-token 2 \
        --criterion sentence_prediction --num-classes $NUM_CLASSES \
        --optimizer lamb \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES \
        --encoder-embed-dim $ENCODER_EMBED_DIM --encoder-layers $ENCODER_LAYERS \
        --save-dir "$CHECKPOINT_DIR" --save-interval 1 --save-interval-updates 100 --keep-interval-updates 5 \
        --distributed-world-size $NUM_GPUS --ddp-backend=no_c10d \
        --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --patience $PATIENCE \
        --log-format simple --log-interval 1000 2>&1 | tee -a "$LOG_FILE"
fi


