#!/bin/bash

if [ $# -ne 12 ]; then
    echo "$0" 'PREFIX NUM_GPUS OUTPUT_DIR DATA_DIR ENCODER_EMBED_DIM' \
	'ENCODER_LAYERS TOTAL_UPDATES' \
	'WARMUP_UPDATES PEAK_LR MAX_SENTENCES' \
	'UPDATE_FREQ PATIENCE'
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
TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512

MAX_SENTENCES=${10}
UPDATE_FREQ=${11}
PATIENCE=${12}

BATCH_SIZE=$((MAX_SENTENCES*UPDATE_FREQ*NUM_GPUS))

PREFIX="$PREFIX.DIM_$ENCODER_EMBED_DIM.LAYERS_$ENCODER_LAYERS.UPDATES_$TOTAL_UPDATES"
PREFIX="$PREFIX.WARMUP_$WARMUP_UPDATES.LR_$PEAK_LR.BATCH_$BATCH_SIZE.PATIENCE_$PATIENCE"

CHECKPOINT_DIR="$OUTPUT_DIR/$PREFIX/checkpoints"
LOG_FILE="$OUTPUT_DIR/$PREFIX/pretrain.log"

mkdir -p "$CHECKPOINT_DIR"

fairseq-train --fp16 --fp16-no-flatten-grads "$DATA_DIR" \
        --task masked_lm --criterion masked_lm --bpe sentencepiece \
        --arch roberta_base --sample-break-mode complete_doc \
        --bpe sentencepiece \
        --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer lamb \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR \
        --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES \
        --encoder-embed-dim $ENCODER_EMBED_DIM --encoder-layers $ENCODER_LAYERS \
        --save-dir "$CHECKPOINT_DIR" \
        --save-interval 1 --save-interval-updates 100 --keep-interval-updates 5 \
        --distributed-world-size "$NUM_GPUS" --ddp-backend no_c10d \
        --patience $PATIENCE \
        --log-format simple --log-interval 1000 2>&1 | tee -a "$LOG_FILE"

