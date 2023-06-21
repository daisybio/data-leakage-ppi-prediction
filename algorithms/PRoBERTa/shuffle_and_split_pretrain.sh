#!/bin/bash
# Shuffle and split up family classification input
INPUT="$1"
OUTPUT="$2"
PREFIX="$3"

SHARD_SPLIT=0.10 
TRAIN_SHARDS=8
VALID_SHARDS=1
TEST_SHARDS=1

mkdir -p "$OUTPUT"

echo "Shuffling data"
grep "\S" "$INPUT" | sort | uniq | shuf > "$OUTPUT/$PREFIX.temp"
TOTAL_LINES=$(wc -l "$OUTPUT/$PREFIX.temp" | cut -f1 -d" ")
# Round up to prevent examples from being lost
SHARD_LINES=$(echo "scale=0; ($TOTAL_LINES/10) + 1" | bc -l)

echo "Splitting data"
split -l $SHARD_LINES -a 1 -d "$OUTPUT/$PREFIX.temp" "$OUTPUT/$PREFIX.temp"

TRAIN_START=0
TRAIN_END=$((TRAIN_START+TRAIN_SHARDS-1))
VALID_START=$((TRAIN_END+1))
VALID_END=$((VALID_START+VALID_SHARDS-1))
TEST_START=$((VALID_END+1))
TEST_END=$((TEST_START+TEST_SHARDS-1))

TEST_PCT=$(echo "$TEST_SHARDS*$SHARD_SPLIT" | bc -l)
echo "Generating testing split files with $TEST_PCT of the data"
for SHARD in $(seq $TEST_START $TEST_END); do
    cat "$OUTPUT/$PREFIX.temp$SHARD" | awk '{print $0"\n"}' >> "$OUTPUT/$PREFIX.split.test$TEST_PCT"
done

VALID_PCT=$(echo "$VALID_SHARDS*$SHARD_SPLIT" | bc -l)
echo "Generating validation split files with $VALID_PCT of the data"
for SHARD in $(seq $VALID_START $VALID_END); do
    cat "$OUTPUT/$PREFIX.temp$SHARD" | awk '{print $0"\n"}' >> "$OUTPUT/$PREFIX.split.valid$VALID_PCT"
done

TRAIN_PCT=$(echo "$TRAIN_SHARDS*$SHARD_SPLIT" | bc -l)
echo "Generating training split files with $TRAIN_PCT of the data"
for SHARD in $(seq $TRAIN_START $TRAIN_END); do
    cat "$OUTPUT/$PREFIX.temp$SHARD" | awk '{print $0"\n"}' >> "$OUTPUT/$PREFIX.split.train$TRAIN_PCT"
done

rm "$OUTPUT/$PREFIX.temp"*
