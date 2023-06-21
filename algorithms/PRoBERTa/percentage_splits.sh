#!/bin/bash
# Generates copies of input file, each with different % of the input file

INPUT="$1"
# Output folder
OUTPUT="$2"
# File name prefix
PREFIX="$3"

# Percentage increments of data in the output file
SHARD_SPLIT=0.10
SHARDS=$(echo "scale=0; 1/$SHARD_SPLIT" | bc -l)

mkdir -p "$OUTPUT"

TOTAL_LINES=$(wc -l "$INPUT" | cut -f1 -d" ")
# Round up to prevent examples from being lost
SHARD_LINES=$(echo "scale=0; ($TOTAL_LINES/$SHARDS) + 1" | bc -l)

echo "Splitting data"
split -l $SHARD_LINES -a 1 -d "$INPUT" "$OUTPUT/$PREFIX.temp"

SHARD_END=$((SHARDS-1))
for SHARD in $(seq 0 $SHARD_END); do
    PCT=$(echo "($SHARD+1)*$SHARD_SPLIT" | bc -l)
    echo "Generating split files with $PCT of the data"
    for INCLUDE in $(seq 0 $SHARD); do
    	cat "$OUTPUT/$PREFIX.temp$INCLUDE" >> "$OUTPUT/$PREFIX.minisplit$PCT"
    done
done

rm "$OUTPUT/$PREFIX.temp"*
