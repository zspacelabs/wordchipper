#!/bin/bash

THREADS=(1 2 4 8 12)

WORKSPACE_ROOT=$(dirname $(cargo locate-project --workspace --message-format plain))

DATA_DIR="$WORKSPACE_ROOT/target/bench-data/python_parallel"

mkdir -p "$DATA_DIR"
rm -rf "$DATA_DIR"/encoding_parallel.*.json

for THREAD in "${THREADS[@]}"; do
    echo "Running benchmark with par_batch and $THREAD threads..."
    RAYON_NUM_THREADS=$THREAD pytest benchmarks/ \
      -k "TestBatchEncode" \
      --benchmark-json="$DATA_DIR/encoding_parallel.$THREAD.json"
done
