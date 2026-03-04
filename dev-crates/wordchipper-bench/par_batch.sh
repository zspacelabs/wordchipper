#!/bin/bash

# Run the benchmark for the par_batch feature

THREADS=(1 2 4 8 16 24 32 48 64)

WORKSPACE_ROOT=$(dirname $(cargo locate-project --workspace --message-format plain))

DATA_DIR="$WORKSPACE_ROOT/dev-crates/wordchipper-bench/bench-data/parallel"

mkdir -p "$DATA_DIR"
rm -rf "$DATA_DIR"/encoding_parallel.*.json

for THREAD in "${THREADS[@]}"; do
    echo "Running benchmark with par_batch and $THREAD threads..."
    RAYON_NUM_THREADS=$THREAD cargo run --bin bench-json -- \
      --bench encoding_parallel \
      --output="$DATA_DIR/encoding_parallel.$THREAD.json" \
      --tee
done
