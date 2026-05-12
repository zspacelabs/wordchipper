# wordchipper-bench

Benchmarks comparing wordchipper's `SpanEncoder` variants against each other and against tiktoken-rs
and HuggingFace tokenizers.

## Benchmarks

| Bench               | What it measures                              |
|---------------------|-----------------------------------------------|
| `encoding_single`   | Single-string encoding (no parallelism)       |
| `encoding_parallel` | Batch encoding via rayon (`try_encode_batch`) |
| `decoding_single`   | Single-string decoding                        |
| `spanning`          | Text spanning (regex vs logos DFA)            |
| `qwen_encoding_single` | Single-string Qwen encode vs HF tokenizers |

### Encoder Variants

- **`buffer_sweep`** - O(n^2) reference implementation using a separate working buffer
- **`tail_sweep`** - O(n^2) linear-scan BPE merge using the output buffer tail as working memory
- **`merge_heap`** - O(n^2) with parallel pair-rank tracking
- **`priority_merge`** - O(n log n) binary min-heap with doubly-linked list (default)

These are selectable via `SpanEncoderSelector`.

## Running

```bash
# All benchmarks
cargo bench -p wordchipper-bench

# Individual benchmarks
cargo bench -p wordchipper-bench --bench encoding_single
cargo bench -p wordchipper-bench --bench encoding_parallel
cargo bench -p wordchipper-bench --bench decoding_single
cargo bench -p wordchipper-bench --bench spanning
cargo bench -p wordchipper-bench --bench qwen_encoding_single

# Filter by name
cargo bench -p wordchipper-bench --bench encoding_single -- diverse
cargo bench -p wordchipper-bench --bench encoding_parallel -- priority_merge
```

### Parallel bench data

`encoding_parallel` uses fineweb-edu parquet shards. The dataset is auto-downloaded on first run.

## JSON output

The `bench-json` binary runs benchmarks and converts divan's human-readable output to JSON.

```bash
# Specific bench target
cargo run --release -p wordchipper-bench --bin bench-json -- --bench spanning

# With divan args
cargo run --release -p wordchipper-bench --bin bench-json -- --bench spanning -- --sample-count 10

# All targets, save to file
cargo run --release -p wordchipper-bench --bin bench-json -- -o results.json

# Echo human output to stderr while parsing
cargo run --release -p wordchipper-bench --bin bench-json -- --bench spanning --tee
```

## Results

See [bench-results/latest.md](bench-results/latest.md) for current numbers.

Previous runs are archived in [bench-results/archive/](bench-results/archive/).
