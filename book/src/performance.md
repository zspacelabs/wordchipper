# Performance

wordchipper is designed for throughput. This chapter covers the knobs you can turn: BPE algorithm
selection, DFA-accelerated pre-tokenization, parallelism, and hash function choice.

## The two bottlenecks

Tokenization has two phases, and each can be the bottleneck:

1. **Pre-tokenization (spanning).** Splitting text into spans using a regex. This is I/O-bound: the
   regex engine scans every byte of the input.
2. **BPE encoding.** Merging byte pairs within each span. This is compute-bound: each span requires
   iterative pair lookups and merges.

For short texts, BPE dominates. For long texts with many spans, pre-tokenization matters more.
wordchipper optimizes both.

## DFA-accelerated spanning (logos)

The biggest single optimization is replacing the regex engine with a compile-time DFA. wordchipper
uses the [logos](https://logos.maciej.codes/) crate to compile pre-tokenization patterns into
deterministic finite automata at build time.

This is enabled by default. When you load a vocabulary whose regex pattern matches a known logos
lexer (cl100k, o200k, r50k), the DFA lexer is used automatically.

### Benchmarks

Spanning throughput on a single thread:

| Model       | Regex    | Logos DFA | Speedup |
|-------------|----------|-----------|---------|
| cl100k_base | ~25 MB/s | ~732 MB/s | **29x** |
| o200k_base  | ~15 MB/s | ~765 MB/s | **52x** |

### Disabling DFA acceleration

If you want to force regex-based spanning (e.g., for testing or debugging):

```rust,no_run
# use wordchipper::{TokenizerOptions, load_vocab, disk_cache::WordchipperDiskCache};
# let mut cache = WordchipperDiskCache::default();
# let (_, vocab) = load_vocab("openai:cl100k_base", &mut cache).unwrap();
let tok = TokenizerOptions::default()
    .with_accelerated_lexers(false)
    .build(vocab);
```

## BPE algorithm selection

wordchipper includes five span encoder implementations. Each trades off differently between
single-threaded speed, concurrent access, memory usage, and implementation complexity. See
[Advanced: Span Encoders](./advanced-span-encoders.md) for a deep dive into how each works.

### The algorithms

| Algorithm       | Best for                    | Notes                                                       |
|-----------------|-----------------------------|-------------------------------------------------------------|
| `MergeHeap`     | Concurrent / multi-threaded | Default for `ConcurrentDefault`. Heap-based merge tracking. |
| `PriorityMerge` | Single-threaded             | Default for `SingleThreadDefault`. Priority-queue merging.  |
| `BufferSweep`   | Testing / reference         | Simple and correct, not optimized.                          |
| `TailSweep`     | Memory-constrained          | Alternative scanning strategy.                              |
| `BpeBacktrack`  | Exact BPE semantics         | O(n) via Aho-Corasick automaton + backtracking.             |

### Selecting an algorithm

```rust,no_run
use wordchipper::{
    TokenizerOptions, TokenEncoderOptions,
    encoders::token_span_encoder::span_encoders::SpanEncoderSelector,
    load_vocab, disk_cache::WordchipperDiskCache,
};

let mut cache = WordchipperDiskCache::default();
let (_, vocab) = load_vocab("openai:cl100k_base", &mut cache).unwrap();

let mut opts = TokenizerOptions::default();
opts.encoder.set_span_encoder(SpanEncoderSelector::BpeBacktrack);
let tok = opts.build(vocab);
```

### Which algorithm should I use?

For most users, the defaults are correct:

- **Multi-threaded workloads** (web servers, batch processing): Use the default `ConcurrentDefault`,
  which selects `MergeHeap`.
- **Single-threaded workloads** (CLI tools, embedded): Use `SingleThreadDefault`, which selects
  `PriorityMerge`.
- **Exact BPE needed**: Use `BpeBacktrack`. It uses an Aho-Corasick automaton pre-built from the
  vocabulary for O(n) encoding that exactly matches the theoretical BPE algorithm. The other
  algorithms also produce correct output for standard vocabularies; `BpeBacktrack` is the
  theoretical gold standard.

## Parallelism with rayon

When the `parallel` feature is enabled (it is by default), batch operations parallelize across
threads:

```rust,no_run
# use wordchipper::{TokenizerOptions, TokenEncoder, load_vocab, disk_cache::WordchipperDiskCache};
# let mut cache = WordchipperDiskCache::default();
# let (_, vocab) = load_vocab("openai:cl100k_base", &mut cache).unwrap();
let tok = TokenizerOptions::default()
    .with_parallel(true)
    .build(vocab);

let texts: Vec<&str> = vec!["hello"; 1000];
let batch = tok.try_encode_batch(&texts).unwrap();
```

Setting `parallel(true)` affects both encoding and decoding. The `concurrent(true)` option
additionally selects a span encoder optimized for concurrent access from multiple threads.

### Controlling thread count

wordchipper uses rayon's global thread pool. Control it with the `RAYON_NUM_THREADS` environment
variable:

```bash
RAYON_NUM_THREADS=8 cargo run --release
```

Or configure it programmatically via `rayon::ThreadPoolBuilder`.

## Hash function selection

wordchipper uses hash maps extensively for vocabulary lookups. The `fast-hash` feature (enabled by
default) swaps in foldhash for faster hashing:

| Feature     | Hash function | Notes                                 |
|-------------|---------------|---------------------------------------|
| `fast-hash` | foldhash      | Good general-purpose, works in no_std |
| (none)      | default       | SipHash (std) or hashbrown default    |

Unlike previous versions, `fast-hash` works in `no_std` environments. See
[Feature Flags](./feature-flags.md) for details.

## End-to-end benchmarks

Encode + decode throughput on 90 MB shards, 48 threads:

| Model         | wordchipper | tiktoken-rs | HuggingFace tokenizers |
|---------------|-------------|-------------|------------------------|
| r50k_base     | 239 MiB/s   | 169 MiB/s   | 22 MiB/s               |
| p50k_base     | 251 MiB/s   | 163 MiB/s   | 22 MiB/s               |
| p50k_edit     | 242 MiB/s   | 170 MiB/s   | 21 MiB/s               |
| cl100k_base   | 214 MiB/s   | 125 MiB/s   | 22 MiB/s               |
| o200k_base    | 119 MiB/s   | 124 MiB/s   | 22 MiB/s               |
| o200k_harmony | 122 MiB/s   | 122 MiB/s   | 22 MiB/s               |

### Running benchmarks yourself

The `sample-timer` tool runs wordchipper vs. tiktoken-rs side-by-side:

```bash
RAYON_NUM_THREADS=48 cargo run --release -p sample-timer -- \
    --dataset-dir $DATASET_DIR --shards 0 --model openai::cl100k_base
```

For criterion-based microbenchmarks:

```bash
cargo bench -p wordchipper-bench
```

## Performance tips

1. **Use the default features.** DFA acceleration and rayon parallelism (`parallel`) are both on by default.
2. **Batch your inputs.** `try_encode_batch` is significantly faster than calling `try_encode` in a
   loop because it amortizes thread pool overhead.
3. **Reuse the tokenizer.** Building a `Tokenizer` pre-computes data structures. Build it once,
   share via `Arc`.
4. **Match the encoder to your workload.** Use `ConcurrentDefault` for multi-threaded,
   `SingleThreadDefault` for single-threaded.
5. **Profile spanning vs. encoding.** If spanning is the bottleneck, make sure DFA acceleration is
   active. If encoding is the bottleneck, experiment with `SpanEncoderSelector` variants.
