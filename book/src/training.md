# Training Your Own Tokenizer

This chapter walks through training a BPE tokenizer from scratch. If you haven't read
[How Tokenizers Work](./how-tokenizers-work.md) yet, start there. This chapter assumes you know what
BPE merges are and why pre-tokenization matters.

## Why train a custom tokenizer?

Pretrained tokenizers like `cl100k_base` and `o200k_base` are trained on broad internet text. They
work well for general English, but they have trade-offs:

- **Domain-specific text.** If your corpus is primarily code, legal documents, or scientific
  notation, a general-purpose tokenizer wastes vocabulary slots on tokens your data rarely uses. A
  tokenizer trained on your data will encode it in fewer tokens.
- **Non-English languages.** General tokenizers are biased toward English. The same sentence in
  Hindi or Korean may produce 3-5x more tokens. Training on your target language fixes this.
- **Smaller vocabularies.** Pretrained models use 50k-200k tokens. If you're building a small model
  or running on constrained hardware, a 4k or 8k vocabulary trained on your data can be more
  efficient than a 100k vocabulary trained on someone else's.
- **Research.** You want to study how vocabulary size, training data, or regex pattern affect
  downstream model performance.

The short version: train your own when the pretrained vocabulary doesn't fit your data well.

## Quick start: the CLI

The fastest way to train a tokenizer is with `wordchipper-cli`. No Rust code needed.

### Training on text files

Create a text file with your training data (one document per line, or free-form text):

```bash
cargo run --release -p wordchipper-cli -- \
    train \
    --vocab-size=8000 \
    --output=my_tokenizer.tiktoken \
    corpus.txt
```

This reads `corpus.txt`, splits it using the default regex pattern (`r50k_base`'s pattern), runs BPE
training to learn 8000 - 256 = 7744 merges, and writes the vocabulary to `my_tokenizer.tiktoken` in
base64-encoded tiktoken format.

### Training on Parquet files

For larger datasets, Parquet is more efficient. The file must have a column named `"text"`:

```bash
cargo run --release -p wordchipper-cli -- \
    train \
    --input-format=parquet \
    --vocab-size=50281 \
    --output=nanochat.tiktoken \
    ~/Data/nanochat/dataset/*.parquet
```

Parquet files are read in streaming record batches, so memory usage stays bounded even for
multi-gigabyte datasets.

### What the output looks like

Training logs progress at 1% intervals:

```text
INFO Reading shards:
INFO 0: shard_00000.parquet
INFO 1: shard_00001.parquet
INFO Training Tokenizer...
INFO Starting BPE training: 50025 merges to compute
INFO Building pair index...
INFO Building heap with 16044 unique pairs
INFO Starting merge loop
INFO Progress: 1% (501/50025 merges) - Last merge: (69, 120) -> 756 (frequency: 166814)
INFO Progress: 2% (1001/50025 merges) - Last merge: (66, 77) -> 1256 (frequency: 17847)
...
INFO Progress: 100% (50025/50025 merges) - Last merge: (302, 3405) -> 50280 (frequency: 18)
INFO Finished training: 50025 merges completed
INFO Vocabulary Size: 50280
```

The early merges have high frequencies (166k+). These are the most common byte pairs in your corpus,
things like `(space, t)` or `(e, space)`. By the end, merges are rare (frequency 18), scraping the
bottom of what's statistically useful.

## The training pipeline

Training has three phases. Understanding them helps you make better choices about regex pattern,
vocabulary size, and data preparation.

### Phase 1: counting spans

Before any merges happen, the trainer splits every input document using the regex pattern, then
counts how many times each unique span appears. This is the same pre-tokenization step that happens
during encoding, but here it builds a frequency table.

```text
Input: "the cat sat on the mat"

Regex splits: ["the", " cat", " sat", " on", " the", " mat"]

Span counts:
  "the"  -> 1
  " the" -> 1
  " cat" -> 1
  " sat" -> 1
  " on"  -> 1
  " mat" -> 1
```

With more data, common spans like `" the"` accumulate large counts. The trainer calls
`update_from_samples()` incrementally, so you can feed it data in batches without loading everything
into memory.

Internally, this is handled by `TextSpanCounter`, which stores span strings as `CompactString` keys
(stack-allocated for short strings, heap-allocated when longer) with `u32` counts.

### Phase 2: building the pair index

Once all data is counted, the trainer converts each unique span into a `TokenSpanBuf`: a buffer of
byte-level token IDs. Every span starts as its raw UTF-8 bytes, mapped through a `ByteMapVocab` (the
initial 256-entry vocabulary where token 0 = byte 0, token 1 = byte 1, etc.).

Then it scans every span to build two structures:

- **Pair counts**: for every adjacent pair `(a, b)`, the total count across all spans (weighted by
  span frequency). If `" the"` appears 50,000 times and contains the pair `(116, 104)` (`t`, `h`),
  that pair gets +50,000.
- **Pair index**: for every pair, which span indices contain it. This avoids scanning all spans when
  a merge only affects a few.

These go into a `PairSpanIndex`, then seed an octonary heap (an 8-ary heap, faster than a binary
heap for this workload) sorted by count descending.

### Phase 3: the merge loop

This is the core of BPE training. It runs until the vocabulary reaches the target size:

```text
while merges_done < num_merges:
    1. Pop the highest-count pair from the heap.
    2. If the count is stale (changed since it was pushed), refresh and re-push.
    3. Allocate a new token ID for this merge.
    4. For every span containing this pair:
       a. Replace all occurrences of (a, b) with the new token.
       b. Update pair counts: decrement removed pairs, increment new pairs.
       c. Track which spans now contain newly created pairs.
    5. Push new pairs onto the heap.
```

The "lazy refresh" in step 2 is important. When pair `(a, b)` is merged, it changes the neighbors of
`a` and `b` in every span where they appeared. This can reduce the count of other pairs in the heap
that haven't been popped yet. Rather than updating every affected heap entry (expensive), the
trainer just checks the current count when a pair is popped. If it's stale, it's re-pushed with the
correct count and the loop continues. This is a standard trick in BPE implementations.

#### A concrete merge example

Given the span `"hello"` (tokens: `[104, 101, 108, 108, 111]`) and the merge `(108, 108) -> 256`
(merging `l`, `l`):

```text
Before: [104, 101, 108, 108, 111]
         h    e    l    l    o

Pairs removed: (101, 108), (108, 108), (108, 111)
Pairs added:   (101, 256), (256, 111)

After:  [104, 101, 256, 111]
         h    e    ll   o
```

The `TokenSpanBuf::merge_pair_cb` method handles this in a single pass through the span, reporting
pair deltas through a callback so the trainer can update global counts incrementally.

## Training via the Rust API

For more control, use `wordchipper-training` directly. Add it to your `Cargo.toml`:

```toml
[dependencies]
wordchipper = "0.8"
wordchipper-training = "0.8"
```

### Minimal example

```rust,no_run
use std::sync::Arc;

use wordchipper::{
    TokenEncoder, TokenDecoder,
    Tokenizer, TokenizerOptions, UnifiedTokenVocab,
    pretrained::openai::OA_CL100K_BASE_PATTERN,
    vocab::ByteMapVocab,
};
use wordchipper_training::BPETRainerOptions;

fn main() {
    // 1. Configure the trainer
    let options = BPETRainerOptions::new(
        OA_CL100K_BASE_PATTERN,  // regex pattern for pre-tokenization
        1000,                     // target vocabulary size
    );
    let mut trainer = options.init();

    // 2. Feed it training data
    let samples = vec![
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
    ];
    trainer.update_from_samples(samples.iter());

    // 3. Train
    let byte_vocab: ByteMapVocab<u32> = Default::default();
    let vocab: Arc<UnifiedTokenVocab<u32>> = trainer
        .train(byte_vocab)
        .expect("training failed")
        .into();

    // 4. Use the tokenizer
    let tokenizer = TokenizerOptions::default().build(vocab);

    let tokens = tokenizer.try_encode("the cat").unwrap();
    let decoded = tokenizer.try_decode_to_string(&tokens).unwrap().unwrap();
    assert_eq!(decoded, "the cat");

    println!("'the cat' -> {:?}", tokens);
}
```

### Feeding data in batches

For large datasets, call `update_from_samples` multiple times. The trainer accumulates span counts
across calls:

```rust,no_run
# use wordchipper::pretrained::openai::OA_CL100K_BASE_PATTERN;
# use wordchipper_training::BPETRainerOptions;
# let mut trainer = BPETRainerOptions::new(OA_CL100K_BASE_PATTERN, 1000).init();
// First batch
let batch1 = vec!["hello world", "hello there"];
trainer.update_from_samples(batch1.iter());

// Second batch
let batch2 = vec!["world peace", "hello again"];
trainer.update_from_samples(batch2.iter());

// Counts from both batches are combined before training
```

This is how the CLI processes Parquet files: each record batch is fed as a separate call to
`update_from_samples`.

### Saving the vocabulary

The trained vocabulary can be saved in tiktoken's base64 format:

```rust,no_run
use std::sync::Arc;

use wordchipper::{
    UnifiedTokenVocab, VocabIndex,
    pretrained::openai::OA_CL100K_BASE_PATTERN,
    vocab::{ByteMapVocab, io::save_base64_span_map_path},
};
use wordchipper_training::BPETRainerOptions;

fn main() {
    let mut trainer = BPETRainerOptions::new(OA_CL100K_BASE_PATTERN, 1000).init();
    trainer.update_from_samples(vec!["hello world"].iter());

    let vocab: Arc<UnifiedTokenVocab<u32>> = trainer
        .train(ByteMapVocab::default())
        .expect("training failed")
        .into();

    // Save as .tiktoken file
    save_base64_span_map_path(
        &vocab.span_vocab().span_map(),
        "my_vocab.tiktoken",
    ).expect("failed to save");
}
```

This produces a file compatible with OpenAI's tiktoken format: one line per token, each line is
`base64(byte_sequence) token_id`.

## Choosing a regex pattern

The regex pattern controls how text is split before BPE runs. This is one of the most consequential
decisions in tokenizer design. The pattern determines what kinds of tokens can exist.

### Available patterns

wordchipper provides the same patterns used by OpenAI's models as constants:

| Constant                 | Used by        | Key behavior                             |
|--------------------------|----------------|------------------------------------------|
| `OA_R50K_BASE_PATTERN`   | GPT-2          | Basic word/number/punctuation splitting  |
| `OA_CL100K_BASE_PATTERN` | GPT-3.5, GPT-4 | Adds case-insensitive contractions       |
| `OA_O200K_BASE_PATTERN`  | GPT-4o         | Unicode-aware, broader letter categories |

All patterns live in `wordchipper::pretrained::openai`.

### What the pattern affects

Consider the difference between splitting on `\w+` versus the cl100k pattern:

```text
Input: "don't stop"

\w+ splits:          ["don", "t", "stop"]
cl100k pattern:      ["don", "'t", " stop"]
```

The cl100k pattern keeps the contraction `'t` together and attaches the leading space to `stop`.
These design choices propagate through the entire vocabulary: a tokenizer trained with `\w+` will
never learn a token for `'t` or ` stop` because the regex never produces those spans.

### Custom patterns

You can use any valid regex:

```rust,no_run
# use wordchipper_training::BPETRainerOptions;
// Split only on whitespace boundaries
let options = BPETRainerOptions::new(r"\S+|\s+", 4000);

// Split on individual characters (character-level BPE)
let options = BPETRainerOptions::new(r".", 1000);

// Domain-specific: keep email addresses and URLs intact
let options = BPETRainerOptions::new(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|https?://\S+|\w+|\s+|.",
    8000,
);
```

As a starting point, use `OA_CL100K_BASE_PATTERN`. It's well-tested and handles contractions,
whitespace, and Unicode reasonably. Change it only when you have a specific reason.

## Choosing a vocabulary size

The vocabulary size is the total number of tokens, including the 256 byte tokens. So
`--vocab-size=1000` means 256 byte tokens + 744 learned merges.

### Trade-offs

| Vocab size | Merges  | Effect                                                          |
|------------|---------|-----------------------------------------------------------------|
| 256        | 0       | Pure byte-level. Every character is 1-4 tokens. Long sequences. |
| 1,000      | 744     | Common bigrams and short words are single tokens.               |
| 8,000      | 7,744   | Most common words are single tokens. Good for small models.     |
| 50,000     | 49,744  | Covers most English words. Standard for GPT-2 era models.       |
| 100,000    | 99,744  | Covers English + common multilingual tokens. GPT-4 range.       |
| 200,000    | 199,744 | Broad multilingual coverage. GPT-4o range.                      |

Larger vocabularies mean shorter token sequences (faster inference, more text per context window)
but larger embedding tables (more parameters, more memory). For a small model on a specific domain,
4k-16k is often a good range. For a general-purpose model, 50k-200k.

### How merges tail off

Early merges capture high-frequency patterns like common letter pairs and short words. Late merges
capture rare combinations. You can see this in the training output: the first merge might have
frequency 166,000 while the last merge has frequency 18.

If your late merges have very low frequencies (single digits), you've likely exhausted the useful
patterns in your data. A smaller vocabulary size would waste less of the embedding table on tokens
that rarely appear.

## Performance expectations

Training speed depends on two factors:

1. **Corpus size.** Counting spans is O(n) in the input size. Expect roughly 1 second per 10 MB of
   input text for the counting phase.
2. **Vocabulary size.** The merge loop runs one iteration per merge (vocab_size - 256 iterations).
   Each iteration touches only the spans affected by that merge, but the total cost grows with both
   vocabulary size and corpus diversity.

For reference, training a 50k vocabulary on the nanochat dataset (8 Parquet shards, roughly 80 MB of
text) takes around 80 CPU-minutes on a single thread. The counting phase is fast; the merge loop
dominates.

The trainer is single-threaded. If your data loading is slow (e.g., reading from disk or network),
run it on a separate thread so the trainer isn't blocked on I/O. The CLI handles this for Parquet by
streaming record batches.

## Putting it all together

Here's a complete workflow: train a tokenizer on your data, save it, and use it for encoding.

```rust,no_run
use std::sync::Arc;

use wordchipper::{
    TokenEncoder, TokenDecoder,
    Tokenizer, TokenizerOptions, UnifiedTokenVocab, VocabIndex,
    pretrained::openai::OA_CL100K_BASE_PATTERN,
    vocab::{ByteMapVocab, io::save_base64_span_map_path},
};
use wordchipper_training::BPETRainerOptions;

fn main() -> wordchipper::WCResult<()> {
    // --- Train ---
    let mut trainer = BPETRainerOptions::new(OA_CL100K_BASE_PATTERN, 2000).init();

    // In practice, feed much more data
    let corpus = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ];
    trainer.update_from_samples(corpus.iter());

    let vocab: Arc<UnifiedTokenVocab<u32>> = trainer
        .train(ByteMapVocab::default())
        .expect("training failed")
        .into();

    // --- Save ---
    save_base64_span_map_path(
        &vocab.span_vocab().span_map(),
        "/tmp/my_vocab.tiktoken",
    )?;

    // --- Use ---
    let tokenizer = TokenizerOptions::default().build(vocab);
    let tokens = tokenizer.try_encode("The quick brown fox")?;
    let decoded = tokenizer.try_decode_to_string(&tokens)?.unwrap();
    assert_eq!(decoded, "The quick brown fox");

    println!("Encoded {} tokens: {:?}", tokens.len(), tokens);

    Ok(())
}
```

## What's next

Once you have a trained tokenizer, you might want to:

- Benchmark it against pretrained tokenizers. See [Performance](./performance.md) for how
  wordchipper measures throughput.
- Build a custom DFA lexer for your regex pattern. See
  [Building Custom Logos Lexers](./custom-logos-lexers.md) for 30-50x faster pre-tokenization.
- Understand the BPE encoding algorithms available. See
  [Advanced: Span Encoders](./advanced-span-encoders.md) for the different encoder strategies.
