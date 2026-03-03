# Getting Started

## Installation

Add wordchipper to your `Cargo.toml`:

```toml
[dependencies]
wordchipper = "0.7"
```

The default features include `std`, `fast-hash` (foldhash hasher), and `parallel` (rayon). Add
`client` for vocabulary download. See [Feature Flags](./feature-flags.md) for all options and
[no_std & Embedded](./no-std.md) for minimal configurations.

## Encode and decode in five lines

```rust,no_run
use wordchipper::{
    Tokenizer, TokenizerOptions, TokenEncoder, TokenDecoder,
    load_vocab, disk_cache::WordchipperDiskCache,
};

fn main() -> wordchipper::WCResult<()> {
    // Load vocabulary (downloads and caches on first run)
    let mut cache = WordchipperDiskCache::default();
    let (_desc, vocab) = load_vocab("openai:cl100k_base", &mut cache)?;

    // Build a tokenizer
    let tok = TokenizerOptions::default().build(vocab);

    // Encode
    let tokens = tok.try_encode("hello world")?;
    assert_eq!(tokens, vec![15339, 1917]);

    // Decode
    let text = tok.try_decode_to_string(&tokens)?;
    assert_eq!(text, "hello world");

    Ok(())
}
```

The first call to `load_vocab` downloads the vocabulary file from OpenAI's CDN and caches it in
`~/.cache/wordchipper/`. Subsequent calls load from disk.

## Available models

```rust,no_run
use wordchipper::list_models;

fn main() {
    for model in list_models() {
        println!("{}", model);
    }
}
```

This prints all registered models with their namespace prefix:

```text
openai::r50k_base
openai::p50k_base
openai::p50k_edit
openai::cl100k_base
openai::o200k_base
openai::o200k_harmony
```

You can also use short names without the `openai::` prefix when loading:

```rust,no_run
# use wordchipper::{load_vocab, disk_cache::WordchipperDiskCache};
# let mut cache = WordchipperDiskCache::default();
let (_desc, vocab) = load_vocab("o200k_base", &mut cache).unwrap();
```

## Batch encoding

For multiple strings, batch encoding runs spans in parallel across threads (when the `parallel`
feature is enabled):

```rust,no_run
use wordchipper::{
    TokenEncoder, TokenizerOptions,
    load_vocab, disk_cache::WordchipperDiskCache,
};

fn main() -> wordchipper::WCResult<()> {
    let mut cache = WordchipperDiskCache::default();
    let (_desc, vocab) = load_vocab("openai:cl100k_base", &mut cache)?;
    let tok = TokenizerOptions::default()
        .with_parallel(true)
        .build(vocab);

    let texts = vec!["hello world", "the cat sat on the mat", "foo bar baz"];
    let batch = tok.try_encode_batch(&texts)?;

    for (text, tokens) in texts.iter().zip(batch.iter()) {
        println!("{:?} -> {:?}", text, tokens);
    }

    Ok(())
}
```

## Choosing a token type

wordchipper is generic over the token integer type. Most users should use `u32`, which is the
default. If your vocabulary fits in 16 bits (< 65,536 entries), you can use `u16` to save memory:

```rust,no_run
use std::sync::Arc;
use wordchipper::{UnifiedTokenVocab, TokenizerOptions, load_vocab, disk_cache::WordchipperDiskCache};

fn main() -> wordchipper::WCResult<()> {
    let mut cache = WordchipperDiskCache::default();
    let (_desc, vocab) = load_vocab("openai:r50k_base", &mut cache)?;

    // Convert to u16 (works for r50k_base with ~50k tokens)
    let vocab_u16: Arc<UnifiedTokenVocab<u16>> = Arc::new(vocab.as_ref().to_token_type());
    let tok = TokenizerOptions::default().build(vocab_u16);

    Ok(())
}
```

For `cl100k_base` (~100k tokens) and `o200k_base` (~200k tokens), use `u32`.

## What happens under the hood

When you call `try_encode("hello world")`, wordchipper runs a two-phase pipeline:

1. **Spanning.** The text is split into coarse segments called _spans_. For OpenAI models, this uses
   a regex pattern (or a faster DFA lexer) that handles whitespace, punctuation, letters, and
   digits. `"hello world"` becomes `["hello", " world"]`.

2. **BPE encoding.** Each span is encoded independently using Byte Pair Encoding. The encoder
   iteratively merges adjacent byte pairs according to the vocabulary's merge table, always picking
   the lowest-ranked (highest priority) pair first. The result is a sequence of token IDs.

Decoding reverses the process: each token ID maps to a byte sequence, and the byte sequences are
concatenated to produce the original text.

The next chapter explains this pipeline in detail, with visual examples of how BPE works.
