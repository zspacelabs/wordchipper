# wordchipper - HPC Rust BPE Tokenizer

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/wordchipper)

## Status

This is ready for alpha users, and is 2x the speed of `tiktoken-rs`
for many current models.

The productionization towards an LTR stable release can be
tracked in the
[Alpha Release Tracking Issue](https://github.com/zspacelabs/wordchipper/issues/2).

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

### Suite Crates

This is the main crate for the [wordchipper](https://github.com/zspacelabs/wordchipper) project.

The core additional user-facing crates are:

* [wchipper](https://crates.io/crates/wchipper) - a multi-tool tokenizer binary; notably:
    * `wchipper cat` - in-line encoder/decoder tool.
    * `wchipper train` - tokenizer training tool.
* [wordchipper-training](https://crates.io/crates/wordchipper-training) - an extension crate for training tokenizers.

## Encode/Decode Side-by-Side Benchmarks

| Model         | wordchipper  | tiktoken-rs  | tokenizers  |
|---------------|--------------|--------------|-------------|
| r50k_base     | 239.19 MiB/s | 169.30 MiB/s | 22.03 MiB/s |
| p50k_base     | 250.55 MiB/s | 163.07 MiB/s | 22.23 MiB/s |
| p50k_edit     | 241.69 MiB/s | 169.76 MiB/s | 21.27 MiB/s |
| cl100k_base   | 214.26 MiB/s | 125.43 MiB/s | 21.62 MiB/s |
| o200k_base    | 119.49 MiB/s | 123.75 MiB/s | 22.03 MiB/s |
| o200k_harmony | 121.80 MiB/s | 121.54 MiB/s | 22.08 MiB/s |

* *Help?* - I'm assuming some bug on my part for `tokenizers` + `rayon`.
* Methodology; 90MB shards of 1024 samples each, 48 threads.

```terminaloutput
$ for m in openai/{r50k_base,p50k_base,p50k_edit,cl100k_base,o200k_base,o200k_harmony}; \
  do RAYON_NUM_THREADS=48 cargo run --release -p sample-timer -- \
   --dataset-dir $DATASET_DIR --shards 0 --model $m; done
```

## Client Usage

### Pretrained Vocabularies

* [OpenAI OATokenizer](https://docs.rs/wordchipper/latest/wordchipper/pretrained/openai/enum.OATokenizer.html)

### Encoders and Decoders

* [Token Encoders](https://docs.rs/wordchipper/latest/wordchipper/encoders/index.html)
* [Token Decoders](https://docs.rs/wordchipper/latest/wordchipper/decoders/index.html)

## Loading Pretrained Models

Loading a pre-trained model requires reading the vocabulary,
as well as configuring the spanning (regex and special words)
configuration.

For a number of pretrained models, simplified constructors are
available to download, cache, and load the vocabulary.

See: [wordchipper::get_model](
https://docs.rs/wordchipper/latest/wordchipper/fn.get_model.html)

```rust,no_run
use std::sync::Arc;

use wordchipper::{
    get_model,
    TokenDecoder,
    TokenEncoder,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
};

fn example() -> wordchipper::errors::Result<(Arc<dyn TokenEncoder<u32>>, Arc<dyn TokenDecoder<u32>>)> {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<u32> = get_model("openai/o200k_harmony", &mut disk_cache)?;

    let encoder = vocab.to_default_encoder();
    let decoder = vocab.to_default_decoder();

    Ok((encoder, decoder))
}
```

## Acknowledgements

* Thank you to [@karpathy](https://github.com/karpathy)
  and [nanochat](https://github.com/karpathy/nanochat)
  for the work on `rustbpe`.
* Thank you to [tiktoken](https://github.com/openai/tiktoken) for their initial work in the rust
  tokenizer space.

