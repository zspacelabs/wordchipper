# wordchipper-training

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper-training)](https://crates.io/crates/wordchipper-training)
[![Documentation](https://img.shields.io/docsrs/wordchipper-training)](https://docs.rs/wordchipper/latest/wordchipper-training/)
[![Test Status](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml)

[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/wordchipper)

Part of the [wordchipper tokenizer suite](https://crates.io/crates/wordchipper).

See: [wordchipper](https://crates.io/crates/wordchipper)

## Status: Working Prototype

This crate is functional but has not received the same level of scrutiny as the rest of the project.

## Overview

This crate provides utilities for training WordChipper tokenizers and vocabularies.

### Suite Crates

This is the training implementation crate for the
[wordchipper tokenizer suite](https://github.com/zspacelabs/wordchipper).

The core additional user-facing crates are:

* [wordchipper](https://crates.io/crates/wordchipper) - the core tokenizer library.
* [wordchipper-cli](https://crates.io/crates/wordchipper-cli) - a multi-tool tokenizer binary;
  notably:
    * `wordchipper-cli cat` - in-line encoder/decoder tool.
    * `wordchipper-cli train` - tokenizer training tool.

## Training Tool

A pre-built training tool is shipped as part of the [
`wordchipper-cli`](https://crates.io/crates/wordchipper-cli) tool.

## Training

This is a code snippet overview of training.

Expect training to take ~1s/10MB of input; and to be slowed primarily by how well the stream logic
of loading the
training samples is parallelized.

Note: currently, training has limited logging and no progress reporting.

A common training binary is probably a good idea; and much of the messiness of supporting many
different training data
sources could be hidden in the isolated deps of such a tool.

Consider the following, to train a tokenizer and export it a "*.tiktoken" file.

* The iterator stream for samples may be quite large.
* Training a nanochat equivalent tokenizer takes ~80 CPU minutes.

```rust,no_run
use std::sync::Arc;

use wordchipper::{
    Tokenizer,
    TokenizerOptions,
    UnifiedTokenVocab,
    pretrained::openai::OA_CL100K_BASE_PATTERN,
    vocab::{ByteMapVocab, io::save_base64_span_map_path},
};
use wordchipper_training::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};

fn example<I, S>(
    vocab_size: usize,
    batches: I,
    vocab_save_path: Option<String>,
) -> Arc<Tokenizer<u32>>
where
    I: IntoIterator,
    I::Item: AsRef<[S]>,
    S: AsRef<str>,
{
    // We can pick any unsigned integer type > vocab_size;
    // See [`wordchipper::TokenType`].
    type T = u32;
    type K = String;
    type C = u64;

    let options = BinaryPairVocabTrainerOptions::new(OA_CL100K_BASE_PATTERN, vocab_size);

    let mut trainer: BinaryPairVocabTrainer<K, C> = options.init();

    for batch in batches {
        // The trainer has no parallelism.
        // The perceived benefits of parallelism in the trainer
        // are insignificant if the IO for the sample source is
        // fed by another thread.
        trainer.update_from_samples(batch.as_ref());
    }

    let byte_vocab: ByteMapVocab<T> = Default::default();

    let vocab: Arc<UnifiedTokenVocab<T>> = trainer
        .train(byte_vocab.clone())
        .expect("training failed")
        .into();

    if let Some(path) = vocab_save_path {
        save_base64_span_map_path(&vocab.span_vocab().span_map(), &path)
            .expect("failed to save vocab");
        println!("- tiktoken vocab: {path:?}");
    }

    let tokenizer: Arc<Tokenizer<u32>> =
        TokenizerOptions::default().with_parallel(true).build(vocab);

    tokenizer
}
```