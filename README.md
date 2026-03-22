<img src="media/logo.png" width="50%" />

[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![Test Status](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml)
[![license](https://shields.io/badge/license-MIT-blue)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/wordchipper)

I am usually available as `@crutcher` on the Burn Discord:

* Burn
  Discord: [![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)

## Status

This is ready for alpha users, and is 2x the speed of `tiktoken-rs`
for many current models.

The productionization towards an LTR stable release can be
tracked in the
[Alpha Release Tracking Issue](https://github.com/zspacelabs/wordchipper/issues/2).

## Overview

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

The primary documentation is for the [wordchipper crate](crates/wordchipper).

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

## `no_std` Support

The core tokenization pipeline (spanning, encoding, decoding, vocabulary lookup) works in `no_std`
environments. This is CI-verified against `wasm32-unknown-unknown` and `thumbv7m-none-eabi` targets.

```toml
[dependencies]
wordchipper = { version = "0.7", default-features = false }
```

Features that require `std` (training, file I/O, download, rayon parallelism) are behind
feature flags that imply `std`.

## Language Bindings

### Python

```bash
pip install wordchipper
```

```python
from wordchipper import Tokenizer

tok = Tokenizer.from_pretrained("cl100k_base")
tokens = tok.encode("hello world")  # [15339, 1917]
text = tok.decode(tokens)  # "hello world"
```

See [bindings/python](bindings/python) for full API and benchmarks.

### JavaScript / TypeScript (WASM)

```js
import {Tokenizer} from "./js/dist/index.js";

const tok = await Tokenizer.fromPretrained("o200k_base");
const tokens = tok.encode("hello world"); // Uint32Array
const text = tok.decode(tokens);          // "hello world"
```

See [bindings/wasm](bindings/wasm) for full API, build instructions, and examples.

## Components

### Published Crates

- [wordchipper](crates/wordchipper) - main crate.
- [wordchipper-training](crates/wordchipper-training) - training library.
- [wordchipper-cli](crates/wordchipper-cli) - `chipper` multifunction command-line tool.

#### Internally Sourced Dep Crates

You should never need to import these directly.

- [wordchipper-disk-cache](crates/wordchipper-disk-cache) - disk cache.

### Bindings

- [wordchipper-python](bindings/python) - Python bindings (PyO3/maturin)
- [wordchipper-wasm](bindings/wasm) - WASM bindings (wasm-bindgen) with TypeScript wrapper

### Development Crates

A Number of internal crates are used for development.

- [dev-crates](dev-crates)

## Acknowledgements

* Thank you to [@karpathy](https://github.com/karpathy)
  and [nanochat](https://github.com/karpathy/nanochat)
  for the work on `rustbpe`.
* Thank you to [tiktoken](https://github.com/openai/tiktoken) for their initial work in the rust
  tokenizer space.

