# wordchipper

[![PyPI](https://img.shields.io/pypi/v/wordchipper)](https://pypi.org/project/wordchipper/)
[![Crates.io Version](https://img.shields.io/crates/v/wordchipper)](https://crates.io/crates/wordchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wordchipper/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)

Python bindings for the [wordchipper](https://github.com/zspacelabs/wordchipper) BPE tokenizer
library, by [ZSpaceLabs](https://zspacelabs.ai).

`wordchipper` is a high-performance Rust byte-pair encoder tokenizer for the OpenAI GPT-2 tokenizer
family. Under Python wrappers, we see a range of ~2x-4x (4 to 64 cores) speedups over
[tiktoken](https://github.com/openai/tiktoken).

| x 64 Core   | r50k python | o200k python |
| ----------- | ----------- | ------------ |
| wordchipper | 110.5 MiB/s | 106.5 MiB/s  |
| tiktoken    | 25.5 MiB/s  | 32.7 MiB/s   |
| tokenizers  | 20.8 MiB/s  | 23.2 MiB/s   |

Read the full performance paper:
[wordchipper: Fast BPE Tokenization with Substitutable Internals](https://zspacelabs.ai/wordchipper/articles/substitutable/)

## Installation

```bash
pip install wordchipper
```

## Usage

```python
from wordchipper import Tokenizer

# See available models
Tokenizer.available_models()
# ['r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base', 'o200k_harmony']

# Load a tokenizer
tok = Tokenizer.from_pretrained("cl100k_base")

# Encode / decode
tokens = tok.encode("hello world")        # [15339, 1917]
text = tok.decode(tokens)                  # "hello world"

# Batch encode / decode (parallel via rayon)
results = tok.encode_batch(["hello", "world", "foo bar"])
texts = tok.decode_batch(results)

# Vocab inspection
tok.vocab_size                             # 100256
tok.token_to_id("hello")                   # 15339
tok.id_to_token(15339)                     # "hello"
tok.token_to_id("nonexistent")             # None

# Special tokens
tok.get_special_tokens()
# [('<|endoftext|>', 100257), ...]

# Save vocab to file (base64 tiktoken format, excludes special tokens)
tok.save_base64_vocab("vocab.tiktoken")
```

## Compatibility wrappers

Drop-in replacements for `tiktoken` and HuggingFace `tokenizers`. Change one import line and the
rest of your code stays the same:

```python
# tiktoken compat
from wordchipper.compat import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("hello world")

# HuggingFace tokenizers compat
from wordchipper.compat.tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("Xenova/gpt-4o")
output = tok.encode("hello world")
output.ids      # [24912, 2375]
```

Parameters that are accepted for API compatibility but not implemented (e.g. `allowed_special`,
`disallowed_special`, `is_pretokenized`) will raise `NotImplementedError` when set to non-default
values.

## Development

Requires [Rust](https://rustup.rs/) and [uv](https://docs.astral.sh/uv/).

```bash
cd bindings/python

# Set up environment and build
uv venv .venv
source .venv/bin/activate
uv pip install maturin pytest
maturin develop

# Run tests
pytest tests/ -v
```

After making changes to `src/lib.rs`, rebuild with `maturin develop` before re-running tests.

## Benchmarks

Compares `wordchipper` against `tiktoken` and HuggingFace `tokenizers` for single and batch encoding
on cl100k_base and o200k_base. Uses the same corpora and methodology as the Rust benchmarks in
`wordchipper-bench`:

- **Single-string**: `english.txt` / `multilingual.txt` repeated 10x
- **Batch**: 1024 samples from fineweb-edu shard 0 (~4.2 MB)

```bash
# Install benchmark dependencies
uv pip install pytest-benchmark tiktoken tokenizers pyarrow

# Build in release mode for meaningful numbers
maturin develop --release

# Run all benchmarks
pytest benchmarks/

# Run only single-encode benchmarks
pytest benchmarks/ -k "TestSingleEncode"

# Run only batch-encode benchmarks
pytest benchmarks/ -k "TestBatchEncode"

# Run only decode benchmarks
pytest benchmarks/ -k "TestSingleDecode"

# Filter by model
pytest benchmarks/ -k "cl100k_base"
```

## License

`wordchipper` is distributed under the terms of both the MIT license and the Apache License (Version
2.0). See [LICENSE-APACHE](https://github.com/zspacelabs/wordchipper/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/zspacelabs/wordchipper/blob/main/LICENSE-MIT) for details.
