# wordchipper

Python bindings for the [wordchipper](https://github.com/zspacelabs/wordchipper) BPE tokenizer
library.

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

Drop-in replacements for `tiktoken` and HuggingFace `tokenizers`. Change one import
line and the rest of your code stays the same:

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

Parameters that are accepted for API compatibility but not implemented (e.g.
`allowed_special`, `disallowed_special`, `is_pretokenized`) will raise
`NotImplementedError` when set to non-default values.

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

Compares `wordchipper` against `tiktoken` and HuggingFace `tokenizers` for single
and batch encoding on cl100k_base and o200k_base. Uses the same corpora and methodology
as the Rust benchmarks in `wordchipper-bench`:

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
