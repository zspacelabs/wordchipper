# AGENTS.md — wordchipper

Context for coding agents working on **wordchipper**. Read this before writing any code.

---

## Project Overview

**wordchipper** is a high-performance Rust BPE (Byte Pair Encoding) tokenizer library targeting
HPC environments. It provides training, encoding, and decoding of BPE vocabularies, with
compatibility for `tiktoken` and `nanochat/rustbpe` formats.

- **Repository**: https://github.com/zspacelabs/wordchipper
- **License**: MIT
- **Rust Edition**: 2024
- **MSRV**: 1.93.0
- **Current Version**: 0.7.3

---

## Workspace Layout

```
wordchipper/
├── crates/
│   ├── wordchipper/              # Core library (published)
│   ├── wordchipper-disk-cache/   # Download cache utility (published, internal dep)
│   ├── wordchipper-training/     # BPE training library (published)
│   └── wordchipper-cli/          # CLI tool (published)
├── bindings/
│   ├── python/                   # PyO3/maturin Python bindings (published to PyPI)
│   └── wasm/                     # wasm-bindgen WASM bindings + TypeScript wrapper
├── dev-crates/
│   ├── wordchipper-bench/        # Divan benchmark suite (unpublished)
│   ├── wordchipper-data/         # Dataset loading for training (unpublished)
│   └── sample-timer/             # Timing comparisons vs tiktoken/tokenizers (unpublished)
└── book/                         # mdBook documentation with interactive WASM demo
```

### Published Crates

| Crate                    | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `wordchipper`            | Core encode/decode/vocab library                             |
| `wordchipper-training`   | BPE vocabulary trainer                                       |
| `wordchipper-cli`        | Command-line interface                                       |
| `wordchipper-disk-cache` | Disk cache for vocab downloads (not intended for direct use) |

---

## Build & Test Commands

```sh
# Format (requires nightly)
cargo +nightly fmt
cargo +nightly fmt --check   # CI check

# Lint
cargo clippy --no-deps

# Test (full workspace)
cargo test --workspace

# Test alternate feature set
cargo test -p wordchipper --no-default-features --features client,fast-hash

# Test no_std surface
cargo test -p wordchipper --no-default-features --tests

# Cross-check no_std targets (wasm32 + ARM Cortex-M3)
cargo check -p wordchipper --target wasm32-unknown-unknown --no-default-features
cargo check -p wordchipper --target thumbv7m-none-eabi --no-default-features

# Generate docs (all features)
cargo doc --no-deps --quiet --all-features
```

### WASM Bindings

```sh
cd bindings/wasm
wasm-pack build --target web        # Build package
wasm-pack test --node               # Run tests (used by CI)
```

### Python Bindings

```sh
cd bindings/python
uv venv .venv && source .venv/bin/activate
uv pip install maturin pytest
maturin develop
pytest tests/ -v
```

### Book (mdBook)

```sh
cargo install mdbook
cd book
./setup-wasm.sh                     # Build WASM demo and download vocab files
mdbook serve                        # Serve locally at http://localhost:3000
mdbook build                        # Build static HTML to book/book/
```

### CI Jobs

All of the following must pass on every push/PR to `main`:
`fmt` → `clippy` → `test` → `cross` → `wasm` → `python`

---

## Feature Flags (`wordchipper` crate)

| Feature      | Default        | Purpose                                               |
|--------------|----------------|-------------------------------------------------------|
| `std`        | ✓              | Standard library support                              |
| `fast-hash`  | ✓              | Fast hashing via foldhash (works in no_std)           |
| `parallel`   | ✓              | Batch parallelism via rayon (implies `concurrent`)    |
| `concurrent` | via `parallel` | Thread pool and concurrency utilities (implies `std`) |
| `client`     |                | Load and run pretrained encoders/decoders             |
| `download`   | via `client`   | Network vocab downloading                             |
| `datagym`    | via `client`   | DataGym I/O for training data                         |
| `tracing`    |                | `tracing` instrumentation points                      |
| `testing`    |                | Utilities for downstream test crates                  |

`wordchipper-training` has no default features; enable `tracing` optionally.

---

## `no_std` Support

The crate uses unconditional `#![no_std]` with `#[cfg(feature = "std")] extern crate std;`
(the "Reddit PSA" pattern). A `pub(crate) mod prelude` re-exports `alloc` types (`Vec`, `String`,
`ToString`, `Box`). Add `use crate::prelude::*;` in any module needing those types.

- `hashbrown` is always present (non-optional) for `HashMap`/`HashSet` in `no_std`.
- All features that require OS threads, file I/O, or network are `std`-gated.
- WASM bindings (`bindings/wasm/`) build with `default-features = false` as the canonical
  `no_std` integration example.

---

## Core Architecture

### Two-Phase Tokenization

1. **Spanning** (pre-tokenization) — splits raw text into word-level spans via regex or Logos DFA.
2. **BPE Encoding** — merges subword pairs within each span using the vocabulary merge table.

### Key Types

| Type                    | Role                                                                                                          |
|-------------------------|---------------------------------------------------------------------------------------------------------------|
| `UnifiedTokenVocab<T>`  | Primary user-facing vocabulary; owns `ByteMapVocab`, `SpanMapVocab`, `PairMapVocab`, and `TextSpanningConfig` |
| `Tokenizer<T>`          | Wraps vocab + encoder + decoder; primary entry point                                                          |
| `TokenizerOptions`      | Builder for `Tokenizer`; controls parallelism and other options                                               |
| `TextSpanningConfig<T>` | Regex patterns + special tokens for pre-tokenization                                                          |
| `SpanMapVocab<T>`       | `Vec<u8> → T` dictionary                                                                                      |
| `PairMapVocab<T>`       | `(T, T) → T` BPE merge rules                                                                                  |
| `ByteMapVocab<T>`       | Bijective `u8 ↔ T` byte map (256 entries)                                                                     |

`T` is generic over `TokenType` (`u16`, `u32`, `u64`). `u32` is standard; use `u16` only for
vocabs ≤ 65535 tokens.

### Encoder/Decoder Implementations

Multiple encoder and decoder implementations are maintained deliberately for cross-benchmarking.
Do **not** consolidate them to a single "best" implementation — the ability to benchmark across
approaches is an architectural requirement.

### Vocabulary Immutability

`UnifiedTokenVocab` is immutable after construction and typically wrapped in `Arc`. This is a
deliberate design choice enabling thread safety and cacheline-friendly access patterns.

Dynamic span caching was tested and abandoned due to cacheline contention under multi-threaded load.

### Logos DFA Lexers

For OpenAI-style pre-tokenization patterns, Logos DFA lexers provide 30–50× speedup over regex by
compiling patterns to DFAs at build time. Post-processing via `TokenRole` and
`for_each_classified_span` corrects the DFA output to match the `\s+(?!\S)` lookahead semantics
that Logos cannot express natively.

---

## Code Conventions

See [`STYLE_GUIDE.md`](STYLE_GUIDE.md) for full details. Key rules:

- **"Style Hints"**: Many types carry `## Style Hints` doc comments prescribing preferred variable
  names. Follow them. Examples: `SpanTokenMap<T>` → `span_map`; `TextSpanningConfig<T>` →
  `spanner_config`.
- **All public items must have `///` doc comments** with `## Arguments`, `## Returns`,
  `## Panics` sections as appropriate.
- **Private fields, public accessors**: struct fields are private; expose via `foo()`, `set_foo()`,
  `with_foo()`.
- **Constructors**: prefer `from_*` and `new()` returning `Self` or `WCResult<Self>`.
- **Builder methods**: `with_*`, consume `self`.
- **`cfg` gating**: feature-gated modules use `#[cfg(feature = "...")]` at `mod` declarations.
- **Error handling**: use `thiserror` in `wordchipper`; `anyhow` is acceptable in binaries and
  dev-crates.
- **Tests**: live in `#[cfg(test)] mod tests` in the same file; use `serial_test::serial` for
  env-var mutation.
- **`missing_docs` and `unused` are warned** at crate root; treat warnings as errors (`deny`).

---

## Dependency Notes

- `regex` and `fancy-regex` are intentionally pinned due to a performance regression under
  concurrent access in newer versions. Do not upgrade without benchmarking.
- `hashbrown` is non-optional so `default-features = false` works out of the box.
- Library crates require broader semver ranges than binaries; validate with
  `cargo +nightly update -Z minimal-versions` before bumping deps.
