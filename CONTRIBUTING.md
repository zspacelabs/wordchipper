# CONTRIBUTING.md — wordchipper

## Before You Start

Read [`STYLE_GUIDE.md`](STYLE_GUIDE.md) for coding conventions. All PRs must pass CI before merge.

---

## Development Setup

You need stable Rust (MSRV 1.93.0) and nightly `rustfmt`:

```sh
rustup toolchain install stable nightly
rustup component add --toolchain nightly rustfmt
```

For WASM work: `cargo install wasm-pack`  
For Python bindings: `uv` + `maturin` (`uv pip install maturin pytest`)  
For the book: `cargo install mdbook`

---

## Workflow

1. Format: `cargo +nightly fmt`
2. Lint: `cargo clippy --no-deps` (fix all warnings — they're treated as errors in CI)
3. Test: `cargo test --workspace`
4. Commit.

A convenience script `fix.sh` runs format and clippy together if it exists at the repo root.

---

## CI Requirements

All of the following must pass. Run them locally before pushing:

```sh
# Format check
cargo +nightly fmt --check

# Lint
cargo clippy --no-deps

# Tests (default features, alternate feature set, no_std surface)
cargo test --workspace
cargo test -p wordchipper --no-default-features --features client,fast-hash
cargo test -p wordchipper --no-default-features --tests

# no_std cross-check
cargo check -p wordchipper --target wasm32-unknown-unknown --no-default-features
cargo check -p wordchipper --target thumbv7m-none-eabi --no-default-features

# WASM bindings
cd bindings/wasm && wasm-pack test --node

# Python bindings
cd bindings/python
uv venv .venv && source .venv/bin/activate
uv pip install maturin pytest && maturin develop
pytest tests/ -v
```

---

## Key Constraints

**Vocabulary immutability.** `UnifiedTokenVocab` is immutable after construction. Do not add
mutation methods or interior mutability to vocabulary types.

**Multiple encoder/decoder implementations.** Several implementations exist for benchmarking
purposes. Do not consolidate them — the cross-benchmark capability is intentional.

**`no_std` cleanliness.** Code in `crates/wordchipper` must not introduce `std`-only constructs
outside of `std`-gated modules. Use `crate::prelude::*` for common alloc types.

**Dependency pins.** `regex` and `fancy-regex` are pinned due to a performance regression. Do not
upgrade these without first benchmarking under concurrent workloads.

**Semver ranges.** Published library crates must support a range of dependency versions, not just
the latest. After any dependency changes, verify minimum versions:

```sh
cargo +nightly update -Z minimal-versions
cargo test --workspace
```

---

## Adding a New Pretrained Model

Pretrained model registrations live in `crates/wordchipper/src/pretrained/`. Each model is
registered via the `inventory` crate at program startup. See existing OpenAI models for the pattern.

---

## Experimental Features

Features that are not ready for the published API belong in a separate unpublished crate, not
behind a feature flag in `wordchipper`. The `dev-crates/` directory is appropriate for internal
tooling; a standalone crate under `crates/` with `publish = false` works for experimental
extensions intended for eventual promotion.

---

## Documentation

The `book/` directory contains an mdBook with user-facing documentation and an interactive WASM
tokenizer demo. When adding or changing public API, update the relevant book chapters alongside the
doc comments.

```sh
cd book
./setup-wasm.sh   # first time only
mdbook serve
```
