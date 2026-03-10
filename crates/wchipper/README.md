# wchipper

[![Crates.io Version](https://img.shields.io/crates/v/wchipper)](https://crates.io/crates/wchipper)
[![Documentation](https://img.shields.io/docsrs/wordchipper)](https://docs.rs/wordchipper/latest/wchipper/)
[![Test Status](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml/badge.svg)](https://github.com/zspacelabs/wordchipper/actions/workflows/ci.yml)

A text LLM tokenizer command line multi-tool.

## Overview

Part of the [wordchipper tokenizer suite](https://crates.io/crates/wordchipper).

### Suite Crates

This is the binary tokenizer multi-tool for the
[wordchipper tokenizer suite](https://github.com/zspacelabs/wordchipper).

The core additional user-facing crates are:

* [wordchipper](https://crates.io/crates/wordchipper) - the core tokenizer library.
* [wordchipper-training](https://crates.io/crates/wordchipper-training) - an extension crate for training tokenizers.

## Installation

```terminaloutput
% cargo install wchipper
```

## Usage

See: [USAGE](USAGE.md) for detailed usage instructions.

### Example: wchipper cat

```terminaloutput
% echo "abc def" | wchipper cat --encode --model=openai::gpt2
39305 825 198

% echo "39305 825 198" | cargo run --release -p wordchipper-cli -- \
    cat --decode --model=openai::gpt2
abc def
```

### Example: wchipper models list

```terminaloutput
 % wchipper models list                  
"openai" - Pretrained vocabularies from OpenAI
  * "openai:gpt2"
    GPT-2 `gpt2` vocabulary
  * "openai:r50k_base"
    GPT-2 `p50k_base` vocabulary
  * "openai:p50k_base"
    GPT-2 `p50k_base` vocabulary
  * "openai:p50k_edit"
    GPT-2 `p50k_edit` vocabulary
  * "openai:cl100k_base"
    GPT-3 `cl100k_base` vocabulary
  * "openai:o200k_base"
    GPT-5 `o200k_base` vocabulary
  * "openai:o200k_harmony"
    GPT-5 `o200k_harmony` vocabulary
```

### Example: wchipper train

```terminal
% wchipper train \
    --output=/tmp/tok.tokenizer \
    --lexer-model=gpt2 \
    --input-format=parquet \
     ~/Data/nanochat/dataset/*.parquet 
INFO Reading shards:
INFO 0: /Users/crutcher/Data/nanochat/dataset/shard_00000.parquet
INFO 1: /Users/crutcher/Data/nanochat/dataset/shard_00001.parquet
...
INFO 6: /Users/crutcher/Data/nanochat/dataset/shard_00006.parquet
INFO 7: /Users/crutcher/Data/nanochat/dataset/shard_00007.parquet
INFO Training Tokenizer...
INFO Starting BPE training: 50025 merges to compute
INFO Building pair index...
INFO Building heap with 16044 unique pairs
INFO Starting merge loop
INFO Progress: 1% (501/50025 merges) - Last merge: (69, 120) -> 756 (frequency: 166814)
INFO Progress: 2% (1001/50025 merges) - Last merge: (66, 77) -> 1256 (frequency: 17847)
INFO Progress: 3% (1501/50025 merges) - Last merge: (402, 104) -> 1756 (frequency: 4811)
...
INFO Progress: 98% (49025/50025 merges) - Last merge: (2376, 347) -> 49280 (frequency: 18)
INFO Progress: 99% (49525/50025 merges) - Last merge: (567, 372) -> 49780 (frequency: 18)
INFO Progress: 100% (50025/50025 merges) - Last merge: (302, 3405) -> 50280 (frequency: 18)
INFO Finished training: 50025 merges completed
INFO Vocabulary Size: 50280
INFO output: /tmp/tok.tokenizer
```
