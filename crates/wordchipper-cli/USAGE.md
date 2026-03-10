# Command-Line Help for `wordchipper-cli`

This document contains the help content for the `wordchipper-cli` command-line program.

**Command Overview:**

* [`wordchipper-cli`↴](#wordchipper-cli)
* [`wordchipper-cli cat`↴](#wordchipper-cli-cat)
* [`wordchipper-cli lexers`↴](#wordchipper-cli-lexers)
* [`wordchipper-cli lexers list`↴](#wordchipper-cli-lexers-list)
* [`wordchipper-cli lexers stress`↴](#wordchipper-cli-lexers-stress)
* [`wordchipper-cli models`↴](#wordchipper-cli-models)
* [`wordchipper-cli models list`↴](#wordchipper-cli-models-list)
* [`wordchipper-cli train`↴](#wordchipper-cli-train)
* [`wordchipper-cli doc`↴](#wordchipper-cli-doc)

## `wordchipper-cli`

Text tokenizer multi-tool

**Usage:** `wordchipper-cli <COMMAND>`

###### **Subcommands:**

* `cat` — Act as a streaming tokenizer
* `lexers` — Lexers sub-menu
* `models` — Models sub-menu
* `train` — Train a new model
* `doc` — Generate markdown documentation

## `wordchipper-cli cat`

Act as a streaming tokenizer

**Usage:** `wordchipper-cli cat [OPTIONS] <--model <MODEL>> <--encode|--decode>`

###### **Options:**

* `--model <MODEL>` — Model to use for encoding

  Default value: `openai:r50k_base`
* `--encode` — Encode from text to tokens
* `--decode` — Decode from tokens to text
* `--input <INPUT>` — Optional input file; "-" may be used to indicate stdin
* `--output <OUTPUT>` — Optional output file; "-" may be used to indicate stdout
* `--cache-dir <CACHE_DIR>` — Cache directory

## `wordchipper-cli lexers`

Lexers sub-menu

**Usage:** `wordchipper-cli lexers <COMMAND>`

###### **Subcommands:**

* `list` — List available lexers
* `stress` — Stress test a regex accelerator

## `wordchipper-cli lexers list`

List available lexers

**Usage:** `wordchipper-cli lexers list [OPTIONS]`

**Command Alias:** `ls`

###### **Options:**

* `-p`, `--patterns` — Display the patterns

## `wordchipper-cli lexers stress`

Stress test a regex accelerator

**Usage:**
`wordchipper-cli lexers stress [OPTIONS] --input-format <INPUT_FORMAT> <--lexer-model <LEXER_MODEL>|--pattern <PATTERN>> [FILES]...`

###### **Arguments:**

* `<FILES>` — Input files

###### **Options:**

* `--input-format <INPUT_FORMAT>` — The input shard file format

  Possible values:
    - `text`:
      Simple text files
    - `parquet`:
      Parquet files

* `--input-batch-size <INPUT_BATCH_SIZE>` — The input batch size

  Default value: `100`
* `-q`, `--quiet` — Silence log messages
* `-v`, `--verbose` — Turn debugging information on (-v, -vv, -vvv)
* `-t`, `--ts` — Enable timestamped logging
* `--lexer-model <LEXER_MODEL>` — Model name for selection
* `--pattern <PATTERN>` — Pattern for selection
* `--pre-context <PRE_CONTEXT>` — Span context before error

  Default value: `8`
* `--post-context <POST_CONTEXT>` — Span context after error

  Default value: `8`

## `wordchipper-cli models`

Models sub-menu

**Usage:** `wordchipper-cli models <COMMAND>`

###### **Subcommands:**

* `list` — List available models

## `wordchipper-cli models list`

List available models

**Usage:** `wordchipper-cli models list`

**Command Alias:** `ls`

## `wordchipper-cli train`

Train a new model

**Usage:**
`wordchipper-cli train [OPTIONS] --input-format <INPUT_FORMAT> <--lexer-model <LEXER_MODEL>|--pattern <PATTERN>> [FILES]...`

###### **Arguments:**

* `<FILES>` — Input files

###### **Options:**

* `--input-format <INPUT_FORMAT>` — The input shard file format

  Possible values:
    - `text`:
      Simple text files
    - `parquet`:
      Parquet files

* `--input-batch-size <INPUT_BATCH_SIZE>` — The input batch size

  Default value: `100`
* `-q`, `--quiet` — Silence log messages
* `-v`, `--verbose` — Turn debugging information on (-v, -vv, -vvv)
* `-t`, `--ts` — Enable timestamped logging
* `--vocab-size <VOCAB_SIZE>` — Max vocab size

  Default value: `50281`
* `--lexer-model <LEXER_MODEL>` — Model name for selection
* `--pattern <PATTERN>` — Pattern for selection
* `--output <OUTPUT>` — Optional output file; "-" may be used to indicate stdout

## `wordchipper-cli doc`

Generate markdown documentation

**Usage:** `wordchipper-cli doc`



<hr/>

<small><i>
This document was generated automatically by
<a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>

