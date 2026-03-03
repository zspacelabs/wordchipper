# Pretrained Models

wordchipper ships with loaders for OpenAI's public BPE vocabularies. Each model defines three
things: a regex pattern for pre-tokenization, a merge table (the vocabulary), and a set of special
tokens.

## Model overview

| Model           | Vocab tokens | Pattern        | Special tokens                                       | Used by          |
|-----------------|--------------|----------------|------------------------------------------------------|------------------|
| `r50k_base`     | ~50k         | GPT-2 pattern  | `<\|endoftext\|>`                                    | GPT-2            |
| `p50k_base`     | ~50k         | GPT-2 pattern  | `<\|endoftext\|>`                                    | Codex            |
| `p50k_edit`     | ~50k         | GPT-2 pattern  | endoftext, fim_prefix/middle/suffix                  | Codex (edit)     |
| `cl100k_base`   | ~100k        | cl100k pattern | endoftext, fim_prefix/middle/suffix, endofprompt     | GPT-3.5, GPT-4   |
| `o200k_base`    | ~200k        | o200k pattern  | endoftext, endofprompt                               | GPT-4o           |
| `o200k_harmony` | ~200k        | o200k pattern  | endoftext, endofprompt, startoftext, + many reserved | GPT-4o (harmony) |

### What changed between models

**r50k / p50k (GPT-2 era).** The simplest pattern. Contractions (`'s`, `'t`, etc.) are matched
literally. Words and numbers are preceded by an optional space. Case-sensitive.

**cl100k (GPT-3.5 / GPT-4).** Case-insensitive contractions. Words can be preceded by a non-letter,
non-newline character (allowing punctuation to attach). Numbers limited to 3 digits at a time.
Newlines handled explicitly.

**o200k (GPT-4o).** Much more sophisticated word patterns. Recognizes casing transitions (CamelCase
splits). Uses Unicode general categories (`\p{Lu}`, `\p{Ll}`, `\p{Lt}`, `\p{Lm}`, `\p{Lo}`, `\p{M}`)
for precise letter classification. Contractions are appended to words rather than matched
separately.

### Shared vocabularies

Some models share the same vocabulary file but differ in special tokens:

- `p50k_edit` uses the same vocab as `p50k_base` but adds FIM (fill-in-middle) tokens.
- `o200k_harmony` uses the same vocab as `o200k_base` but adds many reserved and named special
  tokens for structured generation.

## Loading a vocabulary

The standard way to load a model:

```rust,no_run
use wordchipper::{load_vocab, disk_cache::WordchipperDiskCache};

let mut cache = WordchipperDiskCache::default();
let (desc, vocab) = load_vocab("openai:cl100k_base", &mut cache).unwrap();
```

`load_vocab` returns a `(VocabDescription, Arc<UnifiedTokenVocab<u32>>)`. The description contains
metadata; the vocab contains everything needed for encoding and decoding.

### Short names

The `openai::` prefix is optional. Both `"cl100k_base"` and `"openai:cl100k_base"` work. Use
`list_vocabs()` for all registered short names and `list_models()` for fully qualified names.

### Loading from a file path

If you have a `.tiktoken` file on disk, you can skip the download:

```rust,no_run
use wordchipper::pretrained::openai::OATokenizer;

let vocab = OATokenizer::Cl100kBase
    .load_path::<u32>("/path/to/cl100k_base.tiktoken")
    .unwrap();
```

### Loading from a reader

For maximum flexibility (e.g., loading from an in-memory buffer or a network stream):

```rust,no_run
use std::io::BufReader;
use wordchipper::pretrained::openai::OATokenizer;

let data: &[u8] = b"..."; // tiktoken base64 format
let reader = BufReader::new(data);
let vocab = OATokenizer::O200kBase.read_vocab::<u32, _>(reader).unwrap();
```

## Special tokens

Special tokens are strings with reserved token IDs that are never produced by BPE encoding. They're
used for control flow: marking end-of-text, fill-in-middle boundaries, prompt boundaries, etc.

```rust,no_run
# use wordchipper::{load_vocab, disk_cache::WordchipperDiskCache, TokenizerOptions, TokenEncoder};
# let mut cache = WordchipperDiskCache::default();
# let (_, vocab) = load_vocab("openai:cl100k_base", &mut cache).unwrap();
# let tok = TokenizerOptions::default().build(vocab);
let specials = tok.special_vocab();
for (bytes, &id) in specials.span_map().iter() {
    let name = String::from_utf8_lossy(bytes);
    println!("{} -> {}", name, id);
}
```

For `cl100k_base`, this prints:

```text
<|endoftext|> -> 100257
<|fim_prefix|> -> 100258
<|fim_middle|> -> 100259
<|fim_suffix|> -> 100260
<|endofprompt|> -> 100276
```

## The OATokenizer enum

For programmatic access to all OpenAI models, use the `OATokenizer` enum:

```rust,no_run
use wordchipper::pretrained::openai::OATokenizer;

// Iterate over all models
# #[cfg(feature = "std")]
# {
use strum::IntoEnumIterator;
for model in OATokenizer::iter() {
    println!("{}", model);
}
# }
```

Each variant provides:

- `pattern()` - the regex pattern for pre-tokenization
- `special_tokens::<T>()` - the special token list
- `load_vocab::<T>(loader)` - load the vocabulary with download support
- `load_path::<T>(path)` - load from a local file

## The tiktoken format

OpenAI's vocabulary files use a simple base64 format. Each line contains a base64-encoded byte
sequence and its integer token ID, separated by a space:

```text
IQ== 0
Ig== 1
Iw== 2
...
```

The base64 decodes to the raw bytes that the token represents. This format is what
`WordchipperDiskCache` downloads and caches from OpenAI's CDN.

## Choosing a model

If you're building a tool that interacts with an OpenAI model, use the matching tokenizer:

| If you use...            | Load...       |
|--------------------------|---------------|
| GPT-4o, GPT-4o-mini      | `o200k_base`  |
| GPT-4, GPT-3.5-turbo     | `cl100k_base` |
| GPT-3 (text-davinci-003) | `p50k_base`   |
| GPT-2                    | `r50k_base`   |

If you're building your own model or just need token counting, `o200k_base` has the largest
vocabulary and handles the widest range of languages efficiently.
