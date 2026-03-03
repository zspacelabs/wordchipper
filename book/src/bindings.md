# Python & WASM Bindings

wordchipper has first-class bindings for Python and JavaScript/TypeScript. Both expose the same core
operations: load a vocabulary, encode text to tokens, decode tokens back to text.

## Python

### Installation

```bash
pip install wordchipper
```

### Basic usage

```python
from wordchipper import Tokenizer

tok = Tokenizer.from_pretrained("cl100k_base")

# Encode and decode
tokens = tok.encode("hello world")       # [15339, 1917]
text = tok.decode(tokens)                 # "hello world"

# Batch operations (parallel via rayon)
results = tok.encode_batch(["hello", "world", "foo bar"])
texts = tok.decode_batch(results)
```

### Vocabulary inspection

```python
tok.vocab_size                             # 100256
tok.token_to_id("hello")                   # 15339
tok.id_to_token(15339)                     # "hello"
tok.token_to_id("nonexistent")             # None

# Special tokens
tok.get_special_tokens()
# [('<|endoftext|>', 100257), ('<|fim_prefix|>', 100258), ...]
```

### Available models

```python
Tokenizer.available_models()
# ['r50k_base',
#  'p50k_base',
#  'p50k_edit',
#  'cl100k_base',
#  'o200k_base',
#  'o200k_harmony']
```

### Saving vocabularies

Export a vocabulary in tiktoken's base64 format:

```python
tok.save_base64_vocab("vocab.tiktoken")
```

### Compatibility wrappers

Drop-in replacements for `tiktoken` and HuggingFace `tokenizers`. Change one import line and
the rest of your code stays the same.

**tiktoken**

```python
# Before
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# After
from wordchipper.compat import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

tokens = enc.encode("hello world")
text = enc.decode(tokens)

# Model lookup works too
enc = tiktoken.encoding_for_model("gpt-4o")
```

The `Encoding` class exposes `encode`, `encode_ordinary`, `encode_batch`, `decode`,
`decode_batch`, and properties `name`, `n_vocab`, `max_token_value`, `eot_token`, and
`special_tokens_set`.

**HuggingFace tokenizers**

```python
# Before
from tokenizers import Tokenizer

# After
from wordchipper.compat.tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("Xenova/gpt-4o")
output = tok.encode("hello world")
output.ids      # [24912, 2375]
output.tokens   # ["hello", " world"]
text = tok.decode(output.ids)
```

The `Tokenizer` class exposes `encode`, `encode_batch`, `decode`, `decode_batch`,
`get_vocab_size`, `token_to_id`, and `id_to_token`. Known HuggingFace identifiers
(e.g. `Xenova/gpt-4o`) are mapped automatically; bare encoding names like `cl100k_base`
also work.

### Building from source

Requires [Rust](https://rustup.rs/) and [uv](https://docs.astral.sh/uv/):

```bash
cd bindings/python
uv venv .venv && source .venv/bin/activate
uv pip install maturin pytest
maturin develop           # debug build
maturin develop --release # release build for benchmarks
pytest tests/ -v
```

## JavaScript / TypeScript (WASM)

wordchipper compiles to WebAssembly and runs in browsers and Node.js. The WASM build uses
`default-features = false` (no `std`, no parallelism, no file I/O), so all core tokenization works in
the browser without a server.

### Quick start

```js
import { Tokenizer } from "./js/dist/index.js";

const tok = await Tokenizer.fromPretrained("o200k_base");

const tokens = tok.encode("hello world"); // Uint32Array [24912, 2375]
const text = tok.decode(tokens); // "hello world"

tok.free(); // release WASM memory when done
```

### Loading

Two ways to load a tokenizer:

```js
// Fetch from OpenAI's CDN (convenience)
const tok1 = await Tokenizer.fromPretrained("cl100k_base");

// Or from your own vocab bytes (no network request)
const data = new Uint8Array(/* .tiktoken file contents */);
const tok2 = await Tokenizer.fromVocabData("cl100k_base", data);
```

`fromPretrained` uses `fetch()` internally, so it works in both browser and Node.js 18+
environments.

### Encode and decode

```js
// Single
const tokens = tok.encode("hello world"); // Uint32Array
const text = tok.decode(tokens); // string

// Batch
const results = tok.encodeBatch(["hello", "world"]); // Uint32Array[]
const texts = tok.decodeBatch(results); // string[]
```

### Vocabulary inspection

```js
tok.vocabSize; // 100256
tok.maxToken; // 100255 (or null)
tok.tokenToId("hello"); // 15339 (or null)
tok.idToToken(15339); // "hello" (or null)
tok.getSpecialTokens(); // [["<|endoftext|>", 100257], ...]
Tokenizer.availableModels(); // ["r50k_base", "p50k_base", ...]
```

### Memory management

WASM objects must be freed manually. Call `tok.free()` when you're done with a tokenizer to release
its WASM memory.

### Building from source

Requires [Rust](https://rustup.rs/), [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/),
and Node.js:

```bash
# Build the WASM package
wasm-pack build bindings/wasm --target web

# Build the TypeScript wrapper
cd bindings/wasm/js
npm install
npm run build
```

### Examples

Working examples are included in the repository:

- **Node.js:** `examples/wasm-node/` - Encode/decode from a Node.js script
- **Browser:** `examples/wasm-browser/` - In-browser tokenization with a simple HTML page
- **Live demo:** [Interactive Tokenizer Demo](./interactive-tokenizer.md) - Try it directly in this book

## API comparison

All three produce identical token sequences for the same input and model.

**Rust**

- Load: `load_vocab("cl100k_base", &mut cache)`
- Encode: `tok.try_encode(text)`
- Decode: `tok.try_decode_to_string(&tokens)`
- Batch: `tok.try_encode_batch(&texts)`
- Vocab size: `tok.vocab().len()`
- Special tokens: `tok.special_vocab().span_map()`

**Python**

- Load: `Tokenizer.from_pretrained("cl100k_base")`
- Encode: `tok.encode(text)`
- Decode: `tok.decode(tokens)`
- Batch: `tok.encode_batch(texts)`
- Vocab size: `tok.vocab_size`
- Special tokens: `tok.get_special_tokens()`

**JavaScript**

- Load: `await Tokenizer.fromPretrained("cl100k_base")`
- Encode: `tok.encode(text)`
- Decode: `tok.decode(tokens)`
- Batch: `tok.encodeBatch(texts)`
- Vocab size: `tok.vocabSize`
- Special tokens: `tok.getSpecialTokens()`
