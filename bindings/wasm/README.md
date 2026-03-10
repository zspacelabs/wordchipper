# wordchipper (WASM)

WebAssembly bindings for the [wordchipper](https://github.com/zspacelabs/wordchipper) BPE tokenizer
library. Works in browsers and Node.js.

## Quick Start

```js
import { Tokenizer } from "./js/dist/index.js";

const tok = await Tokenizer.fromPretrained("o200k_base");

const tokens = tok.encode("hello world"); // Uint32Array [24912, 2375]
const text = tok.decode(tokens); // "hello world"

tok.free();
```

## API

### Loading

```js
// Fetch vocab from OpenAI CDN
const tok = await Tokenizer.fromPretrained("cl100k_base");

// Or from your own vocab bytes
const data = new Uint8Array(/* .tiktoken file bytes */);
const tok = await Tokenizer.fromVocabData("cl100k_base", data);
```

### Encode / Decode

```js
const tokens = tok.encode("hello world"); // Uint32Array
const text = tok.decode(tokens); // string

// Batch
const results = tok.encodeBatch(["hello", "world"]); // Uint32Array[]
const texts = tok.decodeBatch(results); // string[]
```

### Vocab Inspection

```js
tok.vocabSize; // 100256
tok.maxToken; // 100255 (or null)
tok.tokenToId("hello"); // 15339 (or null)
tok.idToToken(15339); // "hello" (or null)
tok.getSpecialTokens(); // [["<|endoftext|>", 100257], ...]
Tokenizer.availableModels(); // ["r50k_base", "p50k_base", ...]
```

### Cleanup

```js
tok.free(); // release WASM memory
```

## Available Models

| Model           | Description                               |
|-----------------|-------------------------------------------|
| `r50k_base`     | GPT-2                                     |
| `p50k_base`     | Codex                                     |
| `p50k_edit`     | Codex (edit, shares p50k_base vocab)      |
| `cl100k_base`   | GPT-3.5 / GPT-4                           |
| `o200k_base`    | GPT-4o                                    |
| `o200k_harmony` | GPT-4o (harmony, shares o200k_base vocab) |

## Building

Requires [Rust](https://rustup.rs/), [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/),
and Node.js.

All commands below run from the repository root.

```bash
# Build the WASM package
wasm-pack build bindings/wasm --target web

# Build the TypeScript wrapper
cd bindings/wasm/js
npm install
npm run build
```

## Architecture

The WASM crate uses `wordchipper` with `default-features = false` (no `std`, no `rayon`, no
`download`). All core types are available: vocab construction, encoding, decoding, and the logos DFA
lexer.

Since `vocab::io` (file-based loading) is `std`-gated, the WASM crate includes its own tiktoken
parser that works from `&[u8]`. The TypeScript wrapper's `fromPretrained()` fetches vocab data via
`fetch()` and passes the raw bytes to the WASM side.

## Examples

See [`examples/wasm-node/`](../../examples/wasm-node/) and
[`examples/wasm-browser/`](../../examples/wasm-browser/).
