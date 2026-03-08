# Feature Flags

These are the Cargo features available on `wordchipper`. Features can be enabled in your
`Cargo.toml`:

```toml
[dependencies]
wordchipper = { version = "0.8", features = ["client"] }
```

The default features are `std`, `fast-hash`, and `parallel`.

---

#### features = ["std"]

*Enabled by default.*

Provide standard library integration: regex-based spanning via `regex` and `fancy-regex`, file I/O,
and `std::collections::HashMap`. Building with `default-features = false` removes the standard
library dependency, leaving a pure `no_std` crate that uses `hashbrown` for hash maps and logos DFA
lexers for pre-tokenization.

#### features = ["fast-hash"]

*Enabled by default.*

Use `foldhash` for faster hash maps. When combined with `std`, the standard library's `HashMap` is
used with foldhash's hasher. Without `std`, `hashbrown::HashMap` is used with foldhash's hasher.
This feature has no `std` requirement, so it works in `no_std` environments.

#### features = ["parallel"]

*Enabled by default. Implies `concurrent`.*

Enable rayon-based parallelism for batch encoding and decoding. Control the thread count with the
`RAYON_NUM_THREADS` environment variable.

#### features = ["concurrent"]

*Implies `std`.*

Enable the thread pool (`PoolToy`), pooled `regex-automata` caches, and concurrency utilities used
for concurrent encoder access. The `regex-automata` spanning backend is always available (even
without this feature, using a single mutex cache), but `concurrent` adds thread-distributed cache
pools for ~4-8x faster spanning under multi-threaded workloads. The `parallel` feature enables this
automatically; use `concurrent` directly if you want the thread pool without pulling in rayon.

#### features = ["client"]

*Implies `download` and `datagym` (both of which imply `std`).*

Everything needed to load pretrained vocabularies: downloading from the network and parsing
DataGym-format files.

#### features = ["download"]

*Implies `std`.*

Download and cache vocabulary files from the internet.

#### features = ["datagym"]

*Implies `std`.*

Load DataGym-format vocabularies (used by older OpenAI models like GPT-2). Pulls in `serde_json`.

#### features = ["tracing"]

Add `tracing` instrumentation points throughout the encoding pipeline. Only useful for profiling
the library itself.

#### features = ["testing"]

Export test utilities for downstream crates to use in their own test suites.
