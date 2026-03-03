# Known logos lexer divergences

Three divergence classes found and fixed via `cargo test -p lexer-equivalence`.

## r50k: contraction after space (5 cases, 1 class)

Pattern: `[any_ws] [space] [apostrophe] [contraction_letter]`

Regex matches `" '"` as punctuation via ` ?[^\s\p{L}\p{N}]++`, leaving `s` as a
separate token. Logos matches `'s` as a Contraction token, leaving `" "` standalone.

All 5 cases are the same bug with different leading whitespace (space, NBSP, tab, CR, LF).

Fix in: `crates/wordchipper/src/spanners/span_lexers/logos/r50k.rs`
and post-processing in `gpt2_family.rs`.

## cl100k: two sub-classes (6,253 cases, 18 classes)

### CR/LF + trailing whitespace

Regex groups `"\r "` as one span via `\s+`. Logos splits into separate Newline and
Whitespace tokens.

### NBSP/tab + Mark + letter

Regex keeps non-space whitespace and Mark+letter separate. Logos post-processing
incorrectly merges them when multi-char whitespace ends with a non-space ws char.

Fix in: `crates/wordchipper/src/spanners/span_lexers/logos/cl100k.rs`
and post-processing in `gpt2_family.rs`.

## o200k: Mark absorption (48,366 cases, 37 classes)

Regex treats standalone Mark chars as their own span. Logos DFA longest-match merges
Mark into the following token because `\p{M}` is in the o200k word pattern
`[\p{Ll}\p{Lm}\p{Lo}\p{M}]`, and DFA always takes longest-match rather than
leftmost-first.

Fix in: `crates/wordchipper/src/spanners/span_lexers/logos/o200k.rs`,
splitting Punctuation into three DFA variants (PunctuationSpaced, PunctuationBare,
PunctuationBareMark) and excluding `\p{M}` from word prefix entrypoints.
