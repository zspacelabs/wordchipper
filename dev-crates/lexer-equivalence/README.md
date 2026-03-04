# lexer-equivalence

Combinatorial equivalence testing for the logos DFA lexers against the regex reference
implementations in wordchipper.

## Why this matters

Wordchipper's logos-based lexers replace regex-based spanning with hand-crafted DFA token
definitions. This delivers 30-50x faster spanning throughput, which directly translates to higher
end-to-end encoding speed in published benchmarks.

However, logos DFA semantics differ from regex in ways that can silently produce wrong span
boundaries:

- **Longest-match vs first-match**: logos always picks the longest DFA match, while the original
  regex patterns use leftmost-first alternation. A single misplaced priority or character class can
  cause tokens to absorb extra bytes.
- **No lookahead/lookbehind**: the regex patterns use `\s+(?!\S)` for whitespace splitting, which
  has no DFA equivalent and must be handled in post-processing.
- **Unicode category overlap**: `\p{M}` (Mark) appears in both word and punctuation character
  classes in o200k, creating ambiguous DFA states that logos resolves differently than regex.

If the lexers diverge from regex on any input, the benchmark numbers are not comparable because the
two code paths tokenize different spans. This crate exists to prove they don't.

## How it works

The three OpenAI patterns (r50k, cl100k, o200k) use a finite set of Unicode predicates (`\p{Lu}`,
`\p{Ll}`, `\p{N}`, `\s`, etc.) that partition the Unicode space into 22 equivalence cells. Two
codepoints in the same cell are indistinguishable to every regex predicate in the patterns, so
testing one representative per cell covers the full Unicode space.

The test generates all k-character strings (k=1..4) from 29 representative codepoints (22 cells + 7
sub-cell extras for logos DFA edge cases) and compares the span output of each logos lexer against
the regex reference.

### Tests

Each lexer (r50k, cl100k, o200k) has a single equivalence test that runs all 29 representatives at
k=1..4 and panics on any divergence from the regex reference.

### Representative validation

The `validate_representative_completeness` test programmatically verifies that the representative
set covers every equivalence cell by evaluating all 21 regex predicates against every Unicode
codepoint and checking that no uncovered bit-signature exists.

## Running

```
cargo test -p lexer-equivalence
```

The full suite tests ~732,540 inputs per lexer (29^1 + 29^2 + 29^3 + 29^4) and runs in under 10
seconds.
