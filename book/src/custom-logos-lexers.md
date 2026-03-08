# Building Custom Logos Lexers

> This chapter assumes familiarity with the two-phase pipeline (spanning + BPE encoding) described
> in [How Tokenizers Work](./how-tokenizers-work.md).

BPE tokenizers split text in two phases: first into words (pre-tokenization), then each word into
subword tokens. The first phase is typically a big regex. OpenAI's `cl100k_base` pattern, for
example, uses alternations with Unicode property classes and lookaheads to segment "hello world"
into `["hello", " world"]`.

Regex is correct but slow. Each match backtracks through Unicode property tables. wordchipper has
three spanning tiers: fancy-regex (fallback), `regex-automata` (middle tier, ~4-8x faster), and
logos DFA (fastest, ~14-21x faster). The [logos](https://logos.maciej.codes/) crate compiles regex
patterns into a deterministic finite automaton (DFA) at build time via a derive macro. No
backtracking, no runtime regex compilation. For wordchipper's cl100k and o200k patterns, this gives
**14-21x speedups** over regex (250-300 MB/s single-threaded). See
[Performance](./performance.md) for full benchmarks across all three tiers.

But logos DFA can't express everything a regex can. The OpenAI patterns use `\s+(?!\S)`, a negative
lookahead that backtracks so the last whitespace character becomes a prefix of the next word. Logos
has no lookaheads. So we need a post-processing step that corrects the token stream after the DFA
runs.

This post-processing is extracted into composable, public building blocks. You supply the patterns.
We handle the rest.

## When to use this

The building blocks in this chapter are designed for tokenizers that use OpenAI-style regex
pre-tokenization with the `\s+(?!\S)` lookahead idiom. This includes:

- **cl100k_base** (GPT-4, GPT-3.5-turbo)
- **o200k_base** (GPT-4o)
- **p50k_base / p50k_edit** (GPT-3, Codex)
- **r50k_base** (GPT-2)
- Any custom tokenizer that copies the OpenAI regex structure

Tokenizers with fundamentally different pre-tokenization don't need this machinery:

- **SentencePiece** (Llama, Gemini) replaces spaces with `▁` before tokenization. No regex
  pre-tokenization.
- **Byte-level BPE** (GPT-NeoX, Falcon) uses HuggingFace's `ByteLevel` pre-tokenizer.
- **Bert-style** tokenizers split on whitespace and punctuation with simple rules, no lookaheads.

If your tokenizer's regex pattern doesn't use `\s+(?!\S)` or a similar whitespace-backtracking
idiom, you can still use logos for DFA speed, but you won't need `Gpt2FamilyTokenRole` or
`for_each_classified_span`. Just implement `SpanLexer` directly.

## The whitespace problem, concretely

To understand why we need post-processing, consider what happens with the input `"hello   world"`.

The regex `\s+(?!\S)` matches whitespace greedily, then backtracks one character. So `"   "` (three
spaces) becomes `"  "` (two spaces as one token) + `" world"` (the last space merges into the next
word). This is how OpenAI's patterns work: whitespace "leaks" into the following word.

Logos has no lookaheads. Its DFA matches `"   "` as a single `Whitespace` token and `"world"` as a
separate `Letters` token. Without correction, you'd get different spans than the regex.

The post-processing engine fixes this. It buffers whitespace tokens and, when the next token
arrives, decides how to split them. The rules depend on what kind of token follows:

- **A letter token** (`world`): split off the last whitespace character and merge it with the word.
  Result: `"  "` + `" world"`. Matches the regex.
- **A punctuation token** (`!`): if the last whitespace character is an ASCII space, merge it with
  the punctuation (matching the ` ?` prefix in patterns like ` ?[^\s\p{L}\p{N}]+`).
- **A digit or newline**: emit the whitespace as its own span. No merging.

You don't need to implement any of this. You just tell the engine what kind of token each logos
variant represents, and it applies the correct rule.

## The building blocks

Three public items in `wordchipper::spanners::span_lexers::logos::gpt2_family`:

### `Gpt2FamilyTokenRole`

An enum that classifies how each logos token interacts with preceding whitespace:

```rust
pub enum Gpt2FamilyTokenRole {
    Whitespace,
    Punctuation,
    Word { check_contraction: bool, first_char_is_letter: bool },
    Standalone,
    Newline,
    Gap,
}
```

Each variant tells the engine a different whitespace rule:

- **`Whitespace`**: this token _is_ whitespace. The engine buffers it and decides later how to split
  it based on what comes next.
- **`Punctuation`**: absorbs a preceding ASCII space. The engine merges the last buffered space
  character into this token's span (matching the ` ?` regex prefix).
- **`Word`**: absorbs a preceding space _if `first_char_is_letter` is true_. If the token starts
  with a non-letter prefix (like `"hello` where `"` is the first char), set `first_char_is_letter`
  to false and the engine handles the prefix separately. The `check_contraction` field enables
  cl100k-style splitting where `'The` becomes `'T` + `he`.
- **`Standalone`**: never absorbs whitespace. Digits, explicit contractions. Any preceding
  whitespace becomes its own span.
- **`Newline`**: newline-containing whitespace (`\s*[\r\n]+`). Buffered separately; at end of
  string, adjacent Newline + Whitespace merge (matching the regex `\s++$` behavior).
- **`Gap`**: unrecognized bytes. Use this for logos `Err(())`.

### `Gpt2FamilyLogos` trait

Maps a logos token enum to `Gpt2FamilyTokenRole`:

```rust
pub trait Gpt2FamilyLogos<'a>: Logos<'a> {
    fn family_role(&self) -> Gpt2FamilyTokenRole;
}
```

Implement this for your token enum and the engine handles all the post-processing.

### `contraction_split`

An optional utility for cl100k-compatible lexers. Logos longest-match picks `'The` as one token, but
cl100k's regex first-match picks `'T` (contraction) then `he` (letters). This function detects the
contraction prefix and returns the split point.

Most custom lexers won't need this. Set `check_contraction: false` and ignore it.

## Building a custom lexer: step by step

Let's build a lexer from scratch. We'll target a simplified pattern that handles letters, digits,
punctuation, and whitespace.

### Step 1: Define the logos token enum

```rust
use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
enum MyToken {
    #[regex(r"\p{Letter}+")]
    Letters,

    #[regex(r"\p{Number}{1,3}")]
    Digits,

    #[regex(r" ?[^\s\p{Letter}\p{Number}]+[\r\n]*")]
    Punctuation,

    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}
```

Each variant maps to a regex fragment. Logos compiles all of them into a single DFA at build time.

### Step 2: Implement `Gpt2FamilyLogos`

Map each token variant to its role. This is where you make design decisions: for each token type,
ask "How should this interact with preceding whitespace?"

```rust
use wordchipper::spanners::span_lexers::logos::gpt2_family::{
    Gpt2FamilyLogos, Gpt2FamilyTokenRole,
};

impl Gpt2FamilyLogos<'_> for MyToken {
    fn family_role(&self) -> Gpt2FamilyTokenRole {
        match self {
            // Whitespace is buffered; last char may merge into next token.
            Self::Whitespace => Gpt2FamilyTokenRole::Whitespace,

            // Letters absorb a preceding space when the token starts with
            // a letter. No contraction splitting needed for our pattern.
            Self::Letters => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
                first_char_is_letter: true,
            },

            // Punctuation absorbs a preceding ASCII space (the ` ?` prefix).
            Self::Punctuation => Gpt2FamilyTokenRole::Punctuation,

            // Newlines are buffered separately.
            Self::Newline => Gpt2FamilyTokenRole::Newline,

            // Digits stand alone. They never merge with preceding whitespace.
            Self::Digits => Gpt2FamilyTokenRole::Standalone,
        }
    }
}
```

The key insight: you don't need to understand the whitespace-splitting algorithm. You just need to
know what each token _is_, and `Gpt2FamilyTokenRole` maps that to the correct behavior.

### Step 3: Use the `logos_lexer!` macro

The `logos_lexer!` macro generates the `SpanLexer` impl and registers the lexer with inventory for
automatic acceleration:

```rust
logos_lexer! {
    /// Logos DFA word scanner for my custom pattern.
    pub struct MyLexer;
    token = MyToken;
    pattern = MY_PATTERN;
}
```

This generates a struct that implements `SpanLexer`, which returns an iterator of word byte ranges.
The macro also registers the lexer via `inventory` so it automatically replaces the regex spanner
when the pattern matches.

If you need a manual `SpanLexer` impl instead (e.g. for a non-standard pattern), you can call
`logos_span_iter` directly:

```rust
use wordchipper::spanners::span_lexers::{SpanLexer, logos::gpt2_family::logos_span_iter};

impl SpanLexer for MyLexer {
    fn find_span_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> Box<dyn Iterator<Item = Range<usize>> + 'a> {
        Box::new(logos_span_iter(text, MyToken::lexer(text).spanned()))
    }
}
```

### The real thing: cl100k in ~65 lines

The built-in `Cl100kLexer` follows exactly this pattern. The token enum has 7 variants. The
`Gpt2FamilyLogos` impl is 15 lines. The `SpanLexer` impl is generated by `logos_lexer!`. You can
read the full source at `crates/wordchipper/src/spanners/span_lexers/logos/cl100k.rs`.

## Gpt2FamilyTokenRole reference

| Variant                                            | Absorbs preceding whitespace?  | Use for                                                     |
| -------------------------------------------------- | ------------------------------ | ----------------------------------------------------------- |
| `Whitespace`                                       | N/A (is whitespace)            | Horizontal whitespace tokens (`[^\S\r\n]+`)                 |
| `Punctuation`                                      | Yes, ASCII space only          | Punctuation with ` ?` prefix (` ?[^\s\p{L}\p{N}]+`)         |
| `Word { check_contraction, first_char_is_letter }` | Yes, if `first_char_is_letter` | Letter/word tokens (`\p{L}+`, case-sensitive word patterns) |
| `Standalone`                                       | No                             | Digits, explicit contractions, anything that stands alone   |
| `Newline`                                          | N/A (buffered separately)      | Newline-containing whitespace (`\s*[\r\n]+`)                |
| `Gap`                                              | No                             | Unrecognized bytes (logos errors). Always use for `Err(())` |

**When in doubt**, use `Standalone`. It's the safest default: the token is emitted as-is, and any
preceding whitespace becomes its own span.

## Testing your lexer

A logos lexer must produce identical span boundaries to the regex it replaces. If they diverge on
any input, benchmark comparisons are invalid because the two code paths tokenize different spans.

### Equivalence testing with `lexer-equivalence`

The `lexer-equivalence` crate in `dev-crates/` provides exhaustive combinatorial testing. It
generates all k-character strings (k=1..4) from a set of Unicode representative codepoints and
compares the span output of each logos lexer against the regex reference.

The representative set is derived from the Unicode predicates used in the OpenAI patterns (`\p{Lu}`,
`\p{Ll}`, `\p{N}`, `\s`, etc.), which partition Unicode into 22 equivalence cells. Two codepoints in
the same cell are indistinguishable to every regex predicate, so testing one representative per cell
covers the full Unicode space. The test suite checks ~732,540 inputs per lexer and runs in under 10
seconds:

```
cargo test -p lexer-equivalence
```

The `validate_representative_completeness` test programmatically verifies that the representative
set covers every equivalence cell, so you can be confident the coverage is complete.

### Adding tests for a custom lexer

To test a custom lexer against its regex, add a test that calls `assert_k_tuple_equivalence` from
the `lexer_equivalence::harness` module with your lexer and the regex pattern it targets. See
`dev-crates/lexer-equivalence/tests/equivalence.rs` for the pattern used by the built-in lexers.
