# How Tokenizers Work

This chapter explains the algorithms behind BPE tokenization. If you already know how BPE works,
skip to [Pretrained Models](./pretrained-models.md). To see tokenization in action, try the
[Interactive Tokenizer Demo](./interactive-tokenizer.md).

## The problem: turning text into numbers

Neural networks operate on fixed-size numerical inputs. A language model needs a way to convert
arbitrary text into a sequence of integers from a finite vocabulary. The simplest approaches have
obvious downsides:

- **Character-level:** Vocabulary is small (~256 for ASCII, ~150k for full Unicode), but sequences
  are long. `"tokenization"` becomes 12 tokens. Long sequences are expensive for attention
  mechanisms.
- **Word-level:** `"tokenization"` is one token, but the vocabulary must include every possible
  word. Rare words, typos, and new terms become `<UNK>` (unknown). Multilingual text explodes the
  vocabulary.
- **Subword:** Split text into pieces that balance vocabulary size against sequence length.
  `"tokenization"` might become `["token", "ization"]`. Common words stay whole. Rare words
  decompose into reusable parts.

BPE is a subword tokenization algorithm. It's used by GPT-2 through GPT-4o, and it's what
wordchipper implements.

## Byte Pair Encoding (BPE)

BPE was originally a data compression algorithm (Gage, 1994). The NLP adaptation (Sennrich et
al., 2016) works in two phases: training and encoding.

### Training: building the vocabulary

Training starts with a base vocabulary of individual bytes (0-255) and iteratively discovers the
most useful multi-byte tokens. In practice, the training corpus is first split into spans by a regex
pattern (see [Pre-tokenization](#pre-tokenization-splitting-before-bpe) below), and pair counting
happens within each span independently. The regex determines what token boundaries are possible. The
steps below describe the core merge algorithm that runs on those spans.

**Step 1.** Start with every unique byte as a token. The vocabulary has 256 entries.

**Step 2.** Count every adjacent pair of tokens across all spans. Find the most frequent pair.

**Step 3.** Create a new token that represents that pair merged together. Add it to the vocabulary.
Record the merge rule: `(token_a, token_b) -> new_token`.

**Step 4.** Replace all occurrences of that pair in the corpus with the new token.

**Step 5.** Repeat from Step 2 until the vocabulary reaches the desired size.

#### Example: training on "aabaa aab"

Start with bytes: `a=0, b=1, space=2`

```text
Corpus: [a, a, b, a, a, ' ', a, a, b]

Round 1: Most frequent pair is (a, a), count=3
  Merge: (a, a) -> aa (token 3)
  Corpus becomes: [aa, b, aa, ' ', aa, b]

Round 2: Most frequent pair is (aa, b), count=2
  Merge: (aa, b) -> aab (token 4)
  Corpus becomes: [aab, aa, ' ', aab]

Round 3: Most frequent pair is (aab, aa), count=1
  (tie with others; pick first)
  ...
```

After training, the vocabulary contains bytes plus all discovered merges, and the merge table
records the order. The rank of a merge (its position in the table) determines priority during
encoding.

### Encoding: applying the merges

Given a trained vocabulary, encoding a string works as follows:

**Step 1.** Convert the input to a sequence of byte tokens.

**Step 2.** Find all adjacent pairs. Look up each pair in the merge table.

**Step 3.** Merge the pair with the lowest rank (highest priority). This is the pair that was
learned earliest during training, meaning it's the most frequent.

**Step 4.** Repeat from Step 2 until no more merges are possible.

#### Example: encoding "aab"

Using the vocabulary from above:

```text
Start:    [a, a, b]       (3 byte tokens)
Pairs:    (a,a)=rank 0    (a,b)=not in table
Merge:    [aa, b]         (merge (a,a) at rank 0)
Pairs:    (aa,b)=rank 1
Merge:    [aab]           (merge (aa,b) at rank 1)
No pairs: done.
Result:   [4]             (token 4 = "aab")
```

The key insight: BPE encoding is deterministic. Given the same vocabulary and merge table, the same
input always produces the same tokens. This is why wordchipper can be validated against tiktoken and
produce identical output.

## Pre-tokenization: splitting before BPE

Running BPE directly on raw text would be wasteful. The string `"hello world"` would start as
individual bytes and need many merge steps. Worse, the space between words could merge with the
surrounding characters, creating tokens that span word boundaries (like `"o w"`), which hurts the
model's ability to generalize.

The solution is **pre-tokenization** (also called **spanning** in wordchipper). Before BPE runs, the
text is split into coarse segments using a regex pattern. Each segment is then encoded
independently.

OpenAI's `cl100k_base` uses this regex pattern:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)
|[^\r\n\p{L}\p{N}]?\p{L}+
|\p{N}{1,3}
| ?[^\s\p{L}\p{N}]+
|\s+(?!\S)
|\s+
```

This splits text into:

- Contractions: `'s`, `'t`, `'re`, etc.
- Words (letters), optionally preceded by a non-letter character
- Numbers, up to 3 digits at a time
- Punctuation sequences, optionally preceded by a space
- Whitespace (with a special rule for trailing space)

For example, `"Hello, world! 123"` becomes:

```text
["Hello", ",", " world", "!", " 123"]
```

Each of these spans is then BPE-encoded independently. This keeps word boundaries clean and makes
encoding much faster (each span is short).

### The whitespace rule

The pattern `\s+(?!\S)` is worth understanding. It matches whitespace greedily, then uses a negative
lookahead to back off one character. So `"   world"` (3 spaces + "world") becomes:

```text
["  ", " world"]
```

The last space "leaks" into the next word. This is an intentional design choice by OpenAI: it means
the model sees leading-space tokens as word starts, improving prediction.

This lookahead is straightforward in a regex engine but impossible in a DFA (deterministic finite
automaton), which is why wordchipper's logos-based lexers need a post-processing step. See
[Building Custom Logos Lexers](./custom-logos-lexers.md) for details.

## The wordchipper pipeline

Putting it all together, here's what happens when you call `try_encode("Hello, world!")`:

```text
Input:     "Hello, world!"
             |
             v
  ┌─────────────────────┐
  │   1. Pre-tokenize   │  (regex or DFA lexer)
  │   Split into spans  │
  └──────────┬──────────┘
             |
   ["Hello", ",", " world", "!"]
             |
             v
  ┌─────────────────────┐
  │   2. BPE encode     │  (per-span, possibly parallel)
  │   Merge byte pairs  │
  └──────────┬──────────┘
             |
   [9906, 11, 1917, 0]
             |
             v
         Token IDs
```

In wordchipper's architecture:

- **Step 1** is handled by a `TextSpanner` (the `spanners` module). The default uses regex. For
  known patterns (cl100k, o200k, r50k), a compile-time DFA lexer is used automatically for 30-50x
  faster spanning.
- **Step 2** is handled by a `SpanEncoder` (the `encoders` module). Multiple BPE algorithms are
  available, optimized for different use cases.

The `Tokenizer` type combines both steps behind a single `try_encode` / `try_decode` interface.

## Decoding: tokens back to text

Decoding is simpler than encoding. Each token ID maps to a byte sequence in the vocabulary. The
decoder concatenates all byte sequences and interprets the result as UTF-8:

```text
[9906,   11,  1917, 0]
  |      |    |     |
  v      v    v     v
"Hello" "," " world" "!"
  |
  v
"Hello, world!"
```

No merging or splitting needed. Decoding is always O(n) in the number of tokens.

## Other tokenization approaches

BPE is not the only subword algorithm. Two others are worth knowing:

### WordPiece (BERT)

WordPiece is similar to BPE but uses a different training objective. Instead of merging the most
frequent pair, it merges the pair that maximizes the likelihood of the training data. Subword pieces
after the first are prefixed with `##` (e.g., `"token" + "##ization"`).

Used by: BERT, DistilBERT, ELECTRA.

### SentencePiece / Unigram (Llama, Gemini)

SentencePiece treats the input as a raw byte stream and replaces spaces with a special `\u2581`
character before tokenization. The Unigram variant starts with a large vocabulary and prunes it down
based on a unigram language model.

Used by: Llama, Gemini, T5, ALBERT.

wordchipper focuses on BPE tokenizers with OpenAI-style regex pre-tokenization. Tokenizers that use
SentencePiece or WordPiece have fundamentally different pre-tokenization and don't need the spanning
machinery described in this book.

## Further reading

- Sennrich et al. (2016), "Neural Machine Translation of Rare Words with Subword Units" (the paper
  that adapted BPE for NLP)
- OpenAI's `tiktoken` repository for the reference regex patterns
- The [Performance](./performance.md) chapter for how wordchipper optimizes BPE encoding
