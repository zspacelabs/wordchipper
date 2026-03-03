//! # Implementation Utils for GPT-2 family regex patterns.
//!
//! Classification of logos tokens for whitespace post-processing.
//!
//! When building a custom accelerated lexer, each logos token variant maps
//! to a [`Gpt2FamilyTokenRole`] that tells the post-processing engine how the token
//! interacts with preceding whitespace.

use core::ops::Range;

use logos::{Logos, SpannedIter};
use ringbuf::traits::{Consumer, Producer};

use crate::spanners::SpanRef;

/// How a logos token interacts with whitespace splitting.
///
/// The `OpenAI` regex patterns use `\s+(?!\S)` which backtracks so the last
/// whitespace character can be absorbed as a prefix by the next pattern
/// (e.g. `[^\r\n\p{L}\p{N}]?\p{L}+`). Logos DFA can't express lookaheads,
/// so we post-process the token stream: when a [`Whitespace`](Self::Whitespace)
/// token precedes certain token kinds, the last character merges into the
/// next span; before other tokens, it becomes a standalone word.
///
/// # Example
///
/// ```
/// use wordchipper::spanners::span_lexers::logos::gpt2_family::Gpt2FamilyTokenRole;
///
/// // Map your logos token to a role:
/// let role = Gpt2FamilyTokenRole::Word {
///     check_contraction: false,
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gpt2FamilyTokenRole {
    /// Horizontal whitespace. Buffered; last char may split to next token.
    Whitespace,
    /// Punctuation with ` ?` prefix. Always absorbs a preceding ASCII space.
    Punctuation,
    /// Letter/word token. Absorbs preceding space if token starts with a letter.
    /// When `check_contraction` is true, applies contraction-prefix splitting
    /// (e.g. `'The` -> `'T` + `he` for cl100k compatibility).
    Word {
        /// Whether to check for and split contraction prefixes.
        check_contraction: bool,
    },
    /// Token that never absorbs whitespace (digits, contractions, newlines).
    Standalone,
    /// Unrecognized bytes.
    Gap,
}

/// Public trait for DFA's which have a `TokenRole`.
pub trait Gpt2FamilyLogos<'a>: Logos<'a> {
    /// The role of this token.
    fn family_role(&self) -> Gpt2FamilyTokenRole;
}

/// Check if a byte slice starts with a cl100k contraction pattern
/// (`'s`, `'t`, `'d`, `'m`, `'re`, `'ve`, `'ll`, case-insensitive)
/// followed by additional bytes. Returns the split point
/// (contraction length) if there are trailing bytes after the
/// contraction, or `None` if the input is just the contraction alone.
///
/// This function does not verify that trailing bytes are letters;
/// callers are expected to only pass word/letter tokens (as the
/// engine does via `TokenRole::Word`).
///
/// This is useful when building cl100k-compatible lexers where logos
/// longest-match picks `'The` as one Letters token, but the regex
/// first-match would pick Contraction `'T` then Letters `he`.
///
/// # Examples
///
/// ```
/// use wordchipper::spanners::span_lexers::logos::gpt2_family::contraction_split;
///
/// assert_eq!(contraction_split(b"'There"), Some(2)); // split after 'T
/// assert_eq!(contraction_split(b"'llama"), Some(3)); // split after 'll
/// assert_eq!(contraction_split(b"'t"), None); // just 't, nothing after
/// assert_eq!(contraction_split(b"'re"), None); // just 're, nothing after
/// assert_eq!(contraction_split(b"hello"), None); // no apostrophe
/// ```
pub fn contraction_split(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 3 || bytes[0] != b'\'' {
        return None;
    }
    let c1 = bytes[1];
    // Single-char suffixes: 's, 't, 'd, 'm
    if matches!(c1, b's' | b'S' | b't' | b'T' | b'd' | b'D' | b'm' | b'M') {
        return (bytes.len() > 2).then_some(2);
    }
    // Two-char suffixes: 're, 've, 'll.
    // Need >= 4 bytes: apostrophe + 2-char suffix + at least 1 trailing letter.
    // A 3-byte input like 're is a standalone contraction, not a split candidate.
    if bytes.len() >= 4 {
        let c2 = bytes[2];
        let is_two = matches!(
            (c1, c2),
            (b'r' | b'R', b'e' | b'E') | (b'v' | b'V', b'e' | b'E') | (b'l' | b'L', b'l' | b'L')
        );
        if is_two {
            return (bytes.len() > 3).then_some(3);
        }
    }
    None
}

/// `.next_span()` iterator for `Gpt2FamilyLogos`.
///
/// Uses a small inline ring buffer (capacity 3) instead of a heap-allocated
/// `VecDeque`. The maximum spans emitted per token is 4 (`flush_ws_split` +
/// merge + contraction split); the first is returned directly, so 3 stash
/// slots suffice.
pub struct Gpt2FamilySpanIter<'source, Token>
where
    Token: Gpt2FamilyLogos<'source>,
{
    text: &'source str,
    last: usize,
    pending_ws: Option<Range<usize>>,

    ring: ringbuf::StaticRb<Range<usize>, 3>,

    iter: Option<SpannedIter<'source, Token>>,
}

impl<'source, Token> Gpt2FamilySpanIter<'source, Token>
where
    Token: Gpt2FamilyLogos<'source>,
{
    /// Create a new iterator.
    pub fn new(
        text: &'source str,
        iter: SpannedIter<'source, Token>,
    ) -> Self {
        Self {
            text,
            last: 0,
            pending_ws: None,
            ring: Default::default(),
            iter: Some(iter),
        }
    }

    fn push(
        &mut self,
        range: Range<usize>,
    ) {
        self.ring.try_push(range).unwrap();
    }

    fn pop(&mut self) -> Option<Range<usize>> {
        self.ring.try_pop()
    }

    fn next_tok(&mut self) -> Option<(Gpt2FamilyTokenRole, Range<usize>)> {
        if let Some(iter) = &mut self.iter {
            if let Some((res, range)) = iter.next() {
                let role = match res {
                    Ok(tok) => tok.family_role(),
                    Err(_) => Gpt2FamilyTokenRole::Gap,
                };
                return Some((role, range));
            }
            self.iter = None;
        }
        None
    }
}

impl<'source, Token> Iterator for Gpt2FamilySpanIter<'source, Token>
where
    Token: Gpt2FamilyLogos<'source>,
{
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();

        macro_rules! emit {
            ($r:expr) => {
                self.push($r)
            };
        }

        macro_rules! flush_ws_split {
            ($ws:expr) => {{
                let ws = $ws;
                debug_assert!(!ws.is_empty(), "flush_ws_split called with empty range");
                // Find start of the last character (may be multi-byte, e.g. NBSP).
                // Safety: text is &str bytes, so valid UTF-8; the scan always finds
                // a leading byte before reaching ws.start.
                let mut trim = ws.end - 1;
                while trim > ws.start && (bytes[trim] & 0xC0) == 0x80 {
                    trim -= 1;
                }
                debug_assert!(
                    (bytes[trim] & 0xC0) != 0x80,
                    "no leading byte found in ws range"
                );
                if ws.start < trim {
                    emit!(ws.start..trim);
                }
                trim
            }};
        }

        // Emit a Letters/Word span, splitting contractions if needed.
        macro_rules! emit_absorbing {
            ($start:expr, $end:expr, $check_contraction:expr) => {
                if $check_contraction {
                    if let Some(split) = contraction_split(&bytes[$start..$end]) {
                        emit!($start..$start + split);
                        emit!($start + split..$end);
                    } else {
                        emit!($start..$end);
                    }
                } else {
                    emit!($start..$end);
                }
            };
        }

        loop {
            if let Some(range) = self.pop() {
                return Some(range);
            }
            if self.iter.is_none() {
                return self.pending_ws.take();
            }

            if let Some((role, Range { start, end })) = self.next_tok() {
                if self.last < start {
                    // We skipped over a gap.
                    // If there was a pending ws, emit it.
                    if let Some(ws) = self.pending_ws.take() {
                        emit!(ws);
                    }
                }

                self.last = end;

                match role {
                    Gpt2FamilyTokenRole::Whitespace => {
                        if let Some(ws) = self.pending_ws.take() {
                            emit!(ws);
                        }
                        self.pending_ws = Some(start..end);
                    }
                    Gpt2FamilyTokenRole::Punctuation => {
                        // Regex ` ?[^\s\p{L}\p{N}]+` absorbs a preceding ASCII
                        // space (literal ` ?`). Non-space whitespace (NBSP, tab)
                        // is NOT absorbed.
                        if let Some(ws) = self.pending_ws.take() {
                            let ws_start = ws.start;
                            let ws_end = ws.end;
                            let trim = flush_ws_split!(ws);
                            if trim == ws_start || bytes[trim] != b' ' {
                                // Single ws char, or last char is not ASCII space.
                                emit!(trim..ws_end);
                                emit!(start..end);
                            } else {
                                emit!(trim..end);
                            }
                        } else {
                            emit!(start..end);
                        }
                    }
                    Gpt2FamilyTokenRole::Word { check_contraction } => {
                        if let Some(ws) = self.pending_ws.take() {
                            let ws_start = ws.start;
                            let ws_end = ws.end;
                            let trim = flush_ws_split!(ws);
                            let single_char = trim == ws_start;

                            // Decode only the first UTF-8 char from bytes to avoid
                            // validating the entire tail (which would be O(n^2) overall).
                            let first_is_letter = {
                                let tail = &bytes[start..];
                                let char_len = match tail.first() {
                                    Some(&b) if b < 0x80 => 1,
                                    Some(&b) if b < 0xE0 => 2,
                                    Some(&b) if b < 0xF0 => 3,
                                    Some(_) => 4,
                                    None => 0,
                                };
                                char_len > 0
                                    && core::str::from_utf8(&tail[..char_len])
                                        .ok()
                                        .and_then(|s| s.chars().next())
                                        .is_some_and(char::is_alphabetic)
                            };

                            if first_is_letter {
                                // Token has no existing prefix; merge last ws char.
                                emit_absorbing!(trim, end, check_contraction);
                            } else if single_char {
                                // Single ws char: emit standalone, token as-is.
                                emit!(trim..ws_end);
                                emit_absorbing!(start, end, check_contraction);
                            } else {
                                // 2+ ws chars: merge last ws char + non-letter
                                // prefix into one span (like Punctuation ` ?X`),
                                // then emit remaining letters separately.
                                let prefix_len = core::str::from_utf8(&bytes[start..end])
                                    .expect("text is &str bytes, always valid UTF-8")
                                    .chars()
                                    .next()
                                    .map_or(1, char::len_utf8);
                                emit!(trim..start + prefix_len);
                                emit_absorbing!(start + prefix_len, end, check_contraction);
                            }
                        } else {
                            emit_absorbing!(start, end, check_contraction);
                        }
                    }
                    Gpt2FamilyTokenRole::Standalone => {
                        if let Some(ws) = self.pending_ws.take() {
                            let ws_end = ws.end;
                            let trim = flush_ws_split!(ws);
                            emit!(trim..ws_end);
                        }
                        emit!(start..end);
                    }
                    Gpt2FamilyTokenRole::Gap => {
                        if let Some(ws) = self.pending_ws.take() {
                            emit!(ws);
                        }
                    }
                }
            }
        }
    }
}

/// Iterate classified logos tokens and emit Word/Gap spans with
/// post-processing corrections for regex compatibility:
///
/// 1. **Whitespace splitting**: the regex `\s+(?!\S)` backtracks so the last
///    whitespace byte becomes a prefix of the next word. We buffer Whitespace
///    tokens and split off the last byte when followed by certain tokens.
///
/// 2. **Prefix handling**: with 2+ whitespace chars before a token starting
///    with a non-letter, we merge the last whitespace byte + the non-letter
///    prefix into one span (matching how Punctuation's ` ?` absorbs a space
///    in the regex). With 1 whitespace char, it stays standalone.
///
/// 3. **Contraction splitting** (when `check_contraction` is true): regex
///    first-match picks Contraction `'T` over Letters `'The`, but logos
///    longest-match picks Letters. We detect the contraction prefix and
///    split the token.
///
/// # Arguments
///
/// * `iter` - iterator of `(TokenRole, Range<usize>)` pairs from logos
/// * `text` - the text being scanned (ranges index into this)
/// * `offset` - byte offset added to emitted span ranges
/// * `f` - callback; return `false` to halt early
///
/// # Returns
///
/// `(completed, consumed)` where `consumed` is the byte count of accepted
/// spans and `completed` indicates all spans were accepted.
///
/// # Example
///
/// ```
/// use wordchipper::spanners::{
///     SpanRef,
///     span_lexers::logos::gpt2_family::{Gpt2FamilyTokenRole, for_each_classified_span},
/// };
///
/// let text = "hello world";
/// let tokens = vec![
///     (
///         Gpt2FamilyTokenRole::Word {
///             check_contraction: false,
///         },
///         0..5,
///     ), // "hello"
///     (Gpt2FamilyTokenRole::Whitespace, 5..6), // " "
///     (
///         Gpt2FamilyTokenRole::Word {
///             check_contraction: false,
///         },
///         6..11,
///     ), // "world"
/// ];
///
/// let mut spans = Vec::new();
/// for_each_classified_span(tokens.into_iter(), text, 0, &mut |span| {
///     spans.push(span);
///     true
/// });
///
/// assert_eq!(spans, vec![SpanRef::Word(0..5), SpanRef::Word(5..11),]);
/// ```
pub fn for_each_classified_span(
    iter: impl Iterator<Item = (Gpt2FamilyTokenRole, Range<usize>)>,
    text: &str,
    offset: usize,
    f: &mut dyn FnMut(SpanRef) -> bool,
) -> (bool, usize) {
    let text = text.as_bytes();
    let mut last = 0;
    let mut pending_ws: Option<Range<usize>> = None;

    macro_rules! emit {
        (word $r:expr) => {
            if !f(SpanRef::Word(offset + $r.start..offset + $r.end)) {
                return (false, $r.start);
            }
        };
        (gap $r:expr) => {
            if !f(SpanRef::Gap(offset + $r.start..offset + $r.end)) {
                return (false, $r.start);
            }
        };
    }

    macro_rules! flush_ws_split {
        ($ws:expr) => {{
            let ws = $ws;
            debug_assert!(!ws.is_empty(), "flush_ws_split called with empty range");
            // Find start of the last character (may be multi-byte, e.g. NBSP).
            // Safety: text is &str bytes, so valid UTF-8; the scan always finds
            // a leading byte before reaching ws.start.
            let mut trim = ws.end - 1;
            while trim > ws.start && (text[trim] & 0xC0) == 0x80 {
                trim -= 1;
            }
            debug_assert!(
                (text[trim] & 0xC0) != 0x80,
                "no leading byte found in ws range"
            );
            if ws.start < trim {
                emit!(word(ws.start..trim));
            }
            trim
        }};
    }

    // Emit a Letters/Word span, splitting contractions if needed.
    macro_rules! emit_absorbing {
        ($start:expr, $end:expr, $check_contraction:expr) => {
            if $check_contraction {
                if let Some(split) = contraction_split(&text[$start..$end]) {
                    emit!(word($start..$start + split));
                    emit!(word($start + split..$end));
                } else {
                    emit!(word($start..$end));
                }
            } else {
                emit!(word($start..$end));
            }
        };
    }

    for (kind, span) in iter {
        let Range { start, end } = span;

        if last < start {
            if let Some(ws) = pending_ws.take() {
                emit!(word ws);
            }
            emit!(gap(last..start));
        }

        last = end;

        match kind {
            Gpt2FamilyTokenRole::Whitespace => {
                if let Some(ws) = pending_ws.take() {
                    emit!(word ws);
                }
                pending_ws = Some(start..end);
            }
            Gpt2FamilyTokenRole::Punctuation => {
                // Regex ` ?[^\s\p{L}\p{N}]+` absorbs a preceding ASCII
                // space (literal ` ?`). Non-space whitespace (NBSP, tab)
                // is NOT absorbed.
                if let Some(ws) = pending_ws.take() {
                    let ws_start = ws.start;
                    let ws_end = ws.end;
                    let trim = flush_ws_split!(ws);
                    if trim == ws_start || text[trim] != b' ' {
                        // Single ws char, or last char is not ASCII space.
                        emit!(word(trim..ws_end));
                        emit!(word(start..end));
                    } else {
                        emit!(word(trim..end));
                    }
                } else {
                    emit!(word(start..end));
                }
            }
            Gpt2FamilyTokenRole::Word { check_contraction } => {
                if let Some(ws) = pending_ws.take() {
                    let ws_start = ws.start;
                    let ws_end = ws.end;
                    let trim = flush_ws_split!(ws);
                    let single_char = trim == ws_start;

                    // Decode only the first UTF-8 char from bytes to avoid
                    // validating the entire tail (which would be O(n^2) overall).
                    let first_is_letter = {
                        let tail = &text[start..];
                        let char_len = match tail.first() {
                            Some(&b) if b < 0x80 => 1,
                            Some(&b) if b < 0xE0 => 2,
                            Some(&b) if b < 0xF0 => 3,
                            Some(_) => 4,
                            None => 0,
                        };
                        char_len > 0
                            && core::str::from_utf8(&tail[..char_len])
                                .ok()
                                .and_then(|s| s.chars().next())
                                .is_some_and(char::is_alphabetic)
                    };

                    if first_is_letter {
                        // Token has no existing prefix; merge last ws char.
                        emit_absorbing!(trim, end, check_contraction);
                    } else if single_char {
                        // Single ws char: emit standalone, token as-is.
                        emit!(word(trim..ws_end));
                        emit_absorbing!(start, end, check_contraction);
                    } else {
                        // 2+ ws chars: merge last ws char + non-letter
                        // prefix into one span (like Punctuation ` ?X`),
                        // then emit remaining letters separately.
                        let prefix_len = core::str::from_utf8(&text[start..end])
                            .expect("text is &str bytes, always valid UTF-8")
                            .chars()
                            .next()
                            .map_or(1, char::len_utf8);
                        emit!(word(trim..start + prefix_len));
                        emit_absorbing!(start + prefix_len, end, check_contraction);
                    }
                } else {
                    emit_absorbing!(start, end, check_contraction);
                }
            }
            Gpt2FamilyTokenRole::Standalone => {
                if let Some(ws) = pending_ws.take() {
                    let ws_end = ws.end;
                    let trim = flush_ws_split!(ws);
                    emit!(word(trim..ws_end));
                }
                emit!(word(start..end));
            }
            Gpt2FamilyTokenRole::Gap => {
                if let Some(ws) = pending_ws.take() {
                    emit!(word ws);
                }
                emit!(gap(start..end));
            }
        }
    }

    if let Some(ws) = pending_ws.take() {
        last = ws.end;
        emit!(word ws);
    }

    if last < text.len() {
        emit!(gap(last..text.len()));
        last = text.len();
    }

    (true, last)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{alloc::vec::Vec, spanners::span_lexers::logos::gpt2_family::Gpt2FamilyTokenRole};

    /// Collect spans from for_each_classified_span for testing.
    fn collect_spans(
        tokens: impl Iterator<Item = (Gpt2FamilyTokenRole, Range<usize>)>,
        text: &str,
        offset: usize,
    ) -> Vec<SpanRef> {
        let mut spans = Vec::new();
        for_each_classified_span(tokens, text, offset, &mut |span| {
            spans.push(span);
            true
        });
        spans
    }

    /// Assert structural invariants on a span sequence over `text`:
    /// contiguous, complete coverage, UTF-8 aligned. Allows empty spans
    /// because random role assignments can produce them (real-lexer tests
    /// in cl100k/o200k use strict non-empty checks).
    fn assert_structural_invariants(
        spans: &[SpanRef],
        text: &str,
        offset: usize,
    ) -> Result<(), TestCaseError> {
        let len = text.len();
        if spans.is_empty() {
            prop_assert_eq!(len, 0, "no spans emitted for non-empty text");
            return Ok(());
        }

        prop_assert_eq!(
            spans[0].range().start,
            offset,
            "first span doesn't start at offset"
        );
        prop_assert_eq!(
            spans.last().unwrap().range().end,
            offset + len,
            "last span doesn't end at offset + text.len()"
        );

        let bytes = text.as_bytes();
        for i in 0..spans.len() {
            let range = spans[i].range();
            prop_assert!(
                range.start <= range.end,
                "inverted span at index {}: {:?}",
                i,
                range
            );
            let local_start = range.start - offset;
            let local_end = range.end - offset;
            prop_assert!(
                core::str::from_utf8(&bytes[local_start..local_end]).is_ok(),
                "non-UTF-8 span at index {}: {:?}",
                i,
                range
            );
            if i + 1 < spans.len() {
                prop_assert_eq!(
                    range.end,
                    spans[i + 1].range().start,
                    "gap between spans {} and {}",
                    i,
                    i + 1
                );
            }
        }
        Ok(())
    }

    /// Partition `text` into char-aligned chunks using `chunk_roles`.
    /// Each entry in `chunk_roles` is `(char_count, role_index)`.
    /// Returns a vec of `(TokenRole, Range<usize>)`.
    fn build_token_stream(
        text: &str,
        chunk_roles: &[(usize, u8)],
    ) -> Vec<(Gpt2FamilyTokenRole, Range<usize>)> {
        let roles = [
            Gpt2FamilyTokenRole::Whitespace,
            Gpt2FamilyTokenRole::Punctuation,
            Gpt2FamilyTokenRole::Word {
                check_contraction: false,
            },
            Gpt2FamilyTokenRole::Word {
                check_contraction: true,
            },
            Gpt2FamilyTokenRole::Standalone,
            Gpt2FamilyTokenRole::Gap,
        ];

        let mut tokens = Vec::new();
        let mut char_iter = text.char_indices().peekable();

        for &(char_count, role_idx) in chunk_roles {
            if char_iter.peek().is_none() {
                break;
            }
            let start = char_iter.peek().unwrap().0;
            let mut end = start;
            for _ in 0..char_count {
                if let Some((_, ch)) = char_iter.next() {
                    end += ch.len_utf8();
                } else {
                    break;
                }
            }
            if start < end {
                let role = roles[role_idx as usize % roles.len()];
                tokens.push((role, start..end));
            }
        }

        // Cover remaining text as a final token
        if let Some(&(pos, _)) = char_iter.peek() {
            let end = text.len();
            if pos < end {
                tokens.push((
                    Gpt2FamilyTokenRole::Word {
                        check_contraction: false,
                    },
                    pos..end,
                ));
            }
        }

        tokens
    }

    // -------------------------------------------------------------------
    // Structural invariant proptests
    // -------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn structural_invariants_multi_role(
            text in "\\PC{0,100}",
            chunks in proptest::collection::vec((1..5usize, 0..6u8), 1..20),
        ) {
            let tokens = build_token_stream(&text, &chunks);
            let spans = collect_spans(tokens.into_iter(), &text, 0);
            assert_structural_invariants(&spans, &text, 0)?;
        }

        #[test]
        fn structural_invariants_with_offset(
            text in "\\PC{1,50}",
            chunks in proptest::collection::vec((1..5usize, 0..6u8), 1..10),
            offset in 0..1000usize,
        ) {
            let tokens = build_token_stream(&text, &chunks);
            let spans = collect_spans(tokens.into_iter(), &text, offset);
            assert_structural_invariants(&spans, &text, offset)?;
        }

        #[test]
        fn structural_invariants_empty_stream(text in "\\PC{0,100}") {
            let spans = collect_spans(core::iter::empty(), &text, 0);

            if text.is_empty() {
                prop_assert!(spans.is_empty());
            } else {
                prop_assert_eq!(spans.len(), 1);
                prop_assert_eq!(spans[0].clone(), SpanRef::Gap(0..text.len()));
            }
        }

        /// Early termination: stop after N spans, verify consumed is
        /// a valid byte position and accepted spans are contiguous.
        #[test]
        fn early_termination(
            text in "\\PC{1,80}",
            chunks in proptest::collection::vec((1..4usize, 0..6u8), 1..15),
            stop_after in 1..10usize,
        ) {
            let tokens = build_token_stream(&text, &chunks);
            let mut accepted = Vec::new();
            let (completed, consumed) = for_each_classified_span(
                tokens.into_iter(),
                &text,
                0,
                &mut |span| {
                    accepted.push(span);
                    accepted.len() < stop_after
                },
            );

            if !completed {
                // Callback rejected a span: it's the last one in accepted
                prop_assert!(!accepted.is_empty());
                prop_assert!(
                    consumed <= text.len(),
                    "consumed {} > text.len() {}",
                    consumed,
                    text.len()
                );
                // Accepted spans should be contiguous from offset 0
                for i in 1..accepted.len() {
                    prop_assert_eq!(
                        accepted[i - 1].range().end,
                        accepted[i].range().start,
                        "gap between accepted spans {} and {}",
                        i - 1,
                        i
                    );
                }
            }
        }

        /// Same input always produces the same output.
        #[test]
        fn deterministic(
            text in "\\PC{0,80}",
            chunks in proptest::collection::vec((1..4usize, 0..6u8), 1..15),
        ) {
            let tokens1 = build_token_stream(&text, &chunks);
            let tokens2 = build_token_stream(&text, &chunks);
            let spans1 = collect_spans(tokens1.into_iter(), &text, 0);
            let spans2 = collect_spans(tokens2.into_iter(), &text, 0);
            prop_assert_eq!(&spans1, &spans2);
        }
    }

    // -------------------------------------------------------------------
    // contraction_split: exhaustive prefix testing
    // -------------------------------------------------------------------

    #[test]
    fn contraction_split_empty_and_short() {
        assert_eq!(contraction_split(b""), None);
        assert_eq!(contraction_split(b"'"), None);
        assert_eq!(contraction_split(b"'s"), None); // exactly a contraction, no trailing
    }

    #[test]
    fn contraction_split_no_apostrophe() {
        assert_eq!(contraction_split(b"hello"), None);
        assert_eq!(contraction_split(b"abc"), None);
    }

    #[test]
    fn contraction_split_single_char_suffixes() {
        // Each single-char suffix, both cases, with trailing letter
        for &suffix in &[b's', b'S', b't', b'T', b'd', b'D', b'm', b'M'] {
            let input = [b'\'', suffix, b'a'];
            assert_eq!(
                contraction_split(&input),
                Some(2),
                "expected split at 2 for {:?}",
                core::str::from_utf8(&input)
            );
        }
    }

    #[test]
    fn contraction_split_single_char_exact_length() {
        // Exactly 2 bytes after apostrophe = standalone contraction, not a split
        for &suffix in &[b's', b'S', b't', b'T', b'd', b'D', b'm', b'M'] {
            let input = [b'\'', suffix];
            assert_eq!(contraction_split(&input), None);
        }
    }

    #[test]
    fn contraction_split_two_char_suffixes() {
        let pairs: &[(u8, u8)] = &[
            (b'r', b'e'),
            (b'R', b'E'),
            (b'r', b'E'),
            (b'R', b'e'),
            (b'v', b'e'),
            (b'V', b'E'),
            (b'v', b'E'),
            (b'V', b'e'),
            (b'l', b'l'),
            (b'L', b'L'),
            (b'l', b'L'),
            (b'L', b'l'),
        ];
        for &(c1, c2) in pairs {
            let input = [b'\'', c1, c2, b'a'];
            assert_eq!(
                contraction_split(&input),
                Some(3),
                "expected split at 3 for {:?}",
                core::str::from_utf8(&input)
            );
        }
    }

    #[test]
    fn contraction_split_two_char_exact_length() {
        // Exactly 3 bytes = standalone contraction
        let pairs: &[(u8, u8)] = &[(b'r', b'e'), (b'v', b'e'), (b'l', b'l')];
        for &(c1, c2) in pairs {
            let input = [b'\'', c1, c2];
            assert_eq!(contraction_split(&input), None);
        }
    }

    #[test]
    fn contraction_split_non_contraction_prefix() {
        // Apostrophe + letter that isn't a contraction suffix
        assert_eq!(contraction_split(b"'abc"), None);
        assert_eq!(contraction_split(b"'xyz"), None);
        assert_eq!(contraction_split(b"'Hello"), None);
    }

    #[test]
    fn contraction_split_real_words() {
        assert_eq!(contraction_split(b"'There"), Some(2));
        assert_eq!(contraction_split(b"'THE"), Some(2));
        assert_eq!(contraction_split(b"'llama"), Some(3));
        assert_eq!(contraction_split(b"'velvet"), Some(3));
        assert_eq!(contraction_split(b"'really"), Some(3));
    }

    // -------------------------------------------------------------------
    // proptest: contraction_split on arbitrary bytes
    // -------------------------------------------------------------------

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]

        /// contraction_split must never panic on arbitrary byte input,
        /// and when it returns Some(n), n must be a valid split point:
        /// 0 < n < input.len().
        #[test]
        fn contraction_split_arbitrary_bytes(bytes in proptest::collection::vec(0..=255u8, 0..50)) {
            let result = contraction_split(&bytes);
            if let Some(n) = result {
                proptest::prop_assert!(n > 0, "split at 0 is invalid");
                proptest::prop_assert!(
                    n < bytes.len(),
                    "split at {} >= len {} leaves nothing after",
                    n,
                    bytes.len()
                );
            }
        }
    }
}
