//! Token classification and whitespace post-processing for GPT-2 family
//! regex patterns.

use core::ops::Range;

use logos::{Logos, SpannedIter};
use ringbuffer::RingBuffer;
use unicode_general_category::{GeneralCategory, get_general_category};

use crate::spanners::SpanRef;

/// True if `c` is `\p{L}`. Unlike `char::is_alphabetic()`, excludes
/// Mc marks that have `Other_Alphabetic`.
fn is_unicode_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// Byte offset of the first `\p{L}` in `token`, or `token.len()` if none.
fn non_letter_prefix_len(token: &str) -> usize {
    token
        .find(|c: char| is_unicode_letter(c))
        .unwrap_or(token.len())
}

/// How a logos token interacts with whitespace splitting.
///
/// The `OpenAI` regex patterns use `\s+(?!\S)` which backtracks so the last
/// whitespace character can be absorbed as a prefix by the next pattern.
/// Logos DFA can't express lookaheads, so we post-process: when a
/// [`Whitespace`](Self::Whitespace) token precedes certain token kinds,
/// the last character merges into the next span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gpt2FamilyTokenRole {
    /// Horizontal whitespace. Buffered; last char may split to next token.
    Whitespace,
    /// Punctuation with ` ?` prefix. Always absorbs a preceding ASCII space.
    Punctuation,
    /// Letter/word token. Absorbs preceding space if token starts with a
    /// letter. When `check_contraction` is true, applies contraction-prefix
    /// splitting (e.g. `'The` -> `'T` + `he` for cl100k compatibility).
    Word {
        /// Whether to check for and split contraction prefixes.
        check_contraction: bool,
        /// Whether the first character is `\p{L}`. When true, preceding
        /// whitespace merges into this span. When false (non-letter prefix),
        /// whitespace stays separate.
        first_char_is_letter: bool,
    },
    /// Token that never absorbs whitespace (digits, standalone punctuation).
    Standalone,
    /// Newline-containing whitespace (`\s*[\r\n]+`). Buffered separately;
    /// at end of string, adjacent Newline + Whitespace merge (regex `\s++$`).
    Newline,
    /// Unrecognized bytes.
    Gap,
}

/// Maps a logos token enum to [`Gpt2FamilyTokenRole`].
pub trait Gpt2FamilyLogos<'a>: Logos<'a> {
    /// Returns the role for this token variant.
    fn family_role(&self) -> Gpt2FamilyTokenRole;
}

/// If `bytes` starts with a contraction prefix (`'s`, `'t`, `'d`, `'m`,
/// `'re`, `'ve`, `'ll`, case-insensitive) followed by more bytes, returns
/// the split point. Returns `None` if there are no trailing bytes or no
/// contraction prefix.
///
/// Used for cl100k where logos longest-match picks `'The` as one token
/// but the regex first-match would split it as `'T` + `he`.
///
/// # Examples
///
/// ```
/// use wordchipper::spanners::span_lexers::logos::gpt2_family::contraction_split;
///
/// assert_eq!(contraction_split(b"'There"), Some(2));
/// assert_eq!(contraction_split(b"'llama"), Some(3));
/// assert_eq!(contraction_split(b"'t"), None);
/// assert_eq!(contraction_split(b"hello"), None);
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

/// Word-range iterator with whitespace post-processing for GPT-2 family
/// patterns.
///
/// Uses `ConstGenericRingBuffer` (power-of-2 bitmask indexing, ~10% faster
/// than `ringbuf::StaticRb` which uses atomics). Max spans per iteration is
/// 5 (pending_newline flush + ws_split prefix + punct/word emit +
/// contraction split = 1+1+1+2).
pub struct Gpt2FamilySpanIter<'source, I> {
    text: &'source str,
    last: usize,
    pending_ws: Option<Range<usize>>,
    pending_newline: Option<Range<usize>>,
    /// Buffered punctuation span from mark-extension that may absorb
    /// trailing `[\r\n/]*` and adjacent `[^\s\p{L}\p{N}]+` body chars
    /// from subsequent DFA tokens.
    pending_punct: Option<Range<usize>>,
    /// When true, `pending_punct` has already absorbed a `\r`/`\n` byte,
    /// so only `[\r\n/]` chars can continue the trailer.
    punct_in_trailer: bool,

    /// Capacity 8 (next power-of-2 above the 5 slots needed) because
    /// `ConstGenericRingBuffer` requires power-of-2 for bitmask indexing.
    ring: ringbuffer::ConstGenericRingBuffer<Range<usize>, 8>,

    iter: Option<I>,
}

impl<'source, I> Gpt2FamilySpanIter<'source, I>
where
    I: Iterator<Item = (Gpt2FamilyTokenRole, Range<usize>)>,
{
    /// Create a new iterator from a stream of classified token roles.
    pub fn new(
        text: &'source str,
        iter: I,
    ) -> Self {
        Self {
            text,
            last: 0,
            pending_ws: None,
            pending_newline: None,
            pending_punct: None,
            punct_in_trailer: false,
            ring: ringbuffer::ConstGenericRingBuffer::new(),
            iter: Some(iter),
        }
    }

    fn push(
        &mut self,
        range: Range<usize>,
    ) {
        self.ring.enqueue(range);
    }

    fn pop(&mut self) -> Option<Range<usize>> {
        self.ring.dequeue()
    }

    fn next_tok(&mut self) -> Option<(Gpt2FamilyTokenRole, Range<usize>)> {
        if let Some(iter) = &mut self.iter {
            if let Some(item) = iter.next() {
                return Some(item);
            }
            self.iter = None;
        }
        None
    }
}

/// Create a [`Gpt2FamilySpanIter`] from a logos [`SpannedIter`], mapping
/// tokens to [`Gpt2FamilyTokenRole`] via the [`Gpt2FamilyLogos`] trait.
pub fn logos_span_iter<'source, Token: Gpt2FamilyLogos<'source> + 'source>(
    text: &'source str,
    iter: SpannedIter<'source, Token>,
) -> Gpt2FamilySpanIter<'source, impl Iterator<Item = (Gpt2FamilyTokenRole, Range<usize>)> + 'source>
{
    Gpt2FamilySpanIter::new(
        text,
        iter.map(|(res, range)| {
            let role = match res {
                Ok(tok) => tok.family_role(),
                Err(_) => Gpt2FamilyTokenRole::Gap,
            };
            (role, range)
        }),
    )
}

impl<'source, I> Iterator for Gpt2FamilySpanIter<'source, I>
where
    I: Iterator<Item = (Gpt2FamilyTokenRole, Range<usize>)>,
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
                if let Some(pp) = self.pending_punct.take() {
                    self.punct_in_trailer = false;
                    return Some(pp);
                }
                // End of stream: merge pending_newline + pending_ws
                // into one span (matching regex `\s++$`).
                if let Some(nl) = self.pending_newline.take() {
                    if let Some(ws) = self.pending_ws.take() {
                        if nl.end == ws.start {
                            return Some(nl.start..ws.end);
                        }
                        self.push(ws);
                    }
                    return Some(nl);
                }
                return self.pending_ws.take();
            }

            if let Some((role, Range { start, end })) = self.next_tok() {
                // Try to absorb the next token into pending_punct.
                // The regex punct branch ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
                // has a body that greedily matches non-L/N/ws chars, then
                // a trailer that matches `[\r\n/]`. Mark-extension creates
                // pending_punct for the initial body (punct + marks). Here
                // we continue matching body chars from PunctuationBare
                // tokens and trailer chars from Newline tokens.
                if let Some(pp) = self.pending_punct.take() {
                    if pp.end == start {
                        let first = bytes[start];
                        let absorb = if self.punct_in_trailer {
                            // Trailer mode: only continue with pure
                            // [\r\n/] content.
                            bytes[start..end]
                                .iter()
                                .all(|&b| matches!(b, b'\r' | b'\n' | b'/'))
                        } else if matches!(role, Gpt2FamilyTokenRole::Punctuation) {
                            // Body mode: absorb bare punctuation
                            // (not PunctuationSpaced which starts
                            // a new regex match with its ` ?` prefix).
                            first != b' '
                        } else {
                            // Body mode: absorb a newline-starting
                            // token (the [\r\n/]* trailer).
                            matches!(first, b'\r' | b'\n')
                        };
                        if absorb {
                            if !self.punct_in_trailer
                                && bytes[start..end]
                                    .iter()
                                    .any(|&b| matches!(b, b'\r' | b'\n'))
                            {
                                self.punct_in_trailer = true;
                            }
                            self.pending_punct = Some(pp.start..end);
                            self.last = end;
                            continue;
                        }
                        // Body mode: PrefixedWord adjacent to pending
                        // punct. The regex `[^\s\p{L}\p{N}]+` keeps
                        // matching non-L/N/ws chars, so the non-letter
                        // prefix and marks belong in pending_punct.
                        if !self.punct_in_trailer
                            && let Gpt2FamilyTokenRole::Word {
                                first_char_is_letter: false,
                                check_contraction,
                            } = role
                            && !self.text[start..].starts_with(char::is_whitespace)
                        {
                            let prefix_len = non_letter_prefix_len(&self.text[start..end]);
                            if start + prefix_len >= end {
                                self.pending_punct = Some(pp.start..end);
                                self.last = end;
                                continue;
                            }
                            emit!(pp.start..start + prefix_len);
                            emit_absorbing!(start + prefix_len, end, check_contraction);
                            self.last = end;
                            continue;
                        }
                    }
                    // Not absorbed: flush pending_punct.
                    self.punct_in_trailer = false;
                    emit!(pp);
                }

                if self.last < start {
                    // We skipped over a gap.
                    if let Some(nl) = self.pending_newline.take() {
                        emit!(nl);
                    }
                    if let Some(ws) = self.pending_ws.take() {
                        emit!(ws);
                    }
                }

                self.last = end;

                // Flush pending_newline for non-ws/non-newline tokens.
                // For Whitespace: keep pending_newline (may merge at EOF).
                // For Newline: handled in its own arm below.
                if let Some(nl) = self.pending_newline.take() {
                    match role {
                        Gpt2FamilyTokenRole::Whitespace if nl.end == start => {
                            // Adjacent Newline + Whitespace: keep both
                            // buffered separately for potential EOF merge.
                            self.pending_newline = Some(nl);
                        }
                        Gpt2FamilyTokenRole::Newline if nl.end == start => {
                            // Chain adjacent newlines.
                            self.pending_newline = Some(nl.start..end);
                            continue;
                        }
                        _ => {
                            emit!(nl);
                        }
                    }
                }

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
                    Gpt2FamilyTokenRole::Word {
                        check_contraction,
                        first_char_is_letter,
                    } => {
                        if let Some(ws) = self.pending_ws.take() {
                            let ws_start = ws.start;
                            let ws_end = ws.end;
                            let trim = flush_ws_split!(ws);
                            let single_char = trim == ws_start;

                            if first_char_is_letter {
                                // Token has no existing prefix; merge last ws char.
                                emit_absorbing!(trim, end, check_contraction);
                            } else if single_char {
                                // Single ws char: emit standalone, token as-is.
                                emit!(trim..ws_end);
                                emit_absorbing!(start, end, check_contraction);
                            } else if bytes[trim] == b' ' {
                                // 2+ ws chars ending in ASCII space: merge
                                // space + non-letter prefix into one span.
                                // Mark-extension extends past combining marks
                                // that the DFA absorbed into the word token.
                                let prefix_len = non_letter_prefix_len(&self.text[start..end]);
                                if start + prefix_len < end {
                                    emit!(trim..start + prefix_len);
                                    emit_absorbing!(start + prefix_len, end, check_contraction);
                                } else {
                                    // Entire token is non-letters. Buffer
                                    // to absorb trailing body/newline chars.
                                    self.pending_punct = Some(trim..start + prefix_len);
                                    self.punct_in_trailer = false;
                                }
                            } else {
                                // 2+ ws chars ending in non-space (NBSP, tab):
                                // don't absorb; emit last ws char standalone.
                                emit!(trim..ws_end);
                                emit_absorbing!(start, end, check_contraction);
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
                    Gpt2FamilyTokenRole::Newline => {
                        if let Some(ws) = self.pending_ws.take() {
                            emit!(ws);
                        }
                        if let Some(nl) = self.pending_newline.take() {
                            if nl.end == start {
                                self.pending_newline = Some(nl.start..end);
                            } else {
                                emit!(nl);
                                self.pending_newline = Some(start..end);
                            }
                        } else {
                            self.pending_newline = Some(start..end);
                        }
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

/// Emit `SpanRef::Word` and `SpanRef::Gap` spans from classified tokens.
///
/// Thin wrapper over [`Gpt2FamilySpanIter`]: iterates word ranges and fills
/// uncovered byte ranges with `Gap` spans. Returns `(completed, consumed)`.
///
/// # Example
///
/// ```
/// use wordchipper::spanners::{
///     SpanRef,
///     span_lexers::logos::gpt2_family::{
///         Gpt2FamilyTokenRole,
///         for_each_classified_span,
///     },
/// };
///
/// let text = "hello world";
/// let tokens = vec![
///     (
///         Gpt2FamilyTokenRole::Word {
///             check_contraction: false,
///             first_char_is_letter: true,
///         },
///         0..5,
///     ),
///     (Gpt2FamilyTokenRole::Whitespace, 5..6),
///     (
///         Gpt2FamilyTokenRole::Word {
///             check_contraction: false,
///             first_char_is_letter: true,
///         },
///         6..11,
///     ),
/// ];
///
/// let mut spans = Vec::new();
/// for_each_classified_span(tokens.into_iter(), text, 0, &mut |span| {
///     spans.push(span);
///     true
/// });
///
/// assert_eq!(spans, vec![SpanRef::Word(0..5), SpanRef::Word(5..11)]);
/// ```
pub fn for_each_classified_span(
    iter: impl Iterator<Item = (Gpt2FamilyTokenRole, Range<usize>)>,
    text: &str,
    offset: usize,
    f: &mut dyn FnMut(SpanRef) -> bool,
) -> (bool, usize) {
    let len = text.len();
    let mut last = 0;

    for range in Gpt2FamilySpanIter::new(text, iter) {
        if last < range.start && !f(SpanRef::Gap(offset + last..offset + range.start)) {
            return (false, last);
        }
        if !f(SpanRef::Word(offset + range.start..offset + range.end)) {
            return (false, range.start);
        }
        last = range.end;
    }

    if last < len {
        if !f(SpanRef::Gap(offset + last..offset + len)) {
            return (false, last);
        }
        last = len;
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
                first_char_is_letter: true,
            },
            Gpt2FamilyTokenRole::Word {
                check_contraction: true,
                first_char_is_letter: false,
            },
            Gpt2FamilyTokenRole::Standalone,
            Gpt2FamilyTokenRole::Gap,
            Gpt2FamilyTokenRole::Newline,
            Gpt2FamilyTokenRole::Word {
                check_contraction: true,
                first_char_is_letter: true,
            },
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
                        first_char_is_letter: true,
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
            chunks in proptest::collection::vec((1..5usize, 0..8u8), 1..20),
        ) {
            let tokens = build_token_stream(&text, &chunks);
            let spans = collect_spans(tokens.into_iter(), &text, 0);
            assert_structural_invariants(&spans, &text, 0)?;
        }

        #[test]
        fn structural_invariants_with_offset(
            text in "\\PC{1,50}",
            chunks in proptest::collection::vec((1..5usize, 0..8u8), 1..10),
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
            chunks in proptest::collection::vec((1..4usize, 0..8u8), 1..15),
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
            chunks in proptest::collection::vec((1..4usize, 0..8u8), 1..15),
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
