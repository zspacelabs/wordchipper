//! # Lexer Text Spanner

use core::ops::Range;

use crate::{
    alloc::sync::Arc,
    spanners::{SpanRef, TextSpanner, span_lexers::SpanLexer},
    support::ranges::offset_range,
};

/// A [`TextSpanner`] composed over [`SpanLexer`] plugins.
///
/// Combines a word-scanning [`SpanLexer`] with an optional special-token
/// scanner. The word lexer handles segmentation within text segments;
/// the special lexer finds special tokens that split the input into
/// those segments.
///
/// The word lexer is pluggable (e.g. regex-based or logos DFA). The special
/// lexer is always regex-based, built from the special token patterns.
#[derive(Clone)]
pub struct LexerTextSpanner {
    word_lexer: Arc<dyn SpanLexer>,
    special_lexer: Option<Arc<dyn SpanLexer>>,
}

impl LexerTextSpanner {
    /// Build a new [`LexerTextSpanner`].
    ///
    /// ## Arguments
    /// * `word_scanner` - The lexer for word splitting.
    /// * `special_scanner` - The optional lexer for special word matching.
    pub fn new(
        word_scanner: Arc<dyn SpanLexer>,
        special_scanner: Option<Arc<dyn SpanLexer>>,
    ) -> Self {
        Self {
            word_lexer: word_scanner,
            special_lexer: special_scanner,
        }
    }

    fn next_special_span(
        &self,
        text: &str,
    ) -> Option<Range<usize>> {
        self.special_lexer
            .as_ref()
            .and_then(|s| s.find_span_iter(text).next())
    }

    /// Scan `text` into [`Word`](SpanRef::Word) and [`Gap`](SpanRef::Gap) spans.
    ///
    /// Loops over [`next_span`](Self::next_span),
    /// classifying matched regions as `Word` and unmatched regions as `Gap`.
    /// Lexers with richer token classification override this directly.
    ///
    /// ## Arguments
    /// * `text` - the text segment to scan (no special tokens).
    /// * `offset` - byte offset to add to emitted span ranges.
    /// * `f` - callback; return `false` to halt early.
    ///
    /// ## Returns
    /// `(completed, consumed)` where `consumed` is the byte count of
    /// accepted spans and `completed` indicates all spans were accepted.
    fn for_each_word(
        &self,
        text: &str,
        offset: usize,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        let mut last = 0;

        for Range { start, end } in self.word_lexer.find_span_iter(text) {
            if last < start {
                if !f(SpanRef::Gap(offset_range::<usize>(last..start, offset))) {
                    return (false, last);
                }
                last = start;
            }

            if !f(SpanRef::Word(offset_range::<usize>(start..end, offset))) {
                return (false, last);
            }
            last = end;
        }

        if last < text.len() {
            if !f(SpanRef::Gap(offset_range::<usize>(
                last..text.len(),
                offset,
            ))) {
                return (false, last);
            }
            last = text.len();
        }

        (true, last)
    }
}

impl TextSpanner for LexerTextSpanner {
    fn for_each_split_span(
        &self,
        text: &str,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize) {
        let mut current = text;
        let mut offset = 0;

        while let Some(Range { start, end }) = self.next_special_span(current) {
            let pre = &current[..start];

            let (cont, used) = self.for_each_word(pre, offset, f);
            if !cont {
                return (false, offset + used);
            }

            if !f(SpanRef::Special(offset_range::<usize>(start..end, offset))) {
                return (false, offset + start);
            }

            current = &current[end..];
            offset += end;
        }

        self.for_each_word(current, offset, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TokenType,
        alloc::{boxed::Box, vec, vec::Vec},
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanners::{SpanRef, TextSpanningConfig},
    };

    const _LEXER_SPANNER_BOX_CHECK: Option<Box<LexerTextSpanner>> = None;
    const _LEXER_SPANNER_ARC_CHECK: Option<Arc<LexerTextSpanner>> = None;

    fn from_config<T: TokenType>(config: &TextSpanningConfig<T>) -> LexerTextSpanner {
        LexerTextSpanner::new(
            Arc::new(config.pattern().clone().compile().unwrap()),
            config
                .special_pattern()
                .map(|p| Arc::new(p.compile().unwrap()) as Arc<dyn SpanLexer>),
        )
    }

    #[test]
    fn test_for_each_split_span() {
        use crate::spanners::text_spanner::SpanRef::*;
        type T = u32;

        let config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(r"\w+")
            .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let spanner = from_config(&config);

        let source = "abc 1<|FNORD|> def  <|NORP|> ghi   ";

        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span(source, &mut |span_ref| {
            spans.push(span_ref);
            true
        });
        assert_eq!(
            spans,
            vec![
                Word(0..3),
                Gap(3..4),
                Word(4..5),
                Special(5..14),
                Gap(14..15),
                Word(15..18),
                Gap(18..20),
                Special(20..28),
                Gap(28..29),
                Word(29..32),
                Gap(32..35),
            ]
        );

        // The following are white-box tests to exercise the different halting points.

        // Test "for_each_split_span" Word Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("   abc", &mut |span_ref| match span_ref {
            Word(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Gap(0..3)]);

        // Test "for_each_split_span" Special Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("abc   def<|FNORD|>", &mut |span_ref| match span_ref {
            Special(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3), Gap(3..6), Word(6..9)]);

        // Test "for_each_word" Leading Gap Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("abc  def", &mut |span_ref| match span_ref {
            Gap(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3)]);

        // Test "for_each_word" Trailing Gap Exit
        let mut spans: Vec<SpanRef> = Vec::new();
        spanner.for_each_split_span("foo  ", &mut |span_ref| match span_ref {
            Gap(_) => false,
            _ => {
                spans.push(span_ref);
                true
            }
        });
        assert_eq!(spans, vec![Word(0..3)]);
    }

    #[test]
    fn test_split_words() {
        type T = u32;

        let config: TextSpanningConfig<T> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN)
                .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let spanner = from_config(&config);

        let buf = "hello<|FNORD|> wor<|NORP|>ld!";

        assert_eq!(
            &spanner.split_spans(buf),
            &vec![
                SpanRef::Word(0..5),
                SpanRef::Special(5..14),
                SpanRef::Word(14..18),
                SpanRef::Special(18..26),
                SpanRef::Word(26..28),
                SpanRef::Word(28..buf.len()),
            ]
        );
    }

    #[test]
    fn test_rewrite() {
        type T = u32;

        let config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(r"\w+");

        let spanner = from_config(&config);

        let buf = vec!["hello world!", "abc def"];
        assert_eq!(
            spanner.batch_remove_gaps(&buf),
            vec!["helloworld", "abcdef"]
        );
    }
}
