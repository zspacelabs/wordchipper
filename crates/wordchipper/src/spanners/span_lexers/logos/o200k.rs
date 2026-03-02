//! # o200k Logos Lexer
//!
//! Compile-time DFA lexer for the `o200k_base` pattern (GPT-4o).
//!
//! This serves as a reference implementation showing how to build an
//! accelerated lexer using [`Gpt2FamilyTokenRole`] and [`for_each_classified_span`].

use core::ops::Range;
use std::prelude::rust_2015::Box;

use logos::Logos;

use crate::{
    alloc::sync::Arc,
    pretrained::openai::OA_O200K_BASE_PATTERN,
    spanners::span_lexers::{
        SpanLexer,
        accelerators::RegexAcceleratorHook,
        logos::gpt2_family::{Gpt2FamilyLogos, Gpt2FamilySpanIter, Gpt2FamilyTokenRole},
    },
};
// Shorthand aliases for the character classes used in o200k:
//   UPPER = [\p{Uppercase_Letter}\p{Titlecase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]
//   LOWER = [\p{Lowercase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]
//   CONTRACTION_SUFFIX = ('[sS]|'[tT]|'[dD]|'[mM]|'[rR][eE]|'[vV][eE]|'[lL][lL])?
//
// These are inlined below because logos derive macros require string literals.

/// Logos token for the `o200k_base` pattern.
///
/// Key difference from cl100k: contractions are attached to the preceding word,
/// and two word `span_encoders` (uppercase-leading vs lowercase-ending) are merged
/// into a single `Word` variant because logos requires unambiguous DFA states.
///
/// | Regex branch                                         | Logos variant  |
/// |------------------------------------------------------|----------------|
/// | `[^\r\n\p{L}\p{N}]?[UPPER]*[LOWER]+CONTRACTION?`    | Word           |
/// | `[^\r\n\p{L}\p{N}]?[UPPER]+[LOWER]*CONTRACTION?`    | Word           |
/// | `\p{N}{1,3}`                                         | Digits         |
/// | ` ?[^\s\p{L}\p{N}]+[\r\n/]*`                        | Punctuation    |
/// | `\s*[\r\n]+`                                         | Newline        |
/// | `\s+`                                                | Whitespace     |
#[derive(Logos, Debug, PartialEq, Clone)]
pub(crate) enum O200kToken {
    // Both word patterns merged via alternation into a single regex.
    // The two overlap on \p{Modifier_Letter}/\p{Other_Letter}/\p{Mark}
    // characters (e.g. CJK), so logos requires a single DFA pattern.
    // CamelCase splitting still works: the DFA's longest-match finds
    // the same boundaries as regex first-match (e.g. "OpenAI" -> "Open"+"AI").
    // Priority > default to beat Punctuation on the [^\r\n\p{L}\p{N}]? prefix.
    #[regex(
        r"[^\r\n\p{Letter}\p{Number}]?([\p{Uppercase_Letter}\p{Titlecase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]*[\p{Lowercase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]+|[\p{Uppercase_Letter}\p{Titlecase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]+[\p{Lowercase_Letter}\p{Modifier_Letter}\p{Other_Letter}\p{Mark}]*)('[sS]|'[tT]|'[dD]|'[mM]|'[rR][eE]|'[vV][eE]|'[lL][lL])?",
        priority = 3
    )]
    Word,

    #[regex(r"\p{Number}{1,3}")]
    Digits,

    // Note: o200k includes '/' in the newline-char set after punctuation.
    #[regex(r" ?[^\s\p{Letter}\p{Number}]+[\r\n/]*")]
    Punctuation,

    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}

impl Gpt2FamilyLogos<'_> for O200kToken {
    fn family_role(&self) -> Gpt2FamilyTokenRole {
        match self {
            Self::Whitespace => Gpt2FamilyTokenRole::Whitespace,
            Self::Word => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
            },
            Self::Punctuation => Gpt2FamilyTokenRole::Punctuation,
            Self::Digits | Self::Newline => Gpt2FamilyTokenRole::Standalone,
        }
    }
}

/// A [`SpanLexer`] for the `o200k_base` pattern (GPT-4o).
///
/// Uses a compile-time logos DFA for word scanning.
///
/// Only matches the regex spans; does not match the special tokens.
#[derive(Clone, Debug)]
pub struct O200kLexer;

inventory::submit! {
    RegexAcceleratorHook::new(OA_O200K_BASE_PATTERN,|| Arc::new(O200kLexer))
}

impl SpanLexer for O200kLexer {
    fn find_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> Box<dyn Iterator<Item = Range<usize>> + 'a> {
        Box::new(Gpt2FamilySpanIter::new(
            text,
            O200kToken::lexer(text).spanned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{sync::Arc, vec::Vec},
        spanners::{SpanRef, TextSpanner, span_lexers::LexerTextSpanner},
    };

    /// Build a `TextSpanner` from a logos lexer with no specials.
    fn spanner(lexer: impl SpanLexer + 'static) -> LexerTextSpanner {
        LexerTextSpanner::new(Arc::new(lexer), None)
    }

    #[test]
    fn test_o200k_contractions_attached() {
        let s = spanner(O200kLexer);
        let text = "don't I'll she's";
        let spans = s.split_spans(text);

        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();

        assert!(
            words.contains(&"don't"),
            "expected \"don't\" as one token, got: {:?}",
            words
        );
        assert!(
            words.contains(&" she's"),
            "expected \" she's\" as one token, got: {:?}",
            words
        );
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_o200k_unicode() {
        use crate::{
            pretrained::openai::OA_O200K_BASE_PATTERN,
            spanners::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(O200kLexer);

        let cases = [
            "Hello world",
            "Bonjour le monde",
            "\u{4f60}\u{597d}\u{4e16}\u{754c}",
            "\u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043c}\u{0438}\u{0440}",
            "price is 100 dollars",
            "caf\u{00e9} na\u{00ef}ve r\u{00e9}sum\u{00e9}",
            "Hello \u{4e16}\u{754c} 123",
            "don't I'll she's",
            "HELLO WORLD",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            assert_eq!(
                regex_spans, logos_spans,
                "o200k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                text, regex_spans, logos_spans
            );
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_o200k_realworld() {
        use crate::{
            pretrained::openai::OA_O200K_BASE_PATTERN,
            spanners::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(O200kLexer);

        let cases = [
            "the Civil War\u{2014}in which",
            "nation\u{2019}s capital",
            "  Like all Choctaw counties",
            "   123 numbers",
            "hello   ",
            "\t\thello",
            "  \nfoo",
            "foo  \nbar",
            "   hello world",
            "foo  \"bar\" baz",
            "$$$!!!...---",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            assert_eq!(
                regex_spans, logos_spans,
                "o200k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                text, regex_spans, logos_spans
            );
        }
    }

    // -------------------------------------------------------------------
    // proptest: structural invariants on real lexer output
    // -------------------------------------------------------------------

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]

        #[test]
        fn structural_invariants(text in "\\PC{0,200}") {
            let s = spanner(O200kLexer);
            let spans = s.split_spans(&text);

            if text.is_empty() {
                proptest::prop_assert!(spans.is_empty());
            } else {
                proptest::prop_assert!(!spans.is_empty());
                proptest::prop_assert_eq!(spans[0].range().start, 0);
                proptest::prop_assert_eq!(spans.last().unwrap().range().end, text.len());

                for i in 0..spans.len() {
                    let r = spans[i].range();
                    proptest::prop_assert!(
                        r.start < r.end,
                        "empty span at index {}: {:?}",
                        i, r
                    );
                    proptest::prop_assert!(
                        core::str::from_utf8(&text.as_bytes()[r.start..r.end]).is_ok(),
                        "non-UTF-8 span at index {}: {:?}",
                        i, r
                    );
                    if i + 1 < spans.len() {
                        proptest::prop_assert_eq!(
                            r.end,
                            spans[i + 1].range().start,
                            "gap between spans {} and {}",
                            i, i + 1
                        );
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // iterator vs for_each oracle equivalence
    // -------------------------------------------------------------------

    /// Collect Word ranges from for_each_classified_span (the oracle).
    fn oracle_word_ranges(text: &str) -> Vec<Range<usize>> {
        use crate::spanners::span_lexers::logos::gpt2_family::{
            Gpt2FamilyTokenRole,
            for_each_classified_span,
        };

        let tokens: Vec<(Gpt2FamilyTokenRole, Range<usize>)> = O200kToken::lexer(text)
            .spanned()
            .map(|(res, range)| {
                let role = match res {
                    Ok(tok) => tok.family_role(),
                    Err(_) => Gpt2FamilyTokenRole::Gap,
                };
                (role, range)
            })
            .collect();

        let mut word_ranges = Vec::new();
        for_each_classified_span(tokens.into_iter(), text, 0, &mut |span| {
            if let SpanRef::Word(r) = span {
                word_ranges.push(r);
            }
            true
        });
        word_ranges
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]

        /// The iterator must produce exactly the same Word ranges as
        /// the for_each_classified_span oracle for any input.
        #[test]
        fn iter_matches_oracle(text in "\\PC{0,200}") {
            use crate::spanners::span_lexers::logos::gpt2_family::Gpt2FamilySpanIter;

            let iter_ranges: Vec<Range<usize>> =
                Gpt2FamilySpanIter::new(text.as_str(), O200kToken::lexer(&text).spanned())
                    .collect();
            let oracle_ranges = oracle_word_ranges(&text);
            proptest::prop_assert_eq!(
                &iter_ranges, &oracle_ranges,
                "o200k iter vs oracle mismatch for {:?}", text
            );
        }
    }

    // -------------------------------------------------------------------
    // proptest oracle: regex vs logos equivalence
    // -------------------------------------------------------------------

    #[test]
    #[cfg(feature = "std")]
    fn proptest_o200k_logos_matches_regex() {
        use proptest::prelude::*;

        use crate::{
            pretrained::openai::OA_O200K_BASE_PATTERN,
            spanners::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(O200kLexer);

        let config = proptest::test_runner::Config::with_cases(2000);
        proptest!(config, |(text in "\\PC{0,200}")| {
            let regex_spans = regex_spanner.split_spans(&text);
            let logos_spans = logos_spanner.split_spans(&text);
            prop_assert_eq!(
                &regex_spans, &logos_spans,
                "o200k mismatch for {:?}",
                text
            );
        });
    }
}
