//! # cl100k Logos Lexer
//!
//! Compile-time DFA lexer for the `cl100k_base` pattern (GPT-4, GPT-3.5).
//!
//! This serves as a reference implementation showing how to build an
//! accelerated lexer using [`Gpt2FamilyTokenRole`] and [`for_each_classified_span`].

use core::ops::Range;

use logos::Logos;

use super::gpt2_family::{Gpt2FamilyLogos, Gpt2FamilySpanIter, Gpt2FamilyTokenRole};
use crate::{
    alloc::{boxed::Box, sync::Arc},
    pretrained::openai::OA_CL100K_BASE_PATTERN,
    spanners::span_lexers::{SpanLexer, accelerators::RegexAcceleratorHook},
};

/// Logos token for the `cl100k_base` pattern.
///
/// | Regex branch                      | Logos variant  |
/// |-----------------------------------|----------------|
/// | `'(?i:[sdmt]\|ll\|ve\|re)`        | Contraction    |
/// | `[^\r\n\p{L}\p{N}]?\p{L}+`       | Letters        |
/// | `\p{N}{1,3}`                      | Digits         |
/// | ` ?[^\s\p{L}\p{N}]+[\r\n]*`      | Punctuation    |
/// | `\s*[\r\n]`                       | Newline        |
/// | `\s+(?!\S)` / `\s`               | Whitespace     |
#[derive(Logos, Debug, PartialEq, Clone)]
pub(crate) enum Cl100kToken {
    #[regex(r"'[sStTdDmM]|'[rR][eE]|'[vV][eE]|'[lL][lL]")]
    Contraction,

    #[regex(r"[^\r\n\p{Letter}\p{Number}]?\p{Letter}+")]
    Letters,

    #[regex(r"\p{Number}{1,3}")]
    Digits,

    #[regex(r" ?[^\s\p{Letter}\p{Number}]+[\r\n]*")]
    Punctuation,

    // The `+` on `[\r\n]+` is equivalent to the regex `\s*[\r\n]` in practice:
    // consecutive newlines are consumed identically because `\s*` in the regex
    // greedily eats preceding newlines. Both produce the same total span.
    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}

impl Gpt2FamilyLogos<'_> for Cl100kToken {
    fn family_role(&self) -> Gpt2FamilyTokenRole {
        match self {
            Self::Whitespace => Gpt2FamilyTokenRole::Whitespace,
            Self::Letters => Gpt2FamilyTokenRole::Word {
                check_contraction: true,
            },
            Self::Punctuation => Gpt2FamilyTokenRole::Punctuation,
            Self::Contraction | Self::Digits | Self::Newline => Gpt2FamilyTokenRole::Standalone,
        }
    }
}

/// A [`SpanLexer`] for the `cl100k_base` pattern (GPT-4, GPT-3.5).
///
/// Uses a compile-time logos DFA for word scanning.
///
/// Only matches the regex spans; does not match the special tokens.
#[derive(Clone, Debug)]
pub struct Cl100kLexer;

inventory::submit! {
    RegexAcceleratorHook::new(OA_CL100K_BASE_PATTERN,|| Arc::new(Cl100kLexer))
}

impl SpanLexer for Cl100kLexer {
    fn find_span_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> Box<dyn Iterator<Item = Range<usize>> + 'a> {
        Box::new(Gpt2FamilySpanIter::new(
            text,
            Cl100kToken::lexer(text).spanned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{string::ToString, sync::Arc, vec, vec::Vec},
        spanners::{SpanRef, TextSpanner, span_lexers::LexerTextSpanner},
    };

    /// Build a `TextSpanner` from a logos lexer with no specials.
    fn spanner(lexer: impl SpanLexer + 'static) -> LexerTextSpanner {
        LexerTextSpanner::new(Arc::new(lexer), None)
    }

    #[test]
    fn test_logos_basic_splitting() {
        let s = spanner(Cl100kLexer);
        let text = "hello world";
        let spans = s.split_spans(text);

        // cl100k-like: " world" is one token (space grouped with letters).
        assert_eq!(spans, vec![SpanRef::Word(0..5), SpanRef::Word(5..11),]);
    }

    #[test]
    fn test_logos_with_specials() {
        let special_pattern = crate::support::regex::alternate_choice_regex_pattern(&[
            "<|FNORD|>".to_string(),
            "<|NORP|>".to_string(),
        ]);
        let s = LexerTextSpanner::new(
            Arc::new(Cl100kLexer),
            Some(Arc::new(special_pattern.compile().unwrap()) as Arc<dyn SpanLexer>),
        );

        let text = "hello<|FNORD|> world<|NORP|>!";
        let spans = s.split_spans(text);

        assert_eq!(
            spans,
            vec![
                SpanRef::Word(0..5),
                SpanRef::Special(5..14),
                SpanRef::Word(14..20),
                SpanRef::Special(20..28),
                SpanRef::Word(28..29),
            ]
        );
    }

    #[test]
    fn test_logos_digits() {
        let s = spanner(Cl100kLexer);
        let text = "abc 123 4567";
        let spans = s.split_spans(text);

        assert_eq!(
            spans,
            vec![
                SpanRef::Word(0..3),
                SpanRef::Word(3..4),   // " " is Word (Whitespace token)
                SpanRef::Word(4..7),   // "123"
                SpanRef::Word(7..8),   // " " is Word (Whitespace token)
                SpanRef::Word(8..11),  // "456"
                SpanRef::Word(11..12), // "7"
            ]
        );
    }

    #[test]
    fn test_logos_contractions() {
        let s = spanner(Cl100kLexer);
        let text = "don't I'll she's";
        let spans = s.split_spans(text);

        // cl100k: "don" is letters, "'t" is contraction (separate tokens).
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();

        assert!(words.contains(&"don"));
        assert!(words.contains(&"'t"));
        assert!(words.contains(&"'ll"));
        assert!(words.contains(&"'s"));
    }

    #[test]
    fn test_logos_empty() {
        let s = spanner(Cl100kLexer);
        let spans = s.split_spans("");
        assert!(spans.is_empty());
    }

    #[test]
    fn test_logos_whitespace_only() {
        let s = spanner(Cl100kLexer);
        let text = "   ";
        let spans = s.split_spans(text);

        assert_eq!(spans, vec![SpanRef::Word(0..3)]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_logos_cl100k_unicode() {
        use crate::{
            pretrained::openai::OA_CL100K_BASE_PATTERN,
            spanners::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(Cl100kLexer);

        let cases = [
            "Hello world",
            "Bonjour le monde",
            "Hallo Welt",
            "\u{4f60}\u{597d}\u{4e16}\u{754c}",
            "\u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043c}\u{0438}\u{0440}",
            "price is 100 dollars",
            "caf\u{00e9} na\u{00ef}ve r\u{00e9}sum\u{00e9}",
            "Hello \u{4e16}\u{754c} 123",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            assert_eq!(
                regex_spans, logos_spans,
                "cl100k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                text, regex_spans, logos_spans
            );
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_logos_cl100k_realworld() {
        use crate::{
            pretrained::openai::OA_CL100K_BASE_PATTERN,
            spanners::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(Cl100kLexer);

        let cases = [
            "the Civil War\u{2014}in which",
            "nation\u{2019}s capital",
            "it\u{2019}s not easy",
            "Wade County\u{2019}s boundaries were",
            "  Like all Choctaw counties",
            "   123 numbers",
            "hello   ",
            "\t\thello",
            "  \nfoo",
            "foo  \nbar",
            "   hello world",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            if regex_spans != logos_spans {
                let regex_words: Vec<&str> = regex_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                let logos_words: Vec<&str> = logos_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                panic!(
                    "cl100k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                    text, regex_words, logos_words
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_logos_cl100k_long_text() {
        use crate::{
            pretrained::openai::OA_CL100K_BASE_PATTERN,
            spanners::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(Cl100kLexer);

        let cases = [
            "'The items we buy are important",
            "hello\n'The quick brown fox",
            "Shakespeare's \"sources,\" then read",
            "said \"hello\" to",
            "  \"sources,\" then",
            "foo  \"bar\" baz",
            "Shakespeare's Macbeth: From Saga to Screen|\nA close reading",
            "  $400 dollars",
            "  (hello)",
            " \"sources,\" then",
            " $hello world",
            " 'The quick",
            "  'The quick",
            "  \u{2014}hello world",
            " \u{2014}hello world",
            "Shakespeare's Macbeth: From Saga to Screen|\nA close reading of Shakespeare's play that will position the play in terms of its historical and political contexts and its relation to early modern discourses on the feminine, witchcraft, and the divinity of kings. We will begin with a consideration of the historical legends that constitute Shakespeare's \"sources,\" then read the play slowly and closely, coupling our discussions with readings from the period, exploring how Shakespeare's contemporaries thought of the political and cultural issues raised in the play",
        ];

        for text in cases {
            let regex_spans = regex_spanner.split_spans(text);
            let logos_spans = logos_spanner.split_spans(text);

            if regex_spans != logos_spans {
                let regex_words: Vec<&str> = regex_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                let logos_words: Vec<&str> = logos_spans
                    .iter()
                    .map(|s| &text[s.range().clone()])
                    .collect();
                panic!(
                    "cl100k mismatch for {:?}:\n  regex: {:?}\n  logos: {:?}",
                    text, regex_words, logos_words
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // proptest: structural invariants on real lexer output
    // -------------------------------------------------------------------

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]

        #[test]
        fn structural_invariants(text in "\\PC{0,200}") {
            let s = spanner(Cl100kLexer);
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

        let tokens: Vec<(Gpt2FamilyTokenRole, Range<usize>)> = Cl100kToken::lexer(text)
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
            let iter_ranges: Vec<Range<usize>> =
                Gpt2FamilySpanIter::new(text.as_str(), Cl100kToken::lexer(&text).spanned())
                    .collect();
            let oracle_ranges = oracle_word_ranges(&text);
            proptest::prop_assert_eq!(
                &iter_ranges, &oracle_ranges,
                "cl100k iter vs oracle mismatch for {:?}", text
            );
        }
    }

    // -------------------------------------------------------------------
    // proptest oracle: regex vs logos equivalence
    // -------------------------------------------------------------------

    #[test]
    #[cfg(feature = "std")]
    fn proptest_cl100k_logos_matches_regex() {
        use proptest::prelude::*;

        use crate::{
            pretrained::openai::OA_CL100K_BASE_PATTERN,
            spanners::{TextSpannerBuilder, TextSpanningConfig},
        };

        let config: TextSpanningConfig<u32> =
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN);
        let regex_spanner = TextSpannerBuilder::new(config).build();
        let logos_spanner = spanner(Cl100kLexer);

        let config = proptest::test_runner::Config::with_cases(2000);
        proptest!(config, |(text in "\\PC{0,200}")| {
            let regex_spans = regex_spanner.split_spans(&text);
            let logos_spans = logos_spanner.split_spans(&text);
            prop_assert_eq!(
                &regex_spans, &logos_spans,
                "cl100k mismatch for {:?}",
                text
            );
        });
    }
}
