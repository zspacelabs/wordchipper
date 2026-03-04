//! Logos DFA lexer for the `r50k_base` pattern (GPT-2).

use logos::Logos;

use super::gpt2_family::{
    Gpt2FamilyLogos,
    Gpt2FamilyTokenRole,
};
use crate::pretrained::openai::OA_R50K_BASE_PATTERN;

/// Logos token variants for `r50k_base`.
#[derive(Logos, Debug, PartialEq, Clone)]
pub(crate) enum R50kToken {
    // Case-sensitive: the r50k regex `'(?:[sdmt]|ll|ve|re)` only matches
    // lowercase suffixes, unlike cl100k which uses `(?i:...)`.
    #[regex(r"'([sdtm]|re|ve|ll)")]
    Contraction,

    #[regex(r" ?\p{Letter}+")]
    Letters,

    #[regex(r" ?\p{Number}+")]
    Digits,

    #[regex(r" ?[^\s\p{Letter}\p{Number}]+")]
    Punctuation,

    // All whitespace including \r\n. The r50k regex treats all \s uniformly
    // via `\s+(?!\S)` and `\s`; there is no separate newline branch.
    #[regex(r"\s+")]
    Whitespace,
}

impl Gpt2FamilyLogos<'_> for R50kToken {
    fn family_role(&self) -> Gpt2FamilyTokenRole {
        match self {
            Self::Whitespace => Gpt2FamilyTokenRole::Whitespace,
            // All three r50k content patterns use ` ?X` which absorbs only
            // literal ASCII space. This matches the Punctuation role behavior.
            Self::Letters | Self::Digits | Self::Punctuation => Gpt2FamilyTokenRole::Punctuation,
            // When preceded by whitespace ending in ASCII space, the regex
            // ` ?[^\s\p{L}\p{N}]+` absorbs the space + apostrophe as punctuation,
            // leaving the suffix letter(s) as a separate span. The Word role's
            // non-letter-prefix path replicates this: it merges the last ws char
            // with the apostrophe and emits the suffix separately.
            Self::Contraction => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
                first_char_is_letter: false,
            },
        }
    }
}

logos_lexer! {
    /// Logos DFA word scanner for `r50k_base` (GPT-2).
    pub struct R50kLexer;
    token = R50kToken;
    pattern = OA_R50K_BASE_PATTERN;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{
            string::ToString,
            sync::Arc,
            vec,
            vec::Vec,
        },
        spanners::{
            SpanRef,
            TextSpanner,
            span_lexers::{
                LexerTextSpanner,
                SpanLexer,
            },
        },
    };

    /// Build a `TextSpanner` from a logos lexer with no specials.
    fn spanner(lexer: impl SpanLexer + 'static) -> LexerTextSpanner {
        LexerTextSpanner::new(Arc::new(lexer), None)
    }

    #[test]
    fn test_logos_basic_splitting() {
        let s = spanner(R50kLexer);
        let text = "hello world";
        let spans = s.split_spans(text);

        assert_eq!(spans, vec![SpanRef::Word(0..5), SpanRef::Word(5..11),]);
    }

    #[test]
    fn test_logos_with_specials() {
        let special_pattern = crate::support::regex::alternate_choice_regex_pattern(&[
            "<|FNORD|>".to_string(),
            "<|NORP|>".to_string(),
        ]);
        let s = LexerTextSpanner::new(
            Arc::new(R50kLexer),
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
        let s = spanner(R50kLexer);
        let text = "abc 123 4567";
        let spans = s.split_spans(text);

        // Space absorbed by ` ?` prefix in each content pattern.
        assert_eq!(
            spans,
            vec![
                SpanRef::Word(0..3),  // "abc"
                SpanRef::Word(3..7),  // " 123"
                SpanRef::Word(7..12), // " 4567"
            ]
        );
    }

    #[test]
    fn test_logos_contractions() {
        let s = spanner(R50kLexer);
        let text = "don't I'll she's";
        let spans = s.split_spans(text);

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
        let s = spanner(R50kLexer);
        let spans = s.split_spans("");
        assert!(spans.is_empty());
    }

    #[test]
    fn test_logos_whitespace_only() {
        let s = spanner(R50kLexer);
        let text = "   ";
        let spans = s.split_spans(text);

        assert_eq!(spans, vec![SpanRef::Word(0..3)]);
    }

    #[test]
    fn test_logos_camel_case() {
        let s = spanner(R50kLexer);
        // r50k uses \p{L}+ so CamelCase is one token.
        assert_eq!(s.split_spans("CamelCase"), vec![SpanRef::Word(0..9)]);
        assert_eq!(s.split_spans("getElementById"), vec![SpanRef::Word(0..14)]);
    }
}
