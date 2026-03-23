//! Logos DFA lexer for the `cl100k_base` pattern (GPT-4, GPT-3.5).

use logos::Logos;

use super::gpt2_family::{
    Gpt2FamilyLogos,
    Gpt2FamilyTokenRole,
};
use crate::pretrained::openai::OA_CL100K_BASE_PATTERN;

/// Logos token variants for `cl100k_base`.
#[derive(Logos, Debug, PartialEq, Clone)]
pub(crate) enum Cl100kToken {
    #[regex(r"'[sStTdDmM]|'[rR][eE]|'[vV][eE]|'[lL][lL]", priority = 5)]
    Contraction,

    #[regex(r"\p{Letter}+")]
    Letters,

    #[regex(r"[^\r\n\p{Letter}\p{Number}]\p{Letter}+")]
    PrefixedLetters,

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
                first_char_is_letter: true,
            },
            Self::PrefixedLetters => Gpt2FamilyTokenRole::Word {
                check_contraction: true,
                first_char_is_letter: false,
            },
            Self::Contraction => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
                first_char_is_letter: false,
            },
            Self::Punctuation => Gpt2FamilyTokenRole::Punctuation,
            Self::Newline => Gpt2FamilyTokenRole::Newline,
            Self::Digits => Gpt2FamilyTokenRole::Standalone,
        }
    }
}

logos_lexer! {
    /// Logos DFA word scanner for `cl100k_base` (GPT-4, GPT-3.5).
    pub struct Cl100kLexer;
    token = Cl100kToken;
    pattern = OA_CL100K_BASE_PATTERN;
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
    fn test_cl100k_common() {
        crate::spanners::span_lexers::logos::testutil::common_lexer_tests(
            crate::alloc::boxed::Box::new(Cl100kLexer),
        );
    }

    #[test]
    fn test_logos_camel_case() {
        let s = spanner(Cl100kLexer);
        // cl100k uses \p{L}+ so CamelCase is one token.
        assert_eq!(s.split_spans("CamelCase"), vec![SpanRef::Word(0..9)]);
        assert_eq!(s.split_spans("getElementById"), vec![SpanRef::Word(0..14)]);
        assert_eq!(s.split_spans("HTMLParser"), vec![SpanRef::Word(0..10)]);
    }
}
