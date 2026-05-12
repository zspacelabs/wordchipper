//! Logos DFA lexer for the Qwen3.5 pattern.
//!
//! Shared by the Qwen3.5 tokenizer family exposed through the Hugging Face
//! loader.

use logos::Logos;

use super::gpt2_family::{
    Gpt2FamilyLogos,
    Gpt2FamilyTokenRole,
};
use crate::pretrained::huggingface::patterns::QWEN35_PATTERN;

/// Logos token variants for Qwen3.5.
#[derive(Logos, Debug, PartialEq, Clone)]
pub(crate) enum Qwen35Token {
    #[regex(r"[\p{L}\p{M}]+")]
    Letters,

    #[regex(r"[^\r\n\p{L}\p{N}][\p{L}\p{M}]+")]
    PrefixedLetters,

    #[regex(r"\p{N}")]
    Digit,

    #[regex(r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*")]
    Punctuation,

    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}

impl Gpt2FamilyLogos<'_> for Qwen35Token {
    fn family_role(&self) -> Gpt2FamilyTokenRole {
        match self {
            Self::Letters => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
                first_char_is_letter: true,
            },
            Self::PrefixedLetters => Gpt2FamilyTokenRole::Word {
                check_contraction: true,
                first_char_is_letter: false,
            },
            Self::Digit | Self::Newline => Gpt2FamilyTokenRole::Standalone,
            Self::Punctuation => Gpt2FamilyTokenRole::Punctuation,
            Self::Whitespace => Gpt2FamilyTokenRole::Whitespace,
        }
    }
}

logos_lexer! {
    /// Logos DFA word scanner for Qwen3.5.
    pub struct Qwen35Lexer;
    token = Qwen35Token;
    pattern = QWEN35_PATTERN;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{
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

    fn spanner(lexer: impl SpanLexer + 'static) -> LexerTextSpanner {
        LexerTextSpanner::new(Arc::new(lexer), None)
    }

    #[test]
    fn test_qwen35_common() {
        crate::spanners::span_lexers::logos::testutil::common_lexer_tests(
            crate::alloc::boxed::Box::new(Qwen35Lexer),
        );
    }

    #[cfg(feature = "testing")]
    #[test]
    fn test_qwen35_matches_reference() {
        use crate::spanners::span_lexers::accelerators::testutil::assert_matches_reference_lexer;
        use crate::support::regex::RegexPattern;

        let ref_lexer = RegexPattern::Fancy(QWEN35_PATTERN.as_str().into())
            .compile()
            .expect("reference pattern compiles");

        let test_lexer = Qwen35Lexer;

        let samples = &[
            "hello world",
            "  hello  world  ",
            "hello   world",
            "It's a test. Don't panic!",
            "I'm she'll they've we'd he's",
            "I'M SHE'LL THEY'VE WE'D HE'S",
            "foo123bar 456 789",
            "abc 1 2 3 def",
            "   ",
            " ",
            "",
            "a",
            "Hello, World! How are you?",
            "price is $100.00!",
            "foo   bar   baz",
            "\t\t\thello",
            "end with spaces   ",
            "\u{4e16}\u{754c}\u{4f60}\u{597d}",
            "mixed\n\n  content\there",
            "foo'bar'baz",
            "don't I'll she's",
            "'There 'The 'really",
            "'t 'T 're 'RE 'll 'll 'd 'D",
            "hello\nworld",
            "hello \n world",
            "  \n  spaces around newline  \n  ",
            "!@#$%",
            "hello!world",
            "test\r\nwindows",
            "\u{00e9}clair caf\u{00e9}",
            "e\u{0301} combining accent",
            "\u{0300}standalone mark",
        ];

        for sample in samples {
            assert_matches_reference_lexer(sample, &ref_lexer, &test_lexer);
        }
    }

    #[test]
    fn test_basic_splitting() {
        let s = spanner(Qwen35Lexer);

        assert_eq!(
            s.split_spans("hello world", None),
            vec![SpanRef::Word(0..5), SpanRef::Word(5..11)],
        );
    }

    #[test]
    fn test_single_digits() {
        let s = spanner(Qwen35Lexer);
        let text = "abc123";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|span| match span {
                SpanRef::Word(range) => Some(&text[range.clone()]),
                _ => None,
            })
            .collect();

        assert_eq!(words, vec!["abc", "1", "2", "3"]);
    }

    #[test]
    fn test_digits_do_not_absorb_space() {
        let s = spanner(Qwen35Lexer);
        let text = "abc 1";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|span| match span {
                SpanRef::Word(range) => Some(&text[range.clone()]),
                _ => None,
            })
            .collect();

        assert_eq!(words, vec!["abc", " ", "1"]);
    }

    #[test]
    fn test_contractions_case_insensitive() {
        let s = spanner(Qwen35Lexer);
        let text = "don't I'll SHE'S THEY'RE";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|span| match span {
                SpanRef::Word(range) => Some(&text[range.clone()]),
                _ => None,
            })
            .collect();

        assert!(words.contains(&"don"), "expected \"don\" in {:?}", words);
        assert!(words.contains(&"'t"), "expected \"'t\" in {:?}", words);
        assert!(words.contains(&"'ll"), "expected \"'ll\" in {:?}", words);
        assert!(words.contains(&"'S"), "expected \"'S\" in {:?}", words);
        assert!(words.contains(&"'RE"), "expected \"'RE\" in {:?}", words);
    }

    #[test]
    fn test_contraction_followed_by_more_letters() {
        let s = spanner(Qwen35Lexer);
        let text = "'There";
        let spans = s.split_spans(text, None);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|span| match span {
                SpanRef::Word(range) => Some(&text[range.clone()]),
                _ => None,
            })
            .collect();

        assert_eq!(words, vec!["'T", "here"]);
    }

    #[test]
    fn test_standalone_contraction() {
        let s = spanner(Qwen35Lexer);

        assert_eq!(s.split_spans("'t", None), vec![SpanRef::Word(0..2)],);
        assert_eq!(s.split_spans("'ll", None), vec![SpanRef::Word(0..3)],);
    }

    #[test]
    fn test_marks_attach_to_letters() {
        let s = spanner(Qwen35Lexer);
        let text = "e\u{0301}clair";
        let spans = s.split_spans(text, None);

        assert_eq!(spans.len(), 1);
        assert!(matches!(&spans[0], SpanRef::Word(range) if range == &(0..text.len())));
    }

    #[test]
    fn test_marks_not_punctuation() {
        let s = spanner(Qwen35Lexer);
        let text = "\u{0300}";
        let spans = s.split_spans(text, None);

        assert_eq!(spans, vec![SpanRef::Word(0..text.len())]);
    }

    #[test]
    fn test_no_case_split() {
        let s = spanner(Qwen35Lexer);

        assert_eq!(s.split_spans("CamelCase", None), vec![SpanRef::Word(0..9)],);
        assert_eq!(
            s.split_spans("getElementById", None),
            vec![SpanRef::Word(0..14)],
        );
        assert_eq!(s.split_spans("HTMLParser", None), vec![SpanRef::Word(0..10)],);
    }

    #[test]
    fn test_newline_absorbs_preceding_whitespace() {
        let s = spanner(Qwen35Lexer);

        assert_eq!(s.split_spans("  \n", None), vec![SpanRef::Word(0..3)],);
    }

    #[test]
    fn test_punctuation_optional_space() {
        let s = spanner(Qwen35Lexer);

        assert_eq!(s.split_spans(" !", None), vec![SpanRef::Word(0..2)],);
        assert_eq!(
            s.split_spans("  !", None),
            vec![SpanRef::Word(0..1), SpanRef::Word(1..3)],
        );
    }

    #[test]
    fn test_punctuation_trailing_newlines() {
        let s = spanner(Qwen35Lexer);

        assert_eq!(s.split_spans("!\n\n", None), vec![SpanRef::Word(0..3)],);
    }
}