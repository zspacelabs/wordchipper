//! Logos DFA lexer for the `o200k_base` pattern (GPT-4o).

use logos::Logos;

use super::gpt2_family::{
    Gpt2FamilyLogos,
    Gpt2FamilyTokenRole,
};
use crate::pretrained::openai::OA_O200K_BASE_PATTERN;
// Shorthand aliases for the character classes used in o200k:
//   UPPER      = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
//   LOWER      = [\p{Ll}\p{Lm}\p{Lo}\p{M}]
//   UPPER_ONLY = [\p{Lu}\p{Lt}]  (excludes Lm, Lo, M)
//   CONTRACTION = ('[sS]|'[tT]|'[dD]|'[mM]|'[rR][eE]|'[vV][eE]|'[lL][lL])?
//
// These are inlined below because logos derive macros require string literals.

/// Logos token variants for `o200k_base`.
///
/// Unlike cl100k, contractions attach to the preceding word. Word branches
/// are split into `WordLower`/`WordUpper` to prevent DFA longest-match from
/// merging Lo/Lm chars (e.g. `º`) with following uppercase letters.
/// `\p{M}` is excluded from prefix entrypoints to preserve regex branch
/// precedence.
#[derive(Logos, Debug, PartialEq, Clone)]
pub(crate) enum O200kToken {
    // Regex Branch 1: UPPER*LOWER+ (no prefix, first char is letter/mark).
    // Highest priority so marks in LOWER are claimed here, not as prefix.
    #[regex(
        r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'(?:s|t|d|m|re|ve|ll))?",
        priority = 4
    )]
    WordLower,

    // Regex Branch 1 with non-letter prefix.
    #[regex(
        r"[^\r\n\p{Letter}\p{Number}\p{Mark}][\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'(?:s|t|d|m|re|ve|ll))?",
        priority = 3
    )]
    PrefixedWordLower,

    // Regex Branch 2: UPPER_ONLY+LOWER* (no prefix).
    #[regex(
        r"[\p{Lu}\p{Lt}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'(?:s|t|d|m|re|ve|ll))?",
        priority = 2
    )]
    WordUpper,

    // Regex Branch 2 with non-letter prefix.
    #[regex(
        r"[^\r\n\p{Letter}\p{Number}\p{Mark}][\p{Lu}\p{Lt}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'(?:s|t|d|m|re|ve|ll))?",
        priority = 1
    )]
    PrefixedWordUpper,

    #[regex(r"\p{Number}{1,3}")]
    Digits,

    // Note: o200k includes '/' in the newline-char set after punctuation.
    //
    // Spaced punctuation (`" !..."`) is always handled by punctuation in the
    // regex alternation.
    #[regex(r" [^\s\p{Letter}\p{Number}\p{Mark}][^\s\p{Letter}\p{Number}]*[\r\n/]*")]
    PunctuationSpaced,

    // Bare punctuation: requires a second non-mark char before the continuation
    // can include marks. This prevents the DFA from grabbing `punct + marks`
    // as punctuation when the regex word branch (first-match) would claim them
    // as a word with the punct char as prefix. The optional group
    // `(?:NON_MARK REST*)?` ensures that single-punct-then-marks inputs like
    // `!\u{0300}` fall to PrefixedWordLower (matching the regex), while
    // multi-punct inputs like `!@\u{0300}` still stay in punctuation.
    #[regex(r"[^\s\p{Letter}\p{Number}\p{Mark}](?:[^\s\p{Letter}\p{Number}\p{Mark}][^\s\p{Letter}\p{Number}]*)?[\r\n/]*")]
    PunctuationBare,

    #[regex(r"\s*[\r\n]+")]
    Newline,

    #[regex(r"[^\S\r\n]+")]
    Whitespace,
}

impl Gpt2FamilyLogos<'_> for O200kToken {
    fn family_role(&self) -> Gpt2FamilyTokenRole {
        match self {
            Self::Whitespace => Gpt2FamilyTokenRole::Whitespace,
            Self::WordLower | Self::WordUpper => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
                first_char_is_letter: true,
            },
            Self::PrefixedWordLower | Self::PrefixedWordUpper => Gpt2FamilyTokenRole::Word {
                check_contraction: false,
                first_char_is_letter: false,
            },
            Self::PunctuationSpaced | Self::PunctuationBare => Gpt2FamilyTokenRole::Punctuation,
            Self::Digits | Self::Newline => Gpt2FamilyTokenRole::Standalone,
        }
    }
}

logos_lexer! {
    /// Logos DFA word scanner for `o200k_base` (GPT-4o).
    pub struct O200kLexer;
    token = O200kToken;
    pattern = OA_O200K_BASE_PATTERN;
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
    fn test_o200k_common() {
        crate::spanners::span_lexers::logos::testutil::common_lexer_tests(
            crate::alloc::boxed::Box::new(O200kLexer),
        );
    }

    #[test]
    fn test_o200k_camel_case() {
        let s = spanner(O200kLexer);

        // o200k splits on case boundaries: UPPER*LOWER+ and UPPER+LOWER*.
        // "CamelCase" -> "Camel" + "Case"
        let spans = s.split_spans("CamelCase");
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&"CamelCase"[r.clone()]),
                _ => None,
            })
            .collect();
        assert_eq!(words, &["Camel", "Case"]);

        // "getElementById" -> "get" + "Element" + "By" + "Id"
        let text = "getElementById";
        let spans = s.split_spans(text);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();
        assert_eq!(words, &["get", "Element", "By", "Id"]);

        // "HTMLParser" -> all-upper run uses UPPER+LOWER* branch.
        let text = "HTMLParser";
        let spans = s.split_spans(text);
        let words: Vec<&str> = spans
            .iter()
            .filter_map(|s| match s {
                SpanRef::Word(r) => Some(&text[r.clone()]),
                _ => None,
            })
            .collect();
        assert_eq!(words, &["HTMLParser"]);
    }

    // Regression: combining mark after punctuation must stay with the
    // punctuation, not join the following word token.
    #[test]
    fn test_mark_groups_with_punctuation() {
        let s = spanner(O200kLexer);

        // "  !\u{0300}a": mark stays with punct, letter is separate
        assert_eq!(
            s.split_spans("  !\u{0300}a"),
            vec![
                SpanRef::Word(0..1), // " "
                SpanRef::Word(1..5), // " !\u{0300}"
                SpanRef::Word(5..6), // "a"
            ]
        );

        // "  !\u{0300}\r": mark + CR stay with punct
        assert_eq!(
            s.split_spans("  !\u{0300}\r"),
            vec![
                SpanRef::Word(0..1), // " "
                SpanRef::Word(1..6), // " !\u{0300}\r"
            ]
        );

        // k=6 class 1: chained punct+mark tokens absorb into one span
        // "  !\u{0300}!\u{0300}": both punct+mark groups merge
        assert_eq!(
            s.split_spans("  !\u{0300}!\u{0300}"),
            vec![
                SpanRef::Word(0..1), // " "
                SpanRef::Word(1..8), // " !\u{0300}!\u{0300}"
            ]
        );

        // k=6 class 2: punct+mark then punct+letter splits at letter
        // "  !\u{0300}!A": prefix absorbed, letter separate
        assert_eq!(
            s.split_spans("  !\u{0300}!A"),
            vec![
                SpanRef::Word(0..1), // " "
                SpanRef::Word(1..6), // " !\u{0300}!"
                SpanRef::Word(6..7), // "A"
            ]
        );

        // k=6 whitespace after punct+mark: space starts new match,
        // not absorbed into pending punct
        assert_eq!(
            s.split_spans("  !\u{0300} A"),
            vec![
                SpanRef::Word(0..1), // " "
                SpanRef::Word(1..5), // " !\u{0300}"
                SpanRef::Word(5..7), // " A"
            ]
        );
    }
}
