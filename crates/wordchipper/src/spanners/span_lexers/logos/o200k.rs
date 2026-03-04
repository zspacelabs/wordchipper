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

    // Bare punctuation (`"!..."`) must not consume an immediate Mark as the
    // 2nd core char; regex branch 1 matches `!◌` before punctuation.
    #[regex(r"[^\s\p{Letter}\p{Number}\p{Mark}]\p{Mark}+", priority = 6)]
    PunctuationBareMark,

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
            Self::PunctuationSpaced | Self::PunctuationBareMark | Self::PunctuationBare => {
                Gpt2FamilyTokenRole::Punctuation
            }
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
}
