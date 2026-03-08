//! # `OpenAI` Patterns

use crate::{
    join_patterns,
    support::regex::ConstRegexPattern,
};

/// The original "`gpt2`" vocabulary word pattern.
///
/// Slower, use [`OA_R50K_BASE_PATTERN`].
pub const OA_GPT2_PATTERN_SLOW: ConstRegexPattern = ConstRegexPattern::Fancy(join_patterns!(
    r"'s",
    r"'t",
    r"'re",
    r"'ve",
    r"'m",
    r"'ll",
    r"'d",
    r" ?[\p{L}]+",
    r" ?[\p{N}]+",
    r" ?[^\s\p{L}\p{N}]+",
    r"\s+(?!\S)",
    r"\s+",
));

/// The optimized "`gpt2`" vocabulary word pattern.
///
/// Faster than [`OA_GPT2_PATTERN_SLOW`], optimized for performance.
pub const OA_GPT2_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy(join_patterns!(
    r"'(?:[sdmt]|ll|ve|re)",
    r" ?\p{L}++",
    r" ?\p{N}++",
    r" ?[^\s\p{L}\p{N}]++",
    r"\s++$",
    r"\s+(?!\S)",
    r"\s",
));

/// The optimized "`gpt2`" vocabulary word pattern.
///
/// Faster than [`OA_GPT2_PATTERN_SLOW`], optimized for performance.
pub const OA_R50K_BASE_PATTERN: ConstRegexPattern = OA_GPT2_PATTERN;

/// The "`p50k_base`" pretrained vocabulary word pattern.
pub const OA_P50K_BASE_PATTERN: ConstRegexPattern = OA_R50K_BASE_PATTERN;

/// The "`cl100k_base`" pretrained vocabulary word pattern.
pub const OA_CL100K_BASE_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy(join_patterns!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"[^\r\n\p{L}\p{N}]?+\p{L}++",
    r"\p{N}{1,3}+",
    r" ?[^\s\p{L}\p{N}]++[\r\n]*+",
    r"\s++$",
    r"\s*[\r\n]",
    r"\s+(?!\S)",
    r"\s",
));

/// The "`o200k_base`" pretrained vocabulary word pattern.
pub const OA_O200K_BASE_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy(join_patterns!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+"
));

// Transformed patterns for `regex-automata` (no lookaheads, no possessive
// quantifiers). Used by the `RegexAutomataLexer`; the `\s+(?!\S)` + `\s`
// lookahead branches are collapsed to `\s+` with post-processing to split
// off the last character (matching lookahead semantics). Other whitespace
// branches (`\s+$`, `\s*[\r\n]`) are preserved as-is.

/// Transformed `r50k_base`/`gpt2` pattern for `regex-automata`.
pub(crate) const OA_R50K_BASE_PATTERN_RA: &str = join_patterns!(
    r"'(?:[sdmt]|ll|ve|re)",
    r" ?\p{L}+",
    r" ?\p{N}+",
    r" ?[^\s\p{L}\p{N}]+",
    r"\s+$",
    r"\s+",
);

/// Transformed `cl100k_base` pattern for `regex-automata`.
pub(crate) const OA_CL100K_BASE_PATTERN_RA: &str = join_patterns!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"[^\r\n\p{L}\p{N}]?\p{L}+",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s+$",
    r"\s*[\r\n]",
    r"\s+",
);

/// Transformed `o200k_base` pattern for `regex-automata`.
pub(crate) const OA_O200K_BASE_PATTERN_RA: &str = join_patterns!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+",
);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_patterns_compile() {
        assert!(OA_R50K_BASE_PATTERN.compile().is_ok());
        assert!(OA_GPT2_PATTERN_SLOW.compile().is_ok());

        assert!(OA_CL100K_BASE_PATTERN.compile().is_ok());

        assert!(OA_CL100K_BASE_PATTERN.compile().is_ok());
        assert!(OA_O200K_BASE_PATTERN.compile().is_ok());
    }
}
