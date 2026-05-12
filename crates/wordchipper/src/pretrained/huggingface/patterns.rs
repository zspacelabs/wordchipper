//! Shared regex patterns for Hugging Face tokenizers.

use crate::{
    join_patterns,
    spanners::span_lexers::accelerators::RegexAutomataTransformHook,
    support::regex::ConstRegexPattern,
};

/// The Qwen3.5 pretrained vocabulary word pattern.
///
/// Shared by the Qwen3.5 tokenizer family loaded via Hugging Face.
pub(crate) const QWEN35_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy(join_patterns!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
));

/// Transformed Qwen3.5 pattern for `regex-automata` (lookahead removed).
///
/// The `\s+(?!\S)` branch is collapsed to `\s+`; post-processing restores
/// the original end-of-whitespace semantics.
pub(crate) const QWEN35_PATTERN_RA: &str = join_patterns!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+",
);

inventory::submit! {
    RegexAutomataTransformHook::new(QWEN35_PATTERN, QWEN35_PATTERN_RA, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns_compile() {
        assert!(QWEN35_PATTERN.compile().is_ok());
    }
}