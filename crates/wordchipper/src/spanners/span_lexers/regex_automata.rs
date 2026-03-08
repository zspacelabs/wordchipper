//! [`SpanLexer`] backed by `regex_automata::meta::Regex` with external cache
//! management.
//!
//! When the `concurrent` feature is enabled, caches are distributed across
//! pool slots via `PoolToy<Mutex<Cache>>`. Without `concurrent`, a single
//! `spin::Mutex<Cache>` is used instead (still much faster than fancy-regex).

use core::{
    num::NonZeroUsize,
    ops::Range,
};

use regex_automata::{
    Input,
    meta::{
        Cache,
        Regex,
    },
};
use spin::Mutex;

#[cfg(feature = "concurrent")]
use crate::support::concurrency::PoolToy;
use crate::{
    alloc::sync::Arc,
    prelude::*,
    pretrained::openai::patterns::{
        OA_CL100K_BASE_PATTERN,
        OA_CL100K_BASE_PATTERN_RA,
        OA_O200K_BASE_PATTERN,
        OA_O200K_BASE_PATTERN_RA,
        OA_R50K_BASE_PATTERN,
        OA_R50K_BASE_PATTERN_RA,
    },
    spanners::span_lexers::SpanLexer,
};

/// Known pattern transforms: (original fancy pattern, transformed RA pattern,
/// `has_newline_branch`).
const KNOWN_TRANSFORMS: &[(&str, &str, bool)] = &[
    (
        OA_R50K_BASE_PATTERN.as_str(),
        OA_R50K_BASE_PATTERN_RA,
        false,
    ),
    (
        OA_CL100K_BASE_PATTERN.as_str(),
        OA_CL100K_BASE_PATTERN_RA,
        true,
    ),
    (
        OA_O200K_BASE_PATTERN.as_str(),
        OA_O200K_BASE_PATTERN_RA,
        true,
    ),
];

/// `SpanLexer` using `regex_automata::meta::Regex` with pooled or single-mutex
/// caches.
struct RegexAutomataLexer {
    regex: Regex,
    #[cfg(feature = "concurrent")]
    cache_pool: PoolToy<Mutex<Cache>>,
    #[cfg(not(feature = "concurrent"))]
    cache: Mutex<Cache>,
    has_newline_branch: bool,
}

impl SpanLexer for RegexAutomataLexer {
    fn find_span_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> Box<dyn Iterator<Item = Range<usize>> + 'a> {
        #[cfg(feature = "concurrent")]
        let slot = self.cache_pool.get();
        #[cfg(not(feature = "concurrent"))]
        let slot = &self.cache;

        let mut cache = slot.lock();
        let mut spans = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let input = Input::new(text).range(pos..);
            let Some(m) = self.regex.search_with(&mut cache, &input) else {
                break;
            };
            let range = m.range();
            if range.is_empty() {
                // Advance by one UTF-8 character, not one byte, to avoid
                // landing mid-sequence on multi-byte characters.
                pos += text[pos..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(1);
                continue;
            }
            // Whitespace truncation: if this is a multi-char all-whitespace
            // match not at EOF, truncate to all-but-last-char and rewind pos
            // so the trailing char can be absorbed by the next match.
            if needs_ws_truncate(text, &range, self.has_newline_branch) {
                let trunc = last_char_boundary(text, &range);
                spans.push(range.start..trunc);
                pos = trunc;
            } else {
                spans.push(range.start..range.end);
                pos = range.end;
            }
        }
        Box::new(spans.into_iter())
    }
}

/// Returns true if this span is all-whitespace and should be truncated
/// (drop the last character) so it can be absorbed by the next match,
/// simulating `\s+(?!\S)` + `\s` behavior.
fn needs_ws_truncate(
    text: &str,
    span: &Range<usize>,
    has_newline_branch: bool,
) -> bool {
    if span.end >= text.len() {
        return false;
    }
    let s = &text[span.clone()];
    // Single pass: check multi-char, all-whitespace, and newline presence.
    let mut char_count = 0u32;
    let mut has_newline = false;
    for c in s.chars() {
        if !c.is_whitespace() {
            return false;
        }
        char_count += 1;
        if c == '\r' || c == '\n' {
            has_newline = true;
        }
    }
    if char_count <= 1 {
        return false;
    }
    // For patterns with a newline branch, skip spans containing \r or \n
    // (those are handled by the newline branch, not the \s+ branch).
    if has_newline_branch && has_newline {
        return false;
    }
    true
}

/// Find the byte offset of the last character boundary in a span.
fn last_char_boundary(
    text: &str,
    span: &Range<usize>,
) -> usize {
    let s = &text[span.clone()];
    let last_char_len = s.chars().next_back().map(|c| c.len_utf8()).unwrap_or(1);
    span.end - last_char_len
}

/// Try to build a `RegexAutomataLexer` for the given pattern.
///
/// Checks against a table of known `OpenAI` pattern transforms. If the pattern
/// matches, builds a `regex_automata::meta::Regex` from the transformed
/// (lookahead-free) pattern with whitespace post-processing.
///
/// If the pattern is not in the known-transform table, tries compiling it
/// directly (for Basic patterns that have no lookaheads).
pub(crate) fn try_build(
    pattern: &str,
    max_pool: Option<NonZeroUsize>,
) -> Option<Arc<dyn SpanLexer>> {
    // Check known transforms.
    for &(original, transformed, has_newline_branch) in KNOWN_TRANSFORMS {
        if pattern == original {
            let regex = match Regex::new(transformed) {
                Ok(r) => r,
                Err(e) => {
                    log::warn!(
                        "regex-automata failed to compile known transform (len={}): {e}",
                        transformed.len(),
                    );
                    return None;
                }
            };
            return Some(build_lexer(regex, has_newline_branch, max_pool));
        }
    }

    // Fallback: try compiling directly (for patterns without lookaheads).
    let regex = Regex::new(pattern).ok()?;
    Some(build_lexer(regex, false, max_pool))
}

fn build_lexer(
    regex: Regex,
    has_newline_branch: bool,
    _max_pool: Option<NonZeroUsize>,
) -> Arc<dyn SpanLexer> {
    #[cfg(feature = "concurrent")]
    {
        let cache_pool = build_cache_pool(&regex, _max_pool);
        Arc::new(RegexAutomataLexer {
            regex,
            cache_pool,
            has_newline_branch,
        })
    }

    #[cfg(not(feature = "concurrent"))]
    {
        let cache = Mutex::new(regex.create_cache());
        Arc::new(RegexAutomataLexer {
            regex,
            cache,
            has_newline_branch,
        })
    }
}

#[cfg(feature = "concurrent")]
fn build_cache_pool(
    regex: &Regex,
    max_pool: Option<NonZeroUsize>,
) -> PoolToy<Mutex<Cache>> {
    let size = crate::support::concurrency::threads::resolve_max_pool(max_pool);
    let pool: Vec<_> = (0..size)
        .map(|_| Mutex::new(regex.create_cache()))
        .collect();
    PoolToy::from_pool(pool)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{
        spanners::span_lexers::accelerators::testutil::assert_matches_reference_lexer,
        support::regex::RegexWrapper,
    };

    fn ref_lexer(pattern: &str) -> RegexWrapper {
        crate::support::regex::RegexPattern::Fancy(pattern.to_string())
            .compile()
            .unwrap()
    }

    fn ra_lexer(pattern: &str) -> Arc<dyn SpanLexer> {
        try_build(pattern, None).expect("should build")
    }

    const TEST_SAMPLES: &[&str] = &[
        "hello world",
        "  hello  world  ",
        "hello   world",
        "hello\tworld",
        "hello\n\nworld",
        "hello\r\nworld",
        "It's a test. Don't panic!",
        "I'm she'll they've we'd he's",
        "foo123bar 456 789",
        "   ",
        " ",
        "",
        "a",
        "Hello, World! How are you?",
        "price is $100.00!",
        "foo   bar   baz",
        "\t\t\thello",
        "end with spaces   ",
        "Unicode: \u{00A0}\u{2003}test",
        "\u{4e16}\u{754c}\u{4f60}\u{597d}",
        "mixed\n\n  content\there",
        "foo'bar'baz",
        "123\n456\n789",
    ];

    fn check_pattern(original: &str) {
        let reference = ref_lexer(original);
        let test = ra_lexer(original);
        for sample in TEST_SAMPLES {
            assert_matches_reference_lexer(sample, &reference, test.as_ref());
        }
    }

    #[test]
    fn test_r50k_matches_reference() {
        check_pattern(OA_R50K_BASE_PATTERN.as_str());
    }

    #[test]
    fn test_cl100k_matches_reference() {
        check_pattern(OA_CL100K_BASE_PATTERN.as_str());
    }

    #[test]
    fn test_o200k_matches_reference() {
        check_pattern(OA_O200K_BASE_PATTERN.as_str());
    }

    #[test]
    fn test_basic_whitespace_truncation() {
        // "hello   world" with r50k: "   " is truncated to "  ", then
        // " world" is matched by ` ?\p{L}+`.
        let lexer = ra_lexer(OA_R50K_BASE_PATTERN.as_str());
        let spans: Vec<_> = lexer.find_span_iter("hello   world").collect();
        let texts: Vec<&str> = spans.iter().map(|r| &"hello   world"[r.clone()]).collect();
        assert_eq!(texts, vec!["hello", "  ", " world"]);
    }

    #[test]
    fn test_trailing_whitespace_no_split() {
        // Whitespace at end of text should NOT be split.
        let lexer = ra_lexer(OA_R50K_BASE_PATTERN.as_str());
        let text = "hello   ";
        let spans: Vec<_> = lexer.find_span_iter(text).collect();
        let texts: Vec<&str> = spans.iter().map(|r| &text[r.clone()]).collect();
        assert_eq!(texts, vec!["hello", "   "]);
    }

    #[test]
    fn test_direct_basic_pattern() {
        // A simple pattern without lookaheads should compile directly.
        let lexer = try_build(r"\w+|\s+", None);
        assert!(lexer.is_some());
        let lexer = lexer.unwrap();
        let spans: Vec<_> = lexer.find_span_iter("hello world").collect();
        let texts: Vec<&str> = spans.iter().map(|r| &"hello world"[r.clone()]).collect();
        assert_eq!(texts, vec!["hello", " ", "world"]);
    }

    #[test]
    fn test_fancy_pattern_returns_none_on_invalid() {
        // A pattern with lookaheads should fail to compile in regex-automata fallback.
        // But the known transform table should handle OpenAI patterns.
        let result = try_build(r"\s+(?!\S)", None);
        // regex-automata can't compile lookaheads, so this should fail.
        assert!(result.is_none());
    }
}
