//! # Regex Wrapper
//! This modules provides mechanisms to mix `regex` and `fancy_regex` types.

use core::{fmt::Debug, ops::Range};

use crate::{alloc::boxed::Box, spanners::span_lexers::SpanLexer, support::regex::RegexPattern};

/// Error wrapper for regex patterns.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ErrorWrapper {
    /// Error from `regex`.
    Basic(Box<regex::Error>),

    /// Error from `fancy_regex`.
    Fancy(Box<fancy_regex::Error>),
}

impl From<regex::Error> for ErrorWrapper {
    fn from(err: regex::Error) -> Self {
        Self::Basic(err.into())
    }
}

impl From<fancy_regex::Error> for ErrorWrapper {
    fn from(err: fancy_regex::Error) -> Self {
        Self::Fancy(err.into())
    }
}

impl core::fmt::Display for ErrorWrapper {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            Self::Basic(err) => core::fmt::Display::fmt(err, f),
            Self::Fancy(err) => core::fmt::Display::fmt(err, f),
        }
    }
}

impl core::error::Error for ErrorWrapper {}

impl From<RegexPattern> for RegexWrapper {
    fn from(pattern: RegexPattern) -> Self {
        pattern.compile().unwrap()
    }
}

/// Wrapper for regex patterns.
#[derive(Debug, Clone)]
pub enum RegexWrapper {
    /// Wrapper for `regex::Regex`.
    Basic(regex::Regex),

    /// Wrapper for `fancy_regex::Regex`.
    Fancy(fancy_regex::Regex),
}

impl SpanLexer for RegexWrapper {
    fn next_span(
        &self,
        text: &str,
    ) -> Option<Range<usize>> {
        self.find_iter(text).next().map(|m| m.range())
    }
}

impl PartialEq for RegexWrapper {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        match (self, other) {
            (Self::Basic(a), Self::Basic(b)) => a.as_str() == b.as_str(),
            (Self::Fancy(a), Self::Fancy(b)) => a.as_str() == b.as_str(),
            _ => false,
        }
    }
}

impl From<regex::Regex> for RegexWrapper {
    fn from(regex: regex::Regex) -> Self {
        Self::Basic(regex)
    }
}

impl From<fancy_regex::Regex> for RegexWrapper {
    fn from(regex: fancy_regex::Regex) -> Self {
        Self::Fancy(regex)
    }
}

impl RegexWrapper {
    /// Is this `Basic`?
    ///
    /// ## Returns
    /// `true` if it wraps a `regex::Regex`, `false` otherwise.
    pub fn is_basic(&self) -> bool {
        match self {
            Self::Basic(_) => true,
            Self::Fancy(_) => false,
        }
    }

    /// Is this `Fancy`?
    ///
    /// ## Returns
    /// `true` if it wraps a `fancy_regex::Regex`, `false` otherwise.
    pub fn is_fancy(&self) -> bool {
        match self {
            Self::Basic(_) => false,
            Self::Fancy(_) => true,
        }
    }

    /// Get the underlying regex pattern.
    ///
    /// ## Returns
    /// The regex pattern string slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Basic(regex) => regex.as_str(),
            Self::Fancy(regex) => regex.as_str(),
        }
    }

    /// Wrapper for `find_iter`.
    ///
    /// ## Arguments
    /// * `haystack` - The string to search in.
    ///
    /// ## Returns
    /// A `MatchesWrapper` iterator over the matches.
    pub fn find_iter<'r, 'h>(
        &'r self,
        haystack: &'h str,
    ) -> MatchesWrapper<'r, 'h> {
        match self {
            Self::Basic(regex) => regex.find_iter(haystack).into(),
            Self::Fancy(regex) => regex.find_iter(haystack).into(),
        }
    }
}

/// Wrapper for regex matches.
pub enum MatchesWrapper<'r, 'h> {
    /// Wrapper for `regex::Matches`.
    Regex(regex::Matches<'r, 'h>),

    /// Wrapper for `fancy_regex::Matches`.
    FancyRegex(fancy_regex::Matches<'r, 'h>),
}

impl<'r, 'h> From<regex::Matches<'r, 'h>> for MatchesWrapper<'r, 'h> {
    fn from(matches: regex::Matches<'r, 'h>) -> Self {
        Self::Regex(matches)
    }
}

impl<'r, 'h> From<fancy_regex::Matches<'r, 'h>> for MatchesWrapper<'r, 'h> {
    fn from(matches: fancy_regex::Matches<'r, 'h>) -> Self {
        Self::FancyRegex(matches)
    }
}

impl<'r, 'h> Iterator for MatchesWrapper<'r, 'h> {
    type Item = regex::Match<'h>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Regex(matches) => matches.next(),
            Self::FancyRegex(matches) => matches
                .next()
                .map(|m| unsafe { core::mem::transmute(m.unwrap()) }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{format, string::ToString},
        join_patterns,
        support::regex::regex_pattern::ConstRegexPattern,
    };

    #[test]
    fn test_partial_eq() {
        let b0 = RegexPattern::Basic("hello world".to_string())
            .compile()
            .unwrap();
        let b1 = RegexPattern::Basic("world".to_string()).compile().unwrap();

        let f0 = RegexPattern::Fancy("hello world".to_string())
            .compile()
            .unwrap();
        let f1 = RegexPattern::Fancy("world".to_string()).compile().unwrap();

        assert_eq!(&b0, &b0);
        assert_eq!(&b1, &b1);
        assert_ne!(&b0, &b1);
        assert_ne!(&b1, &b0);

        assert_eq!(&f0, &f0);
        assert_eq!(&f1, &f1);
        assert_ne!(&f0, &f1);
        assert_ne!(&f1, &f0);

        assert_ne!(&b0, &f0);
        assert_ne!(&f0, &b0);

        assert_ne!(&b1, &f1);
        assert_ne!(&f1, &b1);
    }

    #[test]
    fn test_const_pattern() {
        const BASIC: ConstRegexPattern = ConstRegexPattern::Basic("hello world");
        assert_eq!(BASIC.as_str(), "hello world");

        let rw = BASIC.compile().unwrap();
        assert_eq!(rw.as_str(), "hello world");
        assert!(rw.is_basic());
        assert!(!rw.is_fancy());

        let pattern = BASIC.to_pattern();
        assert_eq!(pattern.as_str(), "hello world");

        const FANCY: ConstRegexPattern = ConstRegexPattern::Fancy("hello");
        assert_eq!(FANCY.as_str(), "hello");

        let rw = FANCY.compile().unwrap();
        assert_eq!(rw.as_str(), "hello");
        assert!(!rw.is_basic());
        assert!(rw.is_fancy());

        let pattern = FANCY.to_pattern();
        assert_eq!(pattern.as_str(), "hello");
    }

    #[test]
    fn test_basic_pattern() {
        let pattern = RegexPattern::Basic("hello world".to_string());
        assert_eq!(pattern.as_str(), "hello world");

        let rw: RegexWrapper = pattern.into();
        assert_eq!(rw.as_str(), "hello world");
        assert!(rw.is_basic());
        assert!(!rw.is_fancy());
    }

    #[test]
    fn test_fancy_pattern() {
        let pattern = RegexPattern::Fancy("hello world".to_string());
        assert_eq!(pattern.as_str(), "hello world");

        let rw = pattern.compile().unwrap();
        assert_eq!(rw.as_str(), "hello world");
        assert!(!rw.is_basic());
        assert!(rw.is_fancy());
    }

    #[test]
    fn test_adaptive_pattern() {
        let pattern: RegexPattern = "hello world".to_string().into();
        assert_eq!(pattern.as_str(), "hello world");

        let rw = pattern.compile().unwrap();
        assert_eq!(rw.as_str(), "hello world");
        assert!(rw.is_basic());
        assert!(!rw.is_fancy());
    }

    const FANCY_PATTERN: &str = join_patterns!(
        r"'(?:[sdmt]|ll|ve|re)",
        r" ?\p{L}++",
        r" ?\p{N}++",
        r" ?[^\s\p{L}\p{N}]++",
        r"\s++$",
        r"\s+(?!\S)",
        r"\s",
    );

    #[test]
    fn test_basic_pattern_failure() {
        let pattern = RegexPattern::Basic(FANCY_PATTERN.to_string());
        let err = pattern.compile().unwrap_err();
        assert!(matches!(err, ErrorWrapper::Basic(_)));

        assert!(format!("{}", err).contains("regex parse error"));
    }

    #[test]
    fn test_fancy_pattern_failure() {
        let pattern = RegexPattern::Fancy(r"[".to_string());
        let err = pattern.compile().unwrap_err();
        assert!(matches!(err, ErrorWrapper::Fancy(_)));

        assert!(format!("{}", err).contains("Parsing error"));
    }

    #[test]
    fn test_adaptive_pattern_fallback() {
        let pattern: RegexPattern = FANCY_PATTERN.into();
        assert!(matches!(pattern, RegexPattern::Adaptive(_)));

        let rw = pattern.compile().unwrap();
        assert!(!rw.is_basic());
        assert!(rw.is_fancy());
    }
}
