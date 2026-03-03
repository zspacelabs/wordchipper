//! # Regex Pattern Labeled Wrapper

use crate::{
    alloc::string::{
        String,
        ToString,
    },
    support::regex::{
        ErrorWrapper,
        RegexWrapper,
    },
};

/// Const Regex Wrapper Pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstRegexPattern {
    /// This is a basic regex pattern, without extensions.
    Basic(&'static str),

    /// This is a regex pattern that requires regex extensions.
    Fancy(&'static str),
}

impl core::fmt::Display for ConstRegexPattern {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl ConstRegexPattern {
    /// Get the underlying regex pattern.
    ///
    /// ## Returns
    /// The regex pattern string slice.
    pub const fn as_str(&self) -> &str {
        match self {
            Self::Basic(pattern) => pattern,
            Self::Fancy(pattern) => pattern,
        }
    }

    /// Convert to [`RegexPattern`]
    ///
    /// ## Returns
    /// A new `RegexWrapperPattern` instance.
    pub fn to_pattern(&self) -> RegexPattern {
        (*self).into()
    }

    /// Compile the regex pattern into a `RegexWrapper`.
    ///
    /// ## Returns
    /// A `Result` containing the compiled `RegexWrapper` or an `ErrorWrapper`.
    pub fn compile(&self) -> Result<RegexWrapper, ErrorWrapper> {
        RegexPattern::from(*self).compile()
    }
}

impl From<ConstRegexPattern> for RegexPattern {
    fn from(pattern: ConstRegexPattern) -> Self {
        use ConstRegexPattern::*;
        match pattern {
            Basic(pattern) => RegexPattern::Basic(pattern.to_string()),
            Fancy(pattern) => RegexPattern::Fancy(pattern.to_string()),
        }
    }
}

/// Labeled wrapper for regex patterns.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RegexPattern {
    /// This is a basic regex pattern, without extensions.
    Basic(String),

    /// This is a regex pattern that requires regex extensions.
    Fancy(String),

    /// The requirements of this pattern and may require regex extensions.
    Adaptive(String),
}

impl<S: AsRef<str>> From<S> for RegexPattern {
    fn from(pattern: S) -> Self {
        Self::Adaptive(pattern.as_ref().to_string())
    }
}

impl RegexPattern {
    /// Get the underlying regex pattern.
    ///
    /// ## Returns
    /// The regex pattern string slice.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Basic(pattern) => pattern,
            Self::Fancy(pattern) => pattern,
            Self::Adaptive(pattern) => pattern,
        }
    }

    /// Compile the regex pattern into a `RegexWrapper`.
    ///
    /// ## Returns
    /// A `Result` containing the compiled `RegexWrapper` or an `ErrorWrapper`.
    pub fn compile(&self) -> Result<RegexWrapper, ErrorWrapper> {
        match self {
            Self::Basic(pattern) => regex::Regex::new(pattern)
                .map(RegexWrapper::from)
                .map_err(ErrorWrapper::from),
            Self::Fancy(pattern) => fancy_regex::Regex::new(pattern)
                .map(RegexWrapper::from)
                .map_err(ErrorWrapper::from),
            Self::Adaptive(pattern) => {
                regex::Regex::new(pattern)
                    .map(RegexWrapper::from)
                    .or_else(|_| {
                        fancy_regex::Regex::new(pattern)
                            .map(RegexWrapper::from)
                            .map_err(ErrorWrapper::from)
                    })
            }
        }
    }
}
