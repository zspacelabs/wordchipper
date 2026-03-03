//! # Regex Utilities
//!
//! A number of popular in-use LLM Tokenizer Regex Patterns require extended
//! regex machinery provided by the [`fancy_regex`] crate; but naturally, this
//! has performance costs. We'd prefer to avoid using the [`fancy_regex`] crate
//! when possible, falling back on the standard [`regex`] crate when patterns
//! permit this.
//!
//! This recurses into two problems:
//!
//! * Labeling Patterns - [`RegexPattern`]
//!   * [`RegexPattern::Basic`] - a pattern which was written for basic regular
//!     expressions.
//!   * [`RegexPattern::Fancy`] - a pattern which was written for regex
//!     extensions.
//!   * [`RegexPattern::Adaptive`] - unknown target, try basic; then fall-up to
//!     fancy.
//! * Wrapping Compiled Regex - [`RegexWrapper`]
//!
//! The [`RegexWrapper`] type supports only one operation, ``find_iter()`` which
//! requires some adaptation of the `Iterator` stream to function.

mod alt_choice;
mod regex_pattern;
mod regex_wrapper;

#[doc(inline)]
pub use alt_choice::alternate_choice_regex_pattern;
#[doc(inline)]
pub use regex_pattern::*;
#[doc(inline)]
pub use regex_wrapper::*;
