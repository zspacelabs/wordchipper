//! # Logos-based lexers
//!
//! Composable building blocks for compile-time DFA word scanners using
//! the [`logos`](https://docs.rs/logos) crate.
//!
//! ## Building a custom lexer
//!
//! 1. Define a `#[derive(Logos)]` enum with your token patterns.
//! 2. Map each variant to a [`Gpt2FamilyTokenRole`] (via a `role()` method or
//!    similar).
//! 3. Implement [`SpanLexer`](super::SpanLexer) by feeding the token stream to
//!    [`for_each_classified_span`].
//!
//! See [`Cl100kLexer`] and [`O200kLexer`] for reference implementations.

pub mod cl100k;
pub mod gpt2_family;
pub mod o200k;
pub mod r50k;
