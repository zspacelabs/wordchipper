//! Lexer equivalence testing utilities.
//!
//! Provides k-tuple combinatorial equivalence testing for
//! [`SpanLexer`](wordchipper::spanners::span_lexers::SpanLexer)
//! implementations, plus Unicode character class representative sets
//! for the `OpenAI` tokenizer patterns.

pub mod harness;
pub mod representatives;
