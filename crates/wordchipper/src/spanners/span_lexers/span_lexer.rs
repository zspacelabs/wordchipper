//! # `SpanLexer` trait

use core::ops::{
    Deref,
    Range,
};

use crate::prelude::*;

/// Word-scanning plugin trait.
///
/// ## Implementation Notes
///
/// Smart pointer types that implement `Deref<Target: SpanLexer>` (such as
/// `Arc<T>`, `Box<T>`,
/// and [`PoolToy<T>`](crate::support::concurrency::PoolToy)) automatically
/// implement `SpanLexer` through a blanket implementation. This is the
/// idiomatic Rust pattern used by the standard library for traits like
/// `Iterator` and `Future`.
pub trait SpanLexer: Send + Sync {
    /// Returns an iter over matching spans in the text.
    fn find_span_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> Box<dyn Iterator<Item = Range<usize>> + 'a>;
}

// Blanket implementation for any type that derefs to a SpanLexer.
// This allows Arc<T>, Box<T>, PoolToy<T>, etc. to automatically implement
// SpanLexer.
impl<D> SpanLexer for D
where
    D: Deref + Send + Sync,
    D::Target: SpanLexer,
{
    fn find_span_iter<'a>(
        &'a self,
        text: &'a str,
    ) -> Box<dyn Iterator<Item = Range<usize>> + 'a> {
        self.deref().find_span_iter(text)
    }
}
