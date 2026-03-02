//! # `SpanLexer` trait

use core::ops::{Deref, Range};

/// Word-scanning plugin trait.
///
/// Implementors provide word-level text segmentation. The default
/// [`for_each_word`](Self::for_each_word) loops over
/// [`next_span`](Self::next_span) matches, emitting `Word` and `Gap` spans.
/// Lexers that produce richer token streams (e.g. logos DFA) override
/// `for_each_word` directly and leave `next_span` at its default.
///
/// ## Implementation Notes
///
/// Smart pointer types that implement `Deref<Target: SpanLexer>` (such as `Arc<T>`, `Box<T>`,
/// and [`PoolToy<T>`](crate::support::concurrency::PoolToy)) automatically implement `SpanLexer` through
/// a blanket implementation. This is the idiomatic Rust pattern used by the standard library
/// for traits like `Iterator` and `Future`.
pub trait SpanLexer: Send + Sync {
    /// Find the next match in `text` starting from `offset`.
    fn next_span(
        &self,
        text: &str,
    ) -> Option<Range<usize>>;
}

// Blanket implementation for any type that derefs to a SpanLexer.
// This allows Arc<T>, Box<T>, PoolToy<T>, etc. to automatically implement SpanLexer.
impl<D> SpanLexer for D
where
    D: Deref + Send + Sync,
    D::Target: SpanLexer,
{
    fn next_span(
        &self,
        text: &str,
    ) -> Option<Range<usize>> {
        self.deref().next_span(text)
    }
}
