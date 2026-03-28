//! Shared test utilities for GPT-2 family logos lexers.
//!
//! Provides [`common_lexer_tests`] which exercises the structural invariants
//! that every `SpanLexer` implementation must satisfy: empty input, non-empty
//! input ordering, and whitespace handling.

use crate::{
    alloc::{
        boxed::Box,
        sync::Arc,
        vec::Vec,
    },
    spanners::{
        SpanRef,
        TextSpanner,
        span_lexers::{
            LexerTextSpanner,
            SpanLexer,
        },
    },
};

/// Run the common lexer test suite against a `SpanLexer` implementation.
///
/// This exercises:
/// - `find_span_iter` on empty input returns no spans
/// - `find_span_iter` on `"hello world"` returns ordered, non-overlapping spans
///   starting at byte 0
/// - `split_spans` on empty input returns no spans
/// - `split_spans` on whitespace-only input returns spans (whitespace is
///   content)
pub fn common_lexer_tests(lexer: Box<dyn SpanLexer>) {
    // ── find_span_iter: empty input ──
    let spans: Vec<_> = lexer.find_span_iter("").collect();
    assert!(spans.is_empty(), "find_span_iter(\"\") should be empty");

    // ── find_span_iter: basic ordering ──
    let spans: Vec<_> = lexer.find_span_iter("hello world").collect();
    assert!(
        !spans.is_empty(),
        "find_span_iter(\"hello world\") should produce spans"
    );
    assert_eq!(spans[0].start, 0, "first span should start at byte 0");
    for pair in spans.windows(2) {
        assert!(
            pair[0].end <= pair[1].start,
            "spans should not overlap: {:?} vs {:?}",
            pair[0],
            pair[1]
        );
    }

    // ── split_spans: empty input ──
    let spanner = LexerTextSpanner::new(Arc::from(lexer), None);
    let spans = spanner.split_spans("", None);
    assert!(spans.is_empty(), "split_spans(\"\") should be empty");

    // ── split_spans: whitespace-only ──
    let spans = spanner.split_spans("   ", None);
    assert!(
        !spans.is_empty(),
        "split_spans(\"   \") should produce spans for whitespace"
    );
    assert_eq!(
        spans,
        crate::alloc::vec![SpanRef::Word(0..3)],
        "whitespace-only input should produce a single Word(0..3)"
    );
}
