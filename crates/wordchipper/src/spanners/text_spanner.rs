//! # Text Spanner

use core::ops::Range;

use crate::{
    alloc::{
        string::String,
        vec::Vec,
    },
    vocab::DEFAULT_BYTE_PER_TOKEN_RATIO,
};

/// Span Label/Range Reference for [`TextSpanner`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpanRef {
    /// A normal word reference.
    Word(Range<usize>),

    /// A special word reference.
    Special(Range<usize>),

    /// A gap reference.
    Gap(Range<usize>),
}

impl SpanRef {
    /// Get the span range.
    pub fn range(&self) -> &Range<usize> {
        match self {
            SpanRef::Word(range) => range,
            SpanRef::Special(range) => range,
            SpanRef::Gap(range) => range,
        }
    }
}

impl From<SpanRef> for Range<usize> {
    fn from(span: SpanRef) -> Self {
        span.range().clone()
    }
}

/// Trait for text spanners.
pub trait TextSpanner: Send + Sync {
    /// Get the expected bytes per span ratio.
    ///
    /// Used for pre-allocating span buffers.
    fn expected_bytes_per_span(&self) -> f32 {
        DEFAULT_BYTE_PER_TOKEN_RATIO
    }

    /// Estimate the number of spans in the text.
    ///
    /// Computed by dividing the text bytes
    /// by the [`Self::expected_bytes_per_span()`].
    fn expected_span_count(
        &self,
        text: &str,
    ) -> usize {
        text.len() / self.expected_bytes_per_span() as usize
    }

    /// Iterate over all split [`SpanRef`]s in the text.
    ///
    /// # Arguments
    /// * `text` - the text to split.
    /// * `f` - the function to apply to each span; halts when the function
    ///   returns `false`.
    ///
    /// Note: a byte is consumed *only if* the function returns `true`;
    /// if the function returns `false`, the byte is not consumed.
    ///
    /// # Returns
    /// ``(completed, consumed)`` where:
    /// - `consumed` is the number of bytes covered by spans accepted by `f`;
    /// - `completed` is if all spans were accepted.
    fn for_each_split_span(
        &self,
        text: &str,
        f: &mut dyn FnMut(SpanRef) -> bool,
    ) -> (bool, usize);

    /// Split text into spans.
    ///
    /// ## Arguments
    /// * `text` - The text to split.
    ///
    /// ## Returns
    /// A vector of `SpanRef` items.
    fn split_spans(
        &self,
        text: &str,
    ) -> Vec<SpanRef> {
        let capacity = self.expected_span_count(text) * 115 / 100;
        let mut words = Vec::with_capacity(capacity);

        self.for_each_split_span(text, &mut |span_ref| {
            words.push(span_ref);
            true
        });

        words
    }

    /// Rewrite text by splitting and re-joining without `Gap` matches.
    ///
    /// ## Arguments
    /// * `text` - The text to rewrite.
    ///
    /// ## Returns
    /// The rewritten string.
    fn remove_gaps(
        &self,
        text: &str,
    ) -> String {
        self.split_spans(text)
            .into_iter()
            .filter_map(|m| match m {
                SpanRef::Gap(_) => None,
                _ => Some(&text[Range::<usize>::from(m)]),
            })
            .collect()
    }

    /// Batch version of [`Self::remove_gaps`]
    fn batch_remove_gaps(
        &self,
        texts: &[&str],
    ) -> Vec<String> {
        texts.iter().map(|t| self.remove_gaps(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{
        boxed::Box,
        sync::Arc,
    };

    const _TEXT_SPANNER_BOX_CHECK: Option<Box<dyn TextSpanner>> = None;
    const _TEXT_SPANNER_ARC_CHECK: Option<Arc<dyn TextSpanner>> = None;

    #[test]
    fn test_spanref() {
        let span = SpanRef::Word(0..3);
        assert_eq!(span.range(), &(0..3));
        assert_eq!(Range::<usize>::from(span), 0..3);

        let span = SpanRef::Gap(0..3);
        assert_eq!(span.range(), &(0..3));
        assert_eq!(Range::<usize>::from(span), 0..3);

        let span = SpanRef::Special(0..3);
        assert_eq!(span.range(), &(0..3));
        assert_eq!(Range::<usize>::from(span), 0..3);
    }
}
