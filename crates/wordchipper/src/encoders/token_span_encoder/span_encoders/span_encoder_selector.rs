//! # Span Encoder Selector

use crate::{
    TokenType,
    alloc::{
        boxed::Box,
        sync::Arc,
    },
    encoders::token_span_encoder::{
        SpanEncoder,
        span_encoders::{
            BufferSweepSpanEncoder,
            MergeHeapSpanEncoder,
            PriorityMergeSpanEncoder,
            TailSweepSpanEncoder,
            bpe_backtrack_encoder::{
                BpeBacktrackSpanEncoder,
                BpeVocab,
            },
        },
    },
    vocab::UnifiedTokenVocab,
};

/// Policy enum for selecting a [`SpanEncoder`] for
/// [`TokenSpanEncoder`](`crate::encoders::token_span_encoder::TokenSpanEncoder`).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, strum::EnumString, strum::EnumIter, strum::Display,
)]
#[non_exhaustive]
pub enum SpanEncoderSelector {
    /// This is the canonical best concurrent encoder.
    ///
    /// It is benchmarked to be the fastest and most efficient encoder for
    /// concurrent use.
    ///
    /// This is currently an alias for: [`MergeHeap`](`Self::MergeHeap`)
    #[default]
    ConcurrentDefault,

    /// This the canonical best single-threaded encoder.
    ///
    /// It is benchmarked to be the fastest and most efficient encoder for
    /// single-threaded use.
    ///
    /// This is currently an alias for: [`PriorityMerge`](`Self::PriorityMerge`)
    SingleThreadDefault,

    /// The canonical reference encoder, [`BufferSweepSpanEncoder`].
    ///
    /// This encoder is meant to be used as a reference implementation for
    /// testing and comparison. The code and behavior are as simple as
    /// possible, but it is not optimized for performance.
    ///
    /// This is currently an alias for: [`BufferSweep`](`Self::BufferSweep`)
    Reference,

    /// Use the [`TailSweepSpanEncoder`] encoder.
    TailSweep,

    /// Use the [`MergeHeapSpanEncoder`] encoder.
    MergeHeap,

    /// Use the [`PriorityMergeSpanEncoder`] encoder.
    PriorityMerge,

    /// Use the [`BufferSweepSpanEncoder`] encoder.
    BufferSweep,

    /// Use the [`BpeBacktrackSpanEncoder`] encoder.
    BpeBacktrack,
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_span_encoder_selector_strum_roundtrip() {
        for variant in <SpanEncoderSelector as strum::IntoEnumIterator>::iter() {
            let s = variant.to_string();
            assert_eq!(
                SpanEncoderSelector::from_str(&s).unwrap(),
                variant,
                "roundtrip failed for variant string: {s}"
            );
        }
    }
}

impl SpanEncoderSelector {
    /// Get a builder for the configured [`SpanEncoder`].
    ///
    /// The `vocab` parameter is needed by encoders that pre-build data
    /// structures from the vocabulary (e.g. BPE automaton).
    pub fn span_encoder_builder<T: TokenType>(
        &self,
        vocab: &UnifiedTokenVocab<T>,
    ) -> Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync> {
        use SpanEncoderSelector::*;
        match self {
            Reference | BufferSweep => {
                Arc::new(|| Box::new(BufferSweepSpanEncoder::<T>::default()))
            }
            TailSweep => Arc::new(|| Box::new(TailSweepSpanEncoder::<T>::default())),
            ConcurrentDefault | MergeHeap => {
                Arc::new(|| Box::new(MergeHeapSpanEncoder::<T>::default()))
            }
            SingleThreadDefault | PriorityMerge => {
                Arc::new(|| Box::new(PriorityMergeSpanEncoder::<T>::default()))
            }
            BpeBacktrack => {
                let bpe_vocab = Arc::new(BpeVocab::from_vocab(vocab));
                Arc::new(move || Box::new(BpeBacktrackSpanEncoder::new(bpe_vocab.clone())))
            }
        }
    }
}
