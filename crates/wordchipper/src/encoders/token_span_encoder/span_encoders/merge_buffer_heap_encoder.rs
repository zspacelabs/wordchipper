//! # Merge Heap based [`SpanEncoder`].
//!
//! Maintains a heap of the best available merges from the pair vocab,
//! iterates until no more merges remain.

use crate::{
    TokenType,
    alloc::vec::Vec,
    encoders::token_span_encoder::span_encoders::span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

/// A [`SpanEncoder`] using a merge heap algorithm.
///
/// This encoder builds and maintains a best-merge heap of potential merges,
/// to avoid secondary lookups in the pair vocab.
#[derive(Default, Debug, Clone)]
pub struct MergeBufferHeapEncoder<T: TokenType> {
    working: Vec<T>,
    pair_ranks: Vec<T>,
}

impl<T: TokenType> SpanEncoder<T> for MergeBufferHeapEncoder<T> {
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // Fill the working vec with the direct byte token translations.
        self.working.clear();
        vocab
            .byte_vocab()
            .append_tokens(span, self.working.as_mut());

        let pr_for_tokens = {
            |tok: &[T], a: usize, b: usize| {
                vocab
                    .lookup_pair(&(tok[a], tok[b]))
                    .unwrap_or(T::max_value())
            }
        };

        // We keep the following property:
        // - pair_ranks[i] = pairs.get(&(CURRENT[i], CURRENT[i + 1]))
        // - pair_ranks.len() = CURRENT.len() - 1 = end - start - 1
        self.pair_ranks.clear();
        self.pair_ranks.extend(
            (0..(self.working.len() - 1)).map(|i| pr_for_tokens(&mut self.working, i, i + 1)),
        );

        while let Some((new_token, i)) = self
            .pair_ranks
            .iter()
            .enumerate()
            .filter_map(|(i, &new_token)| {
                if new_token != T::max_value() {
                    Some((new_token, i))
                } else {
                    None
                }
            })
            .min()
        {
            // At this point, i selects CURRENT[i], PAIR_RANKS[i] such that:
            // - PAIR_RANKS[i] != max_value
            // - PAIR_RANKS[i] is smallest

            // Set CURRENT[i] to the new target rank.
            self.working[i] = new_token;

            if i > 0 {
                // If there is a preceding token, recompute PAIR_RANKS[i-1].
                self.pair_ranks[i - 1] = pr_for_tokens(&mut self.working, i - 1, i);
            }

            if i + 2 < self.working.len() {
                // If this pair rank exists,
                // it will become PAIR_RANKS[i] following the remove below.
                self.pair_ranks[i + 1] = pr_for_tokens(&mut self.working, i, i + 2);
            }

            // Drop PAIR_RANKS[i] and CURRENT[i+1].
            self.pair_ranks.remove(i);
            self.working.remove(i + 1);
        }

        tokens.extend_from_slice(self.working.as_slice());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TokenEncoder,
        TokenType,
        alloc::sync::Arc,
        encoders::{
            testing::{
                common_encoder_test_vocab,
                common_encoder_tests,
            },
            token_span_encoder::{
                SpanEncoderSelector,
                TokenSpanEncoder,
            },
        },
        spanners::TextSpannerBuilder,
    };

    fn test_encoder<T: TokenType>() {
        let vocab: Arc<UnifiedTokenVocab<T>> = common_encoder_test_vocab().into();
        let encoder = TokenSpanEncoder::<T>::new_with_selector(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            SpanEncoderSelector::MergeBufferHeap,
        );
        let encoder: Arc<dyn TokenEncoder<T>> = Arc::new(encoder);
        common_encoder_tests(vocab.into(), encoder)
    }

    #[test]
    fn test_encoder_u16() {
        test_encoder::<u16>();
    }

    #[test]
    fn test_encoder_u32() {
        test_encoder::<u32>();
    }
}
