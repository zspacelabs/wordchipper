//! # Working Buffer Incremental Sweep [`SpanEncoder`].

use crate::{
    TokenType,
    alloc::vec::Vec,
    encoders::token_span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

/// A [`SpanEncoder`] which incrementally scans for merges.
///
/// This encoder uses a persistent working buffer as working memory.
///
/// It fills working memory with the direct byte-token translations;
/// and then iteratively sweeps the buffer to apply the best
/// available merge until no more merges remain.
///
/// It then copies the working buffer to the output buffer.
#[derive(Default, Debug, Clone)]
pub struct BufferSweepSpanEncoder<T: TokenType> {
    // The working token buffer.
    working: Vec<T>,
}

impl<T: TokenType> SpanEncoder<T> for BufferSweepSpanEncoder<T> {
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

        // Incrementally shrink the working memory
        // Until we can no longer find pairs to merge.
        while self.working.len() > 1 {
            // Find the lowest ranked merge available.
            if let Some((token, idx)) = self
                .working
                .windows(2)
                .enumerate()
                .filter_map(|(idx, w)| vocab.lookup_pair(&(w[0], w[1])).map(|token| (token, idx)))
                .min()
            {
                // buf[idx..=idx+1] (a, b) -> buf[idx] t
                self.working[idx] = token;
                self.working.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
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
            SpanEncoderSelector::Reference,
        );
        let encoder: Arc<dyn TokenEncoder<T>> = Arc::new(encoder);
        common_encoder_tests(vocab, encoder)
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
