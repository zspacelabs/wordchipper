//! # Slab Index Decoder

use core::marker::PhantomData;

use crate::{
    TokenType,
    WCResult,
    alloc::{
        sync::Arc,
        vec,
        vec::Vec,
    },
    decoders::{
        DecodeResult,
        TokenDecoder,
    },
    vocab::{
        DEFAULT_BYTE_PER_TOKEN_RATIO,
        TokenSpanMap,
        UnifiedTokenVocab,
    },
};

/// A [`TokenDecoder<T>`] which keeps a dense array index into a shared slab.
///
/// It is expected that all tokens (single-byte and multibyte words,
/// and special tokens) are stored in the slab index.
///
/// ## Style Hints
///
/// When there is no local ambiguity, instance names should prefer `decoder`;
/// and expand to `dict_decoder` when there is ambiguity.
#[derive(Clone)]
pub struct SlabIndexDecoder<T: TokenType> {
    index: Vec<(usize, usize)>,
    slab: Vec<u8>,

    expected_bytes_per_token: f32,
    _marker: PhantomData<T>,
}

impl<T: TokenType> SlabIndexDecoder<T> {
    /// Build a [`SlabIndexDecoder`] from this [`UnifiedTokenVocab`].
    ///
    /// ## Arguments
    /// * `unified_vocab` - The unified token vocabulary to build the decoder
    ///   from.
    pub fn from_vocab(vocab: Arc<UnifiedTokenVocab<T>>) -> Self {
        Self::new(vocab.unified_dictionary())
    }

    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `token_spans` - The token to word mapping.
    pub fn new(token_spans: TokenSpanMap<T>) -> Self {
        let max_token = token_spans.keys().max().unwrap().to_usize().unwrap();
        let mut index = vec![(0, 0); max_token + 1];

        let total_bytes = token_spans.values().map(|span| span.len()).sum();
        let mut slab = Vec::with_capacity(total_bytes);

        let mut tokens: Vec<T> = token_spans.keys().copied().collect();
        tokens.sort_unstable();

        for token in tokens {
            let idx = token.to_usize().unwrap();
            let span = token_spans.get(&token).unwrap();
            index[idx] = (slab.len(), slab.len() + span.len());
            slab.extend_from_slice(span);
        }

        Self {
            index,
            slab,
            expected_bytes_per_token: DEFAULT_BYTE_PER_TOKEN_RATIO,
            _marker: PhantomData,
        }
    }

    /// Get the expected bytes per token.
    pub fn expected_bytes_per_token(&self) -> f32 {
        self.expected_bytes_per_token
    }

    /// Sets the expected bytes per token.
    ///
    /// This is used to bias the capacity of the output buffer in
    /// `try_decode_to_bytes`.
    pub fn with_expected_bytes_per_token(
        mut self,
        expected: f32,
    ) -> Self {
        self.expected_bytes_per_token = expected;
        self
    }

    /// Predict the capacity needed when pre-allocating output buffers.
    pub fn predicted_byte_buffer_size(
        &self,
        tokens: &[T],
    ) -> usize {
        (tokens.len() as f32 * 1.1 * self.expected_bytes_per_token) as usize
    }

    /// Lookup a token.
    pub fn lookup_span(
        &self,
        token: &T,
    ) -> Option<&[u8]> {
        let idx = token.to_usize().unwrap();
        if idx >= self.index.len() {
            return None;
        }
        let (start, end) = &self.index[idx];
        if end > start {
            Some(&self.slab[*start..*end])
        } else {
            None
        }
    }
}

impl<T: TokenType> TokenDecoder<T> for SlabIndexDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, tokens)))]
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<Vec<u8>>> {
        let capacity = self.predicted_byte_buffer_size(tokens);
        let mut value = Vec::with_capacity(capacity);

        let mut consumed = 0;
        for t in tokens {
            if let Some(w) = self.lookup_span(t) {
                value.extend(w);
                consumed += 1;
            } else {
                break;
            }
        }
        Ok(DecodeResult::new(value, Some(tokens.len() - consumed)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::sync::Arc,
        decoders::utility::testing::common_decoder_tests,
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        spanners::TextSpanningConfig,
        vocab::{
            UnifiedTokenVocab,
            utility::testing::{
                build_test_shift_byte_vocab,
                build_test_vocab,
            },
        },
    };

    #[test]
    fn test_decoder() {
        type T = u16;

        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN),
        )
        .into();

        let decoder =
            SlabIndexDecoder::from_vocab(vocab.clone()).with_expected_bytes_per_token(7.5);

        assert_eq!(decoder.expected_bytes_per_token(), 7.5);

        common_decoder_tests(vocab, Arc::new(decoder));
    }
}
