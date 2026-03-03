//! # Pair Expansion ``{ T -> (T, T) }`` Token Decoder

use crate::{
    TokenType,
    WCResult,
    alloc::{
        vec,
        vec::Vec,
    },
    decoders::{
        DecodeResult,
        TokenDecoder,
    },
    vocab::{
        ByteMapVocab,
        DEFAULT_BYTE_PER_TOKEN_RATIO,
        PairMapVocab,
        TokenPairMap,
    },
};

/// A stack-based pair map `{T -> (T, T) }` incremental stack [`TokenDecoder`].
///
/// ## Style Hints
///
/// When there is no local ambiguity, instance names should prefer `decoder`;
/// and expand to `pair_decoder` when there is ambiguity.
#[derive(Clone)]
pub struct PairExpansionDecoder<T: TokenType> {
    /// Byte/token mapping table.
    byte_vocab: ByteMapVocab<T>,

    /// Token to pair mapping.
    token_pairs: TokenPairMap<T>,
}

impl<T: TokenType> PairExpansionDecoder<T> {
    /// Build a [`PairExpansionDecoder`] from this [`PairMapVocab`].
    ///
    /// ## Arguments
    /// * `pair_vocab` - The pair vocabulary mapping to build the decoder from.
    ///
    /// ## Returns
    /// A new `PairExpansionDecoder` instance.
    pub fn from_pair_vocab(pair_vocab: &PairMapVocab<T>) -> Self {
        let token_pairs = pair_vocab
            .pair_map()
            .iter()
            .map(|(&pair, &token)| (token, pair))
            .collect();
        Self::new(pair_vocab.byte_vocab().clone(), token_pairs)
    }

    /// Creates a new Decoder.
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `token_pairs` - The token to pair mapping.
    ///
    /// ## Returns
    /// A new `PairExpansionDecoder` instance.
    pub fn new(
        byte_vocab: ByteMapVocab<T>,
        token_pairs: TokenPairMap<T>,
    ) -> Self {
        Self {
            byte_vocab,
            token_pairs,
        }
    }

    /// Get the [`ByteMapVocab`].
    pub fn byte_vocab(&self) -> &ByteMapVocab<T> {
        &self.byte_vocab
    }

    /// Get the [`TokenPairMap`].
    pub fn token_pairs(&self) -> &TokenPairMap<T> {
        &self.token_pairs
    }
}

impl<T: TokenType> TokenDecoder<T> for PairExpansionDecoder<T> {
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<Vec<u8>>> {
        let capacity = (tokens.len() as f32 * DEFAULT_BYTE_PER_TOKEN_RATIO) as usize;
        let mut value = Vec::with_capacity(capacity);

        let mut stack = vec![];
        let mut consumed = 0;

        for t in tokens {
            stack.push(*t);

            while let Some(t) = stack.pop() {
                if let Some(b) = self.byte_vocab.get_byte(t) {
                    value.push(b);
                } else if let Some((a, b)) = self.token_pairs.get(&t) {
                    stack.push(*b);
                    stack.push(*a);
                } else {
                    stack.push(t);
                    break;
                }
            }

            if stack.is_empty() {
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
        decoders::utility::testing::common_decoder_unit_test,
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
    fn test_pair_decoder() {
        type T = u16;

        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(OA_CL100K_BASE_PATTERN),
        )
        .into();

        let decoder = PairExpansionDecoder::from_pair_vocab(&vocab.pair_vocab());

        assert_eq!(decoder.byte_vocab(), &decoder.byte_vocab);
        assert_eq!(decoder.token_pairs(), &decoder.token_pairs);
        assert_eq!(&decoder.byte_vocab, vocab.byte_vocab());

        common_decoder_unit_test(vocab, &decoder);
    }
}
