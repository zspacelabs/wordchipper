use crate::{
    TokenDecoder,
    TokenEncoder,
    TokenType,
    UnifiedTokenVocab,
    WCResult,
    alloc::sync::Arc,
    decoders::{
        BatchDecodeResult,
        DecodeResult,
    },
    prelude::*,
    spanners::TextSpanner,
};

/// Unified Tokenizer.
///
/// Combines:
///  * [`UnifiedTokenVocab`],
///  * [`TokenEncoder`], and
///  * [`TokenDecoder`] wrappers.
#[derive(Clone)]
pub struct Tokenizer<T: TokenType> {
    vocab: Arc<UnifiedTokenVocab<T>>,
    encoder: Arc<dyn TokenEncoder<T>>,
    decoder: Arc<dyn TokenDecoder<T>>,
}

impl<T: TokenType> Tokenizer<T> {
    /// Create a new tokenizer.
    pub fn new(
        vocab: Arc<UnifiedTokenVocab<T>>,
        encoder: Arc<dyn TokenEncoder<T>>,
        decoder: Arc<dyn TokenDecoder<T>>,
    ) -> Self {
        Self {
            vocab,
            encoder,
            decoder,
        }
    }

    /// Get the underlying vocabulary.
    pub fn vocab(&self) -> &Arc<UnifiedTokenVocab<T>> {
        &self.vocab
    }

    /// Get the underlying encoder.
    pub fn encoder(&self) -> &Arc<dyn TokenEncoder<T>> {
        &self.encoder
    }

    /// Get the underlying decoder.
    pub fn decoder(&self) -> &Arc<dyn TokenDecoder<T>> {
        &self.decoder
    }
}

impl<T: TokenType> TokenEncoder<T> for Tokenizer<T> {
    fn spanner(&self) -> &Arc<dyn TextSpanner> {
        self.encoder.spanner()
    }

    fn special_vocab(&self) -> &crate::vocab::SpecialVocab<T> {
        self.encoder.special_vocab()
    }

    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> WCResult<()> {
        self.encoder.try_encode_append(text, tokens)
    }

    fn try_encode(
        &self,
        text: &str,
    ) -> WCResult<Vec<T>> {
        self.encoder.try_encode(text)
    }

    fn try_encode_batch(
        &self,
        batch: &[&str],
    ) -> WCResult<Vec<Vec<T>>> {
        self.encoder.try_encode_batch(batch)
    }
}

impl<T: TokenType> TokenDecoder<T> for Tokenizer<T> {
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<Vec<u8>>> {
        self.decoder.try_decode_to_bytes(tokens)
    }

    fn try_decode_batch_to_bytes(
        &self,
        batch: &[&[T]],
    ) -> WCResult<BatchDecodeResult<Vec<u8>>> {
        self.decoder.try_decode_batch_to_bytes(batch)
    }

    fn try_decode_to_string(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<String>> {
        self.decoder.try_decode_to_string(tokens)
    }

    fn try_decode_batch_to_strings(
        &self,
        batch: &[&[T]],
    ) -> WCResult<BatchDecodeResult<String>> {
        self.decoder.try_decode_batch_to_strings(batch)
    }
}
