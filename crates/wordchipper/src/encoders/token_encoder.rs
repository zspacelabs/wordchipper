//! # Token Encoder Trait

use crate::{
    TokenType,
    WCResult,
    alloc::{
        sync::Arc,
        vec::Vec,
    },
    spanners::TextSpanner,
    vocab::SpecialVocab,
};

/// The common trait for `String/&[u8] -> Vec<T>` encoders.
///
/// ## Style Hints
///
/// When there is no local ambiguity with other encoders,
/// instance names for implementing types should prefer `encoder`;
/// and use the preferred name for the implementing type
/// when there is a conflict.
pub trait TokenEncoder<T: TokenType>: Send + Sync {
    /// Return the attached text segmentor.
    fn spanner(&self) -> &Arc<dyn TextSpanner>;

    /// Return the attached special vocab.
    ///
    /// ## Returns
    /// A reference to the internal `SpecialVocab`.
    fn special_vocab(&self) -> &SpecialVocab<T>;

    /// Return the expected bytes per token ratio.
    ///
    /// This is used by [`TokenEncoder::expected_token_count`] to predict
    /// the size needed when pre-allocating token buffers.
    fn expected_bytes_per_token(&self) -> f32 {
        self.spanner().expected_bytes_per_span()
    }

    /// Predict the capacity needed when pre-allocating token buffers.
    ///
    /// See: [`TokenEncoder::expected_bytes_per_token`].
    fn expected_token_count(
        &self,
        text: &str,
    ) -> usize {
        (text.len() as f32 / self.expected_bytes_per_token()) as usize
    }

    /// Encode bytes into tokens.
    ///
    /// There are significant performance gains to pre-allocating the target
    /// buffer to an appropriate size.
    ///
    /// ## Arguments
    /// * `text` - The string slice to encode.
    /// * `tokens` - The target token buffer to append to.
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> WCResult<()>;

    /// Encode text into tokens, returning an error if the encoding fails.
    ///
    /// ## Arguments
    /// * `text` - The text to encode.
    ///
    /// ## Returns
    /// A `Result` containing the vector of tokens or an error.
    fn try_encode(
        &self,
        text: &str,
    ) -> WCResult<Vec<T>> {
        let capacity = self.expected_token_count(text) * 115 / 100;
        let mut tokens = Vec::with_capacity(capacity);

        self.try_encode_append(text, &mut tokens)?;
        Ok(tokens)
    }

    /// Encode a batch of text into tokens, returning an error if the encoding
    /// fails.
    ///
    /// ## Arguments
    /// * `batch` - A slice of strings to encode.
    ///
    /// ## Returns
    /// A `Result` containing the vector of token vectors or an error.
    fn try_encode_batch(
        &self,
        batch: &[&str],
    ) -> WCResult<Vec<Vec<T>>> {
        batch.iter().map(|s| self.try_encode(s)).collect()
    }
}
