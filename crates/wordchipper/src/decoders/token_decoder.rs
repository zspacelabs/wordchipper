//! # Token Decoder Trait

use crate::{
    TokenType,
    WCResult,
    alloc::{
        string::String,
        vec::Vec,
    },
    decoders::{
        BatchDecodeResult,
        DecodeResult,
    },
    support::strings::string_from_utf8_lossy,
};

/// The common trait for `&[T] -> Vec<u8>/String>` decoders.
///
/// ## Style Hints
///
/// When there is no local ambiguity with other decoders,
/// instance names for implementing types should prefer `decoder`;
/// and use the preferred name for the implementing type
/// when there is conflict with other encoders.
pub trait TokenDecoder<T: TokenType>: Send + Sync {
    /// Decodes tokens into bytes.
    ///
    /// ## Arguments
    /// * `tokens` - A slice of tokens to decode.
    ///
    /// ## Returns
    /// A `Result<DecodeResult<Vec<u8>>>`.
    fn try_decode_to_bytes(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<Vec<u8>>>;

    /// Decodes a batch of tokens.
    ///
    /// ## Arguments
    /// * `batch` - A batch of tokens.
    ///
    /// ## Returns
    /// A `Result<Vec<DecodeResult<Vec<u8>>>>`.
    fn try_decode_batch_to_bytes(
        &self,
        batch: &[&[T]],
    ) -> WCResult<BatchDecodeResult<Vec<u8>>> {
        batch
            .iter()
            .map(|tokens| self.try_decode_to_bytes(tokens))
            .collect::<WCResult<Vec<_>>>()
            .map(BatchDecodeResult::from)
    }

    /// Decodes tokens into a string.
    ///
    /// UTF-8 lossy decoding is used to handle invalid UTF-8 sequences.
    ///
    /// ## Arguments
    /// * `tokens` - A slice of tokens to decode.
    ///
    /// ## Returns
    /// A `Result<Vec<DecodeResult<String>>>`.
    fn try_decode_to_string(
        &self,
        tokens: &[T],
    ) -> WCResult<DecodeResult<String>> {
        self.try_decode_to_bytes(tokens)
            .map(|res| res.convert(string_from_utf8_lossy))
    }

    /// Decodes a batch of tokens.
    ///
    /// UTF-8 lossy decoding is used to handle invalid UTF-8 sequences.
    ///
    /// ## Arguments
    /// * `batch` - A batch of tokens.
    ///
    /// ## Returns
    /// A `Result<BatchDecodeResult<String>>`.
    fn try_decode_batch_to_strings(
        &self,
        batch: &[&[T]],
    ) -> WCResult<BatchDecodeResult<String>> {
        batch
            .iter()
            .map(|tokens| self.try_decode_to_string(tokens))
            .collect::<WCResult<Vec<_>>>()
            .map(BatchDecodeResult::from)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::FromPrimitive;

    use super::*;
    use crate::{
        alloc::{
            string::ToString,
            vec,
        },
        decoders::utility::ByteDecoder,
    };

    #[test]
    fn test_decode_context() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

        let mut tokens = vec![];
        tokens.extend(
            "hello world"
                .as_bytes()
                .iter()
                .map(|&b| decoder.byte_vocab().get_token(b)),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let result = decoder.try_decode_to_bytes(&tokens).unwrap();
        assert_eq!(result.value, "hello world".as_bytes().to_vec());
        assert_eq!(result.remaining, Some(2));
    }

    #[test]
    fn test_decode_to_strings() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

        let str_samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let token_batch: Vec<Vec<T>> = str_samples
            .iter()
            .map(|s| {
                s.as_bytes()
                    .iter()
                    .map(|b| T::from_u8(*b).unwrap())
                    .collect()
            })
            .collect();

        // Test the batch interfaces.
        let string_batch = decoder
            .try_decode_batch_to_strings(
                &token_batch
                    .iter()
                    .map(|v| v.as_ref())
                    .collect::<Vec<&[T]>>(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(string_batch, str_samples);

        // Test the single-sample interfaces.
        for (sample, tokens) in str_samples.iter().zip(token_batch.iter()) {
            assert_eq!(
                decoder.try_decode_to_string(tokens).unwrap().unwrap(),
                sample.to_string()
            );
        }
    }
}
