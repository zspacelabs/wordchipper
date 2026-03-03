//! # Vocab Testing Tools

use crate::{
    TokenType,
    alloc::vec::Vec,
    spanners::TextSpanningConfig,
    vocab::{
        ByteMapVocab,
        SpanMapVocab,
        SpanTokenMap,
        UnifiedTokenVocab,
    },
};

/// Create a test [`UnifiedTokenVocab`].
pub fn build_test_vocab<T: TokenType, C>(
    byte_vocab: ByteMapVocab<T>,
    segmentation: C,
) -> UnifiedTokenVocab<T>
where
    C: Into<TextSpanningConfig<T>>,
{
    let mut span_map: SpanTokenMap<T> = Default::default();
    span_map.extend(
        [
            ("at", 300),
            ("ate", 301),
            ("th", 302),
            ("the", 303),
            (", ", 304),
            ("on", 305),
            ("he", 306),
            ("ll", 307),
            ("hell", 308),
            ("hello", 309),
            ("wo", 310),
            ("ld", 311),
            ("rld", 312),
            ("world", 313),
            ("fo", 314),
            ("for", 315),
            ("all", 316),
            (". ", 317),
        ]
        .into_iter()
        .map(|(k, v)| (k.as_bytes().to_vec(), T::from_usize(v).unwrap())),
    );

    let span_vocab = SpanMapVocab::new(byte_vocab, span_map).unwrap();

    UnifiedTokenVocab::from_span_vocab(segmentation.into(), span_vocab).unwrap()
}

/// Build a [`ByteMapVocab`] with all tokens shifted by `shift`.
///
/// This is a purposely stupid byte map; useful for testing.
pub fn build_test_shift_byte_vocab<T: TokenType>(shift: usize) -> ByteMapVocab<T> {
    // This is a purposely stupid byte map.
    ByteMapVocab::<T>::from_byte_to_token(
        &ByteMapVocab::<T>::default()
            .byte_tokens()
            .iter()
            .map(|&t| t + T::from_usize(shift).unwrap())
            .collect::<Vec<T>>(),
    )
}
