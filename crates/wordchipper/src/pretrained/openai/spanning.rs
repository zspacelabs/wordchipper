//! # Spanning Configuration for `OpenAI` Tokenizers

use crate::{
    TokenType,
    pretrained::openai::{
        OA_CL100K_BASE_PATTERN,
        OA_O200K_BASE_PATTERN,
        OA_P50K_BASE_PATTERN,
        OA_R50K_BASE_PATTERN,
        specials,
    },
    spanners::TextSpanningConfig,
};
/// Get the [`TextSpanningConfig`] for the "`r50k_base`" pretrained vocabulary.
pub fn oa_r50k_base_spanning_config<T: TokenType>() -> TextSpanningConfig<T> {
    TextSpanningConfig::<T>::from_pattern(OA_R50K_BASE_PATTERN)
        .with_special_words(specials::oa_r50k_base_special_tokens())
}

/// Get the [`TextSpanningConfig`] for the "`p50k_base`" pretrained vocabulary.
pub fn oa_p50k_base_spanning_config<T: TokenType>() -> TextSpanningConfig<T> {
    TextSpanningConfig::<T>::from_pattern(OA_P50K_BASE_PATTERN)
        .with_special_words(specials::oa_p50k_base_special_tokens())
}

/// Get the [`TextSpanningConfig`] for the "`p50k_edit`" pretrained vocabulary.
pub fn oa_p50k_edit_spanning_config<T: TokenType>() -> TextSpanningConfig<T> {
    TextSpanningConfig::<T>::from_pattern(OA_P50K_BASE_PATTERN)
        .with_special_words(specials::oa_p50k_edit_special_tokens())
}

/// Get the [`TextSpanningConfig`] for the "`cl100k_base`" pretrained
/// vocabulary.
pub fn oa_cl100k_base_spanning_config<T: TokenType>() -> TextSpanningConfig<T> {
    TextSpanningConfig::<T>::from_pattern(OA_CL100K_BASE_PATTERN)
        .with_special_words(specials::oa_cl100k_edit_special_tokens())
}

/// Get the [`TextSpanningConfig`] for the "`o200k_base`" pretrained vocabulary.
pub fn oa_o200k_base_spanning_config<T: TokenType>() -> TextSpanningConfig<T> {
    TextSpanningConfig::<T>::from_pattern(OA_O200K_BASE_PATTERN)
        .with_special_words(specials::oa_o200k_base_special_tokens())
}

/// Get the [`TextSpanningConfig`] for the "`o200k_harmony`" pretrained
/// vocabulary.
pub fn oa_o200k_harmony_spanning_config<T: TokenType>() -> TextSpanningConfig<T> {
    TextSpanningConfig::<T>::from_pattern(OA_O200K_BASE_PATTERN)
        .with_special_words(specials::oa_o200k_harmony_special_tokens())
}
