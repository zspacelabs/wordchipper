//! # Text Normalization

use crate::alloc::{
    borrow::Cow,
    string::String,
    vec::Vec,
};
use unicode_normalization::{
    UnicodeNormalization,
    is_nfc,
    is_nfd,
    is_nfkc,
    is_nfkd,
};

/// Text normalizers that can be applied before spanning.
#[derive(Debug, Clone, PartialEq)]
pub enum TextNormalizer {
    /// Normalize with Unicode NFC.
    NFC,

    /// Normalize with Unicode NFD.
    NFD,

    /// Normalize with Unicode NFKC.
    NFKC,

    /// Normalize with Unicode NFKD.
    NFKD,

    /// Apply the normalizers in-order.
    Sequence(Vec<TextNormalizer>),
}

impl TextNormalizer {
    /// Normalize `text`, borrowing the input when no rewrite is needed.
    pub fn normalize<'a>(
        &self,
        text: &'a str,
    ) -> Cow<'a, str> {
        match self {
            Self::NFC => normalize_if_needed(text, is_nfc, |s| s.nfc().collect()),
            Self::NFD => normalize_if_needed(text, is_nfd, |s| s.nfd().collect()),
            Self::NFKC => normalize_if_needed(text, is_nfkc, |s| s.nfkc().collect()),
            Self::NFKD => normalize_if_needed(text, is_nfkd, |s| s.nfkd().collect()),
            Self::Sequence(normalizers) => {
                let mut current: Option<String> = None;

                for normalizer in normalizers {
                    let input = current.as_deref().unwrap_or(text);
                    if let Cow::Owned(next) = normalizer.normalize(input) {
                        current = Some(next);
                    }
                }

                current.map(Cow::Owned).unwrap_or_else(|| Cow::Borrowed(text))
            }
        }
    }
}

fn normalize_if_needed<'a, F, G>(
    text: &'a str,
    is_normalized: F,
    normalize: G,
) -> Cow<'a, str>
where
    F: Fn(&str) -> bool,
    G: Fn(&str) -> String,
{
    if is_normalized(text) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(normalize(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfc_normalizer_recomposes_decomposed_unicode() {
        let normalized = TextNormalizer::NFC.normalize("e\u{301}clair cafe\u{301}");
        assert_eq!(normalized.as_ref(), "éclair café");
    }

    #[test]
    fn test_nfc_normalizer_borrows_already_normalized_text() {
        let normalized = TextNormalizer::NFC.normalize("éclair café");
        assert!(matches!(normalized, Cow::Borrowed(_)));
    }

    #[test]
    fn test_sequence_normalizer_applies_in_order() {
        let normalized = TextNormalizer::Sequence(vec![TextNormalizer::NFD, TextNormalizer::NFC])
            .normalize("éclair café");
        assert_eq!(normalized.as_ref(), "éclair café");
    }
}