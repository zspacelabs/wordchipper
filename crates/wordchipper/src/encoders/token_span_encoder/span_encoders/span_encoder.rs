use crate::{
    TokenType,
    UnifiedTokenVocab,
    alloc::vec::Vec,
    spanners::SpanRef,
};

/// A trait for encoding text spans into tokens.
pub trait SpanEncoder<T: TokenType>: Send {
    /// Encodes a single [`SpanRef`]".
    ///
    /// ## Arguments
    /// * `vocab` - The reference vocabulary.
    /// * `span` - The byte span.
    /// * `tokens` - The target token buffer to append to.
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    );

    /// Encodes a single [`SpanRef`]".
    ///
    /// ## Arguments
    /// * `vocab` - The reference vocabulary.
    /// * `text` - The source slice.
    /// * `span_ref` - The labeling and sub-slicing of a span in `text`.
    /// * `tokens` - The target token buffer to append to.
    fn encode_append_span_ref(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        text: &str,
        span_ref: SpanRef,
        tokens: &mut Vec<T>,
    ) {
        match span_ref {
            SpanRef::Word(range) => {
                let span = &text[range].as_bytes();
                if let Some(token) = vocab.lookup_token(span) {
                    // 1. Faster;
                    // 2. Correct-or: Some words may not exist in the pair mappings.
                    tokens.push(token);
                } else {
                    self.encode_append_compound_span(vocab, span, tokens);
                }
            }
            SpanRef::Special(range) => {
                let span = &text[range].as_bytes();
                let special_token = vocab.special_vocab().lookup_token(span).unwrap();
                tokens.push(special_token);
            }
            _ => (),
        }
    }
}
