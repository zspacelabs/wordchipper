use crate::{
    TokenType,
    UnifiedTokenVocab,
    alloc::vec::Vec,
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
}
