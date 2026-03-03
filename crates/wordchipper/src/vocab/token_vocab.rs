//! # Token Vocabulary Index

use crate::{
    alloc::vec::Vec,
    types::{
        TokenType,
        WCHashSet,
    },
};

/// Common traits for token vocabularies.
pub trait VocabIndex<T: TokenType>: Clone + Send + Sync {
    /// The token type: T.
    type Token: TokenType;

    /// Returns a set of all tokens.
    fn tokens(&self) -> WCHashSet<T>;

    /// Returns the number of tokens in the vocabulary.
    fn len(&self) -> usize;

    /// Returns true if the vocabulary is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the highest ranked token.
    ///
    /// ## Returns
    /// The maximum token value, or None.
    fn max_token(&self) -> Option<T> {
        self.tokens().iter().max().copied()
    }

    /// Generate all ``(Vec<u8>, T)`` pairs in the vocabulary.
    ///
    /// ## Returns
    /// An iterator over pairs of byte vectors and their corresponding tokens.
    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)>;
}
