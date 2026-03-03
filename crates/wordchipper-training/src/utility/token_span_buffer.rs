//! # Token Span Buffer

use core::hash::Hash;

use wordchipper::{
    Pair,
    TokenType,
    vocab::ByteMapVocab,
};

/// A mutable span of tokens (a chunk or "word").
///
/// Iteratively rewritten during BPE vocabulary training.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenSpanBuf<T: TokenType> {
    tokens: Vec<T>,
}

impl<T: TokenType, S: AsRef<[T]>> From<S> for TokenSpanBuf<T> {
    fn from(tokens: S) -> Self {
        Self::from_tokens(tokens)
    }
}

impl<T: TokenType> TokenSpanBuf<T> {
    const DEC: i32 = -1;
    const INC: i32 = 1;

    /// Create a new span buffer from tokens.
    pub fn from_tokens<S>(tokens: S) -> Self
    where
        S: AsRef<[T]>,
    {
        Self {
            tokens: tokens.as_ref().to_vec(),
        }
    }

    /// Create a new span buf from a byte slice.
    ///
    /// # Arguments
    /// * `bytes` - the bytes to translate to byte-level tokens.
    /// * `byte_vocab` - the translation for the byte tokens.
    pub fn from_bytes<B: AsRef<[u8]>>(
        bytes: B,
        byte_vocab: &ByteMapVocab<T>,
    ) -> Self {
        Self {
            tokens: bytes
                .as_ref()
                .iter()
                .map(|&b| byte_vocab.get_token(b))
                .collect(),
        }
    }

    /// Create a new span buf from a string slice.
    ///
    /// # Arguments
    /// * `text` - the text to turn into UTF-8 bytes, and translate to
    ///   byte-level tokens.
    /// * `byte_vocab` - the translation for the byte tokens.
    pub fn from_string<S: AsRef<str>>(
        text: S,
        byte_vocab: &ByteMapVocab<T>,
    ) -> Self {
        Self::from_bytes(text.as_ref().as_bytes(), byte_vocab)
    }

    /// View the tokens as a slice.
    pub fn tokens(&self) -> &[T] {
        &self.tokens
    }

    /// Get the length of the span.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Is this span empty?
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get an iterator over [`Pair<T>`] windows of this span.
    pub fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair<T>> + 'a {
        self.tokens.windows(2).map(|w| (w[0], w[1]))
    }

    /// Reduce the capacity of the internal vector to fit its contents.
    pub fn shrink_to_fit(&mut self) {
        self.tokens.shrink_to_fit();
    }

    /// Merge all non-overlapping occurrences of `pair -> replacement`.
    ///
    /// # Arguments
    /// * `pair` - the pair to merge.
    /// * `replacement` - the token to replace `pair` with.
    /// * `on_merge` - a callback function to invoke for each incremental pair
    ///   delta. The function is called with:
    ///   - `pair` - the pair that was merged.
    ///   - `delta` - the pair count delta: `+1` for an added pair, `-1` for a
    ///     removed pair.
    pub fn merge_pair_cb<F>(
        &mut self,
        pair: Pair<T>,
        replacement: T,
        on_merge: &mut F,
    ) where
        F: FnMut(Pair<T>, i32),
    {
        let (a, b) = pair;
        let n = self.tokens.len();

        if n < 2 {
            // Single-token words have no pairs to merge.
            return;
        }

        let mut new_tokens: Vec<T> = Vec::with_capacity(n);

        let mut i = 0;
        while i < n {
            let current = self.tokens[i];

            if i + 1 < n && pair == (current, self.tokens[i + 1]) {
                // Remove Previous Pair?
                if let Some(&x) = new_tokens.last() {
                    on_merge((x, a), Self::DEC);
                    on_merge((x, replacement), Self::INC);
                }

                // Remove Current Pair.
                on_merge(pair, Self::DEC);

                // Remove Next Pair?
                if i + 2 < n {
                    let y = self.tokens[i + 2];
                    on_merge((b, y), Self::DEC);
                    on_merge((replacement, y), Self::INC);
                };

                new_tokens.push(replacement);

                // Skip 'a' and 'b'.
                i += 2;
            } else {
                new_tokens.push(current);
                i += 1;
            }
        }

        self.tokens = new_tokens;
    }

    /// Merge all non-overlapping occurrences of `pair -> replacement`.
    ///
    /// # Arguments
    /// * `pair` - the pair to merge.
    /// * `replacement` - the token to replace `pair` with.
    ///
    /// # Returns
    /// a delta list of pair count deltas for this span:
    /// * `(Pair, +1)` - for each instance of an added `Pair`.
    /// * `(Pair, -1)` - for each instance of a removed `Pair`.
    pub fn merge_pair(
        &mut self,
        pair: Pair<T>,
        replacement: T,
    ) -> Vec<(Pair<T>, i32)> {
        let mut deltas: Vec<(Pair<T>, i32)> = Vec::with_capacity(6);
        self.merge_pair_cb(pair, replacement, &mut |p, d| deltas.push((p, d)));
        deltas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_tokens() {
        let span: TokenSpanBuf<u32> = TokenSpanBuf::from_tokens(vec![1, 2, 3]);
        assert_eq!(span.tokens(), &[1, 2, 3]);
        assert_eq!(span.len(), 3);
        assert!(!span.is_empty());
    }

    #[test]
    fn test_into_span() {
        let span: TokenSpanBuf<u32> = vec![1, 2, 3].into();
        assert_eq!(span.tokens(), &[1, 2, 3]);

        let span: TokenSpanBuf<u32> = [1, 2, 3].into();
        assert_eq!(span.tokens(), &[1, 2, 3]);

        let span: TokenSpanBuf<u32> = (&[1, 2, 3]).into();
        assert_eq!(span.tokens(), &[1, 2, 3]);
    }

    #[test]
    fn test_span_from_str() {
        type T = u32;

        let byte_vocab: ByteMapVocab<T> = Default::default();

        let span: TokenSpanBuf<T> = TokenSpanBuf::from_string("hello", &byte_vocab);
        assert_eq!(span.tokens(), &[104, 101, 108, 108, 111]);

        let shifted_tokens = byte_vocab
            .byte_tokens()
            .iter()
            .map(|&token| token + 10)
            .collect::<Vec<_>>();

        let shift_table: ByteMapVocab<T> = ByteMapVocab::from_byte_to_token(&shifted_tokens);

        let span: TokenSpanBuf<T> = TokenSpanBuf::from_string("hello", &shift_table);
        assert_eq!(span.tokens(), &[114, 111, 118, 118, 121]);
    }

    #[test]
    fn test_span_pairs() {
        let span: TokenSpanBuf<u32> = TokenSpanBuf::from_tokens(vec![1, 2, 3]);
        assert_eq!(span.pairs().collect::<Vec<_>>(), vec![(1, 2), (2, 3)]);
    }

    #[test]
    fn test_span_merge_pair() {
        let mut span: TokenSpanBuf<u32> = TokenSpanBuf::from_tokens(vec![1, 2, 3, 1, 2, 2, 1]);

        let deltas = span.merge_pair((1, 2), 1);
        assert_eq!(span.tokens(), &[1, 3, 1, 2, 1]);

        assert_eq!(
            deltas,
            vec![
                // first match
                ((1, 2), -1),
                ((2, 3), -1),
                // second match
                ((1, 3), 1),
                ((3, 1), -1),
                ((3, 1), 1),
                // third match
                ((1, 2), -1),
                ((2, 2), -1),
                ((1, 2), 1),
            ]
        );
    }

    #[test]
    fn test_span_merge_pair_cb() {
        let mut span: TokenSpanBuf<u32> = TokenSpanBuf::from_tokens(vec![1, 2, 3, 1, 2, 2, 1]);
        let mut deltas = vec![];

        span.merge_pair_cb((1, 2), 1, &mut |p, d| {
            deltas.push((p, d));
        });
        assert_eq!(span.tokens(), &[1, 3, 1, 2, 1]);

        assert_eq!(
            deltas,
            vec![
                // first match
                ((1, 2), -1),
                ((2, 3), -1),
                // second match
                ((1, 3), 1),
                ((3, 1), -1),
                ((3, 1), 1),
                // third match
                ((1, 2), -1),
                ((2, 2), -1),
                ((1, 2), 1),
            ]
        );

        span.shrink_to_fit();
        assert_eq!(span.tokens().len(), span.tokens.capacity());
    }
}
