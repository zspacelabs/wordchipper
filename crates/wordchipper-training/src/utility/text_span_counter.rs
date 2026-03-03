//! # Word Counter

use core::fmt::Debug;

use wordchipper::{
    TokenType,
    WCHashMap,
    hash_map_with_capacity,
    support::regex::RegexWrapper,
    vocab::ByteMapVocab,
};

use crate::{
    CountType,
    StringChunkType,
    utility::TokenSpanBuf,
};

/// Expected average word length in characters.
pub const EXPECTED_WORD_LENGTH: usize = 5;

/// Options for [`TextSpanCounter`].
#[derive(Debug, Clone)]
pub struct TextSpanCounterOptions {
    /// Expected average word length in characters.
    /// Used when pre-allocating buffers.
    pub avg_word_len: usize,
}

impl Default for TextSpanCounterOptions {
    fn default() -> Self {
        Self {
            avg_word_len: EXPECTED_WORD_LENGTH,
        }
    }
}

impl TextSpanCounterOptions {
    /// Set the expected average word length in characters.
    /// Used when pre-allocating buffers.
    pub fn with_avg_word_len(
        self,
        avg_word_len: usize,
    ) -> Self {
        Self { avg_word_len }
    }
}

/// Word counter structure.
pub struct TextSpanCounter<K, C>
where
    K: StringChunkType,
    C: CountType,
{
    /// The config options.
    pub options: TextSpanCounterOptions,

    /// The compiled regex pattern.
    pub regex: RegexWrapper,

    /// The word counts.
    pub word_counts: WCHashMap<K, C>,
}

impl<K, C> TextSpanCounter<K, C>
where
    K: StringChunkType,
    C: CountType,
{
    /// Create a new word counter.
    pub fn new(
        regex: RegexWrapper,
        options: TextSpanCounterOptions,
    ) -> Self {
        Self {
            options,
            regex,
            word_counts: hash_map_with_capacity(100_000),
        }
    }

    /// Release the word counts and return them.
    pub fn release(self) -> WCHashMap<K, C> {
        self.word_counts
    }

    /// Update word counts inplace from text.
    pub fn update_from_text<S: AsRef<str>>(
        &mut self,
        text: S,
    ) {
        let word_counts = &mut self.word_counts;
        let regex = &self.regex;
        for mat in regex.find_iter(text.as_ref()) {
            let piece = mat.as_str();
            let k: K = piece.into();
            *word_counts.entry(k).or_default() += C::one();
        }
    }

    /// Update word counts inplace from a sample iterator.
    pub fn update_from_samples<I>(
        &mut self,
        samples: I,
    ) where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        for sample in samples {
            self.update_from_text(sample);
        }
    }

    /// Convert the word counter to a [`TokenSpanBuf<T>`] count iterator.
    ///
    /// # Arguments
    /// * `byte_vocab` - the byte table to use for byte translation.
    pub fn to_text_span_counts_iter<T: TokenType>(
        &self,
        byte_vocab: &ByteMapVocab<T>,
    ) -> impl Iterator<Item = (TokenSpanBuf<T>, C)> {
        self.word_counts
            .iter()
            .map(|(k, v)| (TokenSpanBuf::from_string(k, byte_vocab), *v))
    }
}

#[cfg(test)]
mod tests {
    use wordchipper::{
        WCHashMap,
        hash_map_new,
        support::regex::RegexPattern,
    };

    use super::*;

    const PATTERN: &str = r"\w+";

    fn get_regex() -> RegexWrapper {
        let pattern: RegexPattern = PATTERN.into();
        pattern.compile().unwrap()
    }

    #[test]
    fn test_word_counter() {
        let mut wc: TextSpanCounter<String, u64> =
            TextSpanCounter::new(get_regex(), TextSpanCounterOptions::default());

        let samples = vec!["Hello world", "Foo world bar world"];
        wc.update_from_samples(samples.iter());

        let counts = wc.release();
        check_common_counts(counts);
    }

    fn check_common_counts<K, C>(counts: WCHashMap<K, C>)
    where
        K: StringChunkType,
        C: CountType,
    {
        let mut counts: Vec<(K, C)> = counts.into_iter().collect::<Vec<_>>();
        counts.sort();
        assert_eq!(
            counts,
            vec![
                ("Foo".into(), C::from_usize(1).unwrap()),
                ("Hello".into(), C::from_usize(1).unwrap()),
                ("bar".into(), C::from_usize(1).unwrap()),
                ("world".into(), C::from_usize(3).unwrap()),
            ]
        );
    }

    #[test]
    fn test_update_from_samples() {
        type K = String;
        type T = usize;
        type C = u64;

        let byte_vocab: ByteMapVocab<T> = Default::default();

        let mut word_counts = TextSpanCounter::<K, C>::new(
            get_regex(),
            TextSpanCounterOptions::default().with_avg_word_len(10),
        );

        let samples = vec!["Hello world", "Foo world bar world"];

        word_counts.update_from_samples(samples.iter());

        let counts: WCHashMap<TokenSpanBuf<T>, C> =
            word_counts.to_text_span_counts_iter(&byte_vocab).collect();
        let mut counts = counts.into_iter().collect::<Vec<_>>();
        counts.sort();

        let mut expected = hash_map_new();
        expected.insert(TokenSpanBuf::from_string("Hello", &byte_vocab), 1);
        expected.insert(TokenSpanBuf::from_string("Foo", &byte_vocab), 1);
        expected.insert(TokenSpanBuf::from_string("bar", &byte_vocab), 1);
        expected.insert(TokenSpanBuf::from_string("world", &byte_vocab), 3);
        let mut expected: Vec<_> = expected.into_iter().collect();
        expected.sort();

        assert_eq!(counts, expected);
    }
}
