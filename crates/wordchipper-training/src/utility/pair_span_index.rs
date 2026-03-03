//! # `PairIndex` Builder

use wordchipper::{
    Pair,
    TokenType,
    WCHashMap,
    WCHashSet,
    hash_map_with_capacity,
};

use crate::{
    CountType,
    utility::TokenSpanBuf,
};

/// A map from [`Pair`] to its occurrence count.
pub type PairCountMap<T, C> = WCHashMap<Pair<T>, C>;

/// A map from [`Pair`] to indices over ``words``.
pub type PairIndexMap<T> = WCHashMap<Pair<T>, WCHashSet<usize>>;

/// An index of ``(T, T)`` pair information relative to a
/// ``&[TokenSpanBuf<T>]``.
#[derive(Debug, Clone)]
pub struct PairSpanIndex<T: TokenType, C: CountType> {
    /// A map from [`Pair`] to its occurrence count.
    ///
    /// ``sum(words[i].non_overlapping_count(pair) * word_counts[i]) for all i``
    pub pair_counts: PairCountMap<T, C>,

    /// A map from [`Pair`] to span indices.
    pub pair_index: PairIndexMap<T>,
}

impl<T: TokenType, C: CountType> PairSpanIndex<T, C> {
    /// Build a [`PairSpanIndex`] from a slice of [`TokenSpanBuf`]s, using a
    /// count table.
    ///
    /// # Arguments
    /// * `spans` - a sequence of text spans; assumed to be unique.
    /// * `counts` - `counts[i]` is the count of `spans[i]`.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(spans, counts)))]
    pub fn from_span_count_table(
        spans: &[TokenSpanBuf<T>],
        counts: &[C],
    ) -> Self {
        let size_hint = spans.len() / 1000;

        let mut pair_index = PairSpanIndex {
            pair_counts: hash_map_with_capacity(size_hint),
            pair_index: hash_map_with_capacity(size_hint),
        };

        let zero = C::zero();

        for (index, span) in spans.iter().enumerate() {
            let count = counts[index];

            if count != zero && span.len() >= 2 {
                for p in span.pairs() {
                    *pair_index.pair_counts.entry(p).or_default() += count;
                    pair_index.pair_index.entry(p).or_default().insert(index);
                }
            }
        }

        pair_index
    }
}

#[cfg(test)]
mod tests {
    use wordchipper::vocab::ByteMapVocab;

    use super::*;

    #[test]
    fn test_pair_index_serial_token_u32_count_usize() {
        test_pair_index::<u32, usize>();
    }

    #[test]
    fn test_pair_index_serial_token_u16_count_i32() {
        test_pair_index::<u16, i32>();
    }

    fn test_pair_index<T: TokenType, C: CountType>() {
        let byte_vocab: ByteMapVocab<T> = Default::default();

        let spans: Vec<TokenSpanBuf<T>> = vec![
            TokenSpanBuf::from_string("hello", &byte_vocab),
            TokenSpanBuf::from_string("world", &byte_vocab),
            TokenSpanBuf::from_string("help", &byte_vocab),
            TokenSpanBuf::from_string("☃", &byte_vocab), // "☃" := [0xE2 0x98] 0x83
        ];

        let counts: Vec<C> = [1, 2, 3, 4]
            .into_iter()
            .map(|c| C::from_u32(c).unwrap())
            .collect();

        let PairSpanIndex {
            pair_counts,
            pair_index: pair_to_word_index,
        } = PairSpanIndex::<T, C>::from_span_count_table(&spans, &counts);

        assert_eq!(
            pair_counts,
            [
                (('e', 'l'), 4),                   // 1 h[el]lo
                (('h', 'e'), 4),                   // 1 [he]llo, 3 [he]lp
                (('l', 'p'), 3),                   // 3 hel[lp]
                (('l', 'd'), 2),                   // 2 wor[ld]
                (('o', 'r'), 2),                   // 2 w[or]ld
                (('r', 'l'), 2),                   // 2 wo[rl]d
                (('w', 'o'), 2),                   // 2 [wo]rld
                (('l', 'l'), 1),                   // 1 he[ll]o
                (('l', 'o'), 1),                   // 1 hel[lo]
                ((0xE2 as char, 0x98 as char), 4), // "☃" := [0xE2 0x98] 0x83
                ((0x98 as char, 0x83 as char), 4), // "☃" := 0xE2 [0x98 0x83]
            ]
            .into_iter()
            .map(|((a, b), c)| (
                (T::from_u8(a as u8).unwrap(), T::from_u8(b as u8).unwrap()),
                C::from_u32(c).unwrap()
            ))
            .collect::<PairCountMap<T, C>>()
        );

        assert_eq!(
            pair_to_word_index,
            [
                (('e', 'l'), vec![0, 2]),                // "h[el]lo world h[el]p ☃"
                (('h', 'e'), vec![0, 2]),                // "[he]llo world [he]lp ☃"
                (('l', 'd'), vec![1]),                   // "hello wor[ld] help ☃"
                (('l', 'l'), vec![0]),                   // "he[ll]o world help ☃"
                (('l', 'o'), vec![0]),                   // "hel[lo] world help ☃"
                (('l', 'p'), vec![2]),                   // "hello world he[lp] ☃"
                (('o', 'r'), vec![1]),                   // "hello w[or]ld help ☃"
                (('r', 'l'), vec![1]),                   // "hello wo[rl]d help ☃"
                (('w', 'o'), vec![1]),                   // "hello [wo]rld help ☃"
                ((0xE2 as char, 0x98 as char), vec![3]), // "hello world help [☃]"
                ((0x98 as char, 0x83 as char), vec![3]), // "hello world help [☃]"
            ]
            .into_iter()
            .map(|((a, b), s)| (
                (T::from_u8(a as u8).unwrap(), T::from_u8(b as u8).unwrap()),
                WCHashSet::from_iter(s)
            ))
            .collect::<PairIndexMap<T>>()
        );
    }
}
