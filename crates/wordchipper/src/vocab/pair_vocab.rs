//! # Pair Map ``{ (T, T) -> T }`` Token Vocabulary

use crate::{
    WCResult,
    alloc::vec::Vec,
    decoders::{
        TokenDecoder,
        utility::PairExpansionDecoder,
    },
    types::{
        Pair,
        TokenType,
        WCHashSet,
    },
    vocab::{
        ByteMapVocab,
        PairTokenMap,
        VocabIndex,
        utility::validators::try_vocab_size,
    },
};

/// Validate that a [`ByteMapVocab`] and [`PairTokenMap`] are compatible.
///
/// - for every ``(a, b) -> t`` entry:
///   - the parents ``(a, b)``:
///     - are either in the `byte_vocab`, or are targets in the map, not both.
///   - the target ``t`` is not in the `byte_vocab`.
///
/// ## Arguments
/// * `byte_vocab` - The byte vocabulary to validate against.
/// * `pairs` - The pair token map to validate.
///
/// ## Returns
/// A `Result` indicating whether the maps are compatible.
pub fn try_validate_pair_map<T: TokenType>(
    byte_vocab: &ByteMapVocab<T>,
    pairs: &PairTokenMap<T>,
) -> WCResult<()> {
    let pair_targets: WCHashSet<T> = pairs.values().copied().collect();

    for t in &pair_targets {
        if let Some(b) = byte_vocab.get_byte(*t) {
            return Err(crate::WCError::VocabConflict(crate::alloc::format!(
                "Target token in pair map {t:?} also mapped to byte {b:0x?}"
            )));
        }
    }

    for (&pair, &t) in pairs.iter() {
        for pt in [pair.0, pair.1] {
            let is_pair_target = pair_targets.contains(&pt);
            let byte_target = byte_vocab.get_byte(pt);

            if is_pair_target && let Some(b) = byte_target {
                return Err(crate::WCError::VocabConflict(crate::alloc::format!(
                    "Pair {pair:?} -> {t:?} parent {pt:?} is a pair target and byte target: {b:0x?}"
                )));
            }
            if !is_pair_target && byte_target.is_none() {
                return Err(crate::WCError::VocabConflict(crate::alloc::format!(
                    "Pair {pair:?} -> {t:?} parent {pt:?} is not defined"
                )));
            }
        }
    }

    Ok(())
}

/// Pair ``(T, T) -> T`` Vocabulary.
///
/// - Grounded in a `ByteTable<T>` for byte-to-token mapping.
/// - Collection of ``(T, T) -> T`` pairs.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct PairMapVocab<T: TokenType> {
    /// Byte/token mapping table.
    byte_vocab: ByteMapVocab<T>,

    /// Map of ``{ (T, T) -> T }``.
    pair_map: PairTokenMap<T>,
}

impl<T: TokenType> PairMapVocab<T> {
    /// Initialize a [`PairMapVocab`].
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `pairs` - The pair token map.
    ///
    /// ## Returns
    /// A `Result` containing the new `PairMapVocab` instance or an error.
    pub fn new(
        byte_vocab: ByteMapVocab<T>,
        mut pairs: PairTokenMap<T>,
    ) -> WCResult<Self> {
        try_validate_pair_map(&byte_vocab, &pairs)?;
        pairs.shrink_to_fit();
        Ok(Self {
            byte_vocab,
            pair_map: pairs,
        })
    }

    /// Convert to a different token type.
    pub fn to_token_type<G: TokenType>(&self) -> WCResult<PairMapVocab<G>> {
        try_vocab_size::<G>(self.max_token().unwrap().to_usize().unwrap())?;

        PairMapVocab::<G>::new(
            self.byte_vocab.to_token_type::<G>()?,
            self.pair_map
                .iter()
                .map(|(&(a, b), &token)| {
                    (
                        (G::from(a).unwrap(), G::from(b).unwrap()),
                        G::from(token).unwrap(),
                    )
                })
                .collect(),
        )
    }

    /// Get the byte vocabulary.
    pub fn byte_vocab(&self) -> &ByteMapVocab<T> {
        &self.byte_vocab
    }

    /// Get the map of pairs.
    pub fn pair_map(&self) -> &PairTokenMap<T> {
        &self.pair_map
    }

    /// Looks up a pair.
    ///
    /// ## Arguments
    /// * `pair` - The pair of tokens to look up.
    ///
    /// ## Returns
    /// An `Option` containing the token corresponding to the pair if it exists.
    pub fn lookup_pair(
        &self,
        pair: &Pair<T>,
    ) -> Option<T> {
        self.pair_map.get(pair).copied()
    }
}

impl<T: TokenType> VocabIndex<T> for PairMapVocab<T> {
    type Token = T;

    fn len(&self) -> usize {
        self.byte_vocab.len() + self.pair_map.len()
    }

    fn tokens(&self) -> WCHashSet<T> {
        self.byte_vocab
            .tokens()
            .iter()
            .copied()
            .chain(self.pair_map.values().copied())
            .collect::<WCHashSet<T>>()
    }

    fn max_token(&self) -> Option<T> {
        let max_t = self.byte_vocab.max_token();
        let max_p = self.pair_map.values().max().copied();
        [max_t, max_p].into_iter().flatten().max()
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        let decoder = PairExpansionDecoder::from_pair_vocab(self);

        self.byte_vocab.span_pairs().chain(
            self.pair_map
                .values()
                .map(move |&t| (decoder.try_decode_to_bytes(&[t]).unwrap().unwrap(), t)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::{
        ByteMapVocab,
        PairTokenMap,
    };

    #[test]
    fn test_tokens_sorted() {
        type T = u32;
        let byte_vocab: ByteMapVocab<T> = Default::default();

        let mut vocab = PairMapVocab::<T> {
            pair_map: PairTokenMap::default(),
            byte_vocab: byte_vocab.clone(),
        };

        assert_eq!(vocab.max_token().unwrap(), 255);

        assert_eq!(&vocab.tokens(), &byte_vocab.tokens());

        vocab.pair_map.insert((1, 2), 300);
        vocab.pair_map.insert((3, 4), 301);
        vocab.pair_map.insert((300, 301), 302);

        assert_eq!(vocab.max_token().unwrap(), 302);
        assert_eq!(vocab.len(), 256 + 3);

        assert_eq!(
            &vocab.tokens(),
            &byte_vocab
                .tokens()
                .into_iter()
                .chain([300_u32, 301, 302].into_iter())
                .collect()
        );
    }
}
