// Algorithm based on github/rust-gems (MIT license, Copyright GitHub Inc. 2023)
// https://github.com/github/rust-gems/tree/main/crates/bpe

//! # BPE backtracking [`SpanEncoder`].
//!
//! Uses an Aho-Corasick automaton for leftmost-longest token matching,
//! combined with a backtracking loop that validates BPE merge boundaries.

use alloc::{
    sync::Arc,
    vec,
    vec::Vec,
};

use aho_corasick::{
    AhoCorasick,
    MatchKind,
};

use crate::{
    TokenType,
    encoders::token_span_encoder::SpanEncoder,
    types::{
        Pair,
        WCHashMap,
    },
    vocab::{
        UnifiedTokenVocab,
        VocabIndex,
    },
};

/// Precomputed BPE vocabulary data for the backtracking encoder.
///
/// Built once from a [`UnifiedTokenVocab`] and shared via [`Arc`].
pub struct BpeVocab<T> {
    /// `{ (T, T) -> T }` merge table (cloned from vocab).
    pair_lookup: WCHashMap<Pair<T>, T>,
    /// Indexed by `T.to_usize()`. Inverse of `pair_lookup`.
    /// Byte-level tokens map to `(self, self)`.
    split_table: Vec<Pair<T>>,
    /// For each token, the next-longest prefix token (or `T::max_value()`
    /// sentinel).
    next_prefix: Vec<T>,
    /// Aho-Corasick automaton over all token byte sequences.
    ac: AhoCorasick,
    /// Maps AC pattern index to token ID.
    ac_tokens: Vec<T>,
    /// Token byte lengths, indexed by `T.to_usize()`.
    token_lens: Vec<usize>,
}

impl<T: TokenType> BpeVocab<T> {
    /// Build from a [`UnifiedTokenVocab`].
    pub fn from_vocab(vocab: &UnifiedTokenVocab<T>) -> Self {
        // Collect all (bytes, token) pairs, sorted by token ID (= rank).
        let mut span_pairs: Vec<(Vec<u8>, T)> = vocab.span_pairs().collect();
        span_pairs.sort_by_key(|(_, t)| *t);

        let max_token = span_pairs
            .last()
            .map(|(_, t)| t.to_usize().unwrap())
            .unwrap_or(0);
        let table_size = max_token + 1;

        // Build token_lens.
        let mut token_lens = vec![0usize; table_size];
        for (bytes, tok) in &span_pairs {
            token_lens[tok.to_usize().unwrap()] = bytes.len();
        }

        // Build AC automaton (leftmost-longest).
        let patterns: Vec<&[u8]> = span_pairs.iter().map(|(b, _)| b.as_slice()).collect();
        let ac_tokens: Vec<T> = span_pairs.iter().map(|(_, t)| *t).collect();
        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .expect("failed to build AhoCorasick automaton");

        // Build next_prefix: for each token, the longest prefix token shorter by 1+
        // bytes.
        let mut next_prefix = vec![T::max_value(); table_size];
        for (bytes, tok) in &span_pairs {
            if bytes.len() > 1
                && let Some(mat) = ac.find(&bytes[..bytes.len() - 1])
            {
                next_prefix[tok.to_usize().unwrap()] = ac_tokens[mat.pattern().as_usize()];
            }
        }

        // Build pair_lookup and split_table incrementally in rank order.
        // For each token, walk the next_prefix chain to find the canonical
        // split (prefix, suffix) where both halves are lower-ranked.
        let mut pair_lookup: WCHashMap<Pair<T>, T> = WCHashMap::default();
        let mut split_table: Vec<Pair<T>> = vec![(T::zero(), T::zero()); table_size];

        for &(ref bytes, tok) in &span_pairs {
            let id = tok.to_usize().unwrap();
            let mut prefix_tok = next_prefix[id];
            let mut found = false;

            while prefix_tok != T::max_value() {
                let prefix_len = token_lens[prefix_tok.to_usize().unwrap()];
                let suffix_bytes = &bytes[prefix_len..];
                // Look up suffix as a token via AC.
                if let Some(mat) = ac.find(suffix_bytes) {
                    let suffix_tok = ac_tokens[mat.pattern().as_usize()];
                    // Verify the AC match covers exactly the suffix bytes.
                    if mat.start() == 0
                        && mat.end() == suffix_bytes.len()
                        && prefix_tok < tok
                        && suffix_tok < tok
                        && is_valid_token_pair(&pair_lookup, &split_table, prefix_tok, suffix_tok)
                    {
                        pair_lookup.insert((prefix_tok, suffix_tok), tok);
                        split_table[id] = (prefix_tok, suffix_tok);
                        found = true;
                        break;
                    }
                }
                // Try a shorter prefix.
                prefix_tok = next_prefix[prefix_tok.to_usize().unwrap()];
            }
            if !found {
                // Leaf token (byte-level or no valid split found).
                split_table[id] = (tok, tok);
            }
        }

        Self {
            pair_lookup,
            split_table,
            next_prefix,
            ac,
            ac_tokens,
            token_lens,
        }
    }

    /// Find the longest matching token starting at the beginning of `text`.
    #[inline]
    fn next_match(
        &self,
        text: &[u8],
    ) -> Option<T> {
        self.ac
            .find(text)
            .map(|m| self.ac_tokens[m.pattern().as_usize()])
    }

    /// Get the next-shorter prefix token, or `None` if at a leaf.
    #[inline]
    fn next_prefix_of(
        &self,
        token: T,
    ) -> Option<T> {
        let p = self.next_prefix[token.to_usize().unwrap()];
        if p == T::max_value() { None } else { Some(p) }
    }

    /// Get the byte length of a token.
    #[inline]
    fn token_len(
        &self,
        token: T,
    ) -> usize {
        self.token_lens[token.to_usize().unwrap()]
    }

    /// Check whether two adjacent tokens form a valid BPE split boundary.
    #[inline]
    fn is_valid_token_pair(
        &self,
        t1: T,
        t2: T,
    ) -> bool {
        is_valid_token_pair(&self.pair_lookup, &self.split_table, t1, t2)
    }
}

/// Check whether two adjacent tokens form a valid BPE boundary.
///
/// Recursively undoes BPE merges to verify that no merge rule would have
/// combined bytes across the split point at a lower rank.
fn is_valid_token_pair<T: TokenType>(
    pair_lookup: &WCHashMap<Pair<T>, T>,
    split_table: &[Pair<T>],
    mut token1: T,
    mut token2: T,
) -> bool {
    let mut limit = T::max_value();
    loop {
        if let Some(&combined) = pair_lookup.get(&(token1, token2))
            && combined < limit
        {
            return false;
        }
        if token1 > token2 {
            limit = token1;
            token1 = split_table[token1.to_usize().unwrap()].1;
            if token1 == limit {
                limit = token2 + T::one();
                token2 = split_table[token2.to_usize().unwrap()].0;
                if token2 + T::one() == limit {
                    return true;
                }
            }
        } else {
            limit = token2 + T::one();
            token2 = split_table[token2.to_usize().unwrap()].0;
            if token2 + T::one() == limit {
                limit = token1;
                token1 = split_table[token1.to_usize().unwrap()].1;
                if token1 == limit {
                    return true;
                }
            }
        }
    }
}

/// Bitfield with all bits initially set to 1.
struct BitField {
    words: Vec<u64>,
}

impl BitField {
    fn new(bits: usize) -> Self {
        Self {
            words: vec![u64::MAX; bits.div_ceil(64)],
        }
    }

    #[inline]
    fn is_set(
        &self,
        bit: usize,
    ) -> bool {
        let (word, bit) = (bit / 64, bit % 64);
        self.words[word] & (1 << bit) != 0
    }

    #[inline]
    fn clear(
        &mut self,
        bit: usize,
    ) {
        let (word, bit) = (bit / 64, bit % 64);
        self.words[word] &= !(1 << bit);
    }

    fn reset(
        &mut self,
        bits: usize,
    ) {
        let needed = bits.div_ceil(64);
        self.words.clear();
        self.words.resize(needed, u64::MAX);
    }
}

/// A [`SpanEncoder`] using BPE backtracking with Aho-Corasick matching.
pub struct BpeBacktrackSpanEncoder<T> {
    vocab: Arc<BpeVocab<T>>,
    bitfield: BitField,
}

impl<T: TokenType> BpeBacktrackSpanEncoder<T> {
    /// Create a new encoder from a shared [`BpeVocab`].
    pub fn new(vocab: Arc<BpeVocab<T>>) -> Self {
        Self {
            vocab,
            bitfield: BitField::new(0),
        }
    }
}

impl<T: TokenType> core::fmt::Debug for BpeBacktrackSpanEncoder<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("BpeBacktrackSpanEncoder").finish()
    }
}

impl<T: TokenType> SpanEncoder<T> for BpeBacktrackSpanEncoder<T> {
    fn encode_append_compound_span(
        &mut self,
        _vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        if span.is_empty() {
            return;
        }

        let bpe = &*self.vocab;
        self.bitfield.reset(span.len() + 1);

        let mut pos = 0usize;
        let mut next_token = bpe.next_match(span);

        // buf holds tokens produced for *this span* so we can backtrack.
        let base = tokens.len();

        while let Some(mut token) = next_token {
            let last = if tokens.len() > base {
                Some(tokens[tokens.len() - 1])
            } else {
                None
            };

            loop {
                let end_pos = pos + bpe.token_len(token);
                if self.bitfield.is_set(end_pos)
                    && last
                        .map(|lt| bpe.is_valid_token_pair(lt, token))
                        .unwrap_or(true)
                {
                    // Accept this token.
                    tokens.push(token);
                    pos = end_pos;
                    next_token = bpe.next_match(&span[end_pos..]);
                    break;
                } else if let Some(shorter) = bpe.next_prefix_of(token) {
                    // Try a shorter prefix.
                    token = shorter;
                } else {
                    // Backtrack: mark position as visited, pop previous token.
                    self.bitfield.clear(pos);
                    if let Some(prev) = last {
                        tokens.pop();
                        pos -= bpe.token_len(prev);
                        next_token = Some(prev);
                    } else {
                        next_token = None;
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use crate::{
        TokenEncoder,
        TokenType,
        encoders::{
            testing::{
                common_encoder_test_vocab,
                common_encoder_tests,
            },
            token_span_encoder::{
                SpanEncoderSelector,
                TokenSpanEncoder,
            },
        },
        spanners::TextSpannerBuilder,
        vocab::UnifiedTokenVocab,
    };

    fn test_encoder<T: TokenType>() {
        let vocab: Arc<UnifiedTokenVocab<T>> = common_encoder_test_vocab().into();
        let encoder = TokenSpanEncoder::<T>::new_with_selector(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            SpanEncoderSelector::BpeBacktrack,
        );
        let encoder: Arc<dyn TokenEncoder<T>> = Arc::new(encoder);
        common_encoder_tests(vocab, encoder)
    }

    #[test]
    fn test_encoder_u16() {
        test_encoder::<u16>();
    }

    #[test]
    fn test_encoder_u32() {
        test_encoder::<u32>();
    }
}
