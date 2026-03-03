//! # Priority-merge [`SpanEncoder`].
//!
//! Uses a binary min-heap over a doubly-linked list for O(n log n) BPE merging,
//! replacing the O(n^2) linear-scan approach in
//! [`super::TailSweepSpanEncoder`].

use alloc::collections::BinaryHeap;
use core::cmp::Reverse;

use crate::{
    TokenType,
    alloc::vec::Vec,
    encoders::token_span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

const NONE: u32 = u32::MAX;

struct Node<T> {
    token: T,
    prev: u32,
    next: u32,
}

/// Heap entry representing a potential merge.
///
/// Ordered by (rank, `left_idx`) so the lowest-rank, leftmost pair is popped
/// first. `left_tok` and `right_tok` are stored for O(1) stale-entry detection.
#[derive(Eq)]
struct MergeEntry<T: Ord> {
    rank: T,
    left_idx: u32,
    left_tok: T,
    right_tok: T,
}

impl<T: Ord> PartialEq for MergeEntry<T> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.rank == other.rank && self.left_idx == other.left_idx
    }
}

impl<T: Ord> Ord for MergeEntry<T> {
    fn cmp(
        &self,
        other: &Self,
    ) -> core::cmp::Ordering {
        self.rank
            .cmp(&other.rank)
            .then(self.left_idx.cmp(&other.left_idx))
    }
}

impl<T: Ord> PartialOrd for MergeEntry<T> {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A [`SpanEncoder`] using a binary min-heap with a doubly-linked list.
///
/// Processes BPE merges in O(n log n) time per span, compared to the
/// O(n^2) linear-scan approach used by other encoders.
pub struct PriorityMergeSpanEncoder<T: TokenType> {
    nodes: Vec<Node<T>>,
    heap: BinaryHeap<Reverse<MergeEntry<T>>>,
}

impl<T: TokenType> Default for PriorityMergeSpanEncoder<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }
}

impl<T: TokenType> core::fmt::Debug for PriorityMergeSpanEncoder<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("PriorityMergeSpanEncoder").finish()
    }
}

impl<T: TokenType> Clone for PriorityMergeSpanEncoder<T> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<T: TokenType> SpanEncoder<T> for PriorityMergeSpanEncoder<T> {
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        let n = span.len();
        let byte_vocab = vocab.byte_vocab();

        if n < 2 {
            for &byte in span {
                tokens.push(byte_vocab.get_token(byte));
            }
            return;
        }

        // Build doubly-linked list of byte tokens.
        self.nodes.clear();
        self.nodes.reserve(n);
        for (i, &byte) in span.iter().enumerate() {
            self.nodes.push(Node {
                token: byte_vocab.get_token(byte),
                prev: if i == 0 { NONE } else { (i - 1) as u32 },
                next: if i + 1 < n { (i + 1) as u32 } else { NONE },
            });
        }

        // Seed the heap with all initially-mergeable adjacent pairs.
        self.heap.clear();
        for i in 0..(n - 1) {
            let left_tok = self.nodes[i].token;
            let right_tok = self.nodes[i + 1].token;
            if let Some(rank) = vocab.lookup_pair(&(left_tok, right_tok)) {
                self.heap.push(Reverse(MergeEntry {
                    rank,
                    left_idx: i as u32,
                    left_tok,
                    right_tok,
                }));
            }
        }

        // Process merges in priority order (lowest rank first).
        while let Some(Reverse(entry)) = self.heap.pop() {
            let li = entry.left_idx as usize;

            // Validate: left node still active with expected right neighbor.
            let ri_u32 = self.nodes[li].next;
            if ri_u32 == NONE {
                continue;
            }
            let ri = ri_u32 as usize;

            // Bidirectional adjacency + token freshness.
            if self.nodes[ri].prev != entry.left_idx
                || self.nodes[li].token != entry.left_tok
                || self.nodes[ri].token != entry.right_tok
            {
                continue;
            }

            // Merge: left absorbs right.
            let new_token = entry.rank;
            self.nodes[li].token = new_token;
            let right_next = self.nodes[ri].next;
            self.nodes[li].next = right_next;
            if right_next != NONE {
                self.nodes[right_next as usize].prev = entry.left_idx;
            }

            // Enqueue new neighbor pairs.
            let left_prev = self.nodes[li].prev;
            if left_prev != NONE {
                let prev_tok = self.nodes[left_prev as usize].token;
                if let Some(rank) = vocab.lookup_pair(&(prev_tok, new_token)) {
                    self.heap.push(Reverse(MergeEntry {
                        rank,
                        left_idx: left_prev,
                        left_tok: prev_tok,
                        right_tok: new_token,
                    }));
                }
            }
            if right_next != NONE {
                let next_tok = self.nodes[right_next as usize].token;
                if let Some(rank) = vocab.lookup_pair(&(new_token, next_tok)) {
                    self.heap.push(Reverse(MergeEntry {
                        rank,
                        left_idx: entry.left_idx,
                        left_tok: new_token,
                        right_tok: next_tok,
                    }));
                }
            }
        }

        // Collect final tokens by walking the linked list.
        let mut idx = 0u32;
        while idx != NONE {
            tokens.push(self.nodes[idx as usize].token);
            idx = self.nodes[idx as usize].next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TokenEncoder,
        TokenType,
        alloc::sync::Arc,
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
    };

    fn test_encoder<T: TokenType>() {
        let vocab: Arc<UnifiedTokenVocab<T>> = common_encoder_test_vocab().into();
        let encoder = TokenSpanEncoder::<T>::new_with_selector(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            SpanEncoderSelector::PriorityMerge,
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
