//! # Vocab Trainer

use core::cmp::Ordering;

use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use num_traits::{
    One,
    Zero,
};
use wordchipper::{
    Pair,
    TokenType,
    UnifiedTokenVocab,
    VocabIndex,
    WCError,
    WCHashSet,
    WCResult,
    hash_map_with_capacity,
    support::regex::RegexPattern,
    vocab::{
        ByteMapVocab,
        PairMapVocab,
        PairTokenMap,
        utility::validators::{
            U8_SIZE,
            expect_vocab_size,
        },
    },
};

use crate::{
    CountType,
    utility::{
        PairIndexMap,
        PairSpanIndex,
        TextSpanCounter,
        TextSpanCounterOptions,
        TokenSpanBuf,
    },
};

/// Options for [`BPETrainer`].
#[derive(Debug, Clone)]
pub struct BPETRainerOptions {
    /// The regex pattern used for text splitting.
    pub pattern: RegexPattern,

    /// The vocab size.
    pub vocab_size: usize,
}

impl BPETRainerOptions {
    /// Create new options.
    ///
    /// ## Arguments
    /// * `pattern` - The word split pattern.
    /// * `vocab_size` - The target vocabulary size.
    ///
    /// ## Returns
    /// A new `BinaryPairVocabTrainerOptions` instance.
    pub fn new<P: Into<RegexPattern>>(
        pattern: P,
        vocab_size: usize,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            vocab_size,
        }
    }
}

impl BPETRainerOptions {
    /// Sets the vocab size.
    ///
    /// ## Arguments
    /// * `vocab_size` - The desired vocabulary size; must be >= 256 (the size
    ///   of the u8 space).
    ///
    /// ## Returns
    /// The updated `BinaryPairVocabTrainerOptions` instance.
    pub fn with_vocab_size(
        self,
        vocab_size: usize,
    ) -> Self {
        Self { vocab_size, ..self }
    }

    /// Sets the regex pattern used for text splitting.
    ///
    /// ## Arguments
    /// * `pattern` - The new word split pattern.
    ///
    /// ## Returns
    /// The updated `BinaryPairVocabTrainerOptions` instance.
    ///
    /// ## Panics
    /// Panics if the regex pattern compilation fails.
    pub fn with_pattern<P: Into<RegexPattern>>(
        self,
        pattern: P,
    ) -> Self {
        let pattern = pattern.into();
        pattern.compile().expect("regex pattern compilation failed");
        Self { pattern, ..self }
    }

    /// Initializes a [`BPETrainer`] from these options.
    ///
    /// ## Returns
    /// A new `BinaryPairVocabTrainer` instance.
    pub fn init(self) -> BPETrainer {
        BPETrainer::new(self)
    }
}

/// Info about a [`Pair`] that could be merged.
#[derive(Debug, Eq)]
pub struct MergeJob<T: TokenType, C: CountType> {
    /// The number of instances of this pair in the corpus.
    pub count: C,

    /// The pair to merge.
    pub pair: Pair<T>,

    /// Word indices that may contain this pair.
    pub word_indices: WCHashSet<usize>,
}

impl<T: TokenType, C: CountType> MergeJob<T, C> {
    /// The job key.
    ///
    /// Max-heap by count; tie-break to ascending pair order (deterministic)
    pub fn heap_key(&self) -> (C, Pair<T>) {
        (self.count, self.pair)
    }
}

impl<T: TokenType, C: CountType> PartialEq for MergeJob<T, C> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.heap_key() == other.heap_key()
    }
}

impl<T: TokenType, C: CountType> PartialOrd for MergeJob<T, C> {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: TokenType, C: CountType> Ord for MergeJob<T, C> {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.heap_key().cmp(&other.heap_key())
    }
}

/// Trainer for learning binary pair encodings.
///
/// # Parameters
/// * `K` - the type used to store strings in the word counts.
/// * `C` - the type used to store counts in the word counts.
pub struct BPETrainer {
    /// Trainer options.
    pub options: BPETRainerOptions,

    /// The text span counter.
    pub span_counter: TextSpanCounter<CompactString, u32>,
}

/// Basic binary pair train results.
#[derive(Debug, Clone)]
pub struct TrainResults<T: TokenType> {
    /// The trained vocab's word split pattern.
    pub pattern: RegexPattern,

    /// The trained vocab's byte/token mapping table.
    pub pair_vocab: PairMapVocab<T>,
}

impl BPETrainer {
    /// Initializes a [`BPETrainer`].
    ///
    /// ## Arguments
    /// * `options` - The trainer options.
    ///
    /// ## Returns
    /// A new `BinaryPairVocabTrainer` instance.
    pub fn new(options: BPETRainerOptions) -> Self {
        let span_counter = TextSpanCounter::<CompactString, u32>::new(
            options
                .pattern
                .compile()
                .expect("regex pattern compilation failed"),
            TextSpanCounterOptions::default(),
        );

        BPETrainer {
            options,
            span_counter,
        }
    }

    /// Update word counts inplace from a sample iterator.
    ///
    /// ## Arguments
    /// * `samples` - An iterator over string-like samples.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, samples)))]
    pub fn update_from_samples<I>(
        &mut self,
        samples: I,
    ) where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        self.span_counter.update_from_samples(samples);
    }

    /// Trains [`UnifiedTokenVocab<T>`].
    ///
    /// The resulting vocab will contain:
    /// * the trainer's word split pattern,
    /// * a ``{(T, T) -> T}`` pair map vocab with the learned binary pair
    ///   merges,
    /// * a ``{Vec<u8> -> T}`` word map that is empty.
    ///
    /// ## Arguments
    /// * `byte_vocab` - the byte/token mapping table to use for training.
    ///
    /// ## Returns
    /// A `Result` containing the `TrainResults<T>` or an error.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, byte_vocab)))]
    fn train_basic_pairs<T>(
        self,
        byte_vocab: ByteMapVocab<T>,
    ) -> WCResult<TrainResults<T>>
    where
        T: TokenType,
    {
        expect_vocab_size::<T>(self.options.vocab_size);

        let num_merges = self.options.vocab_size - U8_SIZE;
        log::info!("Starting BPE training: {} merges to compute", num_merges);

        self.options
            .pattern
            .compile()
            .map_err(|e| WCError::External(e.to_string()))?;

        let mut pairs: PairTokenMap<T> = hash_map_with_capacity(num_merges);

        let (mut words, word_counts): (Vec<TokenSpanBuf<T>>, Vec<C>) = self
            .span_counter
            .to_text_span_counts_iter(&byte_vocab)
            .unzip();

        log::info!("Building pair index...");

        let PairSpanIndex {
            mut pair_counts,
            pair_index: table_pair_index,
        } = PairSpanIndex::from_span_count_table(&words, &word_counts);

        type C = u32;
        let zero = C::zero();
        let one = C::one();

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, word_indices) in table_pair_index.into_iter() {
            let count = *pair_counts.get(&pair).unwrap_or(&zero);
            if count > zero {
                heap.push(MergeJob {
                    pair,
                    count,
                    word_indices,
                });
            }
        }
        // ---- Merge loop ----
        log::info!("Starting merge loop");
        let mut merges_done = 0;
        let mut last_log_percent = 0;

        // The first token we'll allocate is after all the byte tokens.
        let mut next_token = byte_vocab.max_token().unwrap() + T::one();

        while merges_done < num_merges {
            let Some(mut job) = heap.pop() else {
                // No more pairs to merge
                break;
            };

            {
                // Lazy refresh the job count.
                let current = *pair_counts.get(&job.pair).unwrap_or(&zero);
                if job.count != current {
                    job.count = current;
                    if job.count > zero {
                        heap.push(job);
                    }
                    continue;
                }
            }

            if job.count == zero {
                // No live matches.
                break;
            }

            // Generate a new token ID for this merge
            let new_token = next_token;
            next_token = next_token + T::one();

            // Record merge
            pairs.insert(job.pair, new_token);

            let mut new_token_pair_map: PairIndexMap<T> = hash_map_with_capacity(16);

            // Merge this pair in all words where it occurs
            for &word_idx in &job.word_indices {
                // Apply merge to this word.
                words[word_idx].merge_pair_cb(job.pair, new_token, &mut |pair, delta| {
                    // Update global pair counts based on this word's count
                    if delta < 0 {
                        // This (a, b) pair was removed from this span.
                        *pair_counts.entry(pair).or_default() -= one;
                    }
                    if delta > 0 {
                        // This (a, b) pair was added to this span.
                        // And either a or b is new_token.
                        *pair_counts.entry(pair).or_default() += one;
                        new_token_pair_map.entry(pair).or_default().insert(word_idx);
                    }
                });
            }

            // These will all contain the new token and are not yet in the heap:
            // * ``(_, T)`` or ``(T, _)``
            // and are not yet in the heap.
            for (pair, word_indices) in new_token_pair_map {
                let count = *pair_counts.get(&pair).unwrap_or(&zero);
                if count > zero {
                    heap.push(MergeJob {
                        pair,
                        count,
                        word_indices,
                    });
                }
            }

            merges_done += 1;

            // Log progress every 1%
            let current_percent = (merges_done * 100) / num_merges;
            if current_percent > last_log_percent {
                log::info!(
                    "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {:?} (frequency: {})",
                    current_percent,
                    merges_done,
                    num_merges,
                    job.pair,
                    new_token,
                    job.count
                );
                last_log_percent = current_percent;
            }
        }

        pairs.shrink_to_fit();

        let pair_vocab: PairMapVocab<T> = PairMapVocab::<T>::new(byte_vocab.clone(), pairs)?;

        log::info!("Finished training: {} merges completed", merges_done);
        Ok(TrainResults {
            pattern: self.options.pattern,
            pair_vocab,
        })
    }

    /// Trains [`UnifiedTokenVocab<T>`].
    ///
    /// The resulting vocab will contain:
    /// * the trainer's word split pattern,
    /// * a ``{(T, T) -> T}`` pair map vocab with the learned binary pair
    ///   merges,
    /// * a ``{Vec<u8> -> T}`` word map that is empty.
    ///
    /// ## Arguments
    /// * `byte_vocab` - the byte/token mapping table to use for training.
    ///
    /// ## Returns
    /// A `Result` containing the `UnifiedTokenVocab<T>` or an error.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, byte_vocab)))]
    pub fn train<T>(
        self,
        byte_vocab: ByteMapVocab<T>,
    ) -> WCResult<UnifiedTokenVocab<T>>
    where
        T: TokenType,
    {
        let results = self.train_basic_pairs(byte_vocab)?;
        UnifiedTokenVocab::from_pair_vocab(results.pattern.into(), results.pair_vocab)
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;
    use std::sync::Arc;

    use wordchipper::{
        TokenDecoder,
        TokenEncoder,
        TokenizerOptions,
        UnifiedTokenVocab,
        pretrained::openai::OA_CL100K_BASE_PATTERN,
        vocab::ByteMapVocab,
    };

    use crate::{
        BPETRainerOptions,
        bpe_trainer::MergeJob,
    };

    #[test]
    fn test_tokenizer_options() {
        let options = BPETRainerOptions::new(OA_CL100K_BASE_PATTERN, 1000);

        assert_eq!(options.vocab_size, 1000);
        assert_eq!(options.pattern, OA_CL100K_BASE_PATTERN.into());

        let options = options.with_vocab_size(2000).with_pattern(r"\S+");

        assert_eq!(options.vocab_size, 2000);
        assert_eq!(options.pattern, r"\S+".into());
    }

    #[test]
    #[should_panic(expected = "regex pattern compilation failed")]
    fn test_tokenizer_options_bad_pattern() {
        let _ = BPETRainerOptions::new(r"(", 1000).init();
    }

    #[test]
    fn test_train_tokenizer() {
        type T = u16;
        let options = BPETRainerOptions::new(OA_CL100K_BASE_PATTERN, 1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let mut trainer = options.init();
        trainer.update_from_samples(samples.iter());

        let byte_vocab: ByteMapVocab<T> = Default::default();

        let vocab: Arc<UnifiedTokenVocab<T>> = trainer.train(byte_vocab.clone()).unwrap().into();

        let tokenizer = TokenizerOptions::default()
            .with_parallel(false)
            .build::<T>(vocab.clone());

        for sample in samples {
            let tokens = tokenizer.try_encode(sample, None).unwrap();
            assert_eq!(
                tokenizer.try_decode_to_string(&tokens).unwrap().unwrap(),
                sample
            );
        }
    }

    #[test]
    fn test_merge_job_heap_key() {
        type T = u32;
        type C = u32;

        let job1: MergeJob<T, C> = MergeJob {
            pair: (1, 2),
            count: 2,
            word_indices: Default::default(),
        };

        let job2 = MergeJob {
            pair: (2, 1),
            count: 1,
            word_indices: Default::default(),
        };
        let job3 = MergeJob {
            pair: (2, 2),
            count: 1,
            word_indices: Default::default(),
        };

        assert_eq!(&job1, &job1);
        assert_ne!(&job1, &job2);

        assert_eq!(job1.heap_key(), (2, (1, 2)));
        assert_eq!(job2.heap_key(), (1, (2, 1)));

        assert_eq!(job1.heap_key().cmp(&job1.heap_key()), Ordering::Equal);
        assert_eq!(
            job1.heap_key().partial_cmp(&job1.heap_key()),
            Some(Ordering::Equal)
        );

        assert_eq!(job2.heap_key().cmp(&job2.heap_key()), Ordering::Equal);

        assert_eq!(job1.heap_key().cmp(&job2.heap_key()), Ordering::Greater);
        assert_eq!(job2.heap_key().cmp(&job1.heap_key()), Ordering::Less);

        assert_eq!(job3.heap_key().cmp(&job2.heap_key()), Ordering::Greater);
        assert_eq!(job2.heap_key().cmp(&job3.heap_key()), Ordering::Less);
    }
}
