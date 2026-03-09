#![allow(missing_docs)]

use std::sync::{
    Arc,
    LazyLock,
};

use arrow::array::Array;
use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
};
use rayon::prelude::*;
use tiktoken_rs::CoreBPE;
use wordchipper::{
    TokenEncoder,
    TokenType,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    encoders::token_span_encoder::{
        SpanEncoderSelector,
        TokenSpanEncoder,
    },
    load_vocab,
    spanners::TextSpannerBuilder,
    support::concurrency::rayon::ParallelRayonEncoder,
};
use wordchipper_bench::{
    HF_CL100K,
    HF_O200K,
    OA_CL100K_BASE,
    OA_O200K_BASE,
    OA_R50K_BASE,
};
use wordchipper_data::dataset::DatasetCacheConfig;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const BATCH_SIZE: usize = 1024;

struct Batch {
    samples: Vec<String>,
    total_bytes: usize,
}

impl Batch {
    fn strs(&self) -> Vec<&str> {
        self.samples.iter().map(|s| s.as_str()).collect()
    }
}

fn load_batch() -> Batch {
    let mut cache = DatasetCacheConfig::default()
        .init()
        .expect("failed to initialize dataset cache");

    let reader = cache
        .read_batches(0, true)
        .expect("failed to read dataset batches");

    let mut samples = Vec::with_capacity(BATCH_SIZE);
    for batch in reader {
        let batch = batch.unwrap();
        let column = batch
            .column_by_name("text")
            .expect("missing 'text' column")
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();

        for val in column {
            samples.push(val.unwrap().to_string());
            if samples.len() >= BATCH_SIZE {
                let total_bytes = samples.iter().map(|s| s.len()).sum();
                return Batch {
                    samples,
                    total_bytes,
                };
            }
        }
    }

    let total_bytes = samples.iter().map(|s| s.len()).sum();
    Batch {
        samples,
        total_bytes,
    }
}

static BATCH: LazyLock<Batch> = LazyLock::new(load_batch);

fn build_encoder<T: TokenType>(
    model: &str,
    selector: SpanEncoderSelector,
    accelerated: bool,
    concurrent: bool,
) -> Arc<dyn TokenEncoder<T>> {
    let vocab: Arc<UnifiedTokenVocab<T>> = load_vocab(model, &mut WordchipperDiskCache::default())
        .unwrap()
        .vocab()
        .to_token_type::<T>()
        .unwrap()
        .into();

    let spanner = TextSpannerBuilder::new(vocab.spanning().clone())
        .with_accelerated_lexers(accelerated)
        .with_concurrent(concurrent)
        .build();

    let enc: Arc<dyn TokenEncoder<T>> = Arc::new(TokenSpanEncoder::<T>::new_with_selector(
        spanner, vocab, selector,
    ));

    Arc::new(ParallelRayonEncoder::new(enc))
}

fn bench_wc(
    bencher: Bencher,
    model: &str,
    selector: SpanEncoderSelector,
    accelerated: bool,
    concurrent: bool,
) {
    let strs = BATCH.strs();
    let encoder = build_encoder::<u32>(model, selector, accelerated, concurrent);

    bencher
        .counter(BytesCount::new(BATCH.total_bytes))
        .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
}

fn bench_tt(
    bencher: Bencher,
    bpe: &CoreBPE,
) {
    let strs = BATCH.strs();
    bencher
        .counter(BytesCount::new(BATCH.total_bytes))
        .bench(|| {
            strs.par_iter()
                .map(|s| bpe.encode_with_special_tokens(s))
                .collect::<Vec<_>>()
        });
}

fn bench_hf(
    bencher: Bencher,
    name: &str,
) {
    let tok = tokenizers::Tokenizer::from_pretrained(name, None).unwrap();
    let strs = BATCH.strs();
    bencher
        .counter(BytesCount::new(BATCH.total_bytes))
        .bench(|| tok.encode_batch(black_box(strs.clone()), true).unwrap());
}

fn bench_bpe_openai(
    bencher: Bencher,
    tok: &::bpe_openai::Tokenizer,
) {
    let strs = BATCH.strs();
    bencher
        .counter(BytesCount::new(BATCH.total_bytes))
        .bench(|| strs.par_iter().map(|s| tok.encode(s)).collect::<Vec<_>>());
}

mod r50k {
    use wordchipper_bench::HF_R50K;

    use super::*;

    #[divan::bench]
    fn tiktoken(bencher: Bencher) {
        bench_tt(bencher, &tiktoken_rs::r50k_base().unwrap())
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, HF_R50K)
    }

    mod wordchipper {
        use super::*;

        mod regex {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::TailSweep,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    false,
                    false,
                );
            }
        }

        mod regex_automata {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::TailSweep,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    false,
                    true,
                );
            }
        }

        mod logos {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::TailSweep,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_R50K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    true,
                    true,
                );
            }
        }
    }
}

mod cl100k {
    use super::*;

    #[divan::bench]
    fn tiktoken(bencher: Bencher) {
        bench_tt(bencher, &tiktoken_rs::cl100k_base().unwrap())
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, HF_CL100K)
    }

    #[divan::bench]
    fn bpe_openai(bencher: Bencher) {
        bench_bpe_openai(bencher, ::bpe_openai::cl100k_base())
    }

    mod wordchipper {
        use super::*;

        mod regex {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::TailSweep,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    false,
                    false,
                );
            }
        }

        mod regex_automata {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::TailSweep,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    false,
                    true,
                );
            }
        }

        mod logos {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::TailSweep,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_CL100K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    true,
                    true,
                );
            }
        }
    }
}

mod o200k {
    use super::*;

    #[divan::bench]
    fn tiktoken(bencher: Bencher) {
        bench_tt(bencher, &tiktoken_rs::o200k_base().unwrap())
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, HF_O200K)
    }

    #[divan::bench]
    fn bpe_openai(bencher: Bencher) {
        bench_bpe_openai(bencher, ::bpe_openai::o200k_base())
    }

    mod wordchipper {
        use super::*;

        mod regex {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::TailSweep,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    false,
                    false,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    false,
                    false,
                );
            }
        }

        mod regex_automata {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::TailSweep,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    false,
                    true,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    false,
                    true,
                );
            }
        }

        mod logos {
            use super::*;

            #[divan::bench]
            fn buffer_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::BufferSweep,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn tail_sweep(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::TailSweep,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn merge_heap(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::MergeHeap,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn priority_merge(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::PriorityMerge,
                    true,
                    true,
                );
            }

            #[divan::bench]
            fn bpe_backtrack(bencher: Bencher) {
                bench_wc(
                    bencher,
                    OA_O200K_BASE,
                    SpanEncoderSelector::BpeBacktrack,
                    true,
                    true,
                );
            }
        }
    }
}
