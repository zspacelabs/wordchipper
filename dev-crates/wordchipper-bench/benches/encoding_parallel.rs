#![allow(missing_docs)]

use std::sync::LazyLock;

use arrow::array::Array;
use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
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

mod wordchipper {
    use ::wordchipper::{
        TokenEncoderOptions,
        encoders::token_span_encoder::SpanEncoderSelector,
    };
    use wordchipper_bench::{
        OA_CL100K_BASE,
        OA_O200K_BASE,
        OA_R50K_BASE,
    };

    use super::*;

    fn bench_variant(
        bencher: Bencher,
        model: &str,
        selector: SpanEncoderSelector,
        accelerated: bool,
    ) {
        let strs = BATCH.strs();

        let encoder = wordchipper_bench::load_encoder::<u32>(
            model,
            TokenEncoderOptions::default()
                .with_span_encoder(selector)
                .with_accelerated_lexers(accelerated)
                .with_parallel(true),
        );

        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
    }

    mod buffer_sweep {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_R50K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            )
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            )
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_R50K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            )
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            )
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            )
        }
    }

    mod tail_sweep {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_variant(bencher, OA_R50K_BASE, SpanEncoderSelector::TailSweep, false)
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            )
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_variant(bencher, OA_R50K_BASE, SpanEncoderSelector::TailSweep, true)
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::TailSweep,
                true,
            )
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_variant(bencher, OA_O200K_BASE, SpanEncoderSelector::TailSweep, true)
        }
    }

    mod merge_heap {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_variant(bencher, OA_R50K_BASE, SpanEncoderSelector::MergeHeap, false)
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            )
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_variant(bencher, OA_R50K_BASE, SpanEncoderSelector::MergeHeap, true)
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::MergeHeap,
                true,
            )
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_variant(bencher, OA_O200K_BASE, SpanEncoderSelector::MergeHeap, true)
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_R50K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            )
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            )
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_R50K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            )
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            )
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            )
        }
    }

    mod bpe_backtrack {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                false,
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                false,
            )
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_CL100K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                true,
            )
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_variant(
                bencher,
                OA_O200K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                true,
            )
        }
    }
}

mod tiktoken {
    use rayon::prelude::*;
    use tiktoken_rs::{
        CoreBPE,
        cl100k_base,
        o200k_base,
        r50k_base,
    };

    use super::*;

    fn bench_variant(
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

    #[divan::bench]
    fn r50k(bencher: Bencher) {
        bench_variant(bencher, &r50k_base().unwrap())
    }

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        bench_variant(bencher, &cl100k_base().unwrap())
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        bench_variant(bencher, &o200k_base().unwrap())
    }
}

mod tokenizers {
    use wordchipper_bench::{
        HF_CL100K,
        HF_O200K,
    };

    use super::*;

    fn bench_variant(
        bencher: Bencher,
        name: &str,
    ) {
        let tok = ::tokenizers::Tokenizer::from_pretrained(name, None).unwrap();
        let strs = BATCH.strs();
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| tok.encode_batch(black_box(strs.clone()), true).unwrap());
    }

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        bench_variant(bencher, HF_CL100K)
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        bench_variant(bencher, HF_O200K)
    }
}

mod bpe_openai {
    use rayon::prelude::*;

    use super::*;

    fn bench_variant(
        bencher: Bencher,
        tok: &::bpe_openai::Tokenizer,
    ) {
        let strs = BATCH.strs();
        bencher
            .counter(BytesCount::new(BATCH.total_bytes))
            .bench(|| strs.par_iter().map(|s| tok.encode(s)).collect::<Vec<_>>());
    }

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        bench_variant(bencher, ::bpe_openai::cl100k_base())
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        bench_variant(bencher, ::bpe_openai::o200k_base())
    }
}
