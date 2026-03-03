#![allow(missing_docs)]

use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
};
use wordchipper::{
    TokenEncoderOptions,
    encoders::token_span_encoder::SpanEncoderSelector,
};
use wordchipper_bench::{
    HF_CL100K,
    HF_O200K,
    OA_CL100K_BASE,
    OA_O200K_BASE,
    OA_R50K_BASE,
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

static DIVERSE_CORPUS: &str = include_str!("data/multilingual.txt");
static ENGLISH_CORPUS: &str = include_str!("data/english.txt");

fn diverse_text() -> String {
    DIVERSE_CORPUS.repeat(10)
}

fn english_text() -> String {
    ENGLISH_CORPUS.repeat(10)
}

pub fn bench_wc(
    bencher: Bencher,
    text: &str,
    model: &str,
    selector: SpanEncoderSelector,
    accelerator: bool,
) {
    let encoder = wordchipper_bench::load_encoder::<u32>(
        model,
        TokenEncoderOptions::default()
            .with_accelerated_lexers(accelerator)
            .with_span_encoder(selector)
            .with_parallel(true)
            .with_concurrent(false),
    );

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| encoder.try_encode(black_box(text)).unwrap());
}

pub fn bench_tt(
    bencher: Bencher,
    text: &str,
    tok: &tiktoken_rs::CoreBPE,
) {
    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| tok.encode_with_special_tokens(black_box(text)));
}

pub fn bench_hf(
    bencher: Bencher,
    text: &str,
    name: &str,
) {
    let tok = tokenizers::Tokenizer::from_pretrained(name, None).unwrap();

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| tok.encode(black_box(text), true).unwrap());
}

pub fn bench_bpe_openai(
    bencher: Bencher,
    text: &str,
    tok: &::bpe_openai::Tokenizer,
) {
    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| tok.encode(black_box(text)));
}

mod english {
    use super::*;

    mod buffer_sweep {
        use super::*;

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }
    }

    mod tail_sweep {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }
    }

    mod merge_heap {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }
    }

    mod bpe_backtrack {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &english_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                true,
            );
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_tt(bencher, &english_text(), &tiktoken_rs::r50k_base().unwrap())
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_tt(
                bencher,
                &english_text(),
                &tiktoken_rs::cl100k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_tt(
                bencher,
                &english_text(),
                &tiktoken_rs::o200k_base().unwrap(),
            )
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_hf(bencher, &english_text(), HF_CL100K)
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_hf(bencher, &english_text(), HF_O200K)
        }
    }

    mod bpe_openai {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_bpe_openai(bencher, &english_text(), ::bpe_openai::cl100k_base())
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_bpe_openai(bencher, &english_text(), ::bpe_openai::o200k_base())
        }
    }
}

mod diverse {
    use super::*;

    mod buffer_sweep {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BufferSweep,
                false,
            );
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BufferSweep,
                true,
            );
        }
    }

    mod tail_sweep {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::TailSweep,
                false,
            );
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::TailSweep,
                true,
            );
        }
    }

    mod merge_heap {
        use super::*;
        use crate::bench_wc;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::MergeHeap,
                false,
            );
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::MergeHeap,
                true,
            );
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::PriorityMerge,
                false,
            );
        }

        #[divan::bench]
        fn r50k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_R50K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::PriorityMerge,
                true,
            );
        }
    }

    mod bpe_backtrack {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                false,
            );
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                false,
            );
        }

        #[divan::bench]
        fn cl100k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_CL100K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                true,
            );
        }

        #[divan::bench]
        fn o200k_fast(bencher: Bencher) {
            bench_wc(
                bencher,
                &diverse_text(),
                OA_O200K_BASE,
                SpanEncoderSelector::BpeBacktrack,
                true,
            );
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn r50k(bencher: Bencher) {
            bench_tt(bencher, &diverse_text(), &tiktoken_rs::r50k_base().unwrap())
        }

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_tt(
                bencher,
                &diverse_text(),
                &tiktoken_rs::cl100k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_tt(
                bencher,
                &diverse_text(),
                &tiktoken_rs::o200k_base().unwrap(),
            )
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_hf(bencher, &diverse_text(), HF_CL100K)
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_hf(bencher, &diverse_text(), HF_O200K)
        }
    }

    mod bpe_openai {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            bench_bpe_openai(bencher, &diverse_text(), ::bpe_openai::cl100k_base())
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            bench_bpe_openai(bencher, &diverse_text(), ::bpe_openai::o200k_base())
        }
    }
}
