#![allow(missing_docs)]
#![allow(unused)]

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
    encoders::token_span_encoder::{
        SpanEncoderSelector,
        TokenSpanEncoder,
    },
    spanners::TextSpannerBuilder,
    support::concurrency::rayon::ParallelRayonEncoder,
};
use wordchipper_bench::{
    HF_CL100K,
    HF_O200K,
    HF_R50K,
    OA_CL100K_BASE,
    OA_O200K_BASE,
    OA_R50K_BASE,
    load_cached_vocab,
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

pub enum LexerMode {
    Regex,
    RegexAutomata,
    Logos,
}

fn build_encoder_wc<T: TokenType>(
    model: &str,
    selector: SpanEncoderSelector,
    mode: LexerMode,
) -> Arc<dyn TokenEncoder<T>> {
    let vocab: Arc<UnifiedTokenVocab<T>> = load_cached_vocab(model).unwrap();

    let mut builder = TextSpannerBuilder::from_vocab(&vocab);
    builder.set_concurrent(true);

    match mode {
        LexerMode::Regex => {
            builder.set_regex_automata(false);
            builder.set_accelerated_lexers(false);
        }
        LexerMode::RegexAutomata => {
            builder.set_regex_automata(true);
            builder.set_accelerated_lexers(false);
        }
        LexerMode::Logos => {
            builder.set_regex_automata(false);
            builder.set_accelerated_lexers(true);
        }
    }

    let spanner = builder.build();

    let enc: Arc<dyn TokenEncoder<T>> = Arc::new(TokenSpanEncoder::<T>::new_with_selector(
        spanner, vocab, selector,
    ));

    Arc::new(ParallelRayonEncoder::new(enc))
}

fn bench_wc(
    bencher: Bencher,
    model: &str,
    selector: SpanEncoderSelector,
    mode: LexerMode,
) {
    let strs = BATCH.strs();
    let encoder = build_encoder_wc::<u32>(model, selector, mode);
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

macro_rules! gen_cases {
    (@wordchipper_mod $base:expr) => {
        mod wordchipper {
            use super::*;

            gen_cases!(@lexer_mod regex, LexerMode::Regex, $base);
            gen_cases!(@lexer_mod regex_automata, LexerMode::RegexAutomata, $base);
            gen_cases!(@lexer_mod logos, LexerMode::Logos, $base);
        }
    };
    (@lexer_mod $name:ident, $lexer:expr, $base:expr) => {
        mod $name {
            use super::*;

            gen_cases!(@enc_case buffer_sweep, SpanEncoderSelector::BufferSweep, $lexer, $base);
            gen_cases!(@enc_case tail_sweep, SpanEncoderSelector::TailSweep, $lexer, $base);
            gen_cases!(@enc_case merge_buffer_heap, SpanEncoderSelector::MergeBufferHeap, $lexer, $base);
            gen_cases!(@enc_case merge_tail_heap, SpanEncoderSelector::MergeTailHeap, $lexer, $base);
            gen_cases!(@enc_case priority_merge, SpanEncoderSelector::PriorityMerge, $lexer, $base);
            gen_cases!(@enc_case bpe_backtrack, SpanEncoderSelector::BpeBacktrack, $lexer, $base);
        }
    };
    (@enc_case $name:ident, $enc:expr, $lexer:expr, $base:expr) => {
        #[divan::bench]
        fn $name(bencher: Bencher) {
            bench_wc(bencher, $base, $enc, $lexer);
        }
    };
}

mod r50k {
    use super::*;

    #[divan::bench]
    fn tiktoken(bencher: Bencher) {
        bench_tt(bencher, &tiktoken_rs::r50k_base().unwrap())
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, HF_R50K)
    }

    gen_cases!(@wordchipper_mod OA_R50K_BASE);
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

    gen_cases!(@wordchipper_mod OA_O200K_BASE);
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

    gen_cases!(@wordchipper_mod OA_O200K_BASE);
}
