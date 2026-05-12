#![allow(missing_docs)]

use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
};
use wordchipper::{
    TokenEncoderOptions,
};
use wordchipper_bench::{
    HF_QWEN35,
    WC_QWEN35,
    load_cached_encoder,
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

fn bench_wc(
    bencher: Bencher,
    text: &str,
) {
    let encoder = load_cached_encoder::<u32>(WC_QWEN35, TokenEncoderOptions::default());

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| encoder.try_encode(black_box(text), None).unwrap());
}

fn bench_hf(
    bencher: Bencher,
    text: &str,
) {
    let tok = tokenizers::Tokenizer::from_pretrained(HF_QWEN35, None).unwrap();

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| tok.encode(black_box(text), true).unwrap());
}

mod english {
    use super::*;

    #[divan::bench]
    fn wordchipper(bencher: Bencher) {
        bench_wc(bencher, &english_text());
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, &english_text());
    }
}

mod diverse {
    use super::*;

    #[divan::bench]
    fn wordchipper(bencher: Bencher) {
        bench_wc(bencher, &diverse_text());
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, &diverse_text());
    }
}