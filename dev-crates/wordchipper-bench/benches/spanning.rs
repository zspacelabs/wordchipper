#![allow(missing_docs)]

use std::sync::Arc;

use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
};
use wordchipper::{
    pretrained::openai::{
        OA_CL100K_BASE_PATTERN,
        OA_O200K_BASE_PATTERN,
        OA_R50K_BASE_PATTERN,
    },
    spanners::{
        TextSpanner,
        TextSpannerBuilder,
        TextSpanningConfig,
        span_lexers::SpanLexer,
    },
    support::regex::ConstRegexPattern,
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

fn build_regex_only_spanner(
    pattern: impl Into<wordchipper::support::regex::RegexPattern>
) -> Arc<dyn TextSpanner> {
    TextSpanningConfig::from_pattern(pattern)
        .with_concurrent(false)
        .with_accelerators(false)
        .with_regex_automata(false)
        .build()
}

fn build_regex_automata_spanner(pattern: ConstRegexPattern) -> Arc<dyn TextSpanner> {
    TextSpanningConfig::from_pattern(pattern)
        .with_concurrent(false)
        .with_accelerators(false)
        .with_regex_automata(true)
        .build()
}

fn build_default_spanner(
    pattern: impl Into<wordchipper::support::regex::RegexPattern>
) -> Arc<dyn TextSpanner> {
    let config: TextSpanningConfig<u32> = TextSpanningConfig::from_pattern(pattern);
    TextSpannerBuilder::new(config)
        .with_concurrent(false)
        .build()
}

mod english {
    use super::*;

    #[divan::bench]
    fn r50k_regex(bencher: Bencher) {
        let text = english_text();
        let spanner = build_regex_only_spanner(OA_R50K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn r50k_regex_automata(bencher: Bencher) {
        let text = english_text();
        let spanner = build_regex_automata_spanner(OA_R50K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn r50k_default(bencher: Bencher) {
        let text = english_text();
        let spanner = build_default_spanner(OA_R50K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn cl100k_regex(bencher: Bencher) {
        let text = english_text();
        let spanner = build_regex_only_spanner(OA_CL100K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn cl100k_regex_automata(bencher: Bencher) {
        let text = english_text();
        let spanner = build_regex_automata_spanner(OA_CL100K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn cl100k_default(bencher: Bencher) {
        let text = english_text();
        let spanner = build_default_spanner(OA_CL100K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn o200k_regex(bencher: Bencher) {
        let text = english_text();
        let spanner = build_regex_only_spanner(OA_O200K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn o200k_regex_automata(bencher: Bencher) {
        let text = english_text();
        let spanner = build_regex_automata_spanner(OA_O200K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn o200k_default(bencher: Bencher) {
        let text = english_text();
        let spanner = build_default_spanner(OA_O200K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }
}

mod diverse {
    use super::*;

    #[divan::bench]
    fn r50k_regex(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_regex_only_spanner(OA_R50K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn r50k_regex_automata(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_regex_automata_spanner(OA_R50K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn r50k_default(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_default_spanner(OA_R50K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn cl100k_regex(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_regex_only_spanner(OA_CL100K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn cl100k_regex_automata(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_regex_automata_spanner(OA_CL100K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn cl100k_default(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_default_spanner(OA_CL100K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn o200k_regex(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_regex_only_spanner(OA_O200K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn o200k_regex_automata(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_regex_automata_spanner(OA_O200K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }

    #[divan::bench]
    fn o200k_default(bencher: Bencher) {
        let text = diverse_text();
        let spanner = build_default_spanner(OA_O200K_BASE_PATTERN);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| spanner.split_spans(black_box(&text)));
    }
}
