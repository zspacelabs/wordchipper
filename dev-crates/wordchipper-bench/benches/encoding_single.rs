#![allow(missing_docs)]

use std::sync::Arc;

use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
};
use wordchipper::{
    TokenEncoder,
    UnifiedTokenVocab,
    encoders::token_span_encoder::{
        SpanEncoderSelector,
        TokenSpanEncoder,
    },
    spanners::TextSpannerBuilder,
};
use wordchipper_bench::{
    HF_CL100K,
    HF_O200K,
    OA_CL100K_BASE,
    OA_O200K_BASE,
    OA_R50K_BASE,
    load_cached_vocab,
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

pub enum LexerMode {
    Regex,
    RegexAutomata,
    Logos,
}

pub fn bench_wc(
    bencher: Bencher,
    text: &str,
    model: &str,
    selector: SpanEncoderSelector,
    mode: LexerMode,
) {
    let vocab: Arc<UnifiedTokenVocab<u32>> = load_cached_vocab(model).unwrap();

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
    let encoder: Arc<dyn TokenEncoder<u32>> = Arc::new(TokenSpanEncoder::<u32>::new_with_selector(
        spanner, vocab, selector,
    ));

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| encoder.try_encode(black_box(text), None).unwrap());
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

    mod r50k {
        use super::*;

        #[divan::bench]
        fn tiktoken(bencher: Bencher) {
            bench_tt(bencher, &english_text(), &tiktoken_rs::r50k_base().unwrap())
        }

        mod wordchipper {
            use super::*;

            mod regex {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Regex,
                    );
                }
            }

            mod regex_automata {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::RegexAutomata,
                    );
                }
            }

            mod logos {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Logos,
                    );
                }
            }
        }
    }

    mod cl100k {
        use super::*;

        #[divan::bench]
        fn tiktoken(bencher: Bencher) {
            bench_tt(
                bencher,
                &english_text(),
                &tiktoken_rs::cl100k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn tokenizers(bencher: Bencher) {
            bench_hf(bencher, &english_text(), HF_CL100K)
        }

        #[divan::bench]
        fn bpe_openai(bencher: Bencher) {
            bench_bpe_openai(bencher, &english_text(), ::bpe_openai::cl100k_base())
        }

        mod wordchipper {
            use super::*;

            mod regex {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Regex,
                    );
                }
            }

            mod regex_automata {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::RegexAutomata,
                    );
                }
            }

            mod logos {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Logos,
                    );
                }
            }
        }
    }

    mod o200k {
        use super::*;

        #[divan::bench]
        fn tiktoken(bencher: Bencher) {
            bench_tt(
                bencher,
                &english_text(),
                &tiktoken_rs::o200k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn tokenizers(bencher: Bencher) {
            bench_hf(bencher, &english_text(), HF_O200K)
        }

        #[divan::bench]
        fn bpe_openai(bencher: Bencher) {
            bench_bpe_openai(bencher, &english_text(), ::bpe_openai::o200k_base())
        }

        mod wordchipper {
            use super::*;

            mod regex {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Regex,
                    );
                }
            }

            mod regex_automata {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::RegexAutomata,
                    );
                }
            }

            mod logos {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &english_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Logos,
                    );
                }
            }
        }
    }
}

mod diverse {
    use super::*;

    mod r50k {
        use super::*;

        #[divan::bench]
        fn tiktoken(bencher: Bencher) {
            bench_tt(bencher, &diverse_text(), &tiktoken_rs::r50k_base().unwrap())
        }

        mod wordchipper {
            use super::*;

            mod regex {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Regex,
                    );
                }
            }

            mod regex_automata {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::RegexAutomata,
                    );
                }
            }

            mod logos {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_R50K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Logos,
                    );
                }
            }
        }
    }

    mod cl100k {
        use super::*;

        #[divan::bench]
        fn tiktoken(bencher: Bencher) {
            bench_tt(
                bencher,
                &diverse_text(),
                &tiktoken_rs::cl100k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn tokenizers(bencher: Bencher) {
            bench_hf(bencher, &diverse_text(), HF_CL100K)
        }

        #[divan::bench]
        fn bpe_openai(bencher: Bencher) {
            bench_bpe_openai(bencher, &diverse_text(), ::bpe_openai::cl100k_base())
        }

        mod wordchipper {
            use super::*;

            mod regex {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Regex,
                    );
                }
            }

            mod regex_automata {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::RegexAutomata,
                    );
                }
            }

            mod logos {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_CL100K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Logos,
                    );
                }
            }
        }
    }

    mod o200k {
        use super::*;

        #[divan::bench]
        fn tiktoken(bencher: Bencher) {
            bench_tt(
                bencher,
                &diverse_text(),
                &tiktoken_rs::o200k_base().unwrap(),
            )
        }

        #[divan::bench]
        fn tokenizers(bencher: Bencher) {
            bench_hf(bencher, &diverse_text(), HF_O200K)
        }

        #[divan::bench]
        fn bpe_openai(bencher: Bencher) {
            bench_bpe_openai(bencher, &diverse_text(), ::bpe_openai::o200k_base())
        }

        mod wordchipper {
            use super::*;

            mod regex {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Regex,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Regex,
                    );
                }
            }

            mod regex_automata {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::RegexAutomata,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::RegexAutomata,
                    );
                }
            }

            mod logos {
                use super::*;

                #[divan::bench]
                fn buffer_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BufferSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn tail_sweep(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::TailSweep,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn merge_heap(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::MergeHeap,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn priority_merge(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::PriorityMerge,
                        LexerMode::Logos,
                    );
                }

                #[divan::bench]
                fn bpe_backtrack(bencher: Bencher) {
                    bench_wc(
                        bencher,
                        &diverse_text(),
                        OA_O200K_BASE,
                        SpanEncoderSelector::BpeBacktrack,
                        LexerMode::Logos,
                    );
                }
            }
        }
    }
}
