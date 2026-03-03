#![allow(missing_docs)]

use std::sync::{
    Arc,
    LazyLock,
};

use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
};
use tiktoken_rs::{
    CoreBPE,
    Rank,
};
use tokenizers::Tokenizer;
use wordchipper::{
    TokenDecoder,
    TokenEncoder,
    TokenizerOptions,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
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

struct WcFixture {
    tokenizer: Arc<wordchipper::Tokenizer<Rank>>,
}

impl WcFixture {
    fn by_name(model: &str) -> Self {
        let mut disk_cache = WordchipperDiskCache::default();
        let (_desc, vocab) = wordchipper::load_vocab(model, &mut disk_cache).unwrap();
        Self::from_vocab(vocab)
    }

    fn from_vocab(vocab: Arc<UnifiedTokenVocab<u32>>) -> Self {
        let tokenizer = TokenizerOptions::default().build(vocab);
        Self::new(tokenizer)
    }

    fn new(tokenizer: Arc<wordchipper::Tokenizer<Rank>>) -> Self {
        Self { tokenizer }
    }
}

struct TiktokenFixture {
    bpe: Arc<CoreBPE>,
}

static WC_CL100K: LazyLock<WcFixture> = LazyLock::new(|| WcFixture::by_name("openai::cl100k_base"));

static WC_O200K: LazyLock<WcFixture> = LazyLock::new(|| WcFixture::by_name("openai::o200k_base"));

static TT_CL100K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::cl100k_base().unwrap()),
});

static TT_O200K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::o200k_base().unwrap()),
});

static HF_CL100K: LazyLock<Arc<Tokenizer>> = LazyLock::new(|| {
    Arc::new(Tokenizer::from_pretrained("Xenova/text-embedding-ada-002", None).unwrap())
});

static HF_O200K: LazyLock<Arc<Tokenizer>> =
    LazyLock::new(|| Arc::new(Tokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap()));

mod english {
    use super::*;

    mod wordchipper {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let tokens = WC_CL100K.tokenizer.try_encode(&text).unwrap();
            bencher.counter(BytesCount::new(text.len())).bench(|| {
                WC_CL100K
                    .tokenizer
                    .try_decode_to_string(black_box(&tokens))
                    .unwrap()
            });
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let tokens = WC_O200K.tokenizer.try_encode(&text).unwrap();
            bencher.counter(BytesCount::new(text.len())).bench(|| {
                WC_O200K
                    .tokenizer
                    .try_decode_to_string(black_box(&tokens))
                    .unwrap()
            });
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let bpe = &TT_CL100K.bpe;
            let tokens = bpe.encode_with_special_tokens(&text);
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.decode(black_box(tokens.clone())).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let bpe = &TT_O200K.bpe;
            let tokens = bpe.encode_with_special_tokens(&text);
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.decode(black_box(tokens.clone())).unwrap());
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let tok = &*HF_CL100K;
            let ids = tok.encode(text.as_str(), true).unwrap();
            let token_ids = ids.get_ids();
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.decode(black_box(token_ids), false).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let tok = &*HF_O200K;
            let ids = tok.encode(text.as_str(), true).unwrap();
            let token_ids = ids.get_ids();
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.decode(black_box(token_ids), false).unwrap());
        }
    }
}

mod diverse {
    use super::*;

    mod wordchipper {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let tokens = WC_CL100K.tokenizer.try_encode(&text).unwrap();
            let decoder = &WC_CL100K.tokenizer.decoder();
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| decoder.try_decode_to_string(black_box(&tokens)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let tokens = WC_O200K.tokenizer.try_encode(&text).unwrap();
            let decoder = &WC_O200K.tokenizer.decoder();
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| decoder.try_decode_to_string(black_box(&tokens)).unwrap());
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let bpe = &TT_CL100K.bpe;
            let tokens = bpe.encode_with_special_tokens(&text);
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.decode(black_box(tokens.clone())).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let bpe = &TT_O200K.bpe;
            let tokens = bpe.encode_with_special_tokens(&text);
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.decode(black_box(tokens.clone())).unwrap());
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let tok = &*HF_CL100K;
            let ids = tok.encode(text.as_str(), true).unwrap();
            let token_ids = ids.get_ids();
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.decode(black_box(token_ids), false).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let tok = &*HF_O200K;
            let ids = tok.encode(text.as_str(), true).unwrap();
            let token_ids = ids.get_ids();
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.decode(black_box(token_ids), false).unwrap());
        }
    }
}
