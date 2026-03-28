#![allow(missing_docs)]
#![cfg(feature = "client")]

use std::sync::Arc;

use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;
use wordchipper::{
    TokenDecoder,
    TokenEncoder,
    TokenEncoderOptions,
    TokenizerOptions,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    encoders::token_span_encoder::SpanEncoderSelector,
    pretrained::openai::OATokenizer,
};

const SAMPLES: &[&str] = &[
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "It's a beautiful day, and I'll be taking my 3 dogs for a walk.",
    "Don't forget: the temperature is 72 degrees!",
    "  multiple   spaces  ",
    "line1\nline2\r\nline3",
    "123 + 456 = 789",
    "caf\u{00e9} na\u{00ef}ve \u{4f60}\u{597d}",
    "Geburtstag 2024: Alles Gute!",
    "$$$!!!...---",
    " ",
    "a",
    "\t\ttabs\tand\tspaces ",
    "emoji: \u{1f600}\u{1f680}\u{1f4a1}",
    "mixed: hello\u{00a0}world\u{2003}wide",
];

fn load_model(model: OATokenizer) -> Arc<wordchipper::Tokenizer<u32>> {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: Arc<UnifiedTokenVocab<u32>> = model.load_vocab(&mut disk_cache).unwrap().into();
    TokenizerOptions::default().build(vocab)
}

fn roundtrip_validation(model: OATokenizer) {
    let tokenizer = load_model(model);

    for text in SAMPLES {
        let tokens = tokenizer.try_encode(text, None).unwrap();
        let decoded = tokenizer.try_decode_to_string(&tokens).unwrap();
        assert_eq!(
            &decoded.value, text,
            "Roundtrip mismatch for {model:?}: {text:?}"
        );
    }
}

fn tiktoken_validation(
    model: OATokenizer,
    tiktoken_bpe: &CoreBPE,
) {
    let tokenizer = load_model(model);

    for text in SAMPLES {
        let wc_tokens = tokenizer.try_encode(text, None).unwrap();
        let tt_tokens: Vec<u32> = tiktoken_bpe
            .encode_with_special_tokens(text)
            .into_iter()
            .map(|t| t as u32)
            .collect();

        assert_eq!(
            wc_tokens, tt_tokens,
            "Encode mismatch (wordchipper vs tiktoken) for {model:?}: {text:?}"
        );
    }
}

fn tokenizers_validation(
    model: OATokenizer,
    hf_tok: &Tokenizer,
) {
    let tokenizer = load_model(model);

    for text in SAMPLES {
        let wc_tokens = tokenizer.try_encode(text, None).unwrap();
        let hf_encoding = hf_tok.encode(*text, true).unwrap();
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();

        assert_eq!(
            wc_tokens, hf_tokens,
            "Encode mismatch (wordchipper vs tokenizers) for {model:?}: {text:?}"
        );
    }
}

#[test]
#[ignore]
fn cl100k_roundtrip() {
    roundtrip_validation(OATokenizer::Cl100kBase);
}

#[test]
#[ignore]
fn o200k_roundtrip() {
    roundtrip_validation(OATokenizer::O200kBase);
}

#[test]
#[ignore]
fn cl100k_vs_tiktoken() {
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    tiktoken_validation(OATokenizer::Cl100kBase, &bpe);
}

#[test]
#[ignore]
fn o200k_vs_tiktoken() {
    let bpe = tiktoken_rs::o200k_base().unwrap();
    tiktoken_validation(OATokenizer::O200kBase, &bpe);
}

#[test]
#[ignore]
fn cl100k_vs_tokenizers() {
    let tok = Tokenizer::from_pretrained("Xenova/text-embedding-ada-002", None).unwrap();
    tokenizers_validation(OATokenizer::Cl100kBase, &tok);
}

#[test]
#[ignore]
fn o200k_vs_tokenizers() {
    let tok = Tokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap();
    tokenizers_validation(OATokenizer::O200kBase, &tok);
}

fn load_vocab(model: OATokenizer) -> Arc<UnifiedTokenVocab<u32>> {
    let mut disk_cache = WordchipperDiskCache::default();
    model.load_vocab(&mut disk_cache).unwrap().into()
}

fn span_encoder_vs_bpe(
    model: OATokenizer,
    selector: SpanEncoderSelector,
) {
    let vocab = load_vocab(model);

    let bpe_encoder = TokenEncoderOptions::default().build(vocab.clone());
    let alt_encoder = TokenEncoderOptions::default()
        .with_span_encoder(selector)
        .build(vocab);

    for text in SAMPLES {
        let bpe_tokens = bpe_encoder.try_encode(text, None).unwrap();
        let alt_tokens = alt_encoder.try_encode(text, None).unwrap();
        assert_eq!(
            bpe_tokens, alt_tokens,
            "{selector:?} mismatch for {model:?}: {text:?}",
        );
    }
}

#[test]
#[ignore]
fn cl100k_bpe_backtrack_vs_bpe() {
    span_encoder_vs_bpe(OATokenizer::Cl100kBase, SpanEncoderSelector::BpeBacktrack);
}

#[test]
#[ignore]
fn o200k_bpe_backtrack_vs_bpe() {
    span_encoder_vs_bpe(OATokenizer::O200kBase, SpanEncoderSelector::BpeBacktrack);
}
