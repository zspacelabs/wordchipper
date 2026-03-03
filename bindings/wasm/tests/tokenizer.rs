//! WASM binding tests using `wasm-bindgen-test`.

use base64::{
    Engine,
    prelude::BASE64_STANDARD,
};
use js_sys::{
    Array,
    JsString,
    Uint32Array,
};
use wasm_bindgen_test::*;
use wordchipper_wasm::Tokenizer;

/// Generate minimal tiktoken-format vocab data with 256 single-byte tokens.
/// Each byte `i` is base64-encoded and mapped to token ID `i`.
fn make_byte_vocab() -> Vec<u8> {
    let mut data = Vec::new();
    for i in 0u32..256 {
        let encoded = BASE64_STANDARD.encode([i as u8]);
        data.extend_from_slice(encoded.as_bytes());
        data.push(b' ');
        data.extend_from_slice(i.to_string().as_bytes());
        data.push(b'\n');
    }
    data
}

fn make_tokenizer() -> Tokenizer {
    Tokenizer::from_vocab_data("cl100k_base", &make_byte_vocab()).unwrap()
}

// --- Construction ---

#[wasm_bindgen_test]
fn from_vocab_data_succeeds() {
    let result = Tokenizer::from_vocab_data("cl100k_base", &make_byte_vocab());
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn from_vocab_data_all_models() {
    let data = make_byte_vocab();
    for model in [
        "r50k_base",
        "p50k_base",
        "p50k_edit",
        "cl100k_base",
        "o200k_base",
        "o200k_harmony",
    ] {
        let result = Tokenizer::from_vocab_data(model, &data);
        assert!(result.is_ok(), "failed for model {model}");
    }
}

// --- Error cases (parse_tiktoken_data + resolve_model) ---

#[wasm_bindgen_test]
fn invalid_model_name() {
    let result = Tokenizer::from_vocab_data("nonexistent", &make_byte_vocab());
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn malformed_data_no_separator() {
    let result = Tokenizer::from_vocab_data("cl100k_base", b"noseparator\n");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn malformed_data_invalid_base64() {
    let result = Tokenizer::from_vocab_data("cl100k_base", b"!!!bad!!! 0\n");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn malformed_data_invalid_token_id() {
    let result = Tokenizer::from_vocab_data("cl100k_base", b"AA== xyz\n");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn empty_data_succeeds() {
    let result = Tokenizer::from_vocab_data("cl100k_base", b"");
    // Empty vocab is valid (only special tokens remain)
    assert!(result.is_ok());
}

// --- Encode / Decode roundtrip ---

#[wasm_bindgen_test]
fn encode_decode_ascii() {
    let tok = make_tokenizer();
    let text = "hello world";
    let tokens = tok.encode(text).unwrap();
    assert!(!tokens.is_empty());
    assert_eq!(tok.decode(&tokens).unwrap(), text);
}

#[wasm_bindgen_test]
fn encode_empty_string() {
    let tok = make_tokenizer();
    let tokens = tok.encode("").unwrap();
    assert!(tokens.is_empty());
    assert_eq!(tok.decode(&tokens).unwrap(), "");
}

#[wasm_bindgen_test]
fn encode_decode_unicode() {
    let tok = make_tokenizer();
    // Combining accent: "cafe" + U+0301
    let text = "caf\u{00e9}";
    let tokens = tok.encode(text).unwrap();
    assert_eq!(tok.decode(&tokens).unwrap(), text);
}

#[wasm_bindgen_test]
fn encode_decode_whitespace_variants() {
    let tok = make_tokenizer();
    for text in [" ", "  ", "\t", "\n", "\r\n", "a b\tc\nd"] {
        let tokens = tok.encode(text).unwrap();
        assert_eq!(tok.decode(&tokens).unwrap(), text, "failed for {text:?}");
    }
}

// --- Batch operations ---

#[wasm_bindgen_test]
fn encode_batch_decode_batch() {
    let tok = make_tokenizer();
    let texts = vec![JsString::from("hello"), JsString::from("world")];
    let encoded: Array = tok.encode_batch(texts).unwrap();
    assert_eq!(encoded.length(), 2);

    let batch: Vec<Uint32Array> = (0..encoded.length())
        .map(|i| Uint32Array::from(encoded.get(i)))
        .collect();
    let decoded: Array = tok.decode_batch(batch).unwrap();
    assert_eq!(decoded.length(), 2);
    assert_eq!(decoded.get(0).as_string().unwrap(), "hello");
    assert_eq!(decoded.get(1).as_string().unwrap(), "world");
}

#[wasm_bindgen_test]
fn encode_batch_empty() {
    let tok = make_tokenizer();
    let encoded = tok.encode_batch(vec![]).unwrap();
    assert_eq!(encoded.length(), 0);
}

// --- Vocab inspection ---

#[wasm_bindgen_test]
fn vocab_size_includes_byte_tokens() {
    let tok = make_tokenizer();
    // 256 byte tokens + cl100k special tokens
    assert!(tok.vocab_size() >= 256);
}

#[wasm_bindgen_test]
fn max_token_present() {
    let tok = make_tokenizer();
    let max = tok.max_token();
    assert!(!max.is_null());
    // At minimum 255 (highest byte token), likely higher due to special tokens
    assert!(max.as_f64().unwrap() >= 255.0);
}

#[wasm_bindgen_test]
fn token_to_id_found() {
    let tok = make_tokenizer();
    // Single byte "a" (0x61 = 97) maps to token ID 97 in our test vocab
    let id = tok.token_to_id("a");
    assert_eq!(id.as_f64().unwrap() as u32, 97);
}

#[wasm_bindgen_test]
fn token_to_id_not_found() {
    let tok = make_tokenizer();
    // Multi-byte string not in our byte-only vocab
    assert!(tok.token_to_id("hello").is_null());
}

#[wasm_bindgen_test]
fn id_to_token_found() {
    let tok = make_tokenizer();
    assert_eq!(tok.id_to_token(97).as_string().unwrap(), "a");
}

#[wasm_bindgen_test]
fn id_to_token_not_found() {
    let tok = make_tokenizer();
    assert!(tok.id_to_token(999_999).is_null());
}

// --- Special tokens ---

#[wasm_bindgen_test]
fn get_special_tokens_not_empty() {
    let tok = make_tokenizer();
    let specials = tok.get_special_tokens();
    // cl100k_base defines special tokens (e.g. <|endoftext|>)
    assert!(specials.length() > 0);

    // Each entry is a [name, id] pair
    let first = Array::from(&specials.get(0));
    assert_eq!(first.length(), 2);
    assert!(first.get(0).is_string());
    assert!(first.get(1).as_f64().is_some());
}

// --- Available models ---

#[wasm_bindgen_test]
fn available_models_lists_all() {
    let models = Tokenizer::available_models();
    let names: Vec<String> = (0..models.length())
        .map(|i| models.get(i).as_string().unwrap())
        .collect();
    for expected in [
        "r50k_base",
        "p50k_base",
        "p50k_edit",
        "cl100k_base",
        "o200k_base",
        "o200k_harmony",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "missing model: {expected}"
        );
    }
}
