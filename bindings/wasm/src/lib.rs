#![no_std]
#![warn(missing_docs)]
//! WebAssembly bindings for the wordchipper tokenizer library.

extern crate alloc;

use alloc::{
    format,
    string::String,
    sync::Arc,
    vec::Vec,
};

use base64::{
    Engine,
    prelude::BASE64_STANDARD,
};
use js_sys::{
    Array,
    JsString,
    Uint32Array,
};
use wasm_bindgen::prelude::*;
use wordchipper::{
    TokenDecoder,
    TokenEncoder,
    Tokenizer as WCTokenizer,
    TokenizerOptions,
    UnifiedTokenVocab,
    VocabIndex,
    pretrained::openai::OATokenizer,
    support::{
        slices::{
            inner_slice_view,
            inner_str_view,
        },
        strings::string_from_utf8_lossy,
    },
    vocab::{
        SpanMapVocab,
        SpanTokenMap,
    },
};

/// Parse tiktoken base64-encoded vocab data from raw bytes.
fn parse_tiktoken_data(data: &[u8]) -> Result<SpanTokenMap<u32>, JsError> {
    let mut map = SpanTokenMap::default();
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let sep = line
            .iter()
            .position(|&b| b == b' ')
            .ok_or_else(|| JsError::new("invalid tiktoken line: no space separator"))?;
        let span = BASE64_STANDARD
            .decode(&line[..sep])
            .map_err(|e| JsError::new(&format!("base64 decode error: {e}")))?;
        let id_str = core::str::from_utf8(&line[sep + 1..])
            .map_err(|e| JsError::new(&format!("invalid utf8 in token id: {e}")))?;
        let id_str = id_str.trim_ascii_end();
        let id: u32 = id_str
            .parse()
            .map_err(|e| JsError::new(&format!("invalid token id '{id_str}': {e}")))?;
        map.insert(span, id);
    }
    Ok(map)
}

/// Resolve a model name string to an `OATokenizer` variant.
fn resolve_model(name: &str) -> Result<OATokenizer, JsError> {
    name.parse::<OATokenizer>().map_err(|_| {
        JsError::new(&format!(
            "unknown model '{name}'. Use Tokenizer.availableModels() to list valid names."
        ))
    })
}

/// A tokenizer for encoding text to tokens and decoding tokens to text.
#[wasm_bindgen]
pub struct Tokenizer {
    inner: Arc<WCTokenizer<u32>>,
}

#[wasm_bindgen]
impl Tokenizer {
    /// Create a tokenizer from a model name and raw tiktoken vocab data.
    ///
    /// The model name determines the regex pattern and special tokens.
    /// The data should be the raw bytes of a `.tiktoken` file.
    #[wasm_bindgen(js_name = "fromVocabData")]
    pub fn from_vocab_data(
        model: &str,
        data: &[u8],
    ) -> Result<Tokenizer, JsError> {
        let oa = resolve_model(model)?;
        let span_map = parse_tiktoken_data(data)?;
        let spanning = oa.spanning_config::<u32>();
        let vocab =
            UnifiedTokenVocab::from_span_vocab(spanning, SpanMapVocab::from_span_map(span_map))
                .map_err(|e| JsError::new(&format!("failed to build vocab: {e}")))?;

        let inner = TokenizerOptions::default().build(Arc::new(vocab));
        Ok(Tokenizer { inner })
    }

    /// Encode a string into token IDs.
    pub fn encode(
        &self,
        text: &str,
    ) -> Result<Vec<u32>, JsError> {
        self.inner
            .try_encode(text)
            .map_err(|e| JsError::new(&format!("encode error: {e}")))
    }

    /// Decode token IDs back into a string.
    pub fn decode(
        &self,
        tokens: &[u32],
    ) -> Result<String, JsError> {
        self.inner
            .try_decode_to_string(tokens)
            .and_then(|r| r.try_result())
            .map_err(|e| JsError::new(&format!("decode error: {e}")))
    }

    /// Encode multiple strings into arrays of token IDs.
    #[wasm_bindgen(js_name = "encodeBatch")]
    pub fn encode_batch(
        &self,
        texts: Vec<JsString>,
    ) -> Result<Array, JsError> {
        let strings: Vec<String> = texts.iter().map(|s| s.into()).collect();
        let refs = inner_str_view(&strings);
        let results = self
            .inner
            .try_encode_batch(&refs)
            .map_err(|e| JsError::new(&format!("encode_batch error: {e}")))?;
        let arr = Array::new();
        for tokens in results {
            let u32_arr = Uint32Array::new_with_length(tokens.len() as u32);
            u32_arr.copy_from(&tokens);
            arr.push(&u32_arr);
        }
        Ok(arr)
    }

    /// Decode multiple arrays of token IDs back into strings.
    #[wasm_bindgen(js_name = "decodeBatch")]
    pub fn decode_batch(
        &self,
        batch: Vec<Uint32Array>,
    ) -> Result<Array, JsError> {
        let vecs: Vec<Vec<u32>> = batch.iter().map(|arr| arr.to_vec()).collect();
        let refs = inner_slice_view(&vecs);
        let results = self
            .inner
            .try_decode_batch_to_strings(&refs)
            .and_then(|r| r.try_results())
            .map_err(|e| JsError::new(&format!("decode_batch error: {e}")))?;
        let arr = Array::new();
        for s in results {
            arr.push(&JsValue::from_str(&s));
        }
        Ok(arr)
    }

    /// Get the vocabulary size.
    #[wasm_bindgen(getter, js_name = "vocabSize")]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }

    /// Get the maximum token ID, or null if the vocabulary is empty.
    #[wasm_bindgen(getter, js_name = "maxToken")]
    pub fn max_token(&self) -> JsValue {
        match self.inner.vocab().max_token() {
            Some(t) => JsValue::from(t),
            None => JsValue::NULL,
        }
    }

    /// Look up the token ID for a given token string. Returns null if not
    /// found.
    #[wasm_bindgen(js_name = "tokenToId")]
    pub fn token_to_id(
        &self,
        token: &str,
    ) -> JsValue {
        match self.inner.vocab().lookup_token(token.as_bytes()) {
            Some(id) => JsValue::from(id),
            None => JsValue::NULL,
        }
    }

    /// Look up the token string for a given token ID. Returns null if not
    /// found.
    #[wasm_bindgen(js_name = "idToToken")]
    pub fn id_to_token(
        &self,
        id: u32,
    ) -> JsValue {
        match self.inner.vocab().unified_dictionary().get(&id) {
            Some(bytes) => JsValue::from_str(&string_from_utf8_lossy(bytes.clone())),
            None => JsValue::NULL,
        }
    }

    /// Get all special tokens as an array of [name, id] pairs.
    #[wasm_bindgen(js_name = "getSpecialTokens")]
    pub fn get_special_tokens(&self) -> Array {
        let arr = Array::new();
        for (bytes, &token) in self.inner.vocab().special_vocab().span_map().iter() {
            let pair = Array::new();
            pair.push(&JsValue::from_str(&string_from_utf8_lossy(bytes.to_vec())));
            pair.push(&JsValue::from(token));
            arr.push(&pair);
        }
        arr
    }

    /// List all available model names.
    #[wasm_bindgen(js_name = "availableModels")]
    pub fn available_models() -> Array {
        let arr = Array::new();
        let models = [
            "r50k_base",
            "p50k_base",
            "p50k_edit",
            "cl100k_base",
            "o200k_base",
            "o200k_harmony",
        ];
        for name in models {
            arr.push(&JsValue::from_str(name));
        }
        arr
    }
}
