pub mod divan_parser;

use std::sync::{
    Arc,
    Mutex,
    OnceLock,
};

use wordchipper::{
    TokenEncoder,
    TokenEncoderOptions,
    TokenType,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    load_vocab,
};

/// `OpenAI` "`r50k_base`" vocab.
pub const OA_R50K_BASE: &str = "openai:r50k_base";
/// `OpenAI` "`cl100k_base`" vocab.
pub const OA_CL100K_BASE: &str = "openai:cl100k_base";
/// `OpenAI` "`o200k_base`" vocab.
pub const OA_O200K_BASE: &str = "openai:o200k_base";

/// The huggingface/tokenizers model to use for `cl100k_base`.
pub const HF_CL100K: &str = "Xenova/text-embedding-ada-002";

/// The huggingface/tokenizers model to use for `o200k_base`.
pub const HF_O200K: &str = "Xenova/gpt-4o";

/// The shared disk cache for benchmarks.
static DISK_CACHE: OnceLock<Mutex<WordchipperDiskCache>> = OnceLock::new();

fn get_disk_cache() -> &'static Mutex<WordchipperDiskCache> {
    DISK_CACHE.get_or_init(|| Mutex::new(WordchipperDiskCache::default()))
}

/// Builds an `Arc<Tokenizer<T>>` for the target model and options.
///
/// Use the default disk cache to load the vocab.
pub fn load_encoder<T: TokenType>(
    model: &str,
    options: TokenEncoderOptions,
) -> Arc<dyn TokenEncoder<T>> {
    let mut guard = get_disk_cache().lock().unwrap();
    let disk_cache = &mut *guard;

    let vocab: Arc<UnifiedTokenVocab<T>> = load_vocab(model, disk_cache)
        .unwrap()
        .vocab()
        .to_token_type::<T>()
        .unwrap()
        .into();

    options.build(vocab)
}
