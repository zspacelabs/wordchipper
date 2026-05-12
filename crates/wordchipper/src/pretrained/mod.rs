//! # Public Vocabulary Information
//!
//! Common Stats, Public Patterns, and Pretrained Model Sources
//!
//! ## Loading Pretrained Models
//!
//! Loading a pre-trained model requires reading the vocabulary,
//! as well as configuring the spanners (regex and special words)
//! configuration.
//!
//! For a number of pretrained models, simplified constructors are
//! available to download, cache, and load the vocabulary.
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     Tokenizer,
//!     TokenizerOptions,
//!     UnifiedTokenVocab,
//!     disk_cache::WordchipperDiskCache,
//!     load_vocab,
//! };
//!
//! fn example() -> wordchipper::WCResult<Arc<Tokenizer<u32>>> {
//!     let mut disk_cache = WordchipperDiskCache::default();
//!     let loaded = load_vocab("openai:o200k_harmony", &mut disk_cache)?;
//!
//!     let tokenizer =
//!         TokenizerOptions::default().build(loaded.vocab().clone());
//!
//!     Ok(tokenizer)
//! }
//! ```

pub mod factory;
pub mod openai;

#[cfg(feature = "huggingface")]
pub mod huggingface;

#[doc(inline)]
pub use factory::{
    LabeledVocab,
    VocabDescription,
    VocabListing,
    VocabQuery,
    list_models,
    list_vocabs,
    load_vocab,
    resolve_vocab,
};
