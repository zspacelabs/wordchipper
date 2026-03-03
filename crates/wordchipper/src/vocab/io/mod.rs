//! # Vocabulary IO
//!
//! ## Loading A Vocab
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use wordchipper::{
//!     Tokenizer,
//!     TokenizerOptions,
//!     UnifiedTokenVocab,
//!     pretrained::openai::OA_O200K_BASE_PATTERN,
//!     spanners::TextSpanningConfig,
//!     vocab::io::load_base64_unified_vocab_path,
//! };
//!
//! fn example() -> wordchipper::WCResult<Arc<Tokenizer<u32>>> {
//!     let vocab: Arc<UnifiedTokenVocab<u32>> =
//!         load_base64_unified_vocab_path(
//!             "vocab.tiktoken",
//!             TextSpanningConfig::from_pattern(OA_O200K_BASE_PATTERN),
//!         )
//!         .expect("failed to load vocab")
//!         .into();
//!
//!     let tokenizer: Arc<Tokenizer<u32>> =
//!         TokenizerOptions::default().with_parallel(true).build(vocab);
//!
//!     Ok(tokenizer)
//! }
//! ```

mod base64_vocab;

#[doc(inline)]
pub use base64_vocab::*;

#[cfg(all(feature = "std", feature = "datagym"))]
mod datagym_vocab;

#[cfg(all(feature = "std", feature = "datagym"))]
pub use datagym_vocab::*;
