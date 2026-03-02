use std::sync::Arc;

use wordchipper::{Tokenizer, UnifiedTokenVocab, disk_cache::WordchipperDiskCache};

/// Model selector arg group.
#[derive(clap::Args, Debug)]
#[group(required = true, multiple = false)]
pub struct ModelSelectorArgs {
    /// Model to use for encoding.
    #[arg(long, default_value = "openai::r50k_base")]
    model: String,
}

impl ModelSelectorArgs {
    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Load the vocabulary.
    pub fn load_vocab(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> Result<Arc<UnifiedTokenVocab<u32>>, Box<dyn std::error::Error>> {
        let (_desc, vocab) = wordchipper::load_vocab(self.model(), disk_cache)?;
        Ok(vocab)
    }

    /// Load the tokenizer.
    pub fn load_tokenizer(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> Result<Arc<Tokenizer<u32>>, Box<dyn std::error::Error>> {
        let vocab = self.load_vocab(disk_cache)?;
        let tokenizer = wordchipper::TokenizerOptions::default().build(vocab);
        Ok(tokenizer)
    }
}
