use std::sync::Arc;

use tiktoken_rs::{
    CoreBPE,
    Rank,
};

use crate::engines::{
    BoxError,
    EncDecEngine,
};

/// [`EncDecEngine`] implementation for [`CoreBPE`].
pub struct TiktokenRsEngine {
    name: String,
    inner: Arc<CoreBPE>,
}

impl TiktokenRsEngine {
    pub fn new(
        name: String,
        inner: Arc<CoreBPE>,
    ) -> Self {
        let name = format!("tiktoken-rs::{name}");
        Self { name, inner }
    }
}

impl EncDecEngine<Rank> for TiktokenRsEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> Result<Vec<Vec<Rank>>, BoxError> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "parallel")] {
                use rayon::prelude::*;
                let it =batch.par_iter();
            } else {
                let it =batch.iter();
            }
        }
        Ok(it
            .map(|s| self.inner.encode_with_special_tokens(s))
            .collect::<Vec<_>>())
    }

    fn decode_batch(
        &self,
        batch: &[&[Rank]],
    ) -> Result<Vec<String>, BoxError> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "parallel")] {
                use rayon::prelude::*;
                let it =batch.par_iter();
            } else {
                let it =batch.iter();
            }
        }
        Ok(it
            .map(|tokens| self.inner.decode(tokens.to_vec()).unwrap())
            .collect::<Vec<_>>())
    }
}

/// Load a tiktoken model from the given `OATokenizer` enum variant.
pub fn load_tiktoken_bpe(model: &str) -> Result<(String, Arc<CoreBPE>), BoxError> {
    let (source, bpe) = match model {
        "openai:gpt2" => (
            "gpt2",
            tiktoken_rs::get_bpe_from_tokenizer(tiktoken_rs::tokenizer::Tokenizer::Gpt2)?,
        ),
        "openai:r50k_base" => ("r50k_base", tiktoken_rs::r50k_base()?),
        "openai:p50k_base" => ("p50k_base", tiktoken_rs::p50k_base()?),
        "openai:p50k_edit" => ("p50k_edit", tiktoken_rs::p50k_edit()?),
        "openai:cl100k_base" => ("cl100k_base", tiktoken_rs::cl100k_base()?),
        "openai:o200k_base" => ("o200k_base", tiktoken_rs::o200k_base()?),
        "openai:o200k_harmony" => ("o200k_harmony", tiktoken_rs::o200k_harmony()?),
        _ => return Err(format!("unsupported model: {:?}", model).into()),
    };
    Ok((source.to_string(), Arc::new(bpe)))
}
