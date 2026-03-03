use std::sync::Arc;

use tokenizers::{
    Encoding,
    tokenizer::Tokenizer,
};

use crate::engines::{
    BoxError,
    EncDecEngine,
};

/// [`EncDecEngine`] implementation for [`Tokenizer`].
pub struct TokenizersEngine {
    name: String,
    inner: Arc<Tokenizer>,
    use_batch: bool,
}

impl TokenizersEngine {
    pub fn new(
        name: String,
        inner: Arc<Tokenizer>,
        use_batch: bool,
    ) -> Self {
        let mode = if use_batch {
            "batch_encode"
        } else {
            "par_iter"
        };

        let name = format!("tokenizers({mode})::{name}");
        Self {
            name,
            inner,
            use_batch,
        }
    }
}

impl EncDecEngine<u32> for TokenizersEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> Result<Vec<Vec<u32>>, BoxError> {
        if self.use_batch {
            let batch = batch.to_vec();
            let br = self.inner.encode_batch(batch, true).unwrap();
            Ok(br
                .iter()
                .map(|e: &Encoding| e.get_ids().to_vec())
                .collect::<Vec<_>>())
        } else {
            use rayon::prelude::*;
            Ok(batch
                .par_iter()
                .map(|s| self.inner.encode(*s, true).unwrap().get_ids().to_vec())
                .collect::<Vec<_>>())
        }
    }

    fn decode_batch(
        &self,
        batch: &[&[u32]],
    ) -> Result<Vec<String>, BoxError> {
        match self.inner.decode_batch(batch, false) {
            Ok(res) => Ok(res),
            Err(e) => Err(format!("failed to decode batch with \"{}\": {}", self.name(), e).into()),
        }
    }
}

pub fn load_tokenizers_tok(model: &str) -> Result<(String, Arc<Tokenizer>), BoxError> {
    let source = match model {
        "openai:gpt2" => "Xenova/gpt2",
        "openai:r50k_base" => "Xenova/gpt-3",
        "openai:p50k_base" | "openai:p50k_edit" => "Xenova/text-davinci-002",
        "openai:cl100k_base" => "Xenova/text-embedding-ada-002",
        "openai:o200k_base" | "openai:o200k_harmony" => "Xenova/gpt-4o",
        _ => return Err(format!("unsupported model: {:?}", model).into()),
    };

    let tok = Tokenizer::from_pretrained(source, None).map_err(|e| -> BoxError {
        format!("failed to load tokenizer from {}: {}", source, e).into()
    })?;

    Ok((source.to_string(), Arc::new(tok)))
}
