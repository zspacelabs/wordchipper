use std::sync::Arc;

use wordchipper::{
    TokenDecoder,
    TokenEncoder,
    TokenType,
    Tokenizer,
    spanners::TextSpanner,
};

use crate::engines::{
    BoxError,
    EncDecEngine,
};

/// [`EncDecEngine`] implementation for [`TokenEncoder`] + [`TokenDecoder`].
pub struct WordchipperEngine<T: TokenType> {
    name: String,
    tokenizer: Arc<Tokenizer<T>>,
}

impl<T: TokenType> WordchipperEngine<T> {
    pub fn new(
        name: String,
        tokenizer: Arc<Tokenizer<T>>,
    ) -> Self {
        let name = format!("wordchipper::{name}");
        Self { name, tokenizer }
    }

    pub fn spanner(&self) -> &Arc<dyn TextSpanner> {
        self.tokenizer.spanner()
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer<T>> {
        &self.tokenizer
    }
}

impl<T: TokenType> EncDecEngine<T> for WordchipperEngine<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> Result<Vec<Vec<T>>, BoxError> {
        Ok(self.tokenizer.try_encode_batch(batch, None)?)
    }

    fn decode_batch(
        &self,
        batch: &[&[T]],
    ) -> Result<Vec<String>, BoxError> {
        let decoded = self.tokenizer.try_decode_batch_to_strings(batch)?;
        Ok(decoded.unwrap())
    }
}
