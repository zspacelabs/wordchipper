use crate::{
    TokenDecoderOptions,
    TokenEncoderOptions,
    TokenType,
    Tokenizer,
    UnifiedTokenVocab,
    alloc::sync::Arc,
};

/// Options for configuring a [`Tokenizer`].
// TODO: serialize/deserialize?
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TokenizerOptions {
    /// Encoder options.
    pub encoder: TokenEncoderOptions,

    /// Decoder options.
    pub decoder: TokenDecoderOptions,
}

impl TokenizerOptions {
    /// Are accelerated lexers enabled?
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanners.
    pub fn accelerated_lexers(&self) -> bool {
        self.encoder.accelerated_lexers()
    }

    /// Set whether accelerated lexers should be enabled.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanners.
    pub fn set_accelerated_lexers(
        &mut self,
        accelerated_lexers: bool,
    ) {
        self.encoder.set_accelerated_lexers(accelerated_lexers);
    }

    /// Set whether accelerated lexers should be enabled.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanners.
    pub fn with_accelerated_lexers(
        mut self,
        accelerated_lexers: bool,
    ) -> Self {
        self.set_accelerated_lexers(accelerated_lexers);
        self
    }

    /// Gets the configured parallelism value.
    ///
    /// Returns true if either encoder or decoder are configured for
    /// parallelism.
    ///
    /// Enabling parallelism will request threaded implementations.
    pub fn parallel(&self) -> bool {
        self.encoder.parallel() || self.decoder.parallel()
    }

    /// Sets the configured parallelism value on both encoder and decoder.
    ///
    /// Enabling parallelism will request threaded implementations.
    pub fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.encoder.set_parallel(parallel);
        self.decoder.set_parallel(parallel);
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request threaded implementations.
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.set_parallel(parallel);
        self
    }

    /// Returns true if either parallel or concurrent is enabled.
    pub fn is_concurrent(&self) -> bool {
        self.concurrent() || self.parallel()
    }

    /// Gets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder optimized for
    /// concurrent thread access.
    pub fn concurrent(&self) -> bool {
        self.encoder.concurrent()
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn set_concurrent(
        &mut self,
        concurrent: bool,
    ) {
        self.encoder.set_concurrent(concurrent);
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select a encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn with_concurrent(
        mut self,
        concurrent: bool,
    ) -> Self {
        self.set_concurrent(concurrent);
        self
    }

    /// Build a [`Tokenizer`] for the given vocab.
    pub fn build<T: TokenType>(
        &self,
        vocab: Arc<UnifiedTokenVocab<T>>,
    ) -> Arc<Tokenizer<T>> {
        Tokenizer::new(
            vocab.clone(),
            self.encoder.build(vocab.clone()),
            self.decoder.build(vocab),
        )
        .into()
    }
}
