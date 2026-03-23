//! Token Encoder Options
//!
//! Options for building a [`TokenEncoder`].

use crate::{
    TokenEncoder,
    TokenType,
    UnifiedTokenVocab,
    alloc::sync::Arc,
    encoders::token_span_encoder::{
        SpanEncoderSelector,
        TokenSpanEncoder,
    },
    spanners::TextSpannerBuilder,
};

/// Options for configuring a [`TokenEncoder`].
// TODO: serialize/deserialize?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenEncoderOptions {
    /// The [`SpanEncoderSelector`] to use.
    ///
    /// When `None`, an appropriate default will be used for the concurrency.
    pub span_encoder: Option<SpanEncoderSelector>,

    /// Whether to use accelerated lexers, when available.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanners.
    pub accelerated_lexers: bool,

    /// Should the encoder be threaded?
    pub parallel: bool,

    /// Should the encoder be concurrent?
    ///
    /// Concurrent encoders select defaults to be called concurrently.
    pub concurrent: bool,
}

impl Default for TokenEncoderOptions {
    fn default() -> Self {
        Self {
            span_encoder: None,
            accelerated_lexers: true,
            parallel: false,
            concurrent: false,
        }
    }
}

impl TokenEncoderOptions {
    /// Gets the effective span encoder selector.
    ///
    /// Will return any explict setting,
    /// otherwise will select based upon parallel and concurrency settings.
    pub fn effective_span_encoder(&self) -> SpanEncoderSelector {
        match self.span_encoder() {
            Some(selector) => selector,
            None if self.is_concurrent() => SpanEncoderSelector::ConcurrentDefault,
            _ => SpanEncoderSelector::SingleThreadDefault,
        }
    }

    /// Get the configured [`SpanEncoderSelector`].
    pub fn span_encoder(&self) -> Option<SpanEncoderSelector> {
        self.span_encoder
    }

    /// Set the configured [`SpanEncoderSelector`].
    pub fn set_span_encoder<E>(
        &mut self,
        span_encoder: E,
    ) where
        E: Into<Option<SpanEncoderSelector>>,
    {
        self.span_encoder = span_encoder.into();
    }

    /// Set the configured [`SpanEncoderSelector`] and return the builder.
    pub fn with_span_encoder<E>(
        mut self,
        span_encoder: E,
    ) -> Self
    where
        E: Into<Option<SpanEncoderSelector>>,
    {
        self.set_span_encoder(span_encoder);
        self
    }

    /// Are accelerated lexers enabled?
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanners.
    pub fn accelerated_lexers(&self) -> bool {
        self.accelerated_lexers
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
        self.accelerated_lexers = accelerated_lexers;
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
    /// Enabling parallelism will request threaded implementations.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request threaded implementations.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.parallel = parallel;
    }

    /// Sets the configured parallelism value.
    ///
    /// Enabling parallelism will request threaded implementations.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn with_parallel(
        mut self,
        parallel: bool,
    ) -> Self {
        self.set_parallel(parallel);
        self
    }

    /// Returns true if either parallel or concurrent is enabled.
    pub fn is_concurrent(&self) -> bool {
        self.concurrent || self.parallel
    }

    /// Gets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder optimized for
    /// concurrent thread access.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn concurrent(&self) -> bool {
        self.concurrent
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder optimized for
    /// concurrent thread access.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn set_concurrent(
        &mut self,
        concurrent: bool,
    ) {
        self.concurrent = concurrent;
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder optimized for
    /// concurrent thread access.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    pub fn with_concurrent(
        mut self,
        concurrent: bool,
    ) -> Self {
        self.set_concurrent(concurrent);
        self
    }

    /// Build a [`TokenEncoder`] for the given vocab.
    pub fn build<T: TokenType>(
        &self,
        vocab: Arc<UnifiedTokenVocab<T>>,
    ) -> Arc<dyn TokenEncoder<T>> {
        let spanner = TextSpannerBuilder::new(vocab.spanning().clone())
            .with_accelerated_lexers(self.accelerated_lexers())
            .with_concurrent(self.is_concurrent())
            .build();

        #[allow(unused_mut)]
        let mut enc: Arc<dyn TokenEncoder<T>> = Arc::new(TokenSpanEncoder::<T>::new_with_selector(
            spanner,
            vocab,
            self.effective_span_encoder(),
        ));

        #[cfg(feature = "parallel")]
        if self.parallel() {
            enc = Arc::new(crate::support::concurrency::rayon::ParallelRayonEncoder::new(enc));
        }

        enc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_span_encoder() {
        let options = TokenEncoderOptions::default();
        assert_eq!(options.span_encoder(), None);
        assert_eq!(options.accelerated_lexers(), true);
        assert_eq!(options.parallel(), false);
        assert_eq!(options.concurrent(), false);

        assert_eq!(
            options.effective_span_encoder(),
            SpanEncoderSelector::SingleThreadDefault
        );

        let options = options.with_parallel(true);

        assert_eq!(
            options.effective_span_encoder(),
            SpanEncoderSelector::ConcurrentDefault,
        );

        let options = options.with_parallel(false).with_concurrent(true);

        assert_eq!(
            options.effective_span_encoder(),
            SpanEncoderSelector::ConcurrentDefault,
        );

        let options = options.with_span_encoder(SpanEncoderSelector::SingleThreadDefault);

        assert_eq!(
            options.effective_span_encoder(),
            SpanEncoderSelector::SingleThreadDefault,
        );
    }

    #[test]
    fn test_encoder_options() {
        let options = TokenEncoderOptions::default();
        assert_eq!(options.span_encoder(), None);
        assert_eq!(options.accelerated_lexers(), true);
        assert_eq!(options.parallel(), false);
        assert_eq!(options.concurrent(), false);

        assert_eq!(
            options.is_concurrent(),
            options.parallel() || options.concurrent()
        );

        let options = options
            .with_span_encoder(SpanEncoderSelector::SingleThreadDefault)
            .with_accelerated_lexers(false)
            .with_parallel(true)
            .with_concurrent(true);
        assert_eq!(
            options.span_encoder(),
            Some(SpanEncoderSelector::SingleThreadDefault)
        );
        assert_eq!(options.accelerated_lexers(), false);
        assert_eq!(options.parallel(), true);
        assert_eq!(options.concurrent(), true);

        assert_eq!(
            options.is_concurrent(),
            options.parallel() || options.concurrent()
        );
    }
}
