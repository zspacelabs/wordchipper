//! Text Spanner Builder

use core::num::NonZeroUsize;

use crate::{
    TokenType,
    UnifiedTokenVocab,
    alloc::sync::Arc,
    spanners::{
        TextSpanner,
        TextSpanningConfig,
        span_lexers::{
            LexerTextSpanner,
            SpanLexer,
            build_regex_lexer,
        },
    },
};

/// Builder for [`TextSpanner`]s.
///
/// The primary tuning knobs here are:
/// * [`set_concurrent`](Self::set_concurrent) - whether to request a
///   concurrency support.
#[derive(Clone, PartialEq)]
pub struct TextSpannerBuilder<T: TokenType> {
    config: TextSpanningConfig<T>,

    accelerated_lexers: bool,
    concurrent: bool,
    max_pool: Option<NonZeroUsize>,
}

impl<T: TokenType> TextSpannerBuilder<T> {
    /// Build a new `Arc<dyn TextSpanner>` with defaults.
    pub fn default(vocab: &UnifiedTokenVocab<T>) -> Arc<dyn TextSpanner> {
        Self::from_vocab(vocab).build()
    }

    /// Create a new [`TextSpannerBuilder`].
    ///
    /// Clones out the spanners configuration from the provided vocabulary.
    pub fn from_vocab(vocab: &UnifiedTokenVocab<T>) -> Self {
        Self::new(vocab.spanning().clone())
    }

    /// Create a new [`TextSpannerBuilder`] with the given configuration.
    pub fn new(config: TextSpanningConfig<T>) -> Self {
        Self {
            config,
            accelerated_lexers: true,
            concurrent: true,
            max_pool: None,
        }
    }

    /// Get the underlying [`TextSpanningConfig`].
    pub fn config(&self) -> &TextSpanningConfig<T> {
        &self.config
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

    /// Get whether the decoder should use parallel decoding.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    pub fn concurrent(&self) -> bool {
        self.concurrent
    }

    /// Set whether the decoder should use parallel decoding.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    pub fn set_concurrent(
        &mut self,
        concurrent: bool,
    ) {
        self.concurrent = concurrent;
    }

    /// Set whether the decoder should use parallel decoding.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    pub fn with_concurrent(
        mut self,
        concurrent: bool,
    ) -> Self {
        self.set_concurrent(concurrent);
        self
    }

    /// Get the max pool size for the [`TextSpanner`].
    pub fn max_pool(&self) -> Option<NonZeroUsize> {
        self.max_pool
    }

    /// Set the max pool size for the [`TextSpanner`].
    pub fn set_max_pool(
        &mut self,
        max_pool: NonZeroUsize,
    ) {
        self.max_pool = Some(max_pool);
    }

    /// Set the max pool size for the [`TextSpanner`].
    pub fn with_max_pool(
        mut self,
        max_pool: NonZeroUsize,
    ) -> Self {
        self.set_max_pool(max_pool);
        self
    }

    /// Build a [`TextSpanner`] with the current configuration.
    ///
    /// Automatically selects the fastest available word lexer for the
    /// configured pattern (e.g. a logos DFA accelerator if the `logos`
    /// feature is enabled and the pattern is recognized).
    /// Falls back to the compiled regex otherwise.
    /// The special lexer (if any) is always built from the regex pattern.
    pub fn build(&self) -> Arc<dyn TextSpanner> {
        let word_lexer: Arc<dyn SpanLexer> = build_regex_lexer(
            self.config().pattern().clone(),
            self.accelerated_lexers,
            self.concurrent,
            self.max_pool,
        );
        let special_lexer: Option<Arc<dyn SpanLexer>> = self
            .config
            .specials()
            .special_pattern()
            .map(|pattern| build_regex_lexer(pattern, false, self.concurrent, self.max_pool));

        Arc::new(LexerTextSpanner::new(word_lexer, special_lexer))
    }
}
