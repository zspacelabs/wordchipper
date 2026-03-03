use pyo3::{pyclass, pymethods};

use crate::wc;

/// Options for configuring a [`Tokenizer`](`super::Tokenizer`).
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerOptions {
    inner: wc::TokenizerOptions,
}

impl Default for TokenizerOptions {
    fn default() -> Self {
        Self {
            inner: wc::TokenizerOptions::default().with_parallel(true),
        }
    }
}

impl TokenizerOptions {
    /// Get the inner wordchipper [`wc::TokenizerOptions`].
    pub fn inner(&self) -> &wc::TokenizerOptions {
        &self.inner
    }
}

#[pymethods]
impl TokenizerOptions {
    /// Create a new default [`TokenizerOptions`].
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: wc::TokenizerOptions::default(),
        }
    }

    /// Are accelerated lexers enabled?
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanners.
    fn accelerated_lexers(&self) -> bool {
        self.inner.accelerated_lexers()
    }

    /// Set whether accelerated lexers should be enabled.
    ///
    /// When enabled, and an accelerated lexer can be
    /// found for a given regex pattern; the regex accelerator
    /// will be used for spanners.
    fn set_accelerated_lexers(
        &mut self,
        accelerated_lexers: bool,
    ) {
        self.inner.set_accelerated_lexers(accelerated_lexers);
    }

    /// Gets the configured parallelism value.
    ///
    /// Returns true if either encoder or decoder are configured for parallelism.
    ///
    /// Enabling parallelism will request threaded implementations.
    fn parallel(&self) -> bool {
        self.inner.parallel()
    }

    /// Sets the configured parallelism value on both encoder and decoder.
    ///
    /// Enabling parallelism will request threaded implementations.
    fn set_parallel(
        &mut self,
        parallel: bool,
    ) {
        self.inner.set_parallel(parallel);
    }

    /// Returns true if either parallel or concurrent is enabled.
    fn is_concurrent(&self) -> bool {
        self.inner.is_concurrent()
    }

    /// Gets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder optimized for
    /// concurrent thread access.
    fn concurrent(&self) -> bool {
        self.inner.concurrent()
    }

    /// Sets the configured concurrent value.
    ///
    /// Enabling concurrency will select an encoder which plays
    /// well when used from multiple threads.
    ///
    /// See: [`is_concurrent`](Self::is_concurrent)
    fn set_concurrent(
        &mut self,
        concurrent: bool,
    ) {
        self.inner.set_concurrent(concurrent);
    }
}
