use crate::{
    TokenEncoder,
    TokenType,
    UnifiedTokenVocab,
    WCResult,
    alloc::{
        boxed::Box,
        sync::Arc,
        vec::Vec,
    },
    encoders::token_span_encoder::{
        SpanEncoder,
        SpanEncoderSelector,
    },
    prelude::*,
    spanners::{
        SpanRef,
        TextSpanner,
    },
    vocab::SpecialVocab,
};

/// A [`TokenEncoder`] that composes a [`TextSpanner`] with a [`SpanEncoder`].
pub struct TokenSpanEncoder<T>
where
    T: TokenType,
{
    /// The reference vocabulary.
    vocab: Arc<UnifiedTokenVocab<T>>,

    /// Text Spanner.
    spanner: Arc<dyn TextSpanner>,

    #[cfg(feature = "concurrent")]
    se_pool: crate::support::concurrency::PoolToy<std::sync::Mutex<Box<dyn SpanEncoder<T>>>>,

    #[cfg(not(feature = "concurrent"))]
    se_builder: Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync>,
}

impl<T: TokenType> TokenSpanEncoder<T> {
    /// Create a new encoder using the selected [`SpanEncoder`].
    pub fn new_with_selector(
        spanner: Arc<dyn TextSpanner>,
        vocab: Arc<UnifiedTokenVocab<T>>,
        selector: SpanEncoderSelector,
    ) -> Self {
        Self::new_with_builder(
            spanner,
            vocab.clone(),
            selector.span_encoder_builder(&vocab),
        )
    }

    /// Create a new encoder.
    pub fn new_with_builder(
        spanner: Arc<dyn TextSpanner>,
        vocab: Arc<UnifiedTokenVocab<T>>,
        se_builder: Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync>,
    ) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(feature = "concurrent")] {
                use crate::support::concurrency::{PoolToy, threads::resolve_max_pool};

                let pool_size = resolve_max_pool(None);
                let pool: Vec<std::sync::Mutex<Box<dyn SpanEncoder<T>>>> =
                    (0..pool_size).map(|_| std::sync::Mutex::new(se_builder())).collect();

                Self {
                    vocab,
                    spanner,
                    se_pool: PoolToy::from_pool(pool),
                }
            } else {
                Self {
                    vocab,
                    spanner,
                    se_builder,
                }
            }
        }
    }
}

impl<T: TokenType> TokenEncoder<T> for TokenSpanEncoder<T> {
    fn spanner(&self) -> &Arc<dyn TextSpanner> {
        &self.spanner
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.vocab.spanning().specials()
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, text, tokens))
    )]
    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> WCResult<()> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "concurrent")] {
                let mut se = self
                    .se_pool
                    .get()
                    .lock()
                    .expect("SpanEncoder mutex poisoned: a thread panicked during encoding");
            } else {
                let mut se = (self.se_builder)();
            }
        }

        self.spanner.for_each_split_span(text, &mut |span_ref| {
            match span_ref {
                SpanRef::Word(range) => {
                    let span = unsafe { text.get_unchecked(range).as_bytes() };
                    if let Some(token) = self.vocab.lookup_token(span) {
                        tokens.push(token);
                    } else {
                        se.encode_append_compound_span(&self.vocab, span, tokens);
                    }
                }
                SpanRef::Special(range) => {
                    let span = unsafe { text.get_unchecked(range).as_bytes() };
                    if let Some(token) = self.vocab.lookup_special_token(span) {
                        tokens.push(token);
                    } else {
                        let span = String::from_utf8_lossy(span).to_string();
                        panic!("special token \"{span}\" not found in vocab");
                    }
                }
                SpanRef::Gap(_) => (),
            }
            true
        });

        Ok(())
    }
}
