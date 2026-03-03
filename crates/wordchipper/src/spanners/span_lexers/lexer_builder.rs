//! Regex-based `SpanLexer`

use core::num::NonZeroUsize;

use crate::{
    alloc::sync::Arc,
    spanners::span_lexers::{
        SpanLexer,
        accelerators,
    },
    support::regex::{
        RegexPattern,
        RegexWrapper,
    },
};

/// Build a regex-based [`SpanLexer`] with the given configuration.
///
/// ## Arguments
/// * `pattern` - the pattern.
/// * `concurrent` - whether to use a concurrent pool.
/// * `max_pool` - the max size of the concurrent pool; `None` will use
///   system/environment defaults.
pub fn build_regex_lexer(
    pattern: RegexPattern,
    accelerated: bool,
    concurrent: bool,
    max_pool: Option<NonZeroUsize>,
) -> Arc<dyn SpanLexer> {
    let _ = accelerated;
    let _ = concurrent;
    let _ = max_pool;

    if accelerated && let Some(lexer) = accelerators::get_regex_accelerator(pattern.as_str()) {
        return lexer;
    }

    let re: RegexWrapper = pattern.into();

    #[cfg(feature = "concurrent")]
    if concurrent {
        return Arc::new(crate::support::concurrency::PoolToy::new(re, max_pool));
    }

    Arc::new(re)
}
