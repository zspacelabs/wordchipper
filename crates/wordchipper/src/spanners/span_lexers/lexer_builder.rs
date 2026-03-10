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
/// * `enable_accelerators` - whether to try DFA-accelerated lexers (logos)
///   first.
/// * `enable_regex_automata` - whether to try `regex_automata` lexers.
/// * `concurrent` - whether to use a concurrent pool.
/// * `max_pool` - the max size of the concurrent pool; `None` will use
///   system/environment defaults.
pub fn build_regex_lexer(
    pattern: RegexPattern,
    enable_accelerators: bool,
    enable_regex_automata: bool,
    concurrent: bool,
    max_pool: Option<NonZeroUsize>,
) -> Arc<dyn SpanLexer> {
    if enable_accelerators
        && let Some(lexer) = accelerators::get_regex_accelerator(pattern.as_str())
    {
        return lexer;
    }

    // regex-automata: when concurrent feature is active, respect the concurrent
    // flag (so benchmarks can isolate the regex fallback). Without the feature,
    // always try it since it's faster than fancy-regex even single-threaded.
    if enable_regex_automata
        && (cfg!(not(feature = "concurrent")) || concurrent)
        && let Some(lexer) = super::regex_automata::try_build(pattern.as_str(), max_pool)
    {
        return lexer;
    }

    let re: RegexWrapper = pattern.into();

    #[cfg(feature = "concurrent")]
    if concurrent {
        return Arc::new(crate::support::concurrency::PoolToy::new(re, max_pool));
    }

    Arc::new(re)
}
