//! Accelerated custom [`SpanLexer`] machinery.

use crate::{
    alloc::sync::Arc,
    spanners::span_lexers::SpanLexer,
    support::regex::ConstRegexPattern,
};

/// The [`inventory`] hook mechanism for registering regex accelerators.
///
/// These accelerators provide compiled lexers as replacements for
/// specific targeted regex patterns.
///
/// See:
/// * [`get_regex_accelerator`]
/// * [`build_regex_lexer`](`super::build_regex_lexer`)
///
/// # Example
///
/// ```rust,ignore
/// 
/// const MY_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy("abc");
///
/// struct MyRegexAccelerator {}
/// impl SpanLexer for MyRegexAccelerator { ... }
///
/// inventory::submit! {
///     RegexAcceleratorHook::new(
///         MY_PATTERN,
///         || Arc::new(MyRegexAccelerator)
///     )
/// }
/// ```
pub struct RegexAcceleratorHook {
    /// The exact regex pattern.
    pub pattern: ConstRegexPattern,

    /// The [`SpanLexer]` builder function.
    pub builder: fn() -> Arc<dyn SpanLexer>,
}
inventory::collect!(RegexAcceleratorHook);

/// Inventory hook for regex-automata pattern transforms.
///
/// Some patterns require a lookahead-free or possessive-free variant before
/// they can be compiled by `regex-automata`.
pub struct RegexAutomataTransformHook {
    /// The exact source regex pattern.
    pub pattern: ConstRegexPattern,

    /// The transformed pattern accepted by `regex-automata`.
    pub transformed_pattern: &'static str,

    /// Whether whitespace truncation should ignore newline-containing spans.
    pub has_newline_branch: bool,
}
inventory::collect!(RegexAutomataTransformHook);

impl RegexAcceleratorHook {
    /// Setup a new regex accelerator hook.
    pub const fn new(
        pattern: ConstRegexPattern,
        builder: fn() -> Arc<dyn SpanLexer>,
    ) -> Self {
        Self { pattern, builder }
    }
}

impl RegexAutomataTransformHook {
    /// Setup a new regex-automata transform hook.
    pub const fn new(
        pattern: ConstRegexPattern,
        transformed_pattern: &'static str,
        has_newline_branch: bool,
    ) -> Self {
        Self {
            pattern,
            transformed_pattern,
            has_newline_branch,
        }
    }
}

/// Get a regex accelerator.
///
/// ## Returns
/// - `Some(Arc<dyn SpanLexer>)` if an accelerator is found,
/// - `None` otherwise.
pub fn get_regex_accelerator(pattern: &str) -> Option<Arc<dyn SpanLexer>> {
    for hook in inventory::iter::<RegexAcceleratorHook> {
        if hook.pattern.as_str() == pattern {
            return Some((hook.builder)());
        }
    }
    None
}

/// Get a registered `regex-automata` transform.
pub fn get_regex_automata_transform(pattern: &str) -> Option<(&'static str, bool)> {
    for hook in inventory::iter::<RegexAutomataTransformHook> {
        if hook.pattern.as_str() == pattern {
            return Some((hook.transformed_pattern, hook.has_newline_branch));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretrained::openai::patterns::{
        OA_CL100K_BASE_PATTERN,
        OA_CL100K_BASE_PATTERN_RA,
    };

    #[test]
    fn test_unknown_pattern_returns_none() {
        assert!(
            get_regex_accelerator("not_a_real_pattern_that_would_ever_be_registered").is_none()
        );
    }

    #[test]
    fn test_known_regex_automata_transform_returns_hook() {
        assert_eq!(
            get_regex_automata_transform(OA_CL100K_BASE_PATTERN.as_str()),
            Some((OA_CL100K_BASE_PATTERN_RA, true))
        );
    }
}

/// Testing utilities for developing accelerated replacement [`SpanLexer`]s.
#[cfg(any(test, feature = "testing"))]
pub mod testutil {
    use core::cmp::max;

    use super::*;
    use crate::{
        alloc::{
            format,
            vec,
        },
        prelude::*,
    };

    /// Testing utility for checking that a sample and an accelerated span lexer
    /// match.
    pub fn assert_matches_reference_lexer(
        sample: &str,
        ref_lexer: &dyn SpanLexer,
        test_lexer: &dyn SpanLexer,
    ) {
        let expected_spans = ref_lexer.find_span_iter(sample).collect::<Vec<_>>();
        let observed_spans = test_lexer.find_span_iter(sample).collect::<Vec<_>>();

        if observed_spans == expected_spans {
            return;
        }

        let mut first_diff = None;
        for i in 0..max(observed_spans.len(), expected_spans.len()) {
            if observed_spans[i] != expected_spans[i] {
                first_diff = Some(i);
                break;
            }
        }
        let first_diff = first_diff.unwrap();

        let mut parts: Vec<String> = vec![
            "Accelerated lexer failed to match reference lexer.".to_string(),
            format!("sample: <<<{}>>>", sample),
            format!("expected: {:?}", expected_spans),
        ];

        for (i, span) in expected_spans.iter().enumerate() {
            parts.push(format!(
                " {}{}: {:?}: <<<{}>>>",
                if i == first_diff { "*" } else { " " },
                i,
                span,
                &sample[span.clone()]
            ));
        }
        log::error!("observed: {:?}", observed_spans);
        for (i, span) in observed_spans.iter().enumerate() {
            parts.push(format!(
                " {}{}: {:?}: <<<{}>>>",
                if i == first_diff { "*" } else { " " },
                i,
                span,
                &sample[span.clone()]
            ));
        }

        for part in &parts {
            log::error!("{}", part);
        }

        let msg = parts.join("\n");
        panic!("{}", msg);
    }
}
