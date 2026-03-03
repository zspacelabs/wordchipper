//! K-tuple combinatorial equivalence testing harness for
//! [`SpanLexer`](wordchipper::spanners::span_lexers::SpanLexer) pairs.

use std::{ops::Range, sync::Arc};

use wordchipper::{
    spanners::span_lexers::{SpanLexer, build_regex_lexer},
    support::regex::ConstRegexPattern,
};

/// Build a regex-based [`SpanLexer`] from a [`ConstRegexPattern`].
pub fn regex_lexer(pattern: ConstRegexPattern) -> Arc<dyn SpanLexer> {
    build_regex_lexer(pattern.to_pattern(), false, false, None)
}

/// Collect all spans from a lexer into a `Vec`.
pub fn collect_spans(
    lexer: &dyn SpanLexer,
    text: &str,
) -> Vec<Range<usize>> {
    lexer.find_span_iter(text).collect()
}

/// Generate all k-tuples from the given character set and test each
/// against both `ref_lexer` (regex) and `test_lexer` (logos).
///
/// Returns `(total_cases, failure_descriptions)`.
pub fn run_k_tuple_equivalence(
    k: usize,
    chars: &[char],
    ref_lexer: &dyn SpanLexer,
    test_lexer: &dyn SpanLexer,
) -> (usize, Vec<String>) {
    let n = chars.len();
    let total = n.pow(k as u32);
    let mut failures = Vec::new();

    for combo in 0..total {
        let mut text = String::new();
        let mut remainder = combo;
        for _ in 0..k {
            text.push(chars[remainder % n]);
            remainder /= n;
        }

        let expected = collect_spans(ref_lexer, &text);
        let observed = collect_spans(test_lexer, &text);

        if expected != observed {
            let expected_strs: Vec<&str> = expected.iter().map(|r| &text[r.clone()]).collect();
            let observed_strs: Vec<&str> = observed.iter().map(|r| &text[r.clone()]).collect();
            failures.push(format!(
                "k={k} text={text:?}\n  regex:  {expected_strs:?}\n  logos:  {observed_strs:?}"
            ));
        }
    }

    (total, failures)
}

/// Run equivalence tests for `k=1..=max_k`, panic on any failures.
pub fn assert_k_tuple_equivalence(
    name: &str,
    max_k: usize,
    representatives: &[(char, &str)],
    ref_lexer: &dyn SpanLexer,
    test_lexer: &dyn SpanLexer,
) {
    let chars: Vec<char> = representatives.iter().map(|(c, _)| *c).collect();
    let mut total_cases = 0;
    let mut all_failures = Vec::new();

    for k in 1..=max_k {
        let (cases, failures) = run_k_tuple_equivalence(k, &chars, ref_lexer, test_lexer);
        total_cases += cases;
        all_failures.extend(failures);
    }

    if !all_failures.is_empty() {
        let sample: Vec<&String> = all_failures.iter().take(20).collect();
        panic!(
            "{name}: {}/{total_cases} cases failed (showing first {}):\n{}",
            all_failures.len(),
            sample.len(),
            sample
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    eprintln!(
        "{name}: all {total_cases} cases passed (k=1..={max_k}, {} representatives)",
        representatives.len()
    );
}

/// Run equivalence tests for `k=1..=max_k`, return summary without panicking.
///
/// Returns `(total_cases, total_failures)`.
pub fn report_k_tuple_divergences(
    name: &str,
    max_k: usize,
    representatives: &[(char, &str)],
    ref_lexer: &dyn SpanLexer,
    test_lexer: &dyn SpanLexer,
) -> (usize, usize) {
    let chars: Vec<char> = representatives.iter().map(|(c, _)| *c).collect();
    let mut total_cases = 0;
    let mut total_failures = 0;

    for k in 1..=max_k {
        let (cases, failures) = run_k_tuple_equivalence(k, &chars, ref_lexer, test_lexer);
        total_cases += cases;
        if !failures.is_empty() {
            eprintln!("{name} k={k}: {}/{cases} divergences", failures.len());
            for f in failures.iter().take(5) {
                eprintln!("  {f}");
            }
            if failures.len() > 5 {
                eprintln!("  ... and {} more", failures.len() - 5);
            }
            total_failures += failures.len();
        }
    }

    eprintln!("{name}: {total_failures}/{total_cases} total divergences (k=1..={max_k})");
    (total_cases, total_failures)
}
