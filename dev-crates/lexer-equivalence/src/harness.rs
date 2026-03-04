//! K-tuple combinatorial equivalence testing harness for
//! [`SpanLexer`](wordchipper::spanners::span_lexers::SpanLexer) pairs.

use std::{
    collections::BTreeMap,
    fmt::Write,
    ops::Range,
    sync::Arc,
};

use wordchipper::{
    spanners::span_lexers::{
        SpanLexer,
        build_regex_lexer,
    },
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

/// A single divergence between regex and logos lexers.
pub struct Divergence {
    /// Representative index for each character position.
    pub rep_indices: Vec<usize>,
    /// The concrete input text.
    pub text: String,
    /// Regex span strings.
    pub regex_strs: Vec<String>,
    /// Logos span strings.
    pub logos_strs: Vec<String>,
    /// Which char indices are grouped into each regex span.
    pub regex_groups: Vec<Vec<usize>>,
    /// Which char indices are grouped into each logos span.
    pub logos_groups: Vec<Vec<usize>>,
}

/// Generate all k-tuples from the representative set and test each
/// against both `ref_lexer` (regex) and `test_lexer` (logos).
///
/// Returns `(total_cases, divergences)`.
pub fn run_k_tuple_equivalence(
    k: usize,
    representatives: &[(char, &str)],
    ref_lexer: &dyn SpanLexer,
    test_lexer: &dyn SpanLexer,
) -> (usize, Vec<Divergence>) {
    let n = representatives.len();
    let total = n.pow(k as u32);
    let mut divergences = Vec::new();

    for combo in 0..total {
        let mut rep_indices = Vec::with_capacity(k);
        let mut text = String::new();
        let mut char_byte_starts = Vec::with_capacity(k);
        let mut remainder = combo;
        for _ in 0..k {
            let idx = remainder % n;
            rep_indices.push(idx);
            char_byte_starts.push(text.len());
            text.push(representatives[idx].0);
            remainder /= n;
        }

        let expected = collect_spans(ref_lexer, &text);
        let observed = collect_spans(test_lexer, &text);

        if expected != observed {
            let to_groups = |spans: &[Range<usize>]| -> Vec<Vec<usize>> {
                spans
                    .iter()
                    .map(|span| {
                        (0..k)
                            .filter(|&ci| {
                                let start = char_byte_starts[ci];
                                start >= span.start && start < span.end
                            })
                            .collect()
                    })
                    .collect()
            };
            divergences.push(Divergence {
                rep_indices,
                text: text.clone(),
                regex_strs: expected
                    .iter()
                    .map(|r| text[r.clone()].to_string())
                    .collect(),
                logos_strs: observed
                    .iter()
                    .map(|r| text[r.clone()].to_string())
                    .collect(),
                regex_groups: to_groups(&expected),
                logos_groups: to_groups(&observed),
            });
        }
    }

    (total, divergences)
}

/// Group divergences by span-boundary pattern and format as classes.
///
/// Two divergences are in the same class if they have identical
/// (`regex_groups`, `logos_groups`). Positions where multiple labels appear
/// are shown as wildcards.
fn format_divergence_classes(
    divergences: &[Divergence],
    representatives: &[(char, &str)],
) -> String {
    let mut classes: BTreeMap<String, Vec<&Divergence>> = BTreeMap::new();
    for d in divergences {
        let key = format!("{:?}|{:?}", d.regex_groups, d.logos_groups);
        classes.entry(key).or_default().push(d);
    }

    let mut out = String::new();
    let _ = writeln!(out, "{} divergence class(es):\n", classes.len());

    for (i, (_key, members)) in classes.iter().enumerate() {
        let d0 = members[0];
        let k = d0.rep_indices.len();

        // Collect which labels appear at each position across all members.
        let mut pos_labels: Vec<BTreeMap<&str, usize>> = vec![BTreeMap::new(); k];
        for d in members {
            for (pos, &ri) in d.rep_indices.iter().enumerate() {
                *pos_labels[pos].entry(representatives[ri].1).or_insert(0) += 1;
            }
        }

        let format_groups = |groups: &[Vec<usize>]| -> String {
            groups
                .iter()
                .map(|group| {
                    let labels: Vec<&str> = group
                        .iter()
                        .map(|&ci| {
                            if pos_labels[ci].len() == 1 {
                                *pos_labels[ci].keys().next().unwrap()
                            } else {
                                "*"
                            }
                        })
                        .collect();
                    format!("[{}]", labels.join(" "))
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let _ = writeln!(
            out,
            "  {}. {} case(s), k={}:\n     regex: {}\n     logos: {}",
            i + 1,
            members.len(),
            k,
            format_groups(&d0.regex_groups),
            format_groups(&d0.logos_groups),
        );

        // Show wildcard expansions.
        for (pos, labels) in pos_labels.iter().enumerate() {
            if labels.len() > 1 {
                let list: Vec<String> = labels
                    .iter()
                    .map(|(label, count)| format!("{label}({count})"))
                    .collect();
                let _ = writeln!(out, "     * at position {pos}: {{{}}}", list.join(", "));
            }
        }

        // One concrete example.
        let _ = writeln!(
            out,
            "     e.g. {:?} -> regex {:?} / logos {:?}\n",
            d0.text, d0.regex_strs, d0.logos_strs,
        );
    }

    out
}

/// Run equivalence tests for `k=1..=max_k`, panic on any failures.
pub fn assert_k_tuple_equivalence(
    name: &str,
    max_k: usize,
    representatives: &[(char, &str)],
    ref_lexer: &dyn SpanLexer,
    test_lexer: &dyn SpanLexer,
) {
    let mut total_cases = 0;
    let mut all_divergences = Vec::new();

    for k in 1..=max_k {
        let (cases, divs) = run_k_tuple_equivalence(k, representatives, ref_lexer, test_lexer);
        total_cases += cases;
        all_divergences.extend(divs);
    }

    if !all_divergences.is_empty() {
        let report = format_divergence_classes(&all_divergences, representatives);
        panic!(
            "{name}: {}/{total_cases} cases failed\n\n{report}",
            all_divergences.len(),
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
    let mut total_cases = 0;
    let mut all_divergences = Vec::new();

    for k in 1..=max_k {
        let (cases, divs) = run_k_tuple_equivalence(k, representatives, ref_lexer, test_lexer);
        total_cases += cases;
        all_divergences.extend(divs);
    }

    let total_failures = all_divergences.len();
    if total_failures > 0 {
        let report = format_divergence_classes(&all_divergences, representatives);
        eprintln!("{name}: {total_failures}/{total_cases} total divergences (k=1..={max_k})");
        eprintln!("{report}");
    } else {
        eprintln!("{name}: 0/{total_cases} divergences (k=1..={max_k})");
    }

    (total_cases, total_failures)
}
