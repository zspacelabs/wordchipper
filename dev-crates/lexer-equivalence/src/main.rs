fn main() {
    println!("Use `cargo test` to run equivalence tests.");
}

#[cfg(test)]
mod tests {
    use std::{ops::Range, sync::Arc};

    use wordchipper::{
        pretrained::openai::{OA_CL100K_BASE_PATTERN, OA_O200K_BASE_PATTERN, OA_R50K_BASE_PATTERN},
        spanners::span_lexers::{
            SpanLexer,
            accelerators::testutil::assert_matches_reference_lexer,
            build_regex_lexer,
            logos::{cl100k::Cl100kLexer, o200k::O200kLexer, r50k::R50kLexer},
        },
        support::regex::ConstRegexPattern,
    };

    // =====================================================================
    // Character class representatives
    // =====================================================================

    /// One representative codepoint per Unicode General Category relevant to
    /// the `OpenAI` tokenizer regex patterns.
    const REPRESENTATIVES: &[(char, &str)] = &[
        ('A', "Lu"),             // Uppercase Letter
        ('a', "Ll"),             // Lowercase Letter
        ('\u{01C5}', "Lt"),      // Titlecase Letter (Dz with caron)
        ('\u{02B0}', "Lm"),      // Modifier Letter
        ('\u{00BA}', "Lo"),      // Other Letter (masculine ordinal)
        ('\u{0300}', "Mn"),      // Nonspacing Mark (combining grave)
        ('\u{0B3E}', "Mc"),      // Spacing Combining Mark (Oriya AA)
        ('\u{20DD}', "Me"),      // Enclosing Mark (combining circle)
        ('1', "Nd"),             // Decimal Digit Number
        (' ', "Zs_ascii"),       // ASCII space
        ('\u{00A0}', "Zs_nbsp"), // Non-breaking space
        ('\t', "Cc_tab"),        // Horizontal tab (whitespace)
        ('\r', "Cc_cr"),         // Carriage return
        ('\n', "Cc_lf"),         // Line feed
        ('!', "Po"),             // Other Punctuation
        ('$', "Sc"),             // Currency Symbol
        ('\u{00AE}', "So"),      // Other Symbol (registered sign)
        ('\'', "apos"),          // Apostrophe (contraction trigger)
        ('/', "slash"),          // Slash (punctuation trailer in o200k)
    ];

    /// Strict representatives for o200k: excludes Mark (Mn, Mc, Me) and
    /// non-space whitespace (NBSP, tab) that cause known divergences.
    ///
    /// Mark divergence: logos DFA longest-match merges standalone Mark chars
    /// with following tokens, while regex leftmost-first keeps them separate.
    ///
    /// Non-space whitespace divergence: when multi-char whitespace ends with
    /// a non-space ws char (NBSP/tab), post-processing merges it with the
    /// following punctuation prefix, while regex keeps them split.
    const REPRESENTATIVES_STRICT_O200K: &[(char, &str)] = &[
        ('A', "Lu"),
        ('a', "Ll"),
        ('\u{01C5}', "Lt"),
        ('\u{02B0}', "Lm"),
        ('\u{00BA}', "Lo"),
        ('1', "Nd"),
        (' ', "Zs_ascii"),
        ('\r', "Cc_cr"),
        ('\n', "Cc_lf"),
        ('!', "Po"),
        ('$', "Sc"),
        ('\u{00AE}', "So"),
        ('\'', "apos"),
        ('/', "slash"),
    ];

    /// Strict representatives for cl100k: excludes Mark, CR/LF, and
    /// non-space whitespace.
    ///
    /// Additional cl100k divergences beyond o200k:
    /// - CR/LF at end of string: regex `\s++$` groups newlines + trailing ws,
    ///   but logos splits by Newline/Whitespace token types.
    const REPRESENTATIVES_STRICT_CL100K: &[(char, &str)] = &[
        ('A', "Lu"),
        ('a', "Ll"),
        ('\u{01C5}', "Lt"),
        ('\u{02B0}', "Lm"),
        ('\u{00BA}', "Lo"),
        ('1', "Nd"),
        (' ', "Zs_ascii"),
        ('!', "Po"),
        ('$', "Sc"),
        ('\u{00AE}', "So"),
        ('\'', "apos"),
        ('/', "slash"),
    ];

    // =====================================================================
    // Test harness
    // =====================================================================

    fn regex_lexer(pattern: ConstRegexPattern) -> Arc<dyn SpanLexer> {
        build_regex_lexer(pattern.to_pattern(), false, false, None)
    }

    fn collect_spans(
        lexer: &dyn SpanLexer,
        text: &str,
    ) -> Vec<Range<usize>> {
        lexer.find_span_iter(text).collect()
    }

    /// Generate all k-tuples from the given character set and test each
    /// against both `ref_lexer` (regex) and `test_lexer` (logos).
    fn run_k_tuple_equivalence(
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
    fn assert_k_tuple_equivalence(
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
    fn report_k_tuple_divergences(
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

    // =====================================================================
    // STRICT EQUIVALENCE TESTS (should pass)
    //
    // These exclude character classes with known divergences to prove
    // equivalence over the remaining (vast majority of) character space.
    // =====================================================================

    #[test]
    fn r50k_equivalence_k1_to_k4() {
        let ref_lexer = regex_lexer(OA_R50K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(R50kLexer);
        // r50k has no known divergences; test with ALL representatives.
        assert_k_tuple_equivalence("r50k", 4, REPRESENTATIVES, &*ref_lexer, &*test_lexer);
    }

    #[test]
    fn cl100k_equivalence_k1_to_k4() {
        let ref_lexer = regex_lexer(OA_CL100K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(Cl100kLexer);
        assert_k_tuple_equivalence(
            "cl100k",
            4,
            REPRESENTATIVES_STRICT_CL100K,
            &*ref_lexer,
            &*test_lexer,
        );
    }

    #[test]
    fn o200k_equivalence_k1_to_k4() {
        let ref_lexer = regex_lexer(OA_O200K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(O200kLexer);
        assert_k_tuple_equivalence(
            "o200k",
            4,
            REPRESENTATIVES_STRICT_O200K,
            &*ref_lexer,
            &*test_lexer,
        );
    }

    // =====================================================================
    // DIVERGENCE INVENTORY (informational, always passes)
    //
    // Documents the known divergence counts when ALL representatives are
    // included. Some individual comparisons diverge between logos and regex;
    // the counts are printed to stderr for manual review. These tests
    // themselves always pass.
    // =====================================================================

    #[test]
    fn cl100k_divergence_inventory() {
        let ref_lexer = regex_lexer(OA_CL100K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(Cl100kLexer);
        let (total, failures) =
            report_k_tuple_divergences("cl100k", 4, REPRESENTATIVES, &*ref_lexer, &*test_lexer);
        // Record divergence count for regression tracking.
        // If this changes, investigate whether the logos post-processing improved
        // or regressed.
        eprintln!("cl100k divergence rate: {failures}/{total}");
    }

    #[test]
    fn o200k_divergence_inventory() {
        let ref_lexer = regex_lexer(OA_O200K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(O200kLexer);
        let (total, failures) =
            report_k_tuple_divergences("o200k", 4, REPRESENTATIVES, &*ref_lexer, &*test_lexer);
        eprintln!("o200k divergence rate: {failures}/{total}");
    }

    // =====================================================================
    // KNOWN TRICKY INPUTS (regression tests from real-world text)
    // =====================================================================

    #[test]
    fn cl100k_known_tricky_inputs() {
        let ref_lexer = regex_lexer(OA_CL100K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(Cl100kLexer);

        let cases = [
            "'The quick brown fox",
            "  'The quick",
            " \u{2014}hello world",
            "Shakespeare's \"sources,\" then read",
            "foo  \nbar",
            "\t\thello",
            "  $400 dollars",
            "caf\u{00e9} na\u{00ef}ve r\u{00e9}sum\u{00e9}",
        ];

        for text in cases {
            assert_matches_reference_lexer(text, &*ref_lexer, &*test_lexer);
        }
    }

    #[test]
    fn o200k_known_tricky_inputs() {
        let ref_lexer = regex_lexer(OA_O200K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(O200kLexer);

        let cases = [
            " average temperature of 21\u{00B0}C (70\u{00BA}F) during the winter.",
            "\u{00BA}",
            "\u{02B0}F",
            "\u{02B0}Hello",
            "\u{00BA}ABC",
            "don't I'll she's",
            "HELLO WORLD",
            "foo  \nbar",
            "$$$!!!...---",
        ];

        for text in cases {
            assert_matches_reference_lexer(text, &*ref_lexer, &*test_lexer);
        }
    }

    #[test]
    fn r50k_known_tricky_inputs() {
        let ref_lexer = regex_lexer(OA_R50K_BASE_PATTERN);
        let test_lexer: Arc<dyn SpanLexer> = Arc::new(R50kLexer);

        let cases = [
            "Pipeline\nThe mode",
            "\nhello",
            "\n\nhello",
            " \nhello",
            "  \r\n  bar",
            "don't I'll she's",
            "  123",
            "  !",
        ];

        for text in cases {
            assert_matches_reference_lexer(text, &*ref_lexer, &*test_lexer);
        }
    }
}
