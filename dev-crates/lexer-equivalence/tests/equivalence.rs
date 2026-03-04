use std::sync::Arc;

use lexer_equivalence::{
    harness::{
        assert_k_tuple_equivalence,
        regex_lexer,
    },
    representatives::REPRESENTATIVES,
};
use wordchipper::{
    pretrained::openai::{
        OA_CL100K_BASE_PATTERN,
        OA_O200K_BASE_PATTERN,
        OA_R50K_BASE_PATTERN,
    },
    spanners::span_lexers::{
        SpanLexer,
        logos::{
            cl100k::Cl100kLexer,
            o200k::O200kLexer,
            r50k::R50kLexer,
        },
    },
};

// =====================================================================
// REPRESENTATIVE SET VALIDATION
//
// Programmatically verifies that REPRESENTATIVES covers every
// equivalence cell in the regex character class partition. This is the
// "better tool" that replaces manual predicate analysis.
// =====================================================================

mod predicates {
    use std::collections::BTreeMap;

    use regex::Regex;

    /// Compiled regex patterns for Unicode General Category testing.
    pub struct Predicates {
        lu: Regex,
        lt: Regex,
        ll: Regex,
        lm: Regex,
        lo: Regex,
        m: Regex,
        n: Regex,
    }

    /// Names for each predicate bit position. Used for diagnostics.
    pub const PREDICATE_NAMES: &[&str] = &[
        "Lu", "Lt", "Ll", "Lm", "Lo", "M", "N", "ws", "CR", "LF", "space", "apos", "slash", "sdmt",
        "l", "rv", "e", "SDMT", "L_up", "RV_up", "E_up",
    ];

    impl Predicates {
        pub fn new() -> Self {
            Self {
                lu: Regex::new(r"^\p{Lu}$").unwrap(),
                lt: Regex::new(r"^\p{Lt}$").unwrap(),
                ll: Regex::new(r"^\p{Ll}$").unwrap(),
                lm: Regex::new(r"^\p{Lm}$").unwrap(),
                lo: Regex::new(r"^\p{Lo}$").unwrap(),
                m: Regex::new(r"^\p{M}$").unwrap(),
                n: Regex::new(r"^\p{N}$").unwrap(),
            }
        }

        /// Compute the predicate signature bitvector for a character.
        ///
        /// Two chars with identical signatures are interchangeable at the
        /// single-character level across all three `OpenAI` regex patterns.
        pub fn signature(
            &self,
            c: char,
        ) -> u32 {
            let s = c.to_string();
            let mut sig = 0u32;
            let mut bit = 0u32;

            macro_rules! pred {
                ($val:expr) => {{
                    if $val {
                        sig |= 1 << bit;
                    }
                    #[allow(unused_assignments)]
                    {
                        bit += 1;
                    }
                }};
            }

            // Unicode General Category (from regex patterns)
            pred!(self.lu.is_match(&s));
            pred!(self.lt.is_match(&s));
            pred!(self.ll.is_match(&s));
            pred!(self.lm.is_match(&s));
            pred!(self.lo.is_match(&s));
            pred!(self.m.is_match(&s));
            pred!(self.n.is_match(&s));
            pred!(c.is_whitespace());

            // Literal characters from regex patterns
            pred!(c == '\r');
            pred!(c == '\n');
            pred!(c == ' ');
            pred!(c == '\'');
            pred!(c == '/');

            // Contraction suffix letters (case-sensitive, r50k)
            pred!(matches!(c, 's' | 'd' | 'm' | 't'));
            pred!(c == 'l');
            pred!(matches!(c, 'r' | 'v'));
            pred!(c == 'e');

            // Contraction suffix letters (uppercase, cl100k/o200k case-insensitive)
            pred!(matches!(c, 'S' | 'D' | 'M' | 'T'));
            pred!(c == 'L');
            pred!(matches!(c, 'R' | 'V'));
            pred!(c == 'E');

            sig
        }

        /// Format a signature as a human-readable predicate list.
        pub fn format_signature(
            &self,
            sig: u32,
        ) -> String {
            let active: Vec<&str> = PREDICATE_NAMES
                .iter()
                .enumerate()
                .filter(|(i, _)| sig & (1 << i) != 0)
                .map(|(_, name)| *name)
                .collect();
            if active.is_empty() {
                "other".to_string()
            } else {
                active.join("+")
            }
        }
    }

    /// Broad candidate set: all ASCII chars plus non-ASCII exemplars from
    /// every relevant Unicode General Category.
    pub fn candidate_chars() -> Vec<char> {
        let mut chars: Vec<char> = (0u8..=127).map(|b| b as char).collect();
        chars.extend([
            // Non-ASCII letters
            '\u{00C0}', // Lu: Latin capital A with grave
            '\u{01C5}', // Lt: Latin capital D with small z with caron
            '\u{00E0}', // Ll: Latin small a with grave
            '\u{02B0}', // Lm: Modifier letter small h
            '\u{00BA}', // Lo: Masculine ordinal indicator
            '\u{4E00}', // Lo: CJK unified ideograph
            '\u{0E01}', // Lo: Thai character ko kai
            // Marks
            '\u{0300}', // Mn: Combining grave accent
            '\u{0B3E}', // Mc: Oriya vowel sign AA
            '\u{20DD}', // Me: Combining enclosing circle
            // Numbers
            '\u{0660}', // Nd: Arabic-Indic digit zero
            '\u{2160}', // Nl: Roman numeral one
            '\u{00B2}', // No: Superscript two
            // Whitespace
            '\u{00A0}', // NBSP
            '\u{2000}', // En Space
            '\u{3000}', // Ideographic Space
            '\u{2028}', // Line Separator (Zl)
            '\u{2029}', // Paragraph Separator (Zp)
            '\u{0085}', // NEL (Cc, whitespace)
            // Punctuation/symbols (non-ASCII)
            '\u{00AE}', // So: Registered sign
            '\u{00A3}', // Sc: Pound sign
            '\u{2014}', // Pd: Em dash
            '\u{00AB}', // Pi: Left guillemet
            // Format characters
            '\u{200B}', // Cf: Zero-width space (NOT whitespace)
            '\u{FEFF}', // Cf: BOM
        ]);
        chars
    }

    /// Verify that every candidate char's signature is covered by at least
    /// one representative. Returns a map of (signature -> representative
    /// labels).
    pub fn validate_coverage(
        representatives: &[(char, &str)]
    ) -> Result<BTreeMap<u32, Vec<(char, String)>>, Vec<(char, u32)>> {
        #![allow(clippy::type_complexity)]
        let preds = Predicates::new();

        // Build map: signature -> representatives that cover it
        let mut cell_map: BTreeMap<u32, Vec<(char, String)>> = BTreeMap::new();
        for &(c, label) in representatives {
            cell_map
                .entry(preds.signature(c))
                .or_default()
                .push((c, label.to_string()));
        }

        // Check every candidate is covered
        let candidates = candidate_chars();
        let mut uncovered = Vec::new();
        for c in candidates {
            let sig = preds.signature(c);
            if !cell_map.contains_key(&sig) {
                uncovered.push((c, sig));
            }
        }

        if uncovered.is_empty() {
            Ok(cell_map)
        } else {
            Err(uncovered)
        }
    }
}

/// Validate that REPRESENTATIVES covers every equivalence cell.
///
/// This test replaces manual predicate analysis with programmatic
/// verification. If a predicate is missing from the analysis, or a
/// character falls into an uncovered cell, this test fails.
#[test]
fn validate_representative_completeness() {
    let preds = predicates::Predicates::new();

    match predicates::validate_coverage(REPRESENTATIVES) {
        Ok(cell_map) => {
            eprintln!(
                "REPRESENTATIVES: {} entries cover {} distinct cells",
                REPRESENTATIVES.len(),
                cell_map.len()
            );
            for (sig, reps) in &cell_map {
                let labels: Vec<&str> = reps.iter().map(|(_, l)| l.as_str()).collect();
                eprintln!(
                    "  sig={sig:#07x} [{}]: {labels:?}",
                    preds.format_signature(*sig),
                );
            }
        }
        Err(uncovered) => {
            let mut msg = format!("REPRESENTATIVES missing {} cell(s):\n", uncovered.len());
            for (c, sig) in &uncovered {
                msg.push_str(&format!(
                    "  {:?} (U+{:04X}) sig={sig:#07x} [{}]\n",
                    c,
                    *c as u32,
                    preds.format_signature(*sig),
                ));
            }
            panic!("{msg}");
        }
    }
}

// =====================================================================
// FULL EQUIVALENCE TESTS
//
// Test all 29 representatives (k=1..4, ~732K inputs per lexer).
// =====================================================================

#[test]
fn r50k_equivalence() {
    let ref_lexer = regex_lexer(OA_R50K_BASE_PATTERN);
    let test_lexer: Arc<dyn SpanLexer> = Arc::new(R50kLexer);
    assert_k_tuple_equivalence("r50k", 4, REPRESENTATIVES, &*ref_lexer, &*test_lexer);
}

#[test]
fn cl100k_equivalence() {
    let ref_lexer = regex_lexer(OA_CL100K_BASE_PATTERN);
    let test_lexer: Arc<dyn SpanLexer> = Arc::new(Cl100kLexer);
    assert_k_tuple_equivalence("cl100k", 4, REPRESENTATIVES, &*ref_lexer, &*test_lexer);
}

#[test]
fn o200k_equivalence() {
    let ref_lexer = regex_lexer(OA_O200K_BASE_PATTERN);
    let test_lexer: Arc<dyn SpanLexer> = Arc::new(O200kLexer);
    assert_k_tuple_equivalence("o200k", 4, REPRESENTATIVES, &*ref_lexer, &*test_lexer);
}
