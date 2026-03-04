//! Unicode character class representative codepoints for `OpenAI` tokenizer
//! patterns.
//!
//! # Derivation
//!
//! The three `OpenAI` patterns (r50k, cl100k, o200k) use these regex predicates
//! over the Unicode space:
//!
//! - `\p{L}` (any letter), refined by o200k into `\p{Lu}`, `\p{Lt}`, `\p{Ll}`,
//!   `\p{Lm}`, `\p{Lo}`
//! - `\p{M}` (mark, o200k only)
//! - `\p{N}` (any number)
//! - `\s` (whitespace), `\r`, `\n`
//! - `[^\s\p{L}\p{N}]` (punctuation/symbols)
//! - Literal chars: `'` (contraction trigger), `/` (o200k trailer), ` ` (r50k
//!   prefix)
//! - Contraction suffixes: `'(?:[sdmt]|ll|ve|re)` (r50k case-sensitive;
//!   cl100k/o200k case-insensitive via `(?i:...)`)
//!
//! Intersecting these predicates partitions Unicode into 22 equivalence cells
//! (validated programmatically by `validate_representative_completeness` test).
//! The contraction patterns split Ll into 5 sub-cells (non-contraction,
//! single-char suffix, double-char `ll`, multi-start `r`/`v`, multi-end `e`)
//! and likewise Lu into 5 sub-cells for case-insensitive patterns.
//!
//! Some cells have multiple entries (Mn/Mc/Me for Mark, Nd/Nl/No for Number,
//! tab/NBSP for other whitespace, `!`/`$`/`\u{00AE}` for punctuation) because
//! logos DFA sub-cell divergences exist within those regex-equivalent cells.

/// One representative codepoint per equivalence cell in the regex character
/// class partition (22 cells), plus 7 sub-cell extras for logos divergence
/// coverage.
///
/// See module-level docs for the derivation.
pub const REPRESENTATIVES: &[(char, &str)] = &[
    // Letters - base categories (5 cells from o200k GC subdivision)
    ('A', "Lu"),        // Uppercase Letter (non-contraction)
    ('a', "Ll"),        // Lowercase Letter (non-contraction)
    ('\u{01C5}', "Lt"), // Titlecase Letter (Dz with caron)
    ('\u{02B0}', "Lm"), // Modifier Letter
    ('\u{00BA}', "Lo"), // Other Letter (masculine ordinal)
    // Letters - contraction sub-cells within Ll (4 cells)
    ('s', "Ll_ctr"), // Single-char contraction: 's/'d/'m/'t
    ('l', "Ll_ll"),  // Double-char contraction: 'll
    ('r', "Ll_rv"),  // Multi-char contraction start: 're/'ve
    ('e', "Ll_e"),   // Multi-char contraction end: 're/'ve
    // Letters - contraction sub-cells within Lu (4 cells, case-insensitive)
    ('S', "Lu_ctr"), // Uppercase single-char contraction (cl100k/o200k)
    ('L', "Lu_ll"),  // Uppercase double-char contraction
    ('R', "Lu_rv"),  // Uppercase multi-char contraction start
    ('E', "Lu_e"),   // Uppercase multi-char contraction end
    // Marks (1 regex cell, 3 sub-cells for logos coverage)
    ('\u{0300}', "Mn"), // Nonspacing Mark (combining grave)
    ('\u{0B3E}', "Mc"), // Spacing Combining Mark (Oriya AA)
    ('\u{20DD}', "Me"), // Enclosing Mark (combining circle)
    // Numbers (3 cells)
    ('1', "Nd"),        // Decimal Digit Number
    ('\u{2160}', "Nl"), // Letter Number (Roman numeral one)
    ('\u{00B2}', "No"), // Other Number (superscript two)
    // Whitespace (4 cells; tab and NBSP are same regex cell, both kept for logos)
    (' ', "Zs_ascii"),       // ASCII space (distinct: r50k ` ?` prefix)
    ('\u{00A0}', "Zs_nbsp"), // Non-breaking space (same cell as tab)
    ('\t', "Cc_tab"),        // Horizontal tab (same cell as NBSP)
    ('\r', "Cc_cr"),         // Carriage return (distinct: `[\r\n]` patterns)
    ('\n', "Cc_lf"),         // Line feed (distinct: `[\r\n]` patterns)
    // Punctuation/symbols (3 cells; !/$/registered are same regex cell, extras for logos)
    ('!', "Po"),        // Other Punctuation
    ('$', "Sc"),        // Currency Symbol
    ('\u{00AE}', "So"), // Other Symbol (registered sign)
    ('\'', "apos"),     // Apostrophe (contraction trigger)
    ('/', "slash"),     // Slash (punctuation trailer in o200k)
];
