//! Unicode character class representative codepoints for `OpenAI` tokenizer patterns.
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
//! - Literal chars: `'` (contraction trigger), `/` (o200k trailer), ` ` (r50k prefix)
//! - Contraction suffixes: `'(?:[sdmt]|ll|ve|re)` (r50k case-sensitive;
//!   cl100k/o200k case-insensitive via `(?i:...)`)
//!
//! Intersecting these predicates partitions Unicode into 22 equivalence cells
//! (validated programmatically by `validate_representative_completeness` test).
//! The contraction patterns split Ll into 5 sub-cells (non-contraction, single-char
//! suffix, double-char `ll`, multi-start `r`/`v`, multi-end `e`) and likewise Lu
//! into 5 sub-cells for case-insensitive patterns.
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

/// Strict representatives for r50k: excludes contraction Ll sub-cells
/// that cause known divergences.
///
/// Contraction divergence: logos treats `' + contraction_letter` as a
/// contraction token even after space-prefix absorption, while regex
/// matches ` ?[^\s\p{L}\p{N}]++` first (absorbing `" '"`), leaving the
/// contraction letter for the next token.
pub const REPRESENTATIVES_STRICT_R50K: &[(char, &str)] = &[
    ('A', "Lu"),
    ('a', "Ll"),
    ('\u{01C5}', "Lt"),
    ('\u{02B0}', "Lm"),
    ('\u{00BA}', "Lo"),
    // Lu contraction sub-cells included (r50k contractions are case-sensitive,
    // so uppercase contraction letters behave like regular Lu).
    ('S', "Lu_ctr"),
    ('L', "Lu_ll"),
    ('R', "Lu_rv"),
    ('E', "Lu_e"),
    ('\u{0300}', "Mn"),
    ('\u{0B3E}', "Mc"),
    ('\u{20DD}', "Me"),
    ('1', "Nd"),
    ('\u{2160}', "Nl"),
    ('\u{00B2}', "No"),
    (' ', "Zs_ascii"),
    ('\u{00A0}', "Zs_nbsp"),
    ('\t', "Cc_tab"),
    ('\r', "Cc_cr"),
    ('\n', "Cc_lf"),
    ('!', "Po"),
    ('$', "Sc"),
    ('\u{00AE}', "So"),
    ('\'', "apos"),
    ('/', "slash"),
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
pub const REPRESENTATIVES_STRICT_O200K: &[(char, &str)] = &[
    ('A', "Lu"),
    ('a', "Ll"),
    ('\u{01C5}', "Lt"),
    ('\u{02B0}', "Lm"),
    ('\u{00BA}', "Lo"),
    ('s', "Ll_ctr"),
    ('l', "Ll_ll"),
    ('r', "Ll_rv"),
    ('e', "Ll_e"),
    ('S', "Lu_ctr"),
    ('L', "Lu_ll"),
    ('R', "Lu_rv"),
    ('E', "Lu_e"),
    ('1', "Nd"),
    ('\u{2160}', "Nl"),
    ('\u{00B2}', "No"),
    (' ', "Zs_ascii"),
    ('\r', "Cc_cr"),
    ('\n', "Cc_lf"),
    ('!', "Po"),
    ('$', "Sc"),
    ('\u{00AE}', "So"),
    ('\'', "apos"),
    ('/', "slash"),
];

/// Strict representatives for cl100k: excludes Mark, CR/LF, non-space
/// whitespace, and contraction Ll/Lu sub-cells.
///
/// Additional cl100k divergences beyond o200k:
/// - CR/LF at end of string: regex `\s++$` groups newlines + trailing ws,
///   but logos splits by Newline/Whitespace token types.
/// - Contraction: same divergence as r50k (` ?[^\s\p{L}\p{N}]++` absorbs
///   `" '"`, splitting the contraction). Affects both Ll and Lu sub-cells
///   since cl100k contractions are case-insensitive.
pub const REPRESENTATIVES_STRICT_CL100K: &[(char, &str)] = &[
    ('A', "Lu"),
    ('a', "Ll"),
    ('\u{01C5}', "Lt"),
    ('\u{02B0}', "Lm"),
    ('\u{00BA}', "Lo"),
    ('1', "Nd"),
    ('\u{2160}', "Nl"),
    ('\u{00B2}', "No"),
    (' ', "Zs_ascii"),
    ('!', "Po"),
    ('$', "Sc"),
    ('\u{00AE}', "So"),
    ('\'', "apos"),
    ('/', "slash"),
];
