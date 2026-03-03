//! Parse divan benchmark output into structured results.
//!
//! Divan has no machine-readable output format. This module parses the
//! human-readable table output line by line and produces [`BenchResult`]
//! values.

use std::collections::BTreeMap;

use serde::Serialize;

/// Statistical values across benchmark iterations.
#[derive(Debug, Clone, Serialize)]
pub struct StatValues {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fastest: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slowest: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
}

/// A counter with its unit string and stat values.
#[derive(Debug, Clone, Serialize)]
pub struct CounterValues {
    pub unit: String,
    #[serde(flatten)]
    pub stats: StatValues,
}

/// A single benchmark result with timing, throughput, allocs, and counters.
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bench: Option<String>,
    pub samples: u64,
    pub iters: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_ns: Option<StatValues>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub throughput_bps: Option<StatValues>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocs: Option<BTreeMap<String, StatValues>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counters: Option<Vec<CounterValues>>,
}

/// What the alloc section expects next.
#[derive(Debug, Clone, Copy)]
enum AllocExpect {
    Count,
    Bytes,
}

/// Streaming parser for divan's table output.
pub struct DivanParser {
    results: Vec<BenchResult>,
    /// Number of `│` separators in the header (expected column count).
    header_sep_count: usize,
    /// Current bench file name from cargo's "Running benches/..." line.
    bench_file: Option<String>,
    /// Depth-to-name mapping for tree hierarchy.
    names: BTreeMap<usize, String>,
    /// Index of the current result being built (into `results`).
    current: Option<usize>,
    /// Current alloc section label (e.g. "alloc", "`max_alloc`").
    alloc_section: Option<String>,
    /// What the alloc section expects next.
    alloc_expect: Option<AllocExpect>,
}

impl Default for DivanParser {
    fn default() -> Self {
        Self::new()
    }
}

impl DivanParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            header_sep_count: 0,
            bench_file: None,
            names: BTreeMap::new(),
            current: None,
            alloc_section: None,
            alloc_expect: None,
        }
    }

    /// Process one line of divan output.
    pub fn feed_line(
        &mut self,
        line: &str,
    ) {
        // Track bench file from cargo output.
        // Cargo prints: "     Running benches/spanning.rs (target/...)"
        if let Some(pos) = line.find("benches/") {
            let after = &line[pos + 8..];
            if let Some(end) = after.find(".rs") {
                self.bench_file = Some(after[..end].to_string());
            }
        }

        // Detect header line.
        if line.contains("fastest") && line.contains("slowest") && line.contains("median") {
            self.header_sep_count = line.chars().filter(|&ch| ch == '\u{2502}').count();

            // Root name is the first word before the first separator.
            let root = line
                .split('\u{2502}')
                .next()
                .and_then(|s| s.split_whitespace().next())
                .unwrap_or("unknown");
            self.names.clear();
            self.names.insert(0, root.to_string());
            self.current = None;
            self.alloc_section = None;
            self.alloc_expect = None;
            return;
        }

        // Skip lines before we've seen a header, or noise lines.
        if self.header_sep_count == 0 {
            return;
        }
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("Timer ")
            || trimmed.starts_with("Running ")
            || trimmed.starts_with("Compiling ")
            || trimmed.starts_with("Finished ")
        {
            return;
        }

        // Split on │ and reconstruct: the last N parts are always data columns,
        // everything before is the first column (may contain tree-drawing │).
        let cols = split_columns(line, self.header_sep_count);
        let first = &cols[0];

        if let Some((depth, name)) = branch_info(first) {
            // Remove names at depths >= this one.
            self.names.retain(|&d, _| d < depth);
            self.names.insert(depth, name.to_string());

            let samples_str = cols.get(4).map(|s| s.trim()).unwrap_or("");
            let iters_str = cols.get(5).map(|s| s.trim()).unwrap_or("");

            if !samples_str.is_empty() && samples_str.chars().all(|c| c.is_ascii_digit()) {
                // Leaf benchmark with data.
                let full_name: String = self.names.values().cloned().collect::<Vec<_>>().join("::");

                // Extract value text after the branch name in the first column.
                let val_text = strip_tree(first).strip_prefix(name).unwrap_or("").trim();

                let row = parse_stat_row(val_text, &cols);

                let samples: u64 = samples_str.parse().unwrap_or(0);
                let iters: u64 = iters_str.parse().unwrap_or(0);

                let mut result = BenchResult {
                    name: full_name,
                    bench: self.bench_file.clone(),
                    samples,
                    iters,
                    time_ns: convert_row(&row, |u| table_lookup(u, TIME_NS)),
                    throughput_bps: None,
                    allocs: None,
                    counters: None,
                };

                if let Some((_, unit)) = &row[0]
                    && throughput_lookup(unit).is_some()
                {
                    result.throughput_bps = convert_row(&row, throughput_lookup);
                }

                self.results.push(result);
                self.current = Some(self.results.len() - 1);
                self.alloc_section = None;
                self.alloc_expect = None;
            } else {
                self.current = None;
                self.alloc_section = None;
                self.alloc_expect = None;
            }
        } else if let Some(idx) = self.current {
            let val_text = strip_tree(first);

            // Detect section labels: text ending with ":" where data columns are empty.
            let data_cols_empty = (1..4).all(|j| {
                cols.get(j)
                    .map(|s| parse_value(s.trim()).is_none())
                    .unwrap_or(true)
            });

            if val_text.ends_with(':') && data_cols_empty {
                let label = val_text.trim_end_matches(':').trim().replace(' ', "_");
                self.alloc_section = Some(label);
                self.alloc_expect = Some(AllocExpect::Count);
                return;
            }

            let row = parse_stat_row(val_text, &cols);

            if let Some((_, ref unit)) = row[0] {
                let unit = unit.as_str();

                if matches!(self.alloc_expect, Some(AllocExpect::Count))
                    && let Some(ref section) = self.alloc_section
                    && unit.is_empty()
                {
                    let key = format!("{section}_count");
                    self.results[idx]
                        .allocs
                        .get_or_insert_with(BTreeMap::new)
                        .insert(key, raw_stat_values(&row));
                    self.alloc_expect = Some(AllocExpect::Bytes);
                    return;
                }

                if matches!(self.alloc_expect, Some(AllocExpect::Bytes))
                    && let Some(ref section) = self.alloc_section
                {
                    if let Some(sv) = convert_row(&row, |u| table_lookup(u, BYTE_UNITS)) {
                        let key = format!("{section}_bytes");
                        self.results[idx]
                            .allocs
                            .get_or_insert_with(BTreeMap::new)
                            .insert(key, sv);
                    }
                    self.alloc_expect = None;
                    return;
                }

                if let Some(sv) = convert_row(&row, throughput_lookup) {
                    self.results[idx].throughput_bps = Some(sv);
                } else if !unit.is_empty() {
                    let cv = CounterValues {
                        unit: unit.to_string(),
                        stats: raw_stat_values(&row),
                    };
                    self.results[idx]
                        .counters
                        .get_or_insert_with(Vec::new)
                        .push(cv);
                }
            }
        }
    }

    /// Finish parsing and return all collected results.
    pub fn finish(self) -> Vec<BenchResult> {
        self.results
    }
}

// ---------------------------------------------------------------------------
// Unit conversion tables
// ---------------------------------------------------------------------------

/// Time unit to nanoseconds.
const TIME_NS: &[(&str, f64)] = &[
    ("ps", 1e-3),
    ("ns", 1.0),
    ("\u{00b5}s", 1e3), // µs
    ("us", 1e3),
    ("ms", 1e6),
    ("s", 1e9),
];

/// Byte size unit to bytes (also used for throughput by stripping the "/s"
/// suffix).
const BYTE_UNITS: &[(&str, f64)] = &[
    ("", 1.0),
    ("B", 1.0),
    ("KB", 1e3),
    ("MB", 1e6),
    ("GB", 1e9),
    ("TB", 1e12),
    ("KiB", 1024.0),
    ("MiB", 1_048_576.0),
    ("GiB", 1_073_741_824.0),
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split a line into columns: first column + data columns.
///
/// Divan uses `│` both for tree drawing and column separators. The header
/// always has `header_sep_count` separators (typically 5). Data lines may
/// have more due to tree-drawing `│` chars. We split on all `│`, then
/// join the leading segments (before the last N) back together as the
/// first column.
fn split_columns(
    line: &str,
    header_sep_count: usize,
) -> Vec<String> {
    let parts: Vec<&str> = line.split('\u{2502}').collect();
    if parts.len() <= header_sep_count {
        return parts.iter().map(|s| s.to_string()).collect();
    }
    let split_at = parts.len() - header_sep_count;
    let mut cols = vec![parts[..split_at].join("\u{2502}")];
    cols.extend(parts[split_at..].iter().map(|s| s.to_string()));
    cols
}

/// Strip leading tree-drawing characters and spaces from a column.
fn strip_tree(text: &str) -> &str {
    text.trim_start_matches(|c: char| {
        matches!(c, '\u{2502}' | '\u{251c}' | '\u{2570}' | '\u{2500}' | ' ')
    })
    .trim()
}

/// If the first column contains a branch marker, return `(depth, name)`.
fn branch_info(first_col: &str) -> Option<(usize, &str)> {
    let (char_idx, byte_idx) = first_col
        .char_indices()
        .enumerate()
        .find_map(|(ci, (bi, ch))| (ch == '\u{251c}' || ch == '\u{2570}').then_some((ci, bi)))?;

    let depth = char_idx / 3 + 1;

    let name = first_col[byte_idx..]
        .trim_start_matches(['\u{251c}', '\u{2570}', '\u{2500}', ' '])
        .split_whitespace()
        .next()?;

    Some((depth, name))
}

/// Parse a "number unit" string. Returns `(number, unit)` or `None`.
fn parse_value(text: &str) -> Option<(f64, String)> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let num_end = text
        .find(|c: char| !c.is_ascii_digit() && c != ',' && c != '.')
        .unwrap_or(text.len());
    if num_end == 0 {
        return None;
    }
    let num: f64 = text[..num_end].replace(',', "").parse().ok()?;
    let unit = text[num_end..].trim().to_string();
    Some((num, unit))
}

/// Parse stat row: first column value + columns 1..3.
fn parse_stat_row(
    val_text: &str,
    cols: &[String],
) -> [Option<(f64, String)>; 4] {
    [
        parse_value(val_text),
        cols.get(1).and_then(|s| parse_value(s.trim())),
        cols.get(2).and_then(|s| parse_value(s.trim())),
        cols.get(3).and_then(|s| parse_value(s.trim())),
    ]
}

/// Look up a unit string in a conversion table.
fn table_lookup(
    unit: &str,
    table: &[(&str, f64)],
) -> Option<f64> {
    table.iter().find(|&&(u, _)| u == unit).map(|&(_, m)| m)
}

/// Look up a throughput unit (e.g. "MB/s") by stripping "/s" and checking byte
/// units.
fn throughput_lookup(unit: &str) -> Option<f64> {
    unit.strip_suffix("/s")
        .and_then(|b| table_lookup(b, BYTE_UNITS))
}

/// Build [`StatValues`] from raw parsed values, using a lookup function for
/// unit conversion. Returns `None` if no values matched.
fn convert_row(
    row: &[Option<(f64, String)>; 4],
    lookup: impl Fn(&str) -> Option<f64>,
) -> Option<StatValues> {
    let mut vals = [None; 4];
    let mut any = false;
    for (i, slot) in row.iter().enumerate() {
        if let Some((num, unit)) = slot
            && let Some(mult) = lookup(unit)
        {
            vals[i] = Some(num * mult);
            any = true;
        }
    }
    any.then(|| StatValues {
        fastest: vals[0],
        slowest: vals[1],
        median: vals[2],
        mean: vals[3],
    })
}

/// Build [`StatValues`] from raw parsed values without unit conversion.
fn raw_stat_values(row: &[Option<(f64, String)>; 4]) -> StatValues {
    StatValues {
        fastest: row[0].as_ref().map(|(n, _)| *n),
        slowest: row[1].as_ref().map(|(n, _)| *n),
        median: row[2].as_ref().map(|(n, _)| *n),
        mean: row[3].as_ref().map(|(n, _)| *n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_text(text: &str) -> Vec<BenchResult> {
        let mut parser = DivanParser::new();
        for line in text.lines() {
            parser.feed_line(line);
        }
        parser.finish()
    }

    #[test]
    fn two_level_spanning() {
        let input = "\
     Running benches/spanning.rs (target/release/deps/spanning-abc123)
spanning                       fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cl100k_base                 │               │               │               │         │
│  ├─ logos_spanner             12.54 µs      │ 20.33 µs      │ 12.88 µs      │ 13.22 µs      │ 100     │ 100
│  ╰─ regex_spanner             383.4 µs      │ 456.2 µs      │ 391.8 µs      │ 397.1 µs      │ 100     │ 100";

        let results = parse_text(input);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "spanning::cl100k_base::logos_spanner");
        assert_eq!(results[0].bench.as_deref(), Some("spanning"));
        assert_eq!(results[0].samples, 100);
        let t = results[0].time_ns.as_ref().unwrap();
        assert!((t.fastest.unwrap() - 12_540.0).abs() < 1.0);
        assert_eq!(results[1].name, "spanning::cl100k_base::regex_spanner");
    }

    #[test]
    fn three_level_encoding() {
        let input = "\
encoding_single                fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cl100k_base                 │               │               │               │         │
│  ├─ bpe                      │               │               │               │         │
│  │  ╰─ encode                 58.32 µs      │ 72.11 µs      │ 60.44 µs      │ 61.05 µs      │ 100     │ 100";

        let results = parse_text(input);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "encoding_single::cl100k_base::bpe::encode");
    }

    #[test]
    fn throughput_line() {
        let input = "\
spanning                       fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ o200k_base                  │               │               │               │         │
   ╰─ logos_spanner             11.88 µs      │ 15.21 µs      │ 12.11 µs      │ 12.45 µs      │ 100     │ 100
                                765.3 MB/s    │ 598.5 MB/s    │ 751.8 MB/s    │ 730.9 MB/s    │         │";

        let results = parse_text(input);
        assert_eq!(results.len(), 1);
        let tp = results[0].throughput_bps.as_ref().unwrap();
        assert!((tp.fastest.unwrap() - 765_300_000.0).abs() < 1.0);
    }

    #[test]
    fn alloc_section() {
        let input = "\
spanning                       fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ bench1                      10 ns         │ 20 ns         │ 15 ns         │ 16 ns         │ 100     │ 100
                               alloc:        │               │               │               │         │
                                5            │ 8             │ 6             │ 6.5           │         │
                                1.2 KB       │ 2.4 KB        │ 1.8 KB        │ 1.9 KB        │         │";

        let results = parse_text(input);
        assert_eq!(results.len(), 1);
        let allocs = results[0].allocs.as_ref().unwrap();
        let count = allocs.get("alloc_count").unwrap();
        assert!((count.fastest.unwrap() - 5.0).abs() < 0.01);
        let bytes = allocs.get("alloc_bytes").unwrap();
        assert!((bytes.fastest.unwrap() - 1200.0).abs() < 1.0);
    }

    #[test]
    fn skips_cargo_noise() {
        let input = "\
   Compiling wordchipper-bench v0.7.3
    Finished `release` profile [optimized] target(s) in 5.32s
     Running benches/spanning.rs (target/release/deps/spanning-abc123)
Timer precision: 41 ns
spanning                       fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ hello                       1 ns          │ 2 ns          │ 1.5 ns        │ 1.6 ns        │ 10      │ 10";

        let results = parse_text(input);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "spanning::hello");
    }

    #[test]
    fn multiple_headers() {
        let input = "\
     Running benches/spanning.rs (target/release/deps/spanning-abc123)
spanning                       fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ test1                       1 ns          │ 2 ns          │ 1 ns          │ 1 ns          │ 5       │ 5

     Running benches/encoding_single.rs (target/release/deps/encoding_single-def456)
encoding_single                fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ test2                       3 ns          │ 4 ns          │ 3 ns          │ 3 ns          │ 5       │ 5";

        let results = parse_text(input);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "spanning::test1");
        assert_eq!(results[0].bench.as_deref(), Some("spanning"));
        assert_eq!(results[1].name, "encoding_single::test2");
        assert_eq!(results[1].bench.as_deref(), Some("encoding_single"));
    }
}
