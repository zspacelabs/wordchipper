//! # Tiktoken Vocabulary IO

use std::{
    fs::File,
    io::{
        BufRead,
        BufReader,
        BufWriter,
        Write,
    },
    path::Path,
};

use base64::{
    Engine,
    prelude::BASE64_STANDARD,
};

use crate::{
    TokenType,
    WCError,
    WCResult,
    prelude::*,
    spanners::TextSpanningConfig,
    vocab::{
        SpanMapVocab,
        UnifiedTokenVocab,
        vocab_types::SpanTokenMap,
    },
};

/// Build a [`UnifiedTokenVocab`] from a pretrained bas64 vocab file.
///
/// ## Arguments
/// * `data_path` - path to the file.
/// * `pattern` - the word split pattern.
/// * `special_tokens` - the special tokens.
pub fn load_base64_unified_vocab_path<T: TokenType>(
    path: impl AsRef<Path>,
    spanning: TextSpanningConfig<T>,
) -> WCResult<UnifiedTokenVocab<T>> {
    let mut reader = BufReader::new(File::open(path)?);
    read_base64_unified_vocab(&mut reader, spanning)
}

/// Build a [`UnifiedTokenVocab`] from a pretrained bas64 vocab file.
///
/// ## Arguments
/// * `data_path` - path to the file.
/// * `pattern` - the word split pattern.
/// * `special_tokens` - the special tokens.
pub fn read_base64_unified_vocab<T: TokenType>(
    reader: &mut dyn BufRead,
    spanning: TextSpanningConfig<T>,
) -> WCResult<UnifiedTokenVocab<T>> {
    UnifiedTokenVocab::from_span_vocab(spanning, read_base64_span_map(reader)?.into())
}

/// Load a [`SpanMapVocab`] from a base64 vocab file.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_base64_span_vocab_path<T, P>(path: P) -> WCResult<SpanMapVocab<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    Ok(load_base64_span_map_path(path)?.into())
}

/// Load a [`SpanTokenMap`] from a base64 vocab file.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `path` - the path to the vocabulary file.
pub fn load_base64_span_map_path<T, P>(path: P) -> WCResult<SpanTokenMap<T>>
where
    T: TokenType,
    P: AsRef<Path>,
{
    let mut reader = BufReader::new(File::open(path)?);
    read_base64_span_map(&mut reader)
}

/// Read a [`SpanTokenMap`] from a base64 vocab line reader.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `span_map` - the vocabulary to extend.
/// * `reader` - the line reader.
pub fn read_base64_span_map<T>(reader: &mut dyn BufRead) -> WCResult<SpanTokenMap<T>>
where
    T: TokenType,
{
    let mut vocab = SpanTokenMap::default();

    let stream = reader.lines();
    for line in stream {
        let line = line?;
        let s: &str = line.as_ref();

        let parts = s.splitn(2, ' ').collect::<Vec<&str>>();
        assert_eq!(parts.len(), 2);

        let span = BASE64_STANDARD
            .decode(parts[0])
            .map_err(|e| WCError::Parse(e.to_string()))?;

        let id: u64 = parts[1]
            .parse()
            .map_err(|e: core::num::ParseIntError| WCError::Parse(e.to_string()))?;
        let token = T::from_u64(id).ok_or(WCError::TokenOutOfRange)?;

        vocab.insert(span, token);
    }

    Ok(vocab)
}

/// Save a [`SpanTokenMap`] to a base64 vocab file.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `span_map` - the vocabulary to save.
/// * `path` - the path to save the vocabulary to.
pub fn save_base64_span_map_path<T: TokenType, P: AsRef<Path>>(
    span_map: &SpanTokenMap<T>,
    path: P,
) -> WCResult<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    write_base64_span_map(span_map, &mut writer)
}

/// Write a [`SpanTokenMap`] to a [`Write`] writer.
///
/// Lines are:
/// ```terminaloutput
/// {BASE64 SPAN} {TOKEN}
/// ```
///
/// # Arguments
/// * `span_map` - the vocabulary to save.
/// * `writer` - the writer to target.
pub fn write_base64_span_map<T>(
    span_map: &SpanTokenMap<T>,
    writer: &mut dyn Write,
) -> WCResult<()>
where
    T: TokenType,
{
    let mut items: Vec<(T, &Vec<u8>)> = span_map
        .iter()
        .map(|(chunk, &token)| (token, chunk))
        .collect();
    items.sort_by_key(|(t, _)| *t);

    for (token, chunk) in items {
        writeln!(
            writer,
            "{} {}",
            BASE64_STANDARD.encode(chunk),
            token.to_u64().unwrap()
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_tiktoken() {
        type T = u32;

        let mut span_map: SpanTokenMap<T> = Default::default();
        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("banana".as_bytes().to_vec(), 301);
        span_map.insert("pear".as_bytes().to_vec(), 302);

        tempdir::TempDir::new("vocab_test")
            .and_then(|dir| {
                let path = dir.path().join("vocab.tiktoken");

                save_base64_span_map_path(&span_map, &path).expect("Failed to save vocab");

                let loaded_vocab = load_base64_span_map_path(&path).expect("Failed to load vocab");

                assert_eq!(&loaded_vocab, &span_map);

                Ok(())
            })
            .unwrap();
    }
}
