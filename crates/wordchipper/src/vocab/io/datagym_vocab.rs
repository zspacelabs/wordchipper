//! # `DataGym` Vocabulary

use std::io::{
    BufRead,
    BufReader,
};

use serde_json::Value;

use crate::{
    TokenType,
    UnifiedTokenVocab,
    WCResult,
    prelude::*,
    pretrained::openai::{
        oa_r50k_base_spanning_config,
        resources::{
            OA_GPT2_ENCODER_JSON_KEYED_RESOURCE,
            OA_GPT2_VOCAB_BPE_KEYED_RESOURCE,
        },
    },
    support::resources::ResourceLoader,
    vocab::{
        SpanMapVocab,
        SpanTokenMap,
    },
};

/// A map from mojibake characters to their byte representation.
pub type MojibakeMap = crate::types::WCHashMap<char, u8>;

/// Trait for decoding mojibake characters.
pub trait MojibakeDecoder {
    /// Decode a string of mojibake characters into a byte vector.
    fn decode_mojibake(
        &self,
        value: &str,
    ) -> Vec<u8>;
}

impl MojibakeDecoder for MojibakeMap {
    fn decode_mojibake(
        &self,
        value: &str,
    ) -> Vec<u8> {
        value
            .chars()
            .map(|c| self.get(&c).copied().unwrap())
            .collect()
    }
}

/// Builds the default byte vocabulary and mojibake map for datagym
/// vocabularies.
///
/// Datagym was encoded using ISO/IEC 8859-1; and so the [`MojibakeMap`] is used
/// to translate scrambled ("mojibake") UTF-8 decoded characters into the
/// bytes they should have been.
fn datagym_base_maps() -> (MojibakeMap, SpanTokenMap<usize>) {
    let mut rank_to_byte: Vec<u8> = vec![];
    rank_to_byte.extend(0x21..=0x7E);
    rank_to_byte.extend(0xA1..0xAD);
    rank_to_byte.extend(0xAE..=0xFF);

    // This map translates mojibake utf8-decoded characters
    // to the character they would have decoded to under ISO/IEC 8859-1.
    //
    // We begin by back-filling the initial contents of rank_to_byte:
    // mojibake
    let mut mojibake_map: MojibakeMap = rank_to_byte.iter().map(|&b| (char::from(b), b)).collect();

    let mut n = 0u32;
    for b in 0..=255 {
        if !rank_to_byte.contains(&b) {
            // b does not map to a printable byte.

            // Assign b the next available rank:
            rank_to_byte.push(b);

            // Add the mojibake key for this utf-8 byte:
            let c = char::from_u32(256 + n).unwrap();
            mojibake_map.insert(c, b);

            n += 1;
        }
    }
    assert_eq!(n, 68);
    assert_eq!(rank_to_byte.len(), 256);
    assert_eq!(mojibake_map.len(), 256);

    // add the single byte tokens
    let span_tokens: SpanTokenMap<usize> = rank_to_byte
        .into_iter()
        .enumerate()
        .map(|(i, b)| (vec![b], i))
        .collect();

    (mojibake_map, span_tokens)
}

/// Read a daty gym "vocab.bpe" file.
///
/// Handle extended ascii (<https://en.wikipedia.org/wiki/Extended_ASCII>)
/// Assume ISO/IEC 8859-1 (<https://en.wikipedia.org/wiki/ISO/IEC_8859-1>)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn read_datagym_vocab_bpe<R>(
    vocab_bpe_reader: R
) -> WCResult<(MojibakeMap, SpanTokenMap<usize>)>
where
    R: BufRead,
{
    let (mojibake_map, mut span_map) = datagym_base_maps();

    let mut bpe_merges: Vec<(String, String)> = vec![];
    for line in vocab_bpe_reader.lines().skip(1) {
        let line = line?;

        if let Some((first, second)) = line.split_once(' ') {
            bpe_merges.push((first.to_string(), second.to_string()))
        }
    }

    let mut n = span_map.len();
    for (first, second) in bpe_merges {
        let mut key = mojibake_map.decode_mojibake(first.as_str());
        key.extend(mojibake_map.decode_mojibake(second.as_str()));
        span_map.insert(key, n);
        n += 1
    }

    Ok((mojibake_map, span_map))
}

/// Parse a data gym "encoder.json" file from contents.
pub fn read_datagym_encoder_json<R>(
    encoder_json_reader: R,
    mojibake_map: &MojibakeMap,
) -> WCResult<SpanTokenMap<usize>>
where
    R: BufRead,
{
    // check that the encoder file matches the merges file
    // this sanity check is important since tiktoken assumes that ranks are ordered
    // the same as merge priority
    let encoder_json: Value = serde_json::from_reader(encoder_json_reader)
        .unwrap_or(Value::Object(serde_json::Map::default()));
    let mut encoder_json_loaded: SpanTokenMap<usize> = encoder_json
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, val)| {
            (
                mojibake_map.decode_mojibake(key),
                val.as_u64().unwrap() as usize,
            )
        })
        .collect();
    encoder_json_loaded.remove("<|endoftext|>".as_bytes());
    encoder_json_loaded.remove("<|endoftext|>".as_bytes());

    Ok(encoder_json_loaded)
}

/// Handle extended ascii (<https://en.wikipedia.org/wiki/Extended_ASCII>)
/// Assume ISO/IEC 8859-1 (<https://en.wikipedia.org/wiki/ISO/IEC_8859-1>)
/// non-whitespace printable character range:
/// [0x21-0x7E], [0xA1-0xAD), (0xAD-0xFF]
pub fn read_datagym_vocab(
    vocab_bpe_reader: &mut dyn BufRead,
    encoder_json_reader: &mut dyn BufRead,
    clobber_one_byte_tokens: bool,
) -> WCResult<SpanTokenMap<usize>> {
    let (mojibake_map, mut span_map) = read_datagym_vocab_bpe(vocab_bpe_reader)?;

    let encoder_json_loaded = read_datagym_encoder_json(encoder_json_reader, &mojibake_map)?;
    if clobber_one_byte_tokens {
        for (k, v) in &encoder_json_loaded {
            if k.len() == 1 {
                span_map.insert(k.clone(), *v);
            }
        }
    }

    assert_eq!(span_map.len(), encoder_json_loaded.len());

    Ok(span_map)
}

/// Load the `DataGym` GPT-2 span map vocabulary.
pub fn load_gpt2_vocab<T: TokenType>(
    loader: &mut dyn ResourceLoader
) -> WCResult<UnifiedTokenVocab<T>> {
    let vocab_path = loader.load_resource_path(&OA_GPT2_VOCAB_BPE_KEYED_RESOURCE.into())?;
    let encoder_path = loader.load_resource_path(&OA_GPT2_ENCODER_JSON_KEYED_RESOURCE.into())?;

    let vocab_file = std::fs::File::open(vocab_path)?;
    let encoder_file = std::fs::File::open(encoder_path)?;

    let mut vocab_reader = BufReader::new(vocab_file);
    let mut encoder_reader = BufReader::new(encoder_file);

    let span_map = read_datagym_vocab(&mut vocab_reader, &mut encoder_reader, false)?;

    UnifiedTokenVocab::from_span_vocab(
        oa_r50k_base_spanning_config(),
        SpanMapVocab::from_span_map(span_map).to_token_type()?,
    )
}

#[cfg(test)]
mod tests {
    use std::{
        println,
        sync::Arc,
    };

    use wordchipper_disk_cache::WordchipperDiskCache;

    use super::*;
    use crate::{
        TokenEncoderOptions,
        UnifiedTokenVocab,
        encoders::testing::common_encoder_tests,
    };

    #[test]
    #[cfg(all(feature = "std", feature = "datagym", feature = "download"))]
    fn test_load_gpt2_vocab() {
        let mut disk_cache: WordchipperDiskCache = Default::default();
        let vocab: Arc<UnifiedTokenVocab<u32>> = load_gpt2_vocab(&mut disk_cache).unwrap().into();
        let encoder = TokenEncoderOptions::default().build(vocab.clone());

        println!("{}", vocab.spanning().pattern().as_str());
        common_encoder_tests(vocab, encoder);
    }
}
