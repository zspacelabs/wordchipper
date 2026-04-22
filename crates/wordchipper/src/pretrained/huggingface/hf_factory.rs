use tokenizers::{
    ModelWrapper::BPE,
    PreTokenizerWrapper,
    PreTokenizerWrapper::{
        ByteLevel,
        Sequence,
        Split,
    },
    pre_tokenizers::split::SplitPattern,
    tokenizer::Tokenizer,
};

use crate::{
    LabeledVocab,
    UnifiedTokenVocab,
    VocabDescription,
    VocabQuery,
    WCError,
    WCHashMap,
    WCHashSet,
    WCResult,
    alloc::sync::Arc,
    prelude::*,
    pretrained::factory::{
        VocabProvider,
        VocabProviderInventoryHook,
    },
    spanners::TextSpanningConfig,
    support::{
        regex::RegexPattern,
        resources::ResourceLoader,
    },
    vocab::{
        ByteMapVocab,
        SpanMapVocab,
        SpanTokenMap,
    },
};

pub struct HFVocabProvider {}

inventory::submit! {
    VocabProviderInventoryHook::new(|| Arc::new(HFVocabProvider{}))
}

impl VocabProvider for HFVocabProvider {
    fn name(&self) -> String {
        "hf".to_string()
    }

    fn description(&self) -> String {
        "HuggingFace vocabularies".to_string()
    }

    fn list_vocabs(&self) -> Vec<VocabDescription> {
        vec![]
    }

    fn load_vocab(
        &self,
        query: &VocabQuery,
        _loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>> {
        match query.schema() {
            None => return Err(WCError::ResourceNotFound(query.to_string())),
            Some(schema) => {
                if schema != "hf" {
                    return Err(WCError::ResourceNotFound(query.to_string()));
                }
            }
        }

        type T = u32;

        let key = format!("{}/{}", query.path().unwrap(), query.name());
        let id = format!("{}/{}", query.path().unwrap(), query.name());

        match Tokenizer::from_pretrained(&key, None) {
            Ok(tok) => {
                let pattern = extract_pattern(tok.get_pre_tokenizer())?;
                let span_config = TextSpanningConfig::from_pattern(RegexPattern::Adaptive(pattern));

                let BPE(bpe) = tok.get_model() else {
                    return Err(WCError::External(
                        format!("{} is not BPE compatible", id).to_string(),
                    ));
                };

                // TODO: Add support for unknown token.
                if let Some(unk) = bpe.get_unk_token() {
                    return Err(WCError::External(format!("BPE has unk_token {unk:?}")));
                }

                let vocab = bpe.get_vocab();

                let specials: WCHashSet<u32> =
                    tok.get_added_tokens_decoder().keys().copied().collect();

                // Forward and inverse bytes_to_unicode maps.
                let b2c = bytes_char();
                let c2b: WCHashMap<char, u8> = b2c.iter().map(|(&b, &c)| (c, b)).collect();

                // Span map: decode every non-special vocab string back to bytes.
                let mut span_map: SpanTokenMap<T> = SpanTokenMap::default();
                for (s, id) in &vocab {
                    if specials.contains(id) {
                        continue;
                    }
                    let mut bytes = Vec::with_capacity(s.len());
                    for ch in s.chars() {
                        match c2b.get(&ch) {
                            Some(&b) => bytes.push(b),
                            None => {
                                return Err(WCError::External(format!(
                                    "token {s:?} (id {id}) has non-byte-level codepoint {ch:?}"
                                )));
                            }
                        }
                    }
                    span_map.insert(bytes, (*id));
                }

                // Byte map: the single-char string for each byte must resolve in the vocab.
                let byte_tokens: Vec<T> = (0u8..=255)
                    .map(|b| {
                        let key: String = std::iter::once(b2c[&b]).collect();
                        vocab.get(&key).copied().ok_or(b)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|b| WCError::External(format!("missing byte token for 0x{b:02x}")))?;

                let byte_map = ByteMapVocab::<T>::from_byte_to_token(&byte_tokens);
                let span_vocab = SpanMapVocab::<T>::new(byte_map, span_map)?;

                let vocab: Arc<UnifiedTokenVocab<T>> =
                    Arc::new(UnifiedTokenVocab::from_span_vocab(span_config, span_vocab)?);

                let id = VocabQuery::new(Some("hf"), query.path(), query.name());

                let descr: VocabDescription = VocabDescription::new(
                    id,
                    &["hf", query.path().unwrap(), query.name()],
                    "Model loaded from hf",
                );

                Ok(LabeledVocab::new(descr, vocab))
            }
            Err(_) => Err(WCError::ResourceNotFound(query.to_string())),
        }
    }
}
// GPT-2 / r50k-style default, used when pretokenizer is a bare ByteLevel.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn extract_pattern(pt: Option<&PreTokenizerWrapper>) -> Result<String, WCError> {
    fn split_regex(s: &tokenizers::pre_tokenizers::split::Split) -> Result<String, WCError> {
        match &s.pattern {
            SplitPattern::Regex(r) => Ok(r.clone()),
            _ => Err(WCError::External("Split without Regex pattern".into())),
        }
    }
    match pt {
        Some(Split(s)) => split_regex(s),
        Some(ByteLevel(_)) => Ok(GPT2_PATTERN.to_string()),
        Some(Sequence(seq)) => {
            let mut found = None;
            for sub in seq.as_ref() {
                match &sub {
                    Split(s) => {
                        if found.is_some() {
                            return Err(WCError::External("Sequence has multiple Splits".into()));
                        }
                        found = Some(split_regex(s)?);
                    }
                    ByteLevel(_) => {} // sibling byte-encoder, fine
                    _ => return Err(WCError::External("unsupported member in Sequence".into())),
                }
            }
            found.ok_or_else(|| WCError::External("Sequence has no Split regex".into()))
        }
        Some(_) => Err(WCError::External("unsupported pre-tokenizer".into())),
        None => Err(WCError::External("no pre-tokenizer".into())),
    }
}

/// Converts bytes to unicode characters.
/// See <https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9>
fn bytes_char() -> WCHashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }

    // Safety: cs contains all values from bs (between 0 and 255),
    // and some values of value 2⁸ + n, where n is between 0 and 255. This is
    // between 255 and 512. Both ranges are valid UTF-32 values (which is fully
    // saturated until 0xD000)
    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}
