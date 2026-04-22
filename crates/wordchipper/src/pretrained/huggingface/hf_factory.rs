use std::println;

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
    VocabIndex,
    VocabQuery,
    WCError,
    WCHashMap,
    WCHashSet,
    WCResult,
    alloc::sync::Arc,
    prelude::*,
    pretrained::{
        factory::{
            VocabProvider,
            VocabProviderInventoryHook,
        },
        openai::OA_GPT2_PATTERN,
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

fn extract_pattern(pt: Option<&PreTokenizerWrapper>) -> Result<RegexPattern, WCError> {
    fn split_regex(s: &tokenizers::pre_tokenizers::split::Split) -> Result<RegexPattern, WCError> {
        match &s.pattern {
            SplitPattern::Regex(r) => Ok(r.clone().into()),
            _ => Err(WCError::External("Split without Regex pattern".into())),
        }
    }
    match pt {
        Some(Split(s)) => split_regex(s),
        Some(ByteLevel(bl)) if bl.use_regex => Ok(OA_GPT2_PATTERN.into()),
        Some(ByteLevel(_)) => Err(WCError::External(
            "ByteLevel with use_regex=false has no splitting regex".into(),
        )),
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

/// Converts bytes to Unicode characters.
/// See <https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9>
///
/// This is from tokenizers; but is private in that crate.
///
/// TODO: Workout what this is doing, relative to the bytemap.
/// This seems to be some default map for gpt2; and might be shared
/// with the `BytMap` code for loading datagym.
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

/// Attempt to convert a `HuggingFace` tokenizer to a `WordChipper` vocabulary.
pub fn vocab_from_hf_tokenizer(tok: &Tokenizer) -> WCResult<Arc<UnifiedTokenVocab<u32>>> {
    type T = u32;

    let pattern = extract_pattern(tok.get_pre_tokenizer())?;
    let mut span_config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(pattern);

    let BPE(bpe) = tok.get_model() else {
        return Err(WCError::External(
            "Tokenizer is not BPE compatible".to_string(),
        ));
    };

    // TODO: Add support for unknown token.
    if let Some(unk) = bpe.get_unk_token() {
        return Err(WCError::External(format!("BPE has unk_token {unk:?}")));
    }

    let hf_vocab = bpe.get_vocab();

    println!(
        "Debug: {:?}",
        hf_vocab.iter().find(|(_, id)| **id == 157513)
    );

    // TODO: This is broken for Qwen/Qwen3.5-9B for some reason.
    let mut special_tokens: WCHashSet<T> = Default::default();

    let decoder = tok.get_added_tokens_decoder();
    println!("Debug: {:#?}", decoder);

    for (t, at) in decoder.iter() {
        span_config.specials_mut().add_str_word(&at.content, *t);
        special_tokens.insert(*t);
    }

    // Forward and inverse bytes_to_unicode maps.
    let b2c = bytes_char();
    let c2b: WCHashMap<char, u8> = b2c.iter().map(|(&b, &c)| (c, b)).collect();

    // Span map: decode every non-special vocab string back to bytes.
    let mut span_map: SpanTokenMap<T> = SpanTokenMap::default();
    for (s, id) in &hf_vocab {
        if special_tokens.contains(id) {
            continue;
        } else {
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
            span_map.insert(bytes, *id);
        }
    }

    if span_config.specials().len() != special_tokens.len() {
        return Err(WCError::External(format!(
            "hf vocab identifies {} special tokens, but only {} special tokens found in span_config",
            special_tokens.len(),
            span_config.specials().len()
        )));
    }

    // Byte map: the single-char string for each byte must resolve in the vocab.
    let byte_tokens: Vec<T> = (0u8..=255)
        .map(|b| {
            let key: String = std::iter::once(b2c[&b]).collect();
            hf_vocab.get(&key).copied().ok_or(b)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|b| WCError::External(format!("missing byte token for 0x{b:02x}")))?;

    let byte_map = ByteMapVocab::<T>::from_byte_to_token(&byte_tokens);
    let span_vocab = SpanMapVocab::<T>::new(byte_map, span_map)?;

    let expected_len = span_vocab.len() + span_config.specials().len();

    let vocab: Arc<UnifiedTokenVocab<T>> =
        Arc::new(UnifiedTokenVocab::from_span_vocab(span_config, span_vocab)?);

    // TODO: should `vocab.len()` include the special len()?
    if vocab.len() + vocab.special_vocab().len() != expected_len {
        return Err(WCError::External(format!(
            "Expected {} tokens, got {}",
            expected_len,
            vocab.len()
        )));
    }

    Ok(vocab)
}

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
        if let Some(schema) = query.schema()
            && schema != "hf"
        {
            return Err(WCError::ResourceNotFound(query.to_string()));
        }

        match Tokenizer::from_pretrained(query.clone().with_schema(None).to_string(), None) {
            Ok(tok) => {
                let vocab = vocab_from_hf_tokenizer(&tok)?;

                let mut context = vec!["hf"];
                if query.path().is_some() {
                    context.push(query.path().unwrap());
                }
                context.push(query.name());

                let id = query.clone().with_schema(Some("hf"));
                let context = id.to_context();

                let descr: VocabDescription =
                    VocabDescription::new(id, &context, "Model loaded from hf");

                Ok(LabeledVocab::new(descr, vocab))
            }
            Err(_) => Err(WCError::ResourceNotFound(query.to_string())),
        }
    }
}
