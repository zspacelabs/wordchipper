use tokenizers::{
    ModelWrapper::BPE,
    PreTokenizerWrapper::Split,
    pre_tokenizers::split::SplitPattern,
    tokenizer::Tokenizer,
};

use crate::{
    LabeledVocab,
    UnifiedTokenVocab,
    VocabDescription,
    VocabQuery,
    WCError,
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
        type T = u32;

        let key = format!("{}/{}", query.path().unwrap(), query.name());
        match Tokenizer::from_pretrained(&key, None) {
            Ok(tok) => {
                let span_config = if let Some(Split(split)) = tok.get_pre_tokenizer() {
                    let pattern = match &split.pattern {
                        SplitPattern::Regex(str) => RegexPattern::Adaptive(str.to_string()),
                        _ => return Err(WCError::External("No regex pattern".to_string())),
                    };

                    TextSpanningConfig::from_pattern(pattern)
                } else {
                    return Err(WCError::External("No pre-tokenizer".to_string()));
                };

                if let BPE(bpe) = tok.get_model() {
                    let vocab = bpe.get_vocab();

                    let span_map: SpanTokenMap<T> = vocab
                        .iter()
                        .map(|(s, t)| (s.as_bytes().to_vec(), *t))
                        .collect();

                    let byte_map: ByteMapVocab<T> = if let Some(byte_tokens) = (0..256)
                        .map(|b| {
                            let k = format!("<{b:#04X}>");
                            vocab.get(&k).copied()
                        })
                        .collect::<Option<Vec<T>>>()
                    {
                        ByteMapVocab::<T>::from_byte_to_token(&byte_tokens)
                    } else {
                        return Err(WCError::External(
                            "Unable to translate: no byte map".to_string(),
                        ));
                    };

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
                } else {
                    Err(WCError::ResourceNotFound(query.to_string()))
                }
            }
            Err(_) => Err(WCError::ResourceNotFound(query.to_string())),
        }
    }
}
