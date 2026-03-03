use crate::{
    WCError,
    WCResult,
    alloc::{
        sync::Arc,
        vec::Vec,
    },
    prelude::*,
    pretrained::{
        LabeledVocab,
        VocabProvider,
        VocabProviderInventoryHook,
        vocab_description::VocabDescription,
        vocab_query::VocabQuery,
    },
    support::resources::ResourceLoader,
};

inventory::submit! {
    VocabProviderInventoryHook::new(|| Arc::new(OpenaiVocabProvider{}))
}

/// [`VocabProvider`] for `OpenAI` models.
pub struct OpenaiVocabProvider {}

impl VocabProvider for OpenaiVocabProvider {
    fn name(&self) -> String {
        "openai".to_string()
    }

    fn description(&self) -> String {
        "Pretrained vocabularies from OpenAI".to_string()
    }

    fn list_vocabs(&self) -> Vec<VocabDescription> {
        #[allow(unused_mut)]
        let mut vs: Vec<VocabDescription> = Default::default();

        #[cfg(feature = "datagym")]
        vs.push(VocabDescription::new(
            "openai:gpt2",
            &["openai", "gpt2"],
            "GPT-2 `gpt2` vocabulary",
        ));

        #[cfg(feature = "std")]
        vs.extend_from_slice(&[
            VocabDescription::new(
                "openai:r50k_base",
                &["openai", "r50k_base"],
                "GPT-2 `p50k_base` vocabulary",
            ),
            VocabDescription::new(
                "openai:p50k_base",
                &["openai", "p50k_base"],
                "GPT-2 `p50k_base` vocabulary",
            ),
            VocabDescription::new(
                "openai:p50k_edit",
                &["openai", "p50k_edit"],
                "GPT-2 `p50k_edit` vocabulary",
            ),
            VocabDescription::new(
                "openai:cl100k_base",
                &["openai", "cl100k_base"],
                "GPT-3 `cl100k_base` vocabulary",
            ),
            VocabDescription::new(
                "openai:o200k_base",
                &["openai", "o200k_base"],
                "GPT-5 `o200k_base` vocabulary",
            ),
            VocabDescription::new(
                "openai:o200k_harmony",
                &["openai", "o200k_harmony"],
                "GPT-5 `o200k_harmony` vocabulary",
            ),
        ]);
        vs
    }

    fn load_vocab(
        &self,
        query: &VocabQuery,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>> {
        let descr = self.resolve_vocab(query)?;

        #[cfg(feature = "datagym")]
        {
            use super::load_gpt2_vocab;

            if descr.id().name() == "gpt2" {
                let vocab = load_gpt2_vocab(loader)?;
                return Ok(LabeledVocab::new(descr, vocab.into()));
            }
        }

        #[cfg(feature = "std")]
        {
            use core::str::FromStr;

            use crate::pretrained::openai::OATokenizer;
            if let Ok(oat) = OATokenizer::from_str(descr.id().name()) {
                let vocab = oat.load_vocab(loader)?;
                return Ok(LabeledVocab::new(descr, vocab.into()));
            }
        }

        Err(WCError::ResourceNotFound(query.to_string()))
    }
}
