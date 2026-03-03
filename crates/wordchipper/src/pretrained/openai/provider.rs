use crate::{
    UnifiedTokenVocab,
    WCError,
    WCResult,
    alloc::{
        sync::Arc,
        vec::Vec,
    },
    prelude::*,
    pretrained::{
        VocabDescription,
        VocabProvider,
        VocabProviderInventoryHook,
    },
    support::resources::ResourceLoader,
};

inventory::submit! {
    VocabProviderInventoryHook::new(|| Arc::new(OpenaiVocabProvider{}))
}

/// [`VocabProvider`] for `OpenAI` models.
pub struct OpenaiVocabProvider {}

impl VocabProvider for OpenaiVocabProvider {
    fn id(&self) -> String {
        "openai".to_string()
    }

    fn description(&self) -> String {
        "Pretrained vocabularies from OpenAI".to_string()
    }

    fn list_vocabs(&self) -> Vec<VocabDescription> {
        #[allow(unused_mut)]
        let mut vs: Vec<VocabDescription> = Default::default();

        #[cfg(feature = "datagym")]
        vs.push(VocabDescription {
            id: "gpt2".to_string(),
            context: vec!["openai".to_string(), "gpt2".to_string()],
            description: "GPT-2 `gpt2` vocabulary".to_string(),
        });

        #[cfg(feature = "std")]
        vs.extend_from_slice(&[
            VocabDescription {
                id: "r50k_base".to_string(),
                context: vec!["openai".to_string(), "r50k_base".to_string()],
                description: "GPT-2 `p50k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "p50k_base".to_string(),
                context: vec!["openai".to_string(), "p50k_base".to_string()],
                description: "GPT-2 `p50k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "p50k_edit".to_string(),
                context: vec!["openai".to_string(), "p50k_edit".to_string()],
                description: "GPT-2 `p50k_edit` vocabulary".to_string(),
            },
            VocabDescription {
                id: "cl100k_base".to_string(),
                context: vec!["openai".to_string(), "cl100k_base".to_string()],
                description: "GPT-3 `cl100k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "o200k_base".to_string(),
                context: vec!["openai".to_string(), "o200k_base".to_string()],
                description: "GPT-5 `o200k_base` vocabulary".to_string(),
            },
            VocabDescription {
                id: "o200k_harmony".to_string(),
                context: vec!["openai".to_string(), "o200k_harmony".to_string()],
                description: "GPT-5 `o200k_harmony` vocabulary".to_string(),
            },
        ]);

        vs
    }

    fn load_vocab(
        &self,
        name: &str,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)> {
        let _ = loader;
        let _descr = self.resolve_vocab(name)?;

        #[cfg(feature = "datagym")]
        {
            use super::load_gpt2_vocab;

            if name == "gpt2" {
                let vocab = load_gpt2_vocab(loader)?;
                return Ok((_descr, vocab.into()));
            }
        }

        #[cfg(feature = "std")]
        {
            use core::str::FromStr;

            use crate::pretrained::openai::OATokenizer;
            if let Ok(oat) = OATokenizer::from_str(name) {
                let vocab = oat.load_vocab(loader)?;
                return Ok((_descr, vocab.into()));
            }
        }

        Err(WCError::ResourceNotFound(name.to_string()))
    }
}
