//! # `OpenAI` Pretrained Vocabulary Loaders

#[cfg(feature = "std")]
use std::io::BufRead;
#[cfg(feature = "std")]
use std::path::Path;

#[allow(unused_imports)]
use crate::TokenType;
#[allow(unused_imports)]
use crate::UnifiedTokenVocab;
#[allow(unused_imports)]
use crate::prelude::*;
#[allow(unused_imports)]
use crate::pretrained::openai::OA_CL100K_BASE_PATTERN;
#[allow(unused_imports)]
use crate::pretrained::openai::OA_O200K_BASE_PATTERN;
#[allow(unused_imports)]
use crate::pretrained::openai::OA_P50K_BASE_PATTERN;
#[allow(unused_imports)]
use crate::pretrained::openai::OA_R50K_BASE_PATTERN;
#[allow(unused_imports)]
use crate::pretrained::openai::resources::OA_CL100K_BASE_TIKTOKEN_RESOURCE;
#[allow(unused_imports)]
use crate::pretrained::openai::resources::OA_O200K_BASE_TIKTOKEN_RESOURCE;
#[allow(unused_imports)]
use crate::pretrained::openai::resources::OA_P50K_BASE_TIKTOKEN_RESOURCE;
#[allow(unused_imports)]
use crate::pretrained::openai::resources::OA_R50K_BASE_TIKTOKEN_RESOURCE;
#[allow(unused_imports)]
use crate::pretrained::openai::specials::oa_cl100k_edit_special_tokens;
#[allow(unused_imports)]
use crate::pretrained::openai::specials::oa_o200k_base_special_tokens;
#[allow(unused_imports)]
use crate::pretrained::openai::specials::oa_o200k_harmony_special_tokens;
#[allow(unused_imports)]
use crate::pretrained::openai::specials::oa_p50k_base_special_tokens;
#[allow(unused_imports)]
use crate::pretrained::openai::specials::oa_p50k_edit_special_tokens;
#[allow(unused_imports)]
use crate::pretrained::openai::specials::oa_r50k_base_special_tokens;
#[allow(unused_imports)]
use crate::spanners::TextSpanningConfig;
#[allow(unused_imports)]
use crate::support::regex::RegexPattern;
#[allow(unused_imports)]
use crate::support::resources::ConstKeyedResource;
#[cfg(feature = "std")]
use crate::support::resources::ResourceLoader;
#[allow(unused_imports)]
use crate::vocab::utility::factories::ConstVocabularyFactory;

/// Load the `DataGym` GPT-2 span map vocabulary.
#[cfg(all(feature = "std", feature = "datagym"))]
pub fn load_gpt2_vocab<T: TokenType>(
    loader: &mut dyn ResourceLoader
) -> crate::WCResult<crate::UnifiedTokenVocab<T>> {
    use std::io::BufReader;

    use crate::{
        pretrained::openai::{
            oa_r50k_base_spanning_config,
            resources::{
                OA_GPT2_ENCODER_JSON_KEYED_RESOURCE,
                OA_GPT2_VOCAB_BPE_KEYED_RESOURCE,
            },
        },
        vocab::{
            SpanMapVocab,
            io::read_datagym_vocab,
        },
    };

    let vocab_path = loader.load_resource_path(&OA_GPT2_VOCAB_BPE_KEYED_RESOURCE.into())?;
    let mut vocab_reader = BufReader::new(std::fs::File::open(vocab_path)?);

    let encoder_path = loader.load_resource_path(&OA_GPT2_ENCODER_JSON_KEYED_RESOURCE.into())?;
    let mut encoder_reader = BufReader::new(std::fs::File::open(encoder_path)?);

    let span_map = read_datagym_vocab(&mut vocab_reader, &mut encoder_reader, false)?;

    UnifiedTokenVocab::from_span_vocab(
        oa_r50k_base_spanning_config(),
        SpanMapVocab::from_span_map(span_map).to_token_type()?,
    )
}

/// `OpenAI` Pretrained Tokenizer types.
#[derive(Clone, Copy, Debug, PartialEq, strum::EnumString, strum::EnumIter, strum::Display)]
#[non_exhaustive]
pub enum OATokenizer {
    /// GPT-2 "`r50k_base`" tokenizer.
    #[strum(serialize = "r50k_base")]
    R50kBase,

    /// GPT-2 "`p50k_base`" tokenizer.
    #[strum(serialize = "p50k_base")]
    P50kBase,

    /// GPT-2 "`p50k_edit`" tokenizer.
    #[strum(serialize = "p50k_edit")]
    P50kEdit,

    /// GPT-3 "`cl100k_base`" tokenizer.
    #[strum(serialize = "cl100k_base")]
    Cl100kBase,

    /// GPT-5 "`o200k_base`" tokenizer.
    #[strum(serialize = "o200k_base")]
    O200kBase,

    /// GPT-5 "`o200k_harmony`" tokenizer.
    #[strum(serialize = "o200k_harmony")]
    O200kHarmony,
}

impl OATokenizer {
    /// Get the tokenizer vocabulary factory.
    pub fn factory(&self) -> &ConstVocabularyFactory {
        use OATokenizer::*;
        match self {
            R50kBase => &OA_R50K_BASE_VOCAB_FACTORY,
            P50kBase => &OA_P50K_BASE_VOCAB_FACTORY,
            P50kEdit => &OA_P50K_EDIT_VOCAB_FACTORY,
            Cl100kBase => &OA_CL100K_BASE_VOCAB_FACTORY,
            O200kBase => &OA_O200K_BASE_VOCAB_FACTORY,
            O200kHarmony => &OA_O200K_HARMONY_VOCAB_FACTORY,
        }
    }

    /// Get the tokenizer regex pattern.
    pub fn pattern(&self) -> RegexPattern {
        self.factory().pattern()
    }

    /// Get the tokenizer special tokens.
    pub fn special_tokens<T: TokenType>(&self) -> Vec<(String, T)> {
        self.factory().special_tokens()
    }

    /// Get the tokenizer spanners config.
    pub fn spanning_config<T: TokenType>(&self) -> TextSpanningConfig<T> {
        self.factory().spanning_config()
    }

    /// Load pretrained `OpenAI` tokenizer vocabulary.
    ///
    /// Downloads and caches resources using the `disk_cache`.
    #[cfg(feature = "std")]
    pub fn load_vocab<T: TokenType>(
        &self,
        loader: &mut dyn ResourceLoader,
    ) -> crate::WCResult<crate::UnifiedTokenVocab<T>> {
        self.factory().load_vocab(loader)
    }

    /// Load pretrained `OpenAI` tokenizer vocabulary from disk.
    #[cfg(feature = "std")]
    pub fn load_path<T: TokenType>(
        &self,
        path: impl AsRef<Path>,
    ) -> crate::WCResult<crate::UnifiedTokenVocab<T>> {
        self.factory().load_vocab_path(path)
    }

    /// Read pretrained `OpenAI` tokenizer vocabulary from a reader.
    #[cfg(feature = "std")]
    pub fn read_vocab<T: TokenType>(
        &self,
        reader: &mut dyn BufRead,
    ) -> crate::WCResult<crate::UnifiedTokenVocab<T>> {
        self.factory().read_vocab(reader)
    }
}

/// Shared download context key.
const OA_KEY: &str = "openai";

/// The "`r50k_base`" tokenizer.
pub const OA_R50K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "r50k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "r50k_base"],
        resource: OA_R50K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_R50K_BASE_PATTERN,
    special_builder: &oa_r50k_base_special_tokens,
};

/// The "`p50k_base`" tokenizer.
pub const OA_P50K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "p50k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "p50k_base"],
        resource: OA_P50K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_P50K_BASE_PATTERN,
    special_builder: &oa_p50k_base_special_tokens,
};

/// The "`p50k_edit`" tokenizer.
pub const OA_P50K_EDIT_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "p50k_edit",
    resource: OA_P50K_BASE_VOCAB_FACTORY.resource,
    pattern: OA_P50K_BASE_VOCAB_FACTORY.pattern,
    special_builder: &oa_p50k_edit_special_tokens,
};

/// The "`cl100k_base`" tokenizer.
pub const OA_CL100K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "cl100k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "cl100k_base"],
        resource: OA_CL100K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_CL100K_BASE_PATTERN,
    special_builder: &oa_cl100k_edit_special_tokens,
};

/// The "`o200k_base`" tokenizer.
pub const OA_O200K_BASE_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "o200k_base",
    resource: ConstKeyedResource {
        key: &[OA_KEY, "o200k_base"],
        resource: OA_O200K_BASE_TIKTOKEN_RESOURCE,
    },
    pattern: OA_O200K_BASE_PATTERN,
    special_builder: &oa_o200k_base_special_tokens,
};

/// The "`o200k_harmony`" tokenizer.
pub const OA_O200K_HARMONY_VOCAB_FACTORY: ConstVocabularyFactory = ConstVocabularyFactory {
    name: "o200k_harmony",
    resource: OA_O200K_BASE_VOCAB_FACTORY.resource,
    pattern: OA_O200K_BASE_VOCAB_FACTORY.pattern,
    special_builder: &oa_o200k_harmony_special_tokens,
};

#[cfg(test)]
mod test {
    #[test]
    fn test_oa_tokenizer() {
        use core::str::FromStr;

        use super::*;

        assert_eq!(OATokenizer::R50kBase.to_string(), "r50k_base");
        assert_eq!(OATokenizer::P50kBase.to_string(), "p50k_base");
        assert_eq!(OATokenizer::P50kEdit.to_string(), "p50k_edit");
        assert_eq!(OATokenizer::Cl100kBase.to_string(), "cl100k_base");
        assert_eq!(OATokenizer::O200kBase.to_string(), "o200k_base");
        assert_eq!(OATokenizer::O200kHarmony.to_string(), "o200k_harmony");

        assert_eq!(
            OATokenizer::from_str("r50k_base").unwrap(),
            OATokenizer::R50kBase
        );
        assert_eq!(
            OATokenizer::from_str("p50k_base").unwrap(),
            OATokenizer::P50kBase
        );
        assert_eq!(
            OATokenizer::from_str("p50k_edit").unwrap(),
            OATokenizer::P50kEdit
        );
        assert_eq!(
            OATokenizer::from_str("cl100k_base").unwrap(),
            OATokenizer::Cl100kBase
        );
        assert_eq!(
            OATokenizer::from_str("o200k_base").unwrap(),
            OATokenizer::O200kBase
        );
        assert_eq!(
            OATokenizer::from_str("o200k_harmony").unwrap(),
            OATokenizer::O200kHarmony
        );
    }

    #[test]
    #[cfg(all(feature = "std", feature = "datagym", feature = "download"))]
    fn test_load_gpt2_vocab() {
        use crate::{
            TokenEncoder,
            TokenEncoderOptions,
            UnifiedTokenVocab,
            alloc::sync::Arc,
            encoders::testing::common_encoder_tests,
        };

        let mut disk_cache: crate::disk_cache::WordchipperDiskCache = Default::default();
        let vocab: Arc<UnifiedTokenVocab<u32>> = crate::vocab::io::load_gpt2_vocab(&mut disk_cache)
            .unwrap()
            .into();
        let encoder: Arc<dyn TokenEncoder<u32>> =
            TokenEncoderOptions::default().build(vocab.clone());

        common_encoder_tests(vocab, encoder);
    }
}
