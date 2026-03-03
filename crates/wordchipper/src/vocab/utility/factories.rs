//! # Vocab Factory Support

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::BufRead;
#[cfg(feature = "std")]
use std::io::BufReader;
#[cfg(feature = "std")]
use std::path::Path;
#[cfg(feature = "std")]
use std::path::PathBuf;

#[cfg(feature = "std")]
use crate::WCResult;
#[cfg(feature = "std")]
use crate::support::resources::ResourceLoader;
#[cfg(feature = "std")]
use crate::vocab::UnifiedTokenVocab;
use crate::{
    TokenType,
    alloc::{
        string::String,
        vec::Vec,
    },
    spanners::TextSpanningConfig,
    support::{
        regex::{
            ConstRegexPattern,
            RegexPattern,
        },
        resources::ConstKeyedResource,
    },
};

/// A pretrained tokenizer bundle.
pub struct ConstVocabularyFactory {
    /// The name of the tokenizer.
    pub name: &'static str,

    /// A (key, resource) pair.
    pub resource: ConstKeyedResource,

    /// The tokenizer regex pattern.
    pub pattern: ConstRegexPattern,

    /// A generator for special tokens.
    pub special_builder: &'static dyn Fn() -> Vec<(String, usize)>,
}

impl ConstVocabularyFactory {
    /// Get the regex pattern for this tokenizer.
    pub fn pattern(&self) -> RegexPattern {
        self.pattern.to_pattern()
    }

    /// List the special tokens for this tokenizer.
    pub fn special_tokens<T: TokenType>(&self) -> Vec<(String, T)> {
        (self.special_builder)()
            .into_iter()
            .map(|(s, t)| (s, T::from_usize(t).unwrap()))
            .collect()
    }

    /// Load the spanners config for this tokenizer.
    pub fn spanning_config<T: TokenType>(&self) -> TextSpanningConfig<T> {
        TextSpanningConfig::from_pattern(self.pattern()).with_special_words(self.special_tokens())
    }

    /// Fetch a path to the resource through the loader.
    #[cfg(feature = "std")]
    fn fetch_resource(
        &self,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<PathBuf> {
        let res: crate::support::resources::KeyedResource = self.resource.clone().into();
        loader.load_resource_path(&res)
    }

    /// Load the pretrained vocabulary through the loader.
    #[cfg(feature = "std")]
    pub fn load_vocab<T: TokenType>(
        &self,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<UnifiedTokenVocab<T>> {
        let path = self.fetch_resource(loader)?;
        self.load_vocab_path(path)
    }

    /// Load the pretrained vocabulary from disk.
    #[cfg(feature = "std")]
    pub fn load_vocab_path<T: TokenType>(
        &self,
        path: impl AsRef<Path>,
    ) -> WCResult<UnifiedTokenVocab<T>> {
        let mut reader = BufReader::new(File::open(path)?);
        self.read_vocab(&mut reader)
    }

    /// Read the pretrained vocabulary from a reader.
    #[cfg(feature = "std")]
    pub fn read_vocab<T: TokenType>(
        &self,
        reader: &mut dyn BufRead,
    ) -> WCResult<UnifiedTokenVocab<T>> {
        crate::vocab::io::read_base64_unified_vocab(reader, self.spanning_config())
    }
}
