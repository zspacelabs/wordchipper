use crate::{
    TokenType,
    UnifiedTokenVocab,
    alloc::sync::Arc,
    prelude::*,
    pretrained::factory::vocab_query::VocabQuery,
};

/// A description of a pretrained tokenizer.
#[derive(Debug, Clone)]
pub struct VocabDescription {
    /// The parsed id.
    id: VocabQuery,

    /// The cache context for the vocabulary.
    context: Vec<String>,

    /// A description of the vocabulary.
    description: String,
}

impl VocabDescription {
    /// Build a new vocabulary description.
    pub fn new<Q, C, D>(
        id: Q,
        context: &[C],
        description: D,
    ) -> Self
    where
        Q: Into<VocabQuery>,
        C: AsRef<str>,
        D: AsRef<str>,
    {
        let id = id.into();
        let context = context.iter().map(|c| c.as_ref().to_string()).collect();
        let description = description.as_ref().to_string();

        Self {
            id,
            context,
            description,
        }
    }

    /// Get the id of the vocabulary.
    pub fn id(&self) -> &VocabQuery {
        &self.id
    }

    /// Get the context of the vocabulary.
    pub fn context(&self) -> &[String] {
        &self.context
    }

    /// Get the description of the vocabulary.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// A listing of known tokenizer.
#[derive(Debug, Clone)]
pub struct VocabListing {
    /// The id of the factory that produced the vocabularies.
    source: String,

    /// A description of the factory.
    description: String,

    /// Explicitly listed vocabularies.
    vocabs: Vec<VocabDescription>,
}

impl VocabListing {
    /// Build a new vocabulary listing.
    pub fn new(
        source: &str,
        description: &str,
        vocabs: Vec<VocabDescription>,
    ) -> Self {
        Self {
            source: source.to_string(),
            description: description.to_string(),
            vocabs,
        }
    }

    /// Get the id of the factory that produced the vocabularies.
    pub fn provider(&self) -> &str {
        &self.source
    }

    /// Get the description of the factory.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the explicit list of vocabularies.
    pub fn vocabs(&self) -> &[VocabDescription] {
        &self.vocabs
    }
}

/// Resolved vocabulary with its description and loaded vocabulary.
#[derive(Clone)]
pub struct LabeledVocab<T: TokenType> {
    description: VocabDescription,
    vocab: Arc<UnifiedTokenVocab<T>>,
}

impl<T: TokenType> LabeledVocab<T> {
    /// Build a new resolved vocabulary.
    pub fn new(
        description: VocabDescription,
        vocab: Arc<UnifiedTokenVocab<T>>,
    ) -> Self {
        Self { description, vocab }
    }

    /// Get the description of the vocabulary.
    pub fn description(&self) -> &VocabDescription {
        &self.description
    }

    /// Get the unified token vocabulary.
    pub fn vocab(&self) -> &Arc<UnifiedTokenVocab<T>> {
        &self.vocab
    }
}
