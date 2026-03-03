use crate::{
    prelude::*,
    pretrained::vocab_query::VocabQuery,
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
    pub fn new<Q>(
        id: Q,
        context: &[&str],
        description: &str,
    ) -> Self
    where
        Q: Into<VocabQuery>,
    {
        let id = id.into();

        Self {
            id,
            context: context.iter().map(|&s| s.to_string()).collect(),
            description: description.to_string(),
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
