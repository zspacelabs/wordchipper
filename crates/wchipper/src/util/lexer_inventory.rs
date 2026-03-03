use wordchipper::{
    pretrained::openai::{
        OA_CL100K_BASE_PATTERN,
        OA_GPT2_PATTERN,
        OA_O200K_BASE_PATTERN,
    },
    spanners::span_lexers::accelerators::get_regex_accelerator,
};

/// Description of a lexer.
#[derive(Debug, Clone, PartialEq)]
pub struct LexerDescription {
    pub schema: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub pattern: String,
    pub accelerated: bool,
}

impl LexerDescription {
    /// Create a new lexer description.
    pub fn new(
        schema: &str,
        aliases: &[&str],
        description: &str,
        pattern: &str,
    ) -> Self {
        let accelerated = get_regex_accelerator(pattern).is_some();
        assert_ne!(aliases.len(), 0);
        Self {
            schema: schema.to_string(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            description: description.to_string(),
            pattern: pattern.to_string(),
            accelerated,
        }
    }

    pub fn name(&self) -> &str {
        &self.aliases[0]
    }

    pub fn id(&self) -> String {
        format!("{}::{}", self.schema, self.name())
    }
}

/// Inventory of lexers.
#[derive(Debug, Clone, PartialEq)]
pub struct LexerInventory {
    pub lexers: Vec<LexerDescription>,
}

impl LexerInventory {
    /// Build the lexer inventory.
    pub fn build() -> Self {
        let mut lexers = vec![
            LexerDescription::new(
                "openai",
                &["gpt2", "r50k", "p50k"],
                "OpenAI's gpt2/r50k/p50k model pattern.",
                OA_GPT2_PATTERN.as_str(),
            ),
            LexerDescription::new(
                "openai",
                &["cl100k", "cl100k_base", "cl100k_edit"],
                "OpenAI's cl100k model pattern.",
                OA_CL100K_BASE_PATTERN.as_str(),
            ),
            LexerDescription::new(
                "openai",
                &["o200k", "o200k_base", "o200k_harmony"],
                "OpenAI's o200k model pattern.",
                OA_O200K_BASE_PATTERN.as_str(),
            ),
        ];
        lexers.sort_by_key(|a| a.id());
        Self { lexers }
    }

    /// Find a lexer by name.
    pub fn find_model(
        &self,
        query: &str,
    ) -> Option<&LexerDescription> {
        if query.contains("::") {
            let (schema, name) = query.split_once("::")?;
            for lexer in &self.lexers {
                if lexer.schema == schema && lexer.aliases.iter().any(|a| a == name) {
                    return Some(lexer);
                }
            }
        }
        self.lexers
            .iter()
            .find(|&lexer| lexer.aliases.iter().any(|a| a == query))
            .map(|v| v as _)
    }
}
