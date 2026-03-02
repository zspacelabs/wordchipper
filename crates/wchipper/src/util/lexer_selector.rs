use crate::commands::lexers::LexerInventory;

#[derive(clap::Args, Debug)]
#[group(required = true, multiple = false)]
pub struct LexerSelectorArgs {
    /// Model name for selection.
    #[arg(long)]
    model: Option<String>,

    /// Pattern for selection.
    #[arg(long)]
    pattern: Option<String>,
}

impl LexerSelectorArgs {
    pub fn resolve(&self) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(p) = &self.pattern {
            return Ok(p.clone());
        }

        let name = self.model.as_ref().unwrap();
        match LexerInventory::build().find_model(name) {
            Some(model) => Ok(model.pattern.clone()),
            None => Err(format!("Model not found: {name}").into()),
        }
    }
}
