use crate::commands::models::ListModelsArgs;

/// Subcommands for the models command.
#[derive(clap::Subcommand, Debug)]
pub enum ModelsSubcommand {
    /// List available models.
    #[clap(visible_alias = "ls")]
    List(ListModelsArgs),
}

/// Args for the model listing command.
#[derive(clap::Args, Debug)]
pub struct ModelsMenu {
    #[clap(subcommand)]
    pub command: ModelsSubcommand,
}

impl ModelsMenu {
    /// List the menu.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.command {
            ModelsSubcommand::List(cmd) => cmd.run(),
        }
    }
}
