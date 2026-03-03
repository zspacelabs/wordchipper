use crate::commands::lexers::{
    lexers_list_cmd,
    lexers_stress_cmd,
};

/// Subcommands for the lexers command.
#[derive(clap::Subcommand, Debug)]
pub enum LexersSubcommand {
    /// List available lexers.
    #[clap(visible_alias = "ls")]
    List(lexers_list_cmd::ListLexersArgs),

    /// Stress test a regex accelerator.
    Stress(lexers_stress_cmd::StressLexerArgs),
}

/// Args for the lexers menu.
#[derive(clap::Args, Debug)]
pub struct LexersMenu {
    #[clap(subcommand)]
    pub command: LexersSubcommand,
}

impl LexersMenu {
    /// List the menu.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        use LexersSubcommand::*;
        match &self.command {
            List(cmd) => cmd.run(),
            Stress(cmd) => cmd.run(),
        }
    }
}
