pub mod commands;

use clap::Parser;
pub use wordchipper_cli_util as util;

/// Text tokenizer multi-tool.
#[derive(clap::Parser, Debug)]
#[command(name = "wordchipper-cli")]
pub struct Args {
    /// Subcommand to run.
    #[clap(subcommand)]
    pub command: Commands,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    args.command.run()
}

/// Subcommands for wordchipper-cli
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Act as a streaming tokenizer.
    Cat(commands::cat_cmd::CatArgs),

    /// Lexers sub-menu.
    Lexers(commands::lexers::LexersMenu),

    /// Models sub-menu.
    Models(commands::models::ModelsMenu),

    /// Train a new model.
    Train(commands::train_cmd::TrainArgs),

    /// Markdown documentation.
    Doc(commands::doc_cmd::DocArgs),
}

impl Commands {
    /// Run the subcommand.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Commands::Cat(cmd) => cmd.run(),
            Commands::Lexers(cmd) => cmd.run(),
            Commands::Models(cmd) => cmd.run(),
            Commands::Train(cmd) => cmd.run(),
            Commands::Doc(cmd) => cmd.run(),
        }
    }
}
