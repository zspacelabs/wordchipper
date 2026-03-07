extern crate core;

pub mod commands;
pub mod util;

use clap::Parser;

/// Text tokenizer multi-tool.
#[derive(clap::Parser, Debug)]
#[command(name = "report-tool")]
pub struct Args {
    /// Subcommand to run.
    #[clap(subcommand)]
    pub command: Commands,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    args.command.run()
}

/// Subcommands for wchipper
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Rust benchmark plots.
    RustBenchPlots(commands::rust_bench_plots_cmd::RustBenchPlots),
}

impl Commands {
    /// Run the subcommand.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Commands::RustBenchPlots(cmd) => cmd.run(),
        }
    }
}
