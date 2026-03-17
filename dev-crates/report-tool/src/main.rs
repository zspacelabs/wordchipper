pub mod commands;
pub mod util;

use clap::Parser;
use commands::benchmark_plots;

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

/// Subcommands for wordchipper-cli
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Build the benchmark plots.
    BenchmarkPlots(benchmark_plots::BenchmarkPlots),
}

impl Commands {
    /// Run the subcommand.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        use Commands::*;
        match self {
            BenchmarkPlots(cmd) => cmd.run(),
        }
    }
}
