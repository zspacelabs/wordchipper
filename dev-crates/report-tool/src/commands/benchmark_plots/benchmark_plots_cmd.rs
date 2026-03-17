use std::fs::read_dir;

use clap::Args;
use wordchipper_cli_util::logging::LogArgs;

use crate::{
    commands::benchmark_plots::{
        graph_style::GraphStyleOptions,
        python_bench_plots_cmd,
        rust_bench_plots_cmd,
    },
    util::bench_data::{
        PythonParBenchData,
        RustParBenchData,
    },
};

/// Args for the rust-bench-plots command.
#[derive(Args, Debug)]
pub struct BenchmarkPlots {
    /// Rust model names.
    #[clap(long, value_delimiter = ',', default_value = "r50k,cl100k,o200k")]
    rust_models: Vec<String>,

    /// Python model names.
    #[clap(
        long,
        value_delimiter = ',',
        default_value = "gpt2,cl100k_base,o200k_base"
    )]
    python_models: Vec<String>,

    /// Path to the benchmark data.
    #[clap(long, default_value = "benchmarks/")]
    data_dir: String,

    /// Path to the output directory.
    #[clap(long, default_value = "target/benchmarks")]
    output_dir: String,

    #[clap(flatten)]
    logging: LogArgs,
}

impl BenchmarkPlots {
    /// Run the command.
    #[allow(unused)]
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;
        log::info!("{:#?}", self);

        let output_dir = std::path::Path::new(&self.output_dir);

        let ident = regex::Regex::new(r"^(\w+)$")?;

        let graph_options = GraphStyleOptions::default();

        for res in read_dir(self.data_dir.clone())? {
            let entry = res?;

            if entry.file_type()?.is_dir() && ident.is_match(entry.file_name().to_str().unwrap()) {
                let arch_name = entry.file_name().to_str().unwrap().to_string();
                let data_dir = entry.path().join("data");
                let plot_dir = output_dir.join(&arch_name).join("plots");

                log::info!("Processing {arch_name}");
                log::info!(" data: {}", data_dir.display());
                log::info!(" plot: {}", plot_dir.display());

                {
                    let rust_par_data =
                        RustParBenchData::load_data(data_dir.join("rust_parallel"))?;

                    let rust_par_output = plot_dir.join("rust_parallel");
                    std::fs::create_dir_all(&rust_par_output)?;

                    for model in self.rust_models.iter() {
                        rust_bench_plots_cmd::build_rust_benchmark_plots(
                            &arch_name,
                            model,
                            &rust_par_output,
                            graph_options,
                            &rust_par_data,
                        )?;
                    }
                }

                {
                    let python_par_data =
                        PythonParBenchData::load_data(data_dir.join("python_parallel"))?;

                    let python_par_output = plot_dir.join("python_parallel");
                    std::fs::create_dir_all(&python_par_output)?;

                    for model in self.python_models.iter() {
                        python_bench_plots_cmd::build_python_benchmark_plots(
                            &arch_name,
                            model,
                            &python_par_output,
                            graph_options,
                            &python_par_data,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }
}
