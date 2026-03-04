use std::path::Path;

use wordchipper_cli_util::logging::LogArgs;

use crate::util::bench_data::par_bench::ParBenchData;

/// Args for the cat command.
#[derive(clap::Args, Debug)]
pub struct DevArgs {
    /// Path to the benchmark data.
    #[clap(long, default_value = "dev-crates/wordchipper-bench/bench-data")]
    data_dir: String,

    #[clap(flatten)]
    logging: LogArgs,
}

impl DevArgs {
    /// Run the dev command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;

        println!("{:?}", self);

        let data_dir = Path::new(&self.data_dir);

        let data = ParBenchData::load_data(data_dir.join("parallel"))?;

        println!("threads: {:?}", data.threads());
        println!("series: {:#?}", data.series_names());

        Ok(())
    }
}
