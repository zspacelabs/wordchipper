use stderrlog::Timestamp;

/// Logging setup arg group.
#[derive(clap::Args, Debug)]
pub struct LogArgs {
    /// Silence log messages.
    #[clap(short, long)]
    pub quiet: bool,

    /// Turn debugging information on (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, default_value = None)]
    verbose: Option<u8>,

    /// Enable timestamped logging.
    #[clap(short, long)]
    pub ts: bool,
}

impl LogArgs {
    pub fn setup_logging(
        &self,
        default: u8,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let level = if let Some(verbose) = self.verbose
            && verbose > 0
        {
            verbose
        } else {
            default
        };

        let log_level = match level {
            0 => stderrlog::LogLevelNum::Off,
            1 => stderrlog::LogLevelNum::Error,
            2 => stderrlog::LogLevelNum::Warn,
            3 => stderrlog::LogLevelNum::Info,
            4 => stderrlog::LogLevelNum::Debug,
            _ => stderrlog::LogLevelNum::Trace,
        };

        stderrlog::new()
            .quiet(self.quiet)
            .verbosity(log_level)
            .timestamp(if self.ts {
                Timestamp::Second
            } else {
                Timestamp::Off
            })
            .init()?;

        Ok(())
    }
}
