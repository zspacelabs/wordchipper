use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter},
};

fn squash_standard_io(path: &Option<String>) -> Option<String> {
    match path {
        Some(p) if p == "-" => None,
        Some(p) => Some(p.clone()),
        None => None,
    }
}

/// Input/output argument group.
#[derive(clap::Args, Debug)]
pub struct InputArgs {
    /// Optional input file; "-" may be used to indicate stdin.
    #[clap(long, default_value = None)]
    pub input: Option<String>,
}

impl InputArgs {
    /// Open a reader for the input.
    pub fn open_reader(&self) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
        Ok(match squash_standard_io(&self.input) {
            None => Box::new(BufReader::new(std::io::stdin().lock())),
            Some(p) => Box::new(BufReader::new(File::open(p)?)),
        })
    }
}

/// Output argument group.
#[derive(clap::Args, Debug)]
pub struct OutputArgs {
    /// Optional output file; "-" may be used to indicate stdout.
    #[clap(long, default_value = None)]
    pub output: Option<String>,
}

impl OutputArgs {
    /// Open a writer for the output.
    pub fn open_writer(&self) -> Result<Box<dyn std::io::Write>, Box<dyn std::error::Error>> {
        Ok(match squash_standard_io(&self.output) {
            Some(p) => Box::new(BufWriter::new(File::create(p)?)),
            None => Box::new(BufWriter::new(std::io::stdout().lock())),
        })
    }
}
