//! Run `cargo bench` and convert divan output to JSON.

use std::{
    io::{
        BufRead,
        BufReader,
        Write,
    },
    process::{
        Command,
        Stdio,
    },
    sync::Mutex,
    thread,
};

use clap::Parser;
use wordchipper_bench::divan_parser::DivanParser;

/// Run divan benchmarks and output results as JSON.
#[derive(Parser)]
#[command(name = "bench-json")]
struct Cli {
    /// Name of a specific bench target to run.
    #[arg(long)]
    bench: Option<String>,

    /// Write JSON output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<String>,

    /// Echo divan output to stderr while parsing.
    #[arg(long)]
    tee: bool,

    /// Extra arguments passed to the bench binary (after `--`).
    #[arg(last = true)]
    divan_args: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    let mut cmd = Command::new("cargo");
    cmd.arg("bench")
        .arg("-p")
        .arg("wordchipper-bench")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(ref name) = cli.bench {
        cmd.arg("--bench").arg(name);
    }

    if !cli.divan_args.is_empty() {
        cmd.arg("--").args(&cli.divan_args);
    }

    let mut child = cmd.spawn().expect("failed to spawn cargo bench");

    let stdout = child.stdout.take().expect("failed to capture stdout");
    let child_stderr = child.stderr.take().expect("failed to capture stderr");

    let parser = Mutex::new(DivanParser::new());
    let tee = cli.tee;

    // Read stdout and stderr concurrently to avoid deadlock. Stderr carries
    // cargo compilation output and "Running benches/..." lines (needed for
    // the `bench` field). Both streams feed into the shared parser.
    thread::scope(|s| {
        s.spawn(|| {
            let mut our_stderr = std::io::stderr();
            for line in BufReader::new(child_stderr).lines() {
                let line = line.expect("failed to read stderr line");
                let _ = writeln!(our_stderr, "{line}");
                parser.lock().unwrap().feed_line(&line);
            }
        });

        let mut stderr = std::io::stderr();
        for line in BufReader::new(stdout).lines() {
            let line = line.expect("failed to read line");
            if tee {
                let _ = writeln!(stderr, "{line}");
            }
            parser.lock().unwrap().feed_line(&line);
        }
    });

    let status = child.wait().expect("failed to wait for cargo bench");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    let results = parser.into_inner().unwrap().finish();
    let json = serde_json::to_string_pretty(&results).expect("failed to serialize JSON");

    if let Some(ref path) = cli.output {
        std::fs::write(path, format!("{json}\n")).expect("failed to write output file");
    } else {
        println!("{json}");
    }
}
