// use std::{
// fs::{
// DirEntry,
// create_dir_all,
// },
// path::Path,
// str::FromStr,
// time::Duration,
// };
//
// use wordchipper::encoders::token_span_encoder::SpanEncoderSelector;
//
//
// fn main() -> Result<(), Box<dyn std::error::Error>> {
// let data_dir = std::env::current_dir()?.join("target").join("criterion");
//
// let output_dir = std::env::current_dir()?.join("target").join("plots");
//
// create_dir_all(output_dir.clone())?;
//
// build(&data_dir, &output_dir)
// }
//
// #[allow(unused)]
// #[derive(Debug, serde::Deserialize)]
// struct Record {
// group: String,
// function: String,
// value: u64,
// throughput_num: u64,
// throughput_type: String,
// sample_measured_value: f64,
// unit: String,
// iteration_count: u64,
// }
//
// pub fn build(
// data_dir: &Path,
// output_dir: &Path,
// ) -> Result<(), Box<dyn std::error::Error>> {
// for model_dir_read in data_dir.read_dir().unwrap() {
// let model_dir = model_dir_read.unwrap();
// let model_dir_name = model_dir.file_name().to_string_lossy().to_string();
// if model_dir_name.starts_with("TokenEncoder_") {
// let model_name = model_dir_name.split("_").nth(1).unwrap();
// println!("model={:?}", model_name);
//
// use plotters::prelude::*;
//
// let plot_path = output_dir.join(format!("{model_dir_name}.png"));
// println!("plot={:?}", plot_path);
//
// let root = BitMapBackend::new(&plot_path, (640, 480)).into_drawing_area();
// root.fill(&WHITE)?;
// let mut chart = ChartBuilder::on(&root)
// .caption(model_name, ("sans-serif", 50).into_font())
// .margin(5)
// .x_label_area_size(40)
// .y_label_area_size(90)
// .build_cartesian_2d((8..64).log_scale(), (1.0e8..2.0e9).log_scale())?;
//
// chart
// .configure_mesh()
// .x_desc("Thread Count")
// .y_desc("Throughput")
// .y_label_formatter(&|&bps| {
// format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
// })
// .draw()?;
//
//
// chart
// .draw_series(LineSeries::new(
// (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
// &RED,
// ))?
// .label("y = x^2")
// .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
// /
//
// let mut seq_dirs = model_dir
// .path()
// .read_dir()
// .unwrap()
// .collect::<Result<Vec<_>, _>>()?;
// seq_dirs.sort_by_key(|e| e.file_name().to_string_lossy().to_string());
//
// for seq_dir in seq_dirs {
// let seq_name = seq_dir.file_name().to_string_lossy().to_string();
//
// let parts = seq_name.splitn(2, "_").collect::<Vec<_>>();
// if parts.len() != 2 {
// continue;
// }
// let span_name = parts[0];
// let lexer_name = parts[1];
//
// if SpanEncoderSelector::from_str(span_name).is_err() {
// continue;
// }
//
// println!("{span_name}/{lexer_name}");
//
// [(thread_count, mean_bps)]
// let mut chart_series: Vec<(u32, f64)> = Vec::new();
//
// let mut dirs = seq_dir
// .path()
// .read_dir()?
// .filter_map(|e| {
// let d = e.unwrap();
// if u32::from_str(d.file_name().to_string_lossy().as_ref()).is_ok() {
// Some(d)
// } else {
// None
// }
// })
// .collect::<Vec<DirEntry>>();
// dirs.sort_by_key(|e| {
// u32::from_str(e.file_name().to_string_lossy().as_ref()).unwrap()
// });
//
// for thread_dir in dirs {
// let dir_name = thread_dir.file_name().to_string_lossy().to_string();
// if let Ok(thread_count) = u32::from_str(&dir_name) {
// let csv_path = thread_dir.path().join("new").join("raw.csv");
//
// if thread_count == 1 {
// continue;
// }
//
// let mut rdr = csv::ReaderBuilder::new()
// .has_headers(true)
// .from_path(csv_path)
// .unwrap();
//
// let mut byte_count = Option::<u64>::None;
// let mut values: Vec<(Duration, u64)> = Vec::new();
//
// for result in rdr.deserialize() {
// let record: Record = result.unwrap();
//
// if byte_count.is_none() {
// byte_count = Some(record.throughput_num);
// }
//
// let duration =
// build_duration(record.sample_measured_value, &record.unit);
// let count = record.iteration_count;
//
// if let Some((prev_dur, prev_count)) = values.last() {
// values.push((duration - *prev_dur, count - *prev_count))
// } else {
// values.push((duration, count))
// }
// }
//
// let mut point_estimates = values
// .iter()
// .map(|(d, c)| (*d) / (*c as u32))
// .collect::<Vec<_>>();
//
// point_estimates.sort_unstable();
// let inliers = point_estimates[2..point_estimates.len() - 2].to_vec();
//
// let mean_time = inliers.iter().sum::<Duration>() / inliers.len() as u32;
// let mean_bps = byte_count.unwrap() as f64 / mean_time.as_secs_f64();
//
// chart_series.push((thread_count, mean_bps));
//
// println!(
// "| {thread_count:>10} | {mean_bps:>8.2e} b/s | {:>8.2?} | {:>11}/s |",
// mean_time,
// humansize::format_size_i(mean_bps, humansize::BINARY)
// );
// }
// }
//
// let color = match span_name {
// "PriorityMerge" => RED,
// "BufferSweep" => BLUE,
// "TailSweep" => GREEN,
// "MergeHeap" => CYAN,
// _ => BLACK,
// };
//
// let style = if lexer_name == "logos" {
// color.filled()
// } else {
// color.into()
// };
//
// chart
// .draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
// chart_series.iter().map(|(x, y)| (*x as i32, *y)),
// 4,
// style,
// ))?
// .label(format!("{span_name}/{lexer_name}"))
// .legend(move |coord| Circle::new(coord, 4, style));
// chart.draw_series(LineSeries::new(
// chart_series.iter().map(|(x, y)| (*x as i32, *y)),
// &color,
// ))?;
// }
//
// chart
// .configure_series_labels()
// .position(SeriesLabelPosition::LowerRight)
// .background_style(WHITE.mix(0.8))
// .border_style(BLACK)
// .draw()?;
//
// root.present()?;
// }
// }
//
// Ok(())
// }
//
// fn build_duration(
// time: f64,
// units: &str,
// ) -> Duration {
// match units {
// "ns" => Duration::from_nanos(time as u64),
// "us" => Duration::from_micros(time as u64),
// "ms" => Duration::from_millis(time as u64),
// "s" => Duration::from_secs(time as u64),
// _ => panic!("unknown units: {}", units),
// }
// }
//
//

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
    /// Dev/WIP command..
    Dev(commands::dev_cmd::DevArgs),
}

impl Commands {
    /// Run the subcommand.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Commands::Dev(cmd) => cmd.run(),
        }
    }
}
