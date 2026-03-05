use core::f64;
use std::{
    collections::BTreeMap,
    path::Path,
};

use divan_parser::BenchResult;
use plotters::{
    prelude::{
        IntoLogRange,
        ShapeStyle,
        *,
    },
    style::full_palette as palette,
};
use wordchipper_cli_util::logging::LogArgs;

use crate::util::bench_data::par_bench::ParBenchData;

/// Args for the cat command.
#[derive(clap::Args, Debug)]
pub struct DevArgs {
    /// Path to the benchmark data.
    #[clap(long, default_value = "dev-crates/wordchipper-bench/bench-data")]
    data_dir: String,

    /// Model name.
    #[clap(long, default_value = "cl100k")]
    model: String,

    /// Path to the output directory.
    #[clap(long, default_value = "target/plots")]
    output_dir: String,

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

        let output_dir = Path::new(&self.output_dir);
        std::fs::create_dir_all(output_dir)?;

        build_external_tgraph(
            &self.model,
            &output_dir.join(format!("tgraph.external.{}.svg", self.model)),
            &data,
        )?;
        for accel in [false, true] {
            build_internal_tgraph(
                &self.model,
                accel,
                &output_dir.join(format!(
                    "tgraph.{}.{}.svg",
                    if accel { "logos" } else { "regex" },
                    self.model
                )),
                &data,
            )?;
        }
        for accel in [false, true] {
            build_internal_rel_tgraph(
                &self.model,
                accel,
                &output_dir.join(format!(
                    "tgraph.rel.{}.{}.svg",
                    if accel { "logos" } else { "regex" },
                    self.model
                )),
                &data,
            )?;
        }

        Ok(())
    }
}

struct Point {
    pub threads: u32,
    pub value: f64,
}

enum SeriesKind {
    Internal,
    External,
}

struct Series {
    pub name: String,
    pub style: ShapeStyle,
    pub kind: SeriesKind,
    pub points: Vec<Point>,
}
impl Series {
    pub fn min_threads(&self) -> u32 {
        self.points.iter().map(|p| p.threads).min().unwrap()
    }

    pub fn max_threads(&self) -> u32 {
        self.points.iter().map(|p| p.threads).max().unwrap()
    }

    pub fn min_bps(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.value)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    pub fn max_bps(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.value)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

fn external_styles() -> BTreeMap<String, ShapeStyle> {
    [
        ("bpe_openai".to_string(), palette::GREEN_400.filled()),
        ("tiktoken".to_string(), palette::RED_200.filled()),
        ("tokenizers".to_string(), palette::BLUE_300.filled()),
    ]
    .iter()
    .cloned()
    .collect()
}

fn fmax(
    a: f64,
    b: f64,
) -> f64 {
    match a.partial_cmp(&b).unwrap() {
        std::cmp::Ordering::Less => b,
        std::cmp::Ordering::Equal => a,
        std::cmp::Ordering::Greater => a,
    }
}

fn bps(br: &BenchResult) -> f64 {
    br.throughput_bps.as_ref().unwrap().mean.unwrap()
}

fn build_internal_rel_tgraph<P: AsRef<Path>>(
    model: &str,
    accel: bool,
    plot_path: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let plot_path = plot_path.as_ref();

    log::info!("Plotting to {}", plot_path.display());

    let span_map: BTreeMap<&str, ShapeStyle> = [
        ("bpe_backtrack", palette::GREY_A400.filled()),
        ("buffer_sweep", palette::GREEN_A400.filled()),
        ("merge_heap", palette::BLUE_A400.filled()),
        ("priority_merge", palette::PINK_A700.filled()),
        ("tail_sweep", palette::DEEPPURPLE_A400.filled()),
    ]
    .iter()
    .cloned()
    .collect();

    let span_key = |span: &str| {
        format!(
            "encoding_parallel::wordchipper::{span}::{model}{}",
            if accel { "_fast" } else { "" },
        )
    };

    let mut baseline: BTreeMap<u32, f64> = Default::default();
    for span in span_map.keys() {
        if let Some(series_data) = data.select_series(&span_key(span)) {
            for (threads, bench_result) in series_data.iter() {
                let bps = bps(bench_result);

                let entry = baseline.entry(*threads).or_default();
                *entry = fmax(*entry, bps);
            }
        }
    }

    let mut plot_series: Vec<Series> = Default::default();
    for (span, style) in span_map.iter() {
        if let Some(series_data) = data.select_series(&span_key(span)) {
            let style = if accel { style.filled() } else { *style };

            plot_series.push(Series {
                name: span.to_string(),
                style,
                kind: SeriesKind::Internal,
                points: series_data
                    .iter()
                    .map(|(threads, bench_result)| Point {
                        threads: *threads,
                        value: bps(bench_result) / baseline[threads],
                    })
                    .collect(),
            })
        }
    }

    let min_threads = plot_series.iter().map(|s| s.min_threads()).min().unwrap();
    let max_threads = plot_series.iter().map(|s| s.max_threads()).max().unwrap();
    let min_bps = plot_series
        .iter()
        .map(|s| s.min_bps())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_bps = plot_series
        .iter()
        .map(|s| s.max_bps())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let root = SVGBackend::new(plot_path, (640, 480)).into_drawing_area();
    root.fill(&palette::WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "encoder vrs max, {} lexer, model: \"{}\"",
                if accel { "logos" } else { "regex" },
                model
            ),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_threads..max_threads).log_scale().base(2.0),
            min_bps..max_bps,
        )?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Relative Throughput")
        .draw()?;

    for pseries in plot_series {
        let name = &pseries.name;
        let style = pseries.style;
        let points: Vec<(u32, f64)> = pseries
            .points
            .iter()
            .map(|p| (p.threads, p.value))
            .collect();

        let size = 4;
        chart
            .draw_series(
                points
                    .iter()
                    .map(|coord| EmptyElement::at(*coord) + Circle::new((0, 0), size, style)),
            )?
            .label(name)
            .legend(move |coord| EmptyElement::at(coord) + Circle::new((0, 0), size, style));

        chart.draw_series(LineSeries::new(
            pseries.points.iter().map(|p| (p.threads, p.value)),
            style,
        ))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
fn build_internal_tgraph<P: AsRef<Path>>(
    model: &str,
    accel: bool,
    plot_path: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let plot_path = plot_path.as_ref();

    log::info!("Plotting to {}", plot_path.display());

    let span_map: BTreeMap<&str, ShapeStyle> = [
        ("bpe_backtrack", palette::GREY_A400.filled()),
        ("buffer_sweep", palette::GREEN_A400.filled()),
        ("merge_heap", palette::BLUE_A400.filled()),
        ("priority_merge", palette::PINK_A700.filled()),
        ("tail_sweep", palette::DEEPPURPLE_A400.filled()),
    ]
    .iter()
    .cloned()
    .collect();

    let mut plot_series: Vec<Series> = Default::default();

    for (span, style) in span_map.iter() {
        let sname = format!(
            "encoding_parallel::wordchipper::{span}::{model}{}",
            if accel { "_fast" } else { "" },
        );

        if let Some(series_data) = data.select_series(&sname) {
            let style = if accel { style.filled() } else { *style };

            plot_series.push(Series {
                name: span.to_string(),
                style,
                kind: SeriesKind::Internal,
                points: series_data
                    .iter()
                    .map(|(threads, bench_result)| Point {
                        threads: *threads,
                        value: bps(bench_result),
                    })
                    .collect(),
            })
        }
    }

    let min_threads = plot_series.iter().map(|s| s.min_threads()).min().unwrap();
    let max_threads = plot_series.iter().map(|s| s.max_threads()).max().unwrap();
    let min_bps = plot_series
        .iter()
        .map(|s| s.min_bps())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_bps = plot_series
        .iter()
        .map(|s| s.max_bps())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let root = SVGBackend::new(plot_path, (640, 480)).into_drawing_area();
    root.fill(&palette::WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "encoder vrs, {} lexer, model: \"{}\"",
                if accel { "logos" } else { "regex" },
                model
            ),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_threads..max_threads).log_scale().base(2.0),
            min_bps..max_bps,
        )?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Throughput")
        .y_label_formatter(&|&bps| {
            format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
        })
        .draw()?;

    for pseries in plot_series {
        let name = &pseries.name;
        let style = pseries.style;
        let points: Vec<(u32, f64)> = pseries
            .points
            .iter()
            .map(|p| (p.threads, p.value))
            .collect();

        let size = 4;
        chart
            .draw_series(
                points
                    .iter()
                    .map(|coord| EmptyElement::at(*coord) + Circle::new((0, 0), size, style)),
            )?
            .label(name)
            .legend(move |coord| EmptyElement::at(coord) + Circle::new((0, 0), size, style));

        chart.draw_series(LineSeries::new(
            pseries.points.iter().map(|p| (p.threads, p.value)),
            style,
        ))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
fn build_external_tgraph<P: AsRef<Path>>(
    model: &str,
    plot_path: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let plot_path = plot_path.as_ref();

    log::info!("Plotting to {}", plot_path.display());

    let mut plot_series: Vec<Series> = Default::default();

    let series_names = data
        .series_names()
        .into_iter()
        .filter(|name| name.contains(model))
        .collect::<Vec<_>>();

    for (ext, style) in external_styles().iter() {
        if let Some(name) = series_names.iter().find(|name| name.contains(ext)) {
            let series_data = data.select_series(name).unwrap();

            plot_series.push(Series {
                name: ext.to_string(),
                style: *style,
                kind: SeriesKind::External,
                points: series_data
                    .iter()
                    .map(|(threads, bench_result)| Point {
                        threads: *threads,
                        value: bench_result.throughput_bps.as_ref().unwrap().mean.unwrap(),
                    })
                    .collect(),
            })
        }
    }

    let span_map: BTreeMap<&str, ShapeStyle> = [("buffer_sweep", palette::CYAN_300.into())]
        .iter()
        .cloned()
        .collect();

    for (span, style) in span_map.iter() {
        for accel in [false, true] {
            let sname = format!(
                "encoding_parallel::wordchipper::{span}::{model}{}",
                if accel { "_fast" } else { "" },
            );

            if let Some(series_data) = data.select_series(&sname) {
                let display_name = format!("wordchipper:{}", if accel { "logos" } else { "regex" });

                let style = if accel { style.filled() } else { *style };

                plot_series.push(Series {
                    name: display_name,
                    style,
                    kind: SeriesKind::Internal,
                    points: series_data
                        .iter()
                        .map(|(threads, bench_result)| Point {
                            threads: *threads,
                            value: bench_result.throughput_bps.as_ref().unwrap().mean.unwrap(),
                        })
                        .collect(),
                })
            }
        }
    }

    let min_threads = plot_series.iter().map(|s| s.min_threads()).min().unwrap();
    let max_threads = plot_series.iter().map(|s| s.max_threads()).max().unwrap();
    let min_bps = plot_series
        .iter()
        .map(|s| s.min_bps())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_bps = plot_series
        .iter()
        .map(|s| s.max_bps())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let root = SVGBackend::new(plot_path, (640, 480)).into_drawing_area();
    root.fill(&palette::WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("library vrs, model: \"{}\"", model),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_threads..max_threads).log_scale().base(2.0),
            (min_bps..max_bps).log_scale().base(2.0),
        )?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Throughput")
        .y_label_formatter(&|&bps| {
            format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
        })
        .draw()?;

    for pseries in plot_series {
        let name = &pseries.name;
        let style = pseries.style;
        let points: Vec<(u32, f64)> = pseries
            .points
            .iter()
            .map(|p| (p.threads, p.value))
            .collect();

        match pseries.kind {
            SeriesKind::Internal => {
                let size = 2;
                chart
                    .draw_series(points.iter().map(|coord| {
                        EmptyElement::at(*coord)
                            + Circle::new((0, 0), size + 2, BLACK.stroke_width(2))
                            + Circle::new((0, 0), size, style)
                    }))?
                    .label(name)
                    .legend(move |coord| {
                        EmptyElement::at(coord)
                            + Circle::new((0, 0), size + 2, BLACK.stroke_width(2))
                            + Circle::new((0, 0), size, style)
                    });
            }
            SeriesKind::External => {
                let size = 4;
                chart
                    .draw_series(points.iter().map(|coord| {
                        EmptyElement::at(*coord)
                            + TriangleMarker::new((0, 0), size + 2, BLACK.stroke_width(3))
                            + TriangleMarker::new((0, 0), size, style)
                    }))?
                    .label(name)
                    .legend(move |coord| {
                        EmptyElement::at(coord)
                            + TriangleMarker::new((0, 0), size + 2, BLACK.stroke_width(3))
                            + TriangleMarker::new((0, 0), size, style)
                    });
            }
        }

        chart.draw_series(LineSeries::new(
            pseries.points.iter().map(|p| (p.threads, p.value)),
            style,
        ))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
