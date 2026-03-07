use std::{
    collections::BTreeMap,
    path::Path,
};

use divan_parser::BenchResult;
use plotters::{
    prelude::{
        IntoLogRange,
        *,
    },
    style::full_palette as colors,
};
use wordchipper_cli_util::logging::LogArgs;

use crate::util::{
    bench_data::par_bench::ParBenchData,
    float_tools,
    float_tools::fiter_range,
    human_format,
    plotting::{
        MarkerLevel,
        MarkerSeries,
        MarkerStyle,
        MarkerType,
    },
};

/// Args for the rust-bench-plots command.
#[derive(clap::Args, Debug)]
pub struct RustBenchPlots {
    /// Model name.
    #[clap(long, value_delimiter = ',', default_value = "r50k,cl100k,o200k")]
    models: Vec<String>,

    /// Path to the benchmark data.
    #[clap(long, default_value = "benchmarks/amd3990X/data")]
    data_dir: String,

    /// Path to the output directory.
    #[clap(long, default_value = "target/plots")]
    output_dir: String,

    #[clap(flatten)]
    logging: LogArgs,
}

impl RustBenchPlots {
    /// Run the command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;
        log::info!("{:#?}", self);

        let data_dir = Path::new(&self.data_dir);
        let data = ParBenchData::load_data(data_dir.join("rust_parallel"))?;

        let output_dir = Path::new(&self.output_dir);
        std::fs::create_dir_all(output_dir)?;

        for model in self.models.iter() {
            build_model_graphs(model, &output_dir, &data)?;
        }

        Ok(())
    }
}

fn build_model_graphs<P: AsRef<Path>>(
    model: &str,
    output_dir: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    build_external_graphs(model, "buffer_sweep", &output_dir, data)?;

    build_internal_graphs(model, &output_dir, data)
}

fn build_internal_graphs<P: AsRef<Path>>(
    model: &str,
    output_dir: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    for accel in [false, true] {
        let lexer = if accel { "logos" } else { "regex" };
        build_internal_tgraph(
            model,
            accel,
            &output_dir.join(format!("span_encoder_vrs.{model}.{lexer}.log.svg")),
            data,
        )?;
        build_internal_rel_tgraph(
            model,
            accel,
            &output_dir.join(format!("span_encoder_vrs.{model}.{lexer}.rel.svg")),
            data,
        )?;
    }

    Ok(())
}

fn median_bps(br: &BenchResult) -> f64 {
    br.throughput_bps.as_ref().unwrap().median.unwrap()
}

fn span_styles() -> BTreeMap<&'static str, MarkerStyle> {
    let base_style = MarkerStyle::default().with_stroke_style(colors::BLACK.stroke_width(2));

    [
        (
            "buffer_sweep",
            base_style
                .with_marker_type(MarkerType::Circle)
                .with_fill_style(Some(colors::GREEN_200.into())),
        ),
        (
            "priority_merge",
            base_style
                .with_marker_type(MarkerType::Square)
                .with_fill_style(Some(colors::PURPLE_200.into())),
        ),
        (
            "tail_sweep",
            base_style
                .with_marker_type(MarkerType::Diamond)
                .with_fill_style(Some(colors::DEEPORANGE_200.into())),
        ),
        (
            "bpe_backtrack",
            base_style
                .with_marker_type(MarkerType::TriUp)
                .with_fill_style(Some(colors::LIGHTBLUE_200.into())),
        ),
        (
            "merge_heap",
            base_style
                .with_marker_type(MarkerType::TriDown)
                .with_fill_style(Some(colors::BLUEGREY_200.into())),
        ),
    ]
    .iter()
    .cloned()
    .collect()
}

fn build_internal_rel_tgraph<P: AsRef<Path>>(
    model: &str,
    accel: bool,
    plot_path: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let plot_path = plot_path.as_ref();

    log::info!("Plotting to {}", plot_path.display());

    let span_key = |span: &str| {
        format!(
            "encoding_parallel::wordchipper::{span}::{model}{}",
            if accel { "_fast" } else { "" },
        )
    };

    let mut plot_series: Vec<MarkerSeries<(u32, BenchResult)>> = Default::default();
    for (name, &style) in span_styles().iter() {
        if let Some(points) = data.select_series(&span_key(name)) {
            plot_series.push(MarkerSeries {
                name: name.to_string(),
                style,
                points,
            })
        }
    }

    fn select((threads, bench_results): &(u32, BenchResult)) -> (u32, f64) {
        (*threads, median_bps(bench_results))
    }

    let render: Vec<MarkerSeries<(u32, f64)>> = plot_series.iter().map(|s| s.map(select)).collect();

    // Normalize the points to the max value.
    let mut baseline: BTreeMap<u32, f64> = Default::default();
    for ms in render.iter() {
        for &(t, v) in ms.points.iter() {
            let entry = baseline.entry(t).or_default();
            *entry = float_tools::fmax(*entry, v);
        }
    }
    let render: Vec<MarkerSeries<(u32, f64)>> = render
        .into_iter()
        .map(|s| s.map(|&(t, v)| (t, v / baseline[&t])))
        .collect();

    let threads = render.iter().flat_map(|s| s.xs()).collect::<Vec<_>>();
    let x_min = *threads.iter().min().unwrap();
    let x_max = *threads.iter().max().unwrap();
    let x_range = x_min..x_max;

    let values = render.iter().flat_map(|s| s.ys()).collect::<Vec<_>>();
    let y_range = fiter_range(&values).unwrap();

    let root = SVGBackend::new(plot_path, (640, 480)).into_drawing_area();
    root.fill(&colors::WHITE)?;

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
        .y_label_area_size(50)
        .build_cartesian_2d(x_range.log_scale().base(2.0), y_range)?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Median Throughput: max relative")
        .draw()?;

    const SIZE: i32 = 8;
    const LINE_WIDTH: u32 = 4;

    for ms in render {
        chart.draw_series(LineSeries::new(
            ms.points.clone(),
            ms.style.line_style().stroke_width(LINE_WIDTH),
        ))?;

        chart
            .draw_series(ms.points.iter().map(|&coord| ms.style.marker(coord, SIZE)))?
            .label(ms.name)
            .legend(move |coord| ms.style.marker(coord, SIZE));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .margin(12)
        .legend_area_size(15)
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

    let mut schedule: Vec<MarkerSeries<(u32, BenchResult)>> = Default::default();

    for (name, &marker_style) in span_styles().iter() {
        let sname = format!(
            "encoding_parallel::wordchipper::{name}::{model}{}",
            if accel { "_fast" } else { "" },
        );

        if let Some(points) = data.select_series(&sname) {
            schedule.push(MarkerSeries {
                name: name.to_string(),
                style: marker_style,
                points,
            })
        }
    }

    fn select((threads, bench_results): &(u32, BenchResult)) -> (u32, f64) {
        (*threads, median_bps(bench_results))
    }

    let render: Vec<MarkerSeries<(u32, f64)>> = schedule.iter().map(|s| s.map(select)).collect();

    let threads = render.iter().flat_map(|s| s.xs()).collect::<Vec<_>>();
    let x_min = *threads.iter().min().unwrap();
    let x_max = *threads.iter().max().unwrap();
    let x_range = x_min..x_max;

    let values = render.iter().flat_map(|s| s.ys()).collect::<Vec<_>>();
    let y_range = fiter_range(&values).unwrap();

    let root = SVGBackend::new(plot_path, (640, 480)).into_drawing_area();
    root.fill(&colors::WHITE)?;

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
        .y_label_area_size(70)
        .build_cartesian_2d(x_range.log_scale().base(2.0), y_range.log_scale().base(2.0))?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Median Throughput: log scale")
        .y_label_formatter(&|&bps| human_format::format_bps(bps))
        .draw()?;

    const SIZE: i32 = 8;
    const LINE_WIDTH: u32 = 4;

    for ms in render {
        chart.draw_series(LineSeries::new(
            ms.points.clone(),
            ms.style.line_style().stroke_width(LINE_WIDTH),
        ))?;

        chart
            .draw_series(ms.points.iter().map(|&coord| ms.style.marker(coord, SIZE)))?
            .label(ms.name)
            .legend(move |coord| ms.style.marker(coord, SIZE));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .margin(12)
        .legend_area_size(15)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn build_external_graphs<P: AsRef<Path>>(
    model: &str,
    span_encoder: &str,
    output_dir: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let base_style = MarkerStyle::default().with_stroke_style(colors::BLACK.stroke_width(1));

    let external: Vec<MarkerSeries<(u32, BenchResult)>> = vec![
        (
            "bpe_openai",
            base_style
                .with_marker_type(MarkerType::Circle)
                .with_fill_style(Some(colors::DEEPORANGE_200.into())),
        ),
        (
            "tiktoken",
            base_style
                .with_marker_type(MarkerType::Square)
                .with_fill_style(Some(colors::PURPLE_200.into())),
        ),
        (
            "tokenizers",
            base_style
                .with_marker_type(MarkerType::Diamond)
                .with_fill_style(Some(colors::BLUEGREY_200.into())),
        ),
    ]
    .into_iter()
    .filter_map(|(name, style)| {
        data.select_series(&format!("encoding_parallel::{name}::{model}"))
            .map(|series_data| MarkerSeries::new(name, style, series_data))
    })
    .collect();

    let regex_series = MarkerSeries::new(
        "wordchipper:regex",
        base_style
            .with_marker_type(MarkerType::TriUp)
            .with_marker_level(MarkerLevel::Para)
            .with_fill_style(colors::GREEN_A200.filled()),
        data.try_select_series(&format!(
            "encoding_parallel::wordchipper::{span_encoder}::{model}"
        ))?,
    );

    let logos_series = MarkerSeries::new(
        "wordchipper:logos",
        base_style
            .with_marker_type(MarkerType::TriDown)
            .with_marker_level(MarkerLevel::Para)
            .with_fill_style(colors::LIGHTBLUE_A200.filled()),
        data.try_select_series(&format!(
            "encoding_parallel::wordchipper::{span_encoder}::{model}_fast"
        ))?,
    );

    const SIZE: i32 = 8;
    const LINE_WIDTH: u32 = 4;

    for include_logos in [false, true] {
        for log_scale in [false, true] {
            let chart_name = if include_logos { "logos" } else { "regex" };
            let scale_desc = if log_scale { "log" } else { "linear" };

            let plot_path = output_dir.join(format!(
                "wc_{chart_name}_vrs_brandx.rust.{model}.{scale_desc}.svg"
            ));
            log::info!("Plotting to {}", plot_path.display());

            let root = SVGBackend::new(&plot_path, (640, 480)).into_drawing_area();
            root.fill(&colors::WHITE)?;

            fn select((threads, bench_results): &(u32, BenchResult)) -> (u32, f64) {
                (*threads, median_bps(bench_results))
            }

            let mut schedule: Vec<MarkerSeries<(u32, f64)>> =
                external.iter().map(|ms| ms.map(select)).collect();

            schedule.push(regex_series.map(select));
            if include_logos {
                schedule.push(logos_series.map(select));
            }

            let threads = schedule.iter().flat_map(|s| s.xs()).collect::<Vec<_>>();
            let x_min = *threads.iter().max().unwrap();
            let x_max = *threads.iter().min().unwrap();
            let x_range = x_min..x_max;

            let values = schedule.iter().flat_map(|s| s.ys()).collect::<Vec<_>>();
            let y_range = fiter_range(&values).unwrap();

            let caption = format!(
                "wordchipper:{chart_name} {scale_desc} throughput, rust, model: \"{model}\"",
            );

            // ATTENTION: This is weird.
            // The plotters chart machinery makes extensive and heavy use of specialized
            // generic builders, *including* the management of the axis range type.
            //
            // As a result, the choice of range ends up polluting the base type.
            macro_rules! draw_chart {
                ($y_axis:expr) => {{
                    let mut chart = ChartBuilder::on(&root)
                        .caption(caption, ("sans-serif", 20).into_font())
                        .margin(10)
                        .x_label_area_size(40)
                        .y_label_area_size(70)
                        .build_cartesian_2d(x_range.log_scale().base(2.0), $y_axis)?;

                    chart
                        .configure_mesh()
                        .x_desc("Thread Count")
                        .y_desc(format!("Median Throughput: {scale_desc} scale"))
                        .y_label_formatter(&|&bps| human_format::format_bps(bps))
                        .draw()?;

                    for ms in schedule {
                        chart.draw_series(LineSeries::new(
                            ms.points.clone(),
                            ms.style.line_style().stroke_width(LINE_WIDTH),
                        ))?;

                        chart
                            .draw_series(
                                ms.points
                                    .into_iter()
                                    .map(|coords| ms.style.marker(coords, SIZE)),
                            )?
                            .label(ms.name)
                            .legend(move |coord| ms.style.marker(coord, SIZE));
                    }

                    chart
                        .configure_series_labels()
                        .position(SeriesLabelPosition::UpperLeft)
                        .background_style(WHITE.mix(0.8))
                        .margin(12)
                        .legend_area_size(15)
                        .border_style(BLACK)
                        .draw()?;

                    Ok::<_, Box<dyn std::error::Error>>(())
                }};
            }

            if log_scale {
                draw_chart!(y_range.log_scale().base(2.0))?;
            } else {
                draw_chart!(y_range)?;
            }

            root.present()?;
        }
    }

    Ok(())
}
