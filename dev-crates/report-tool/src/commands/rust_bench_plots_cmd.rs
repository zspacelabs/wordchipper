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
use plotters_backend::text_anchor::{
    HPos,
    Pos,
    VPos,
};
use wordchipper_cli_util::logging::LogArgs;

use crate::util::{
    bench_data::{
        RustParBenchData,
        rust_bench_median_bps,
    },
    bounds_tools,
    bounds_tools::{
        iter_frange,
        iter_range,
    },
    human_format,
    plotting::{
        DashStyle,
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

    /// Machine arch name.
    #[clap(long, default_value = "amd3990X")]
    arch: String,

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
        let output_dir = Path::new(&self.output_dir);

        let shape = (800, 600);

        let par_data = RustParBenchData::load_data(data_dir.join("rust_parallel"))?;

        let par_output = output_dir.join("rust_parallel");
        std::fs::create_dir_all(&par_output)?;

        for model in self.models.iter() {
            build_model_graphs(&self.arch, model, &par_output, shape, &par_data)?;
        }

        Ok(())
    }
}

const SIZE: i32 = 10;
const LINE_WIDTH: u32 = 6;

fn build_model_graphs<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    output_dir: &P,
    shape: (u32, u32),
    data: &RustParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let (w, h) = shape;

    let tall_shape = (w, h * 3 / 2);

    build_throughput_graph(arch, model, "buffer_sweep", tall_shape, &output_dir, data)?;

    build_rel_span_encoder_graphs(arch, model, tall_shape, &output_dir, data)
}

fn build_rel_span_encoder_graphs<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    shape: (u32, u32),
    output_dir: &P,
    data: &RustParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let plot_path = output_dir.join(format!("span_encoder_relative.rust.{model}.svg"));
    log::info!("Plotting to {}", plot_path.display());
    let root = SVGBackend::new(&plot_path, shape).into_drawing_area();
    root.fill(&colors::WHITE)?;
    let (title_area, chart_area) = root.split_vertically(60);
    let title_style = TextStyle {
        font: ("sans-serif", 24).into_font(),
        color: BLACK.to_backend_color(),
        pos: Pos::new(HPos::Center, VPos::Top),
    };
    let subtitle_style = TextStyle {
        font: ("sans-serif", 18).into_font(),
        color: BLACK.to_backend_color(),
        pos: Pos::new(HPos::Center, VPos::Top),
    };

    title_area.draw(&Text::new(
        "SpanEncoder Relative Throughput",
        (title_area.dim_in_pixel().0 as i32 / 2, 10),
        title_style,
    ))?;
    title_area.draw(&Text::new(
        format!("arch: \"{arch}\", model: \"{model}\"",),
        (title_area.dim_in_pixel().0 as i32 / 2, 40),
        subtitle_style,
    ))?;

    let charts = chart_area.margin(10, 10, 10, 10);

    let sub_charts = charts.split_evenly((3, 1));

    for (idx, (lexer_label, lexer_key)) in [
        ("logos", "logos"),
        ("regex-automata", "regex_automata"),
        ("fast-regex", "regex"),
    ]
    .into_iter()
    .enumerate()
    {
        let drawing_area = &sub_charts[idx];

        let span_key =
            |span: &str| format!("encoding_parallel::{model}::wordchipper::{lexer_key}::{span}");

        let span_styles = [
            (
                "buffer_sweep",
                MarkerStyle::default()
                    .with_marker_type(MarkerType::Circle)
                    .with_fill_style(Some(colors::GREEN_200.into())),
            ),
            (
                "tail_sweep",
                MarkerStyle::default()
                    .with_marker_type(MarkerType::Diamond)
                    .with_fill_style(Some(colors::DEEPORANGE_200.into())),
            ),
            (
                "merge_heap",
                MarkerStyle::default()
                    .with_marker_type(MarkerType::TriDown)
                    .with_fill_style(Some(colors::BLUEGREY_200.into())),
            ),
            (
                "priority_merge",
                MarkerStyle::default()
                    .with_marker_type(MarkerType::Square)
                    .with_fill_style(Some(colors::PURPLE_200.into())),
            ),
            (
                "bpe_backtrack",
                MarkerStyle::default()
                    .with_marker_type(MarkerType::TriUp)
                    .with_fill_style(Some(colors::LIGHTBLUE_200.into())),
            ),
        ];

        let mut plot_series: Vec<MarkerSeries<(u32, BenchResult)>> = Default::default();
        for (name, style) in span_styles.iter() {
            if let Some(points) = data.select_series(&span_key(name)) {
                plot_series.push(MarkerSeries {
                    name: name.to_string(),
                    style: *style,
                    points,
                })
            }
        }

        fn select((threads, bench_results): &(u32, BenchResult)) -> (u32, f64) {
            (*threads, rust_bench_median_bps(bench_results))
        }

        let render: Vec<MarkerSeries<(u32, f64)>> =
            plot_series.iter().map(|s| s.map(select)).collect();

        // Normalize the points to the max value.
        let mut baseline: BTreeMap<u32, f64> = Default::default();
        for ms in render.iter() {
            for &(t, v) in ms.points.iter() {
                let entry = baseline.entry(t).or_default();
                *entry = bounds_tools::fmax(*entry, v);
            }
        }
        let render: Vec<MarkerSeries<(u32, f64)>> = render
            .into_iter()
            .map(|s| s.map(|&(t, v)| (t, v / baseline[&t])))
            .collect();

        let x_range = match iter_range(render.iter().flat_map(|s| s.xs())) {
            Some(r) => r,
            None => {
                log::warn!("No data for {}::{}", model, lexer_label);
                return Ok(());
            }
        };

        let y_range = match iter_frange(render.iter().flat_map(|s| s.ys())) {
            Some(r) => r,
            None => {
                log::warn!("No data for {}::{}", model, lexer_label);
                return Ok(());
            }
        };

        let show_x_label = idx == 2;

        let mut chart = ChartBuilder::on(drawing_area)
            .caption(
                format!("lexer: {}", lexer_label),
                ("sans-serif", 20).into_font(),
            )
            .margin(10)
            .x_label_area_size(if show_x_label { 60 } else { 0 })
            .y_label_area_size(70)
            .build_cartesian_2d(x_range.log_scale().base(2.0), y_range)?;

        if show_x_label {
            chart
                .configure_mesh()
                .x_desc("Thread Count")
                .x_label_style(("sans-serif", 20.0).into_font())
                .y_desc("Relative Median Throughput")
                .y_label_style(("sans-serif", 20.0).into_font())
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .x_label_style(("sans-serif", 20.0).into_font())
                .y_desc("Relative Median Throughput")
                .y_label_style(("sans-serif", 20.0).into_font())
                .draw()?;
        }

        // Render the lines under the markers.
        for ms in render.iter() {
            chart.draw_series(LineSeries::new(
                ms.points.clone(),
                ms.style.line_style().stroke_width(LINE_WIDTH),
            ))?;
        }

        for ms in render.iter() {
            chart
                .draw_series(ms.points.iter().map(|&coord| ms.style.marker(coord, SIZE)))?
                .label(ms.name.clone())
                .legend(move |coord| ms.style.marker(coord, SIZE));
        }

        if idx == 0 {
            chart
                .configure_series_labels()
                .label_font(("sans-serif", 24).into_font())
                .position(SeriesLabelPosition::LowerLeft)
                .margin(12)
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()?;
        }
    }

    root.present()?;

    Ok(())
}

fn build_throughput_graph<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    span_encoder: &str,
    shape: (u32, u32),
    output_dir: &P,
    data: &RustParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let base_style = MarkerStyle::default().with_stroke_style(colors::BLACK.stroke_width(1));

    let external: Vec<MarkerSeries<(u32, BenchResult)>> = vec![
        (
            "bpe_openai",
            base_style
                .with_marker_type(MarkerType::CrossCircle)
                .with_fill_style(Some(colors::PINK_100.into())),
        ),
        (
            "tiktoken",
            base_style
                .with_marker_type(MarkerType::CrossSquare)
                .with_fill_style(Some(colors::PURPLE_100.into())),
        ),
        (
            "tokenizers",
            base_style
                .with_marker_type(MarkerType::CrossDiamond)
                .with_fill_style(Some(colors::BLUEGREY_100.into())),
        ),
    ]
    .into_iter()
    .filter_map(|(name, style)| {
        data.select_series(&format!("encoding_parallel::{model}::{name}"))
            .map(|series_data| MarkerSeries::new(name, style, series_data))
    })
    .collect();

    let fr_series = MarkerSeries::new(
        "wc:fancy-regex (fallback)",
        base_style
            .with_marker_type(MarkerType::TriUp)
            .with_marker_level(MarkerLevel::Para)
            .with_fill_style(colors::AMBER_A200.filled()),
        data.try_select_series(&format!(
            "encoding_parallel::{model}::wordchipper::regex::{span_encoder}"
        ))?,
    );

    let ra_series = MarkerSeries::new(
        "wc:regex-automata (default)",
        base_style
            .with_marker_type(MarkerType::Diamond)
            .with_marker_level(MarkerLevel::Para)
            .with_fill_style(colors::GREEN_A200.filled()),
        data.try_select_series(&format!(
            "encoding_parallel::{model}::wordchipper::regex_automata::{span_encoder}"
        ))?,
    );

    let logos_series = MarkerSeries::new(
        "wc:logos (custom per pattern)",
        base_style
            .with_marker_type(MarkerType::TriDown)
            .with_marker_level(MarkerLevel::Para)
            .with_fill_style(colors::LIGHTBLUE_A200.filled())
            .with_dash_style(DashStyle {
                size: 4,
                spacing: 8,
            }),
        data.try_select_series(&format!(
            "encoding_parallel::{model}::wordchipper::logos::{span_encoder}"
        ))?,
    );

    for (chart_name, group) in [
        ("fast_regex", vec![&fr_series]),
        ("ra", vec![&ra_series, &fr_series]),
        ("logos", vec![&logos_series, &ra_series, &fr_series]),
    ] {
        let plot_path = output_dir.join(format!("wc_{chart_name}_vrs_brandx.rust.{model}.svg"));
        log::info!("Plotting to {}", plot_path.display());

        let root = SVGBackend::new(&plot_path, shape).into_drawing_area();
        root.fill(&colors::WHITE)?;
        let (title_area, chart_area) = root.split_vertically(60);

        let charts = chart_area.margin(10, 10, 10, 10);

        let title_style = TextStyle {
            font: ("sans-serif", 24).into_font(),
            color: BLACK.to_backend_color(),
            pos: Pos::new(HPos::Center, VPos::Top),
        };
        let subtitle_style = TextStyle {
            font: ("sans-serif", 18).into_font(),
            color: BLACK.to_backend_color(),
            pos: Pos::new(HPos::Center, VPos::Top),
        };

        title_area.draw(&Text::new(
            "wordchipper rust throughput",
            (title_area.dim_in_pixel().0 as i32 / 2, 10),
            title_style,
        ))?;
        title_area.draw(&Text::new(
            format!("arch: \"{arch}\", model: \"{model}\"",),
            (title_area.dim_in_pixel().0 as i32 / 2, 40),
            subtitle_style,
        ))?;

        /*
        let caption = format!(
            "wordchipper:{chart_name} {scale_desc} throughput, rust, model: \"{model}\"",
        );
         */
        let (top, bottom) = charts.split_vertically(charts.dim_in_pixel().1 / 2);

        for (is_top, log_scale, da) in [(true, false, top), (false, true, bottom)] {
            let scale_desc = if log_scale { "log" } else { "linear" };

            fn select((threads, bench_results): &(u32, BenchResult)) -> (u32, f64) {
                (*threads, rust_bench_median_bps(bench_results))
            }

            let mut schedule: Vec<MarkerSeries<(u32, BenchResult)>> = Default::default();
            for &g in &group {
                schedule.push(g.clone());
            }
            schedule.extend(external.clone());

            // Range over all thread values.
            let x_range = bounds_tools::iter_range(schedule.iter().flat_map(|s| s.xs())).unwrap();
            // Range over all bps values.
            let y_range = iter_frange(schedule.iter().flat_map(|s| {
                s.ys()
                    .iter()
                    .map(rust_bench_median_bps)
                    .collect::<Vec<f64>>()
            }))
            .unwrap();

            // ATTENTION: This is weird.
            // The plotters chart machinery makes extensive and heavy use of specialized
            // generic builders, *including* the management of the axis range type.
            //
            // As a result, the choice of range ends up polluting the base type.
            macro_rules! draw_chart {
                ($y_range:expr) => {{
                    let mut chart = ChartBuilder::on(&da)
                        //               .caption(caption, ("sans-serif", 20).into_font())
                        .margin(10)
                        .x_label_area_size(40)
                        .y_label_area_size(120)
                        .build_cartesian_2d(x_range.log_scale().base(2.0), $y_range)?;

                    if is_top {
                        chart
                            .configure_mesh()
                            .x_label_style(("sans-serif", 20.0).into_font())
                            .y_label_style(("sans-serif", 20.0).into_font())
                            .y_desc(format!("Median Throughput: {scale_desc}"))
                            .y_label_formatter(&|&bps| human_format::format_bps(bps))
                            .draw()?;
                    } else {
                        chart
                            .configure_mesh()
                            .x_desc("Thread Count")
                            .y_desc(format!("Median Throughput: {scale_desc}"))
                            .x_label_style(("sans-serif", 20.0).into_font())
                            .y_label_style(("sans-serif", 20.0).into_font())
                            .y_label_formatter(&|&bps| human_format::format_bps(bps))
                            .draw()?;
                    }

                    // Render the lines under the markers.
                    for ms in schedule.iter() {
                        if let Some(dash_style) = ms.style.dash_style {
                            chart.draw_series(DashedLineSeries::new(
                                ms.map_points(select),
                                dash_style.size,
                                dash_style.spacing,
                                ms.style.line_style().stroke_width(LINE_WIDTH),
                            ))?;
                        } else {
                            chart.draw_series(LineSeries::new(
                                ms.map_points(select),
                                ms.style.line_style().stroke_width(LINE_WIDTH),
                            ))?;
                        }
                    }

                    for ms in schedule.iter() {
                        chart
                            .draw_series(ms.points.iter().map(|(threads, bench_results)| {
                                let bps = rust_bench_median_bps(bench_results);
                                let coord = (*threads, bps);
                                ms.style.marker(coord, SIZE)
                            }))?
                            .label(ms.name.clone())
                            .legend(move |coord| ms.style.marker(coord, SIZE));
                    }

                    if is_top {
                        chart
                            .configure_series_labels()
                            .label_font(("sans-serif", 22).into_font())
                            .position(SeriesLabelPosition::UpperLeft)
                            .background_style(WHITE.mix(0.8))
                            .margin(SIZE * 2)
                            .border_style(BLACK)
                            .draw()?;
                    }

                    Ok::<_, Box<dyn std::error::Error>>(())
                }};
            }

            if log_scale {
                draw_chart!(y_range.log_scale().base(2.0))?;
            } else {
                draw_chart!(y_range)?;
            }
        }

        root.present()?;
    }

    Ok(())
}
