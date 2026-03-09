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
    bench_data,
    bench_data::par_bench::ParBenchData,
    bounds_tools,
    bounds_tools::iter_frange,
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
        let output_dir = Path::new(&self.output_dir);

        let shape = (800, 600);

        let par_data = ParBenchData::load_data(data_dir.join("rust_parallel"))?;
        let par_output = output_dir.join("rust_parallel");
        std::fs::create_dir_all(&par_output)?;
        for model in self.models.iter() {
            build_model_graphs(model, &par_output, shape, &par_data)?;
        }

        Ok(())
    }
}

fn build_model_graphs<P: AsRef<Path>>(
    model: &str,
    output_dir: &P,
    shape: (u32, u32),
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let (w, h) = shape;

    let tall_shape = (w, h * 3 / 2);

    build_throughput_graph(model, "buffer_sweep", tall_shape, &output_dir, data)?;

    build_rel_span_encoder_graphs(model, tall_shape, &output_dir, data)
}

fn build_rel_span_encoder_graphs<P: AsRef<Path>>(
    model: &str,
    shape: (u32, u32),
    output_dir: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    const SIZE: i32 = 6;
    const LINE_WIDTH: u32 = 4;

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
        format!("model: \"{model}\"",),
        (title_area.dim_in_pixel().0 as i32 / 2, 40),
        subtitle_style,
    ))?;

    let charts = chart_area.margin(10, 10, 10, 10);

    let sub_charts = charts.split_evenly((3, 1));

    for (idx, (lexer_label, model_suffix)) in [
        ("fast-regex", ""),
        ("regex-automata", "_ra"),
        ("logos", "_fast"),
    ]
    .into_iter()
    .enumerate()
    {
        let drawing_area = &sub_charts[idx];

        let sel_model = format!("{model}{model_suffix}");

        let span_key = |span: &str| format!("encoding_parallel::wordchipper::{span}::{sel_model}",);

        let base_style = MarkerStyle::default().with_stroke_style(colors::BLACK.stroke_width(2));
        let span_styles = [
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
            (*threads, bench_data::median_bps(bench_results))
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

        let x_range = match bounds_tools::iter_range(render.iter().flat_map(|s| s.xs())) {
            Some(r) => r,
            None => {
                log::warn!("No data for {}::{}::{}", model, lexer_label, model_suffix);
                return Ok(());
            }
        };

        let mut chart = ChartBuilder::on(drawing_area)
            .caption(lexer_label, ("sans-serif", 20).into_font())
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range.log_scale().base(2.0), 0.8..1.0)?;

        if idx == sub_charts.len() - 1 {
            chart
                .configure_mesh()
                .x_desc("Thread Count")
                .y_desc("Relative Median Throughput")
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .y_desc("Relative Median Throughput")
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
                .position(SeriesLabelPosition::LowerLeft)
                .margin(12)
                .legend_area_size(15)
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()?;
        }
    }

    root.present()?;

    Ok(())
}

fn build_throughput_graph<P: AsRef<Path>>(
    model: &str,
    span_encoder: &str,
    shape: (u32, u32),
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
                .with_marker_type(MarkerType::CrossDiamond)
                .with_fill_style(Some(colors::BLUEGREY_100.into())),
        ),
    ]
    .into_iter()
    .filter_map(|(name, style)| {
        data.select_series(&format!("encoding_parallel::{name}::{model}"))
            .map(|series_data| MarkerSeries::new(name, style, series_data))
    })
    .collect();

    let fr_series = MarkerSeries::new(
        "wordchipper:fancy-regex",
        base_style
            .with_marker_type(MarkerType::TriUp)
            .with_marker_level(MarkerLevel::Para)
            .with_fill_style(colors::GREEN_A200.filled()),
        data.try_select_series(&format!(
            "encoding_parallel::wordchipper::{span_encoder}::{model}"
        ))?,
    );

    let ra_series = MarkerSeries::new(
        "wordchipper:regex-automata",
        base_style
            .with_marker_type(MarkerType::Diamond)
            .with_marker_level(MarkerLevel::Para)
            .with_fill_style(colors::AMBER_A200.filled()),
        data.try_select_series(&format!(
            "encoding_parallel::wordchipper::{span_encoder}::{model}_ra"
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

    const SIZE: i32 = 6;
    const LINE_WIDTH: u32 = 4;

    for (chart_name, group) in [
        ("fast_regex", vec![&fr_series]),
        ("ra", vec![&fr_series, &ra_series]),
        ("logos", vec![&fr_series, &ra_series, &logos_series]),
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
            format!("model: \"{model}\"",),
            (title_area.dim_in_pixel().0 as i32 / 2, 40),
            subtitle_style,
        ))?;

        /*
        let caption = format!(
            "wordchipper:{chart_name} {scale_desc} throughput, rust, model: \"{model}\"",
        );
         */
        let (top, bottom) = charts.split_vertically(root.dim_in_pixel().1 / 3);

        for (is_top, log_scale, da) in [(true, false, top), (false, true, bottom)] {
            let scale_desc = if log_scale { "log" } else { "linear" };

            fn select((threads, bench_results): &(u32, BenchResult)) -> (u32, f64) {
                (*threads, bench_data::median_bps(bench_results))
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
                    .map(bench_data::median_bps)
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
                        .margin(0)
                        .x_label_area_size(40)
                        .y_label_area_size(70)
                        .build_cartesian_2d(x_range.log_scale().base(2.0), $y_range)?;

                    if is_top {
                        chart
                            .configure_mesh()
                            .y_desc(format!("Median Throughput: {scale_desc}"))
                            .y_label_formatter(&|&bps| human_format::format_bps(bps))
                            .draw()?;
                    } else {
                        chart
                            .configure_mesh()
                            .x_desc("Thread Count")
                            .y_desc(format!("Median Throughput: {scale_desc}"))
                            .y_label_formatter(&|&bps| human_format::format_bps(bps))
                            .draw()?;
                    }

                    // Render the lines under the markers.
                    for ms in schedule.iter() {
                        chart.draw_series(LineSeries::new(
                            ms.map_points(select),
                            ms.style.line_style().stroke_width(LINE_WIDTH),
                        ))?;
                    }

                    for ms in schedule.iter() {
                        chart
                            .draw_series(ms.points.iter().map(|(threads, bench_results)| {
                                let bps = bench_data::median_bps(bench_results);
                                let coord = (*threads, bps);
                                ms.style.marker(coord, SIZE)
                            }))?
                            .label(ms.name.clone())
                            .legend(move |coord| ms.style.marker(coord, SIZE));
                    }

                    /*
                    if log_scale {
                        for ms in schedule.iter() {
                            chart
                                .draw_series(ms.points.iter().map(|(threads, bench_results)| {
                                    let bps = median_bps(bench_results);
                                    let coord = (*threads, bps);
                                    let allocs = alloc_count(bench_results).unwrap_or(0);
                                    let deallocs = dealloc_count(bench_results).unwrap_or(0);

                                    EmptyElement::at(coord)
                                        + Text::new(
                                            format!("{}:{}", allocs, deallocs),
                                            (5, 10),
                                            ("sans-serif", 12).into_font(),
                                        )
                                }))?
                                .label(ms.name.clone())
                                .legend(move |coord| ms.style.marker(coord, SIZE));
                        }
                    }
                     */

                    if is_top {
                        chart
                            .configure_series_labels()
                            .position(SeriesLabelPosition::UpperLeft)
                            .background_style(WHITE.mix(0.8))
                            .margin(SIZE * 2)
                            .legend_area_size(15)
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
