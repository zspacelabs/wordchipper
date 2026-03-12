use std::path::Path;

use plotters::{
    backend::SVGBackend,
    element::Text,
    prelude::{
        BLACK,
        IntoFont,
        TextStyle,
        *,
    },
    style::full_palette as colors,
};
use plotters_backend::text_anchor::{
    HPos,
    Pos,
    VPos,
};
use serde_json::Value;
use wordchipper_cli_util::logging::LogArgs;

use crate::util::{
    bench_data::{
        PythonParBenchData,
        py_bench_median_bps,
    },
    bounds_tools,
    bounds_tools::iter_frange,
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
pub struct PythonBenchPlots {
    /// Model name.
    #[clap(
        long,
        value_delimiter = ',',
        default_value = "gpt2,cl100k_base,o200k_base"
    )]
    models: Vec<String>,

    /// Path to the benchmark data.
    #[clap(long, default_value = "benchmarks/amd3990X/data")]
    data_dir: String,

    /// Path to the output directory.
    #[clap(long, default_value = "target/plots")]
    output_dir: String,

    #[clap(flatten)]
    logging: LogArgs,

    /// Display the parsed series.
    #[clap(long)]
    debug_series: bool,
}

impl PythonBenchPlots {
    /// Run the command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;
        log::info!("{:#?}", self);

        const SHAPE: (u32, u32) = (800, 600);
        const PAR_DIR: &str = "python_parallel";

        let data_dir = Path::new(&self.data_dir);
        let output_dir = Path::new(&self.output_dir);

        let par_data = PythonParBenchData::load_data(data_dir.join(PAR_DIR))?;

        if self.debug_series {
            let names = par_data.series_names();
            log::info!("Series: {:#?}", names);
        }

        let par_output = output_dir.join(PAR_DIR);
        std::fs::create_dir_all(&par_output)?;

        for model in self.models.iter() {
            build_model_graphs(model, &par_output, SHAPE, &par_data)?;
        }

        Ok(())
    }
}

fn build_model_graphs<P: AsRef<Path>>(
    model: &str,
    output_dir: &P,
    shape: (u32, u32),
    data: &PythonParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let (w, h) = shape;

    let tall_shape = (w, h * 3 / 2);

    build_throughput_graph(model, tall_shape, &output_dir, data)?;

    Ok(())
}

#[allow(unused)]
fn build_throughput_graph<P: AsRef<Path>>(
    model: &str,
    shape: (u32, u32),
    output_dir: &P,
    data: &PythonParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let mut groups: Vec<MarkerSeries<(u32, Value)>> = Default::default();

    if let Some(series) = data.select_series(&format!("tokenizers[{model}]")) {
        groups.push(MarkerSeries::new(
            "tokenizers",
            MarkerStyle::default()
                .with_marker_type(MarkerType::Diamond)
                .with_fill_style(Some(colors::BLUEGREY_100.into())),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("tiktoken[{model}]")) {
        groups.push(MarkerSeries::new(
            "tiktoken",
            MarkerStyle::default()
                .with_marker_type(MarkerType::CrossSquare)
                .with_fill_style(Some(colors::PURPLE_100.into())),
            series,
        ));
    }
    /*
    if let Some(series) = data.select_series(&format!("tokenizers_threadpool[{model}]")) {
        groups.push(MarkerSeries::new(
            "tokenizers::threadpool",
            MarkerStyle::default()
                .with_marker_type(MarkerType::CrossDiamond)
                .with_fill_style(Some(colors::BLUEGREY_100.into())),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("tiktoken_threadpool[{model}]")) {
        groups.push(MarkerSeries::new(
            "tiktoken::threadpool",
            MarkerStyle::default()
                .with_marker_type(MarkerType::Square)
                .with_fill_style(Some(colors::PURPLE_200.into())),
            series,
        ));
    }
     */

    if let Some(series) = data.select_series(&format!("wordchipper_parallel[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::rayon::regex-automata (default)",
            MarkerStyle::default()
                .with_marker_type(MarkerType::TriUp)
                .with_marker_level(MarkerLevel::Para)
                .with_fill_style(Some(colors::GREEN_A200.into())),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("wordchipper_parallel_accel[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::rayon::logos (custom per pattern)",
            MarkerStyle::default()
                .with_marker_type(MarkerType::TriDown)
                .with_marker_level(MarkerLevel::Para)
                .with_fill_style(Some(colors::LIGHTBLUE_A200.into()))
                .with_dash_style(DashStyle {
                    size: 4,
                    spacing: 8,
                }),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("wordchipper_threadpool[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::threadpool::regex-automata (default)",
            MarkerStyle::default()
                .with_marker_type(MarkerType::TriUp)
                .with_fill_style(Some(colors::LIGHTGREEN_A200.into())),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("wordchipper_threadpool_accel[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::threadpool::logos (custom per pattern)",
            MarkerStyle::default()
                .with_marker_type(MarkerType::TriDown)
                .with_fill_style(Some(colors::BLUEGREY_A200.into()))
                .with_dash_style(DashStyle {
                    size: 4,
                    spacing: 8,
                }),
            series,
        ));
    }

    log::info!(
        "loaded: {:#?}",
        groups
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    );

    const SIZE: i32 = 7;
    const LINE_WIDTH: u32 = 5;

    let plot_path = output_dir.join(format!("wc_vrs_brandx.py.{model}.svg"));
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
        "wordchipper python throughput",
        (title_area.dim_in_pixel().0 as i32 / 2, 10),
        title_style,
    ))?;
    title_area.draw(&Text::new(
        format!("model: \"{model}\"",),
        (title_area.dim_in_pixel().0 as i32 / 2, 40),
        subtitle_style,
    ))?;

    let (top, bottom) = charts.split_vertically(charts.dim_in_pixel().1 / 2);
    for (is_top, log_scale, da) in [(true, false, top), (false, true, bottom)] {
        let scale_desc = if log_scale { "log" } else { "linear" };

        fn select((threads, bench_results): &(u32, Value)) -> (u32, f64) {
            (*threads, py_bench_median_bps(bench_results))
        }

        // Range over all thread values.
        let x_range = bounds_tools::iter_range(groups.iter().flat_map(|s| s.xs())).unwrap();
        // Range over all bps values.
        let y_range = iter_frange(
            groups
                .iter()
                .flat_map(|s| s.ys().iter().map(py_bench_median_bps).collect::<Vec<f64>>()),
        )
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
                    .y_label_area_size(80)
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
                for ms in groups.iter() {
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

                for ms in groups.iter() {
                    chart
                        .draw_series(ms.points.iter().map(|(threads, bench_results)| {
                            let bps = py_bench_median_bps(bench_results);
                            let coord = (*threads, bps);
                            ms.style.marker(coord, SIZE)
                        }))?
                        .label(ms.name.clone())
                        .legend(move |coord| ms.style.marker(coord, SIZE));
                }

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

    Ok(())
}
