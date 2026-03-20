use std::path::Path;

use plotters::{
    chart::SeriesLabelPosition,
    style::full_palette as colors,
};
use serde_json::Value;

use crate::{
    commands::benchmark_plots::common_plots::{
        LegendLocation,
        build_throughput_plot,
    },
    util::{
        bench_data::{
            PythonParBenchData,
            py_bench_median_bps,
        },
        plotting::{
            DashStyle,
            GraphStyleOptions,
            MarkerLevel,
            MarkerSeries,
            MarkerStyle,
            MarkerType,
        },
    },
};

pub(crate) fn build_python_benchmark_plots<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    output_dir: &P,
    options: GraphStyleOptions,
    data: &PythonParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    build_python_throughput_graph(arch, model, options, &output_dir, data)?;

    Ok(())
}

fn build_python_throughput_graph<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    options: GraphStyleOptions,
    output_dir: &P,
    data: &PythonParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let mut groups: Vec<MarkerSeries<(u32, Value)>> = Default::default();

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

    if let Some(series) = data.select_series(&format!("wordchipper_parallel_accel[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::rayon::logos",
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
    if let Some(series) = data.select_series(&format!("wordchipper_parallel[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::rayon::regex-automata",
            MarkerStyle::default()
                .with_marker_type(MarkerType::TriUp)
                .with_marker_level(MarkerLevel::Para)
                .with_fill_style(Some(colors::GREEN_A200.into())),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("wordchipper_threadpool_accel[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::threadpool::logos",
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
    if let Some(series) = data.select_series(&format!("wordchipper_threadpool[{model}]")) {
        groups.push(MarkerSeries::new(
            "wc::threadpool::regex-automata",
            MarkerStyle::default()
                .with_marker_type(MarkerType::TriUp)
                .with_fill_style(Some(colors::LIGHTGREEN_A200.into()))
                .with_dash_style(DashStyle {
                    size: 4,
                    spacing: 8,
                }),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("tiktoken[{model}]")) {
        groups.push(MarkerSeries::new(
            "tiktoken",
            MarkerStyle::default()
                .with_marker_type(MarkerType::CrossSquare)
                .with_fill_style(Some(colors::PURPLE_A100.into())),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("bpe_openai[{model}]")) {
        groups.push(MarkerSeries::new(
            "bpe-openai",
            MarkerStyle::default()
                .with_marker_type(MarkerType::Circle)
                .with_fill_style(Some(colors::ORANGE_A100.into())),
            series,
        ));
    }
    if let Some(series) = data.select_series(&format!("tokenizers[{model}]")) {
        groups.push(MarkerSeries::new(
            "tokenizers",
            MarkerStyle::default()
                .with_marker_type(MarkerType::Diamond)
                .with_fill_style(Some(colors::BLUEGREY_A100.into())),
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

    let series: Vec<MarkerSeries<(u32, f64)>> = groups
        .into_iter()
        .map(|ms| ms.map(|(t, br)| (*t, py_bench_median_bps(br))))
        .collect();

    let plot_path_stem = output_dir.join(format!("wc_vrs_brandx.py.{model}"));
    build_throughput_plot(
        "wordchipper python throughput",
        &format!("arch: \"{arch}\", model: \"{model}\""),
        options,
        &plot_path_stem,
        &series,
        LegendLocation {
            top: false,
            label_pos: SeriesLabelPosition::LowerRight,
        },
    )
}
