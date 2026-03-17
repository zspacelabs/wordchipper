use std::path::Path;

use divan_parser::BenchResult;
use plotters::{
    prelude::*,
    style::full_palette as colors,
};
use plotters_backend::text_anchor::{
    HPos,
    Pos,
    VPos,
};

use crate::{
    commands::benchmark_plots::{
        graph_style::GraphStyleOptions,
        plots,
    },
    util::{
        bench_data::{
            RustParBenchData,
            rust_bench_median_bps,
        },
        plotting::{
            DashStyle,
            MarkerLevel,
            MarkerSeries,
            MarkerStyle,
            MarkerType,
        },
    },
};

pub fn build_rust_benchmark_plots<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    output_dir: &P,
    options: GraphStyleOptions,
    data: &RustParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    build_rust_throughput_plots(arch, model, "buffer_sweep", options, &output_dir, data)?;

    build_rust_relative_span_encoder_plots(arch, model, options, &output_dir, data)
}

#[allow(clippy::type_complexity)]
fn build_rust_relative_span_encoder_plots<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    options: GraphStyleOptions,
    output_dir: &P,
    data: &RustParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let mut lexer_groups: Vec<(String, Vec<MarkerSeries<(u32, f64)>>)> = Default::default();

    for (lexer_label, lexer_key) in [
        ("logos", "logos"),
        ("regex-automata", "regex_automata"),
        ("fast-regex", "regex"),
    ]
    .into_iter()
    {
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

        lexer_groups.push((
            lexer_label.to_string(),
            plot_series
                .iter()
                .map(|s| s.map(|(t, br)| (*t, rust_bench_median_bps(br))))
                .collect(),
        ));
    }

    let view: Vec<(&str, &[MarkerSeries<(u32, f64)>])> = lexer_groups
        .iter()
        .map(|(lexer_label, group)| (lexer_label.as_str(), group.as_slice()))
        .collect();

    let plot_path = output_dir.join(format!("span_encoder_relative.rust.{model}.svg"));

    plots::build_relative_span_encoder_plot(
        "SpanEncoder Relative Throughput",
        &format!("arch: \"{arch}\", model: \"{model}\""),
        options,
        &plot_path,
        &view,
    )
}

fn build_rust_throughput_plots<P: AsRef<Path>>(
    arch: &str,
    model: &str,
    span_encoder: &str,
    options: GraphStyleOptions,
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
        let mut schedule: Vec<MarkerSeries<(u32, BenchResult)>> = Default::default();
        schedule.extend(group.clone());
        schedule.extend(external.clone());

        let series: Vec<MarkerSeries<(u32, f64)>> = schedule
            .into_iter()
            .map(|ms| ms.map(|(t, br)| (*t, rust_bench_median_bps(br))))
            .collect();

        let plot_path = output_dir.join(format!("wc_{chart_name}_vrs_brandx.rust.{model}.svg"));

        plots::build_throughput_plot(
            "wordchipper rust throughput",
            &format!("arch: \"{arch}\", model: \"{model}\""),
            &series,
            options,
            &plot_path,
        )?;
    }

    Ok(())
}
