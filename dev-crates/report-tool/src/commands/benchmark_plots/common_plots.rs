use std::{
    collections::BTreeMap,
    fs::File,
    io::{
        BufWriter,
        Write,
    },
    path::Path,
};

use plotters::{
    backend::SVGBackend,
    chart::{
        ChartBuilder,
        SeriesLabelPosition,
    },
    drawing::IntoDrawingArea,
    element::Text,
    prelude::{
        BLACK,
        Color,
        DashedLineSeries,
        ErrorBar,
        IntoFont,
        IntoLogRange,
        LineSeries,
        TextStyle,
        WHITE,
        full_palette,
    },
};
use plotters_backend::text_anchor::{
    HPos,
    Pos,
    VPos,
};

use crate::util::{
    bounds_tools,
    bounds_tools::{
        iter_frange,
        iter_range,
    },
    human_format,
    plotting::{
        GraphStyleOptions,
        MarkerSeries,
    },
};

#[derive(Debug, Clone)]
pub struct LegendLocation {
    pub top: bool,
    pub label_pos: SeriesLabelPosition,
}

impl Default for LegendLocation {
    fn default() -> Self {
        Self {
            top: true,
            label_pos: SeriesLabelPosition::UpperRight,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MinMaxStat {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

pub fn build_minmax_relative_encoder_table<P: AsRef<Path>>(
    stem_path: &P,
    lexer_groups: &[(&str, Vec<MarkerSeries<(u32, MinMaxStat)>>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let stem_path = stem_path.as_ref();
    let table_path = stem_path.with_added_extension("csv");

    log::info!("Writing table to {}", table_path.display());
    let mut writer = BufWriter::new(File::create(table_path)?);

    let threads = lexer_groups.first().unwrap().1.first().unwrap().xs();

    let mut header = vec!["Lexer".to_string(), "Encoder".to_string()];
    header.extend(threads.iter().map(|t| format!("Thread {}", t)));
    writeln!(writer, "{}", header.join(", "))?;

    for (lexer_label, group) in lexer_groups.iter() {
        for series in group.iter() {
            assert_eq!(threads, series.xs());

            let mut row = vec![lexer_label.to_string(), series.name.clone()];
            row.extend(
                series
                    .ys()
                    .iter()
                    .map(|stat| format!("{:.2}:{:.2}:{:.2}", stat.min, stat.mean, stat.max)),
            );

            writeln!(writer, "{}", row.join(", "))?;
        }
    }

    writer.flush()?;

    drop(writer);

    Ok(())
}

pub fn build_minmax_relative_span_encoder_plot<P: AsRef<Path>>(
    title: &str,
    caption: &str,
    options: GraphStyleOptions,
    stem_path: &P,
    lexer_groups: &[(&str, &[MarkerSeries<(u32, MinMaxStat)>])],
) -> Result<(), Box<dyn std::error::Error>> {
    let stem_path = stem_path.as_ref();

    let mut relative_groups: Vec<(&str, Vec<MarkerSeries<(u32, MinMaxStat)>>)> = Default::default();

    for &(lexer_label, group) in lexer_groups.iter() {
        // Normalize the points to the max value.
        let mut baseline: BTreeMap<u32, f64> = Default::default();
        for ms in group.iter() {
            for &(t, stat) in ms.points.iter() {
                let entry = baseline.entry(t).or_default();
                *entry = bounds_tools::fmax(*entry, stat.mean);
            }
        }
        let render: Vec<MarkerSeries<(u32, MinMaxStat)>> = group
            .iter()
            .map(|s| {
                s.map(|&(t, stat)| {
                    let scale = 1.0 / baseline[&t];
                    (
                        t,
                        MinMaxStat {
                            min: stat.min * scale,
                            max: stat.max * scale,
                            mean: stat.mean * scale,
                        },
                    )
                })
            })
            .collect();

        relative_groups.push((lexer_label, render));
    }

    build_minmax_relative_encoder_table(&stem_path, &relative_groups)?;

    let plot_path = stem_path.with_added_extension("svg");
    log::info!("Plotting to {}", plot_path.display());

    let root = SVGBackend::new(&plot_path, options.shape).into_drawing_area();
    root.fill(&full_palette::WHITE)?;

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
        title,
        (title_area.dim_in_pixel().0 as i32 / 2, 10),
        title_style,
    ))?;
    title_area.draw(&Text::new(
        caption,
        (title_area.dim_in_pixel().0 as i32 / 2, 40),
        subtitle_style,
    ))?;

    let charts = chart_area.margin(10, 10, 10, 10);

    let sub_charts = charts.split_evenly((lexer_groups.len(), 1));

    for (idx, (lexer_label, group)) in relative_groups.iter().enumerate() {
        let drawing_area = &sub_charts[idx];

        let x_range = iter_range(group.iter().flat_map(|s| s.xs())).unwrap();
        let y_range = iter_frange(
            group
                .iter()
                .flat_map(|s| s.ys().into_iter().flat_map(|stat| vec![stat.min, stat.max])),
        )
        .unwrap();

        let show_x_label = idx == lexer_groups.len() - 1;

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
        for ms in group.iter() {
            let mut line_style = ms.style.line_style().stroke_width(options.line_width);
            line_style.color.3 = 0.7;

            let points = ms
                .points
                .iter()
                .map(|(t, s)| (*t, s.mean))
                .collect::<Vec<_>>();

            chart.draw_series(LineSeries::new(points, line_style))?;
        }

        for ms in group.iter() {
            let mut line_style = ms.style.line_style().stroke_width(options.line_width);
            line_style.color.3 = 0.6;

            chart.draw_series(
                ms.points
                    .iter()
                    .map(|(threads, stat)| {
                        ErrorBar::new_vertical(
                            *threads,
                            stat.min,
                            stat.mean,
                            stat.max,
                            line_style,
                            options.size,
                        )
                    })
                    .collect::<Vec<_>>(),
            )?;
        }

        for ms in group.iter() {
            chart
                .draw_series(ms.points.iter().map(|(threads, stat)| {
                    let coord = (*threads, stat.mean);
                    ms.style.marker(coord, options.size)
                }))?
                .label(ms.name.clone())
                .legend(move |coord| ms.style.marker(coord, options.size));
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

pub fn build_minmax_throughput_table<P: AsRef<Path>>(
    stem_path: &P,
    series: &[MarkerSeries<(u32, MinMaxStat)>],
) -> Result<(), Box<dyn std::error::Error>> {
    let stem_path = stem_path.as_ref();
    let table_path = stem_path.with_added_extension("csv");

    log::info!("Writing table to {}", table_path.display());
    let mut writer = BufWriter::new(File::create(table_path)?);

    let threads = series.first().unwrap().xs();
    let mut header = vec!["Name".to_string()];
    header.extend(threads.iter().map(|t| format!("Thread {}", t)));
    writeln!(writer, "{}", header.join(", "))?;

    for series in series.iter() {
        assert_eq!(threads, series.xs());

        let mut row = vec![series.name.clone()];
        row.extend(series.ys().iter().map(|stat| {
            format!(
                "{}:{}:{}",
                human_format::format_bps(stat.min),
                human_format::format_bps(stat.mean),
                human_format::format_bps(stat.max)
            )
        }));

        writeln!(writer, "{}", row.join(", "))?;
    }

    writer.flush()?;

    drop(writer);

    Ok(())
}

pub fn build_minmax_throughput_plot<P: AsRef<Path>>(
    title: &str,
    caption: &str,
    options: GraphStyleOptions,
    stem_path: &P,
    series: &[MarkerSeries<(u32, MinMaxStat)>],
    legend_loc: LegendLocation,
) -> Result<(), Box<dyn std::error::Error>> {
    let stem_path = stem_path.as_ref();

    build_minmax_throughput_table(&stem_path, series)?;

    let plot_path = stem_path.with_added_extension("svg");
    log::info!("Plotting to {}", plot_path.display());

    let root = SVGBackend::new(&plot_path, options.shape).into_drawing_area();
    root.fill(&full_palette::WHITE)?;

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
        title,
        (title_area.dim_in_pixel().0 as i32 / 2, 10),
        title_style,
    ))?;
    title_area.draw(&Text::new(
        caption,
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

        let x_range = iter_range(series.iter().flat_map(|s| s.xs())).unwrap();
        let y_range = iter_frange(
            series
                .iter()
                .flat_map(|s| s.ys().into_iter().flat_map(|stat| vec![stat.min, stat.max])),
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
                    .margin(10)
                    .x_label_area_size(40)
                    .y_label_area_size(120)
                    .build_cartesian_2d(x_range.log_scale().base(2.0), $y_range)?;

                if is_top {
                    chart
                        .configure_mesh()
                        .x_label_style(("sans-serif", 20.0).into_font())
                        .y_label_style(("sans-serif", 20.0).into_font())
                        .y_desc(format!("Throughput: {scale_desc}"))
                        .y_label_formatter(&|&bps| human_format::format_bps(bps))
                        .draw()?;
                } else {
                    chart
                        .configure_mesh()
                        .x_desc("Thread Count")
                        .y_desc(format!("Throughput: {scale_desc}"))
                        .x_label_style(("sans-serif", 20.0).into_font())
                        .y_label_style(("sans-serif", 20.0).into_font())
                        .y_label_formatter(&|&bps| human_format::format_bps(bps))
                        .draw()?;
                }

                // Render the lines under the markers.
                for ms in series.iter() {
                    let mut line_style = ms.style.line_style().stroke_width(options.line_width);
                    line_style.color.3 = 0.6;

                    let points = ms
                        .points
                        .iter()
                        .map(|(t, s)| (*t, s.mean))
                        .collect::<Vec<_>>();

                    if let Some(dash_style) = ms.style.dash_style {
                        chart.draw_series(DashedLineSeries::new(
                            points,
                            dash_style.size,
                            dash_style.spacing,
                            line_style,
                        ))?;
                    } else {
                        chart.draw_series(LineSeries::new(points, line_style))?;
                    }
                }

                for ms in series.iter() {
                    let mut line_style = ms.style.line_style().stroke_width(options.line_width);
                    line_style.color.3 = 0.6;

                    chart.draw_series(
                        ms.points
                            .iter()
                            .map(|(threads, stat)| {
                                ErrorBar::new_vertical(
                                    *threads,
                                    stat.min,
                                    stat.mean,
                                    stat.max,
                                    line_style,
                                    options.size,
                                )
                            })
                            .collect::<Vec<_>>(),
                    )?;
                }

                for ms in series.iter() {
                    chart
                        .draw_series(
                            ms.points
                                .iter()
                                .map(|(threads, stat)| {
                                    let coord = (*threads, stat.mean);
                                    ms.style.marker(coord, options.size)
                                })
                                .collect::<Vec<_>>(),
                        )?
                        .label(ms.name.clone())
                        .legend(move |coord| ms.style.marker(coord, options.size));
                }

                if is_top == legend_loc.top {
                    chart
                        .configure_series_labels()
                        .label_font(("sans-serif", 20).into_font())
                        .position(legend_loc.label_pos.clone())
                        .margin(options.size * 2)
                        .background_style(WHITE.mix(0.4))
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
