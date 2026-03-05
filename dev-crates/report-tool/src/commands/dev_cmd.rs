use std::{
    collections::BTreeMap,
    path::Path,
};

use divan_parser::BenchResult;
use plotters::{
    element::{
        Drawable,
        PathElement,
        Polygon,
    },
    prelude::{
        IntoLogRange,
        ShapeStyle,
        *,
    },
    style::full_palette as colors,
};
use plotters_backend::{
    BackendCoord,
    DrawingErrorKind,
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

        build_demo_graph(&output_dir)?;

        build_external_tgraph(&self.model, "buffer_sweep", &output_dir, &data)?;

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

struct Series {
    pub name: String,
    pub glyph: String,
    pub style: ShapeStyle,
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

fn external_styles() -> BTreeMap<String, (String, ShapeStyle)> {
    [
        (
            "bpe_openai".to_string(),
            ("♣︎".to_string(), colors::DEEPORANGE_200.filled()),
        ),
        (
            "tiktoken".to_string(),
            ("♥︎".to_string(), colors::PURPLE_200.filled()),
        ),
        (
            "tokenizers".to_string(),
            ("♦︎".to_string(), colors::PINK_200.filled()),
        ),
    ]
    .iter()
    .cloned()
    .collect()
}

fn fmin(
    a: f64,
    b: f64,
) -> f64 {
    match a.partial_cmp(&b).unwrap() {
        std::cmp::Ordering::Less => a,
        std::cmp::Ordering::Equal => a,
        std::cmp::Ordering::Greater => b,
    }
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

fn median_bps(br: &BenchResult) -> f64 {
    br.throughput_bps.as_ref().unwrap().median.unwrap()
}

fn as_points<F>(
    obs: &[(u32, BenchResult)],
    f: F,
) -> Vec<Point>
where
    F: Fn(u32, &BenchResult) -> f64,
{
    obs.iter()
        .map(|(threads, br)| Point {
            threads: *threads,
            value: f(*threads, br),
        })
        .collect()
}

pub struct Glyph<Coord>
where
    Coord: Copy,
{
    points: Vec<Coord>,
    stroke: Option<ShapeStyle>,
    fill: Option<ShapeStyle>,
}

impl<Coord> Glyph<Coord>
where
    Coord: Copy,
{
    pub fn new<P: Into<Vec<Coord>>, S: Into<Option<ShapeStyle>>, F: Into<Option<ShapeStyle>>>(
        points: P,
        stroke: S,
        fill: F,
    ) -> Self {
        Self {
            points: points.into(),
            stroke: stroke.into(),
            fill: fill.into(),
        }
    }
}

impl<DB: DrawingBackend, Coord> Drawable<DB> for Glyph<Coord>
where
    Coord: Copy,
{
    fn draw<I: Iterator<Item = BackendCoord>>(
        &self,
        _pos: I,
        backend: &mut DB,
        parent_dim: (u32, u32),
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        if let Some(fill) = self.fill {
            Polygon::new(self.points.clone(), fill).draw(
                std::iter::empty(),
                backend,
                parent_dim,
            )?;
        }
        if let Some(stroke) = self.stroke {
            let mut path = self.points.clone();
            path.push(self.points[0]);
            PathElement::new(path, stroke).draw(std::iter::empty(), backend, parent_dim)?;
        }
        Ok(())
    }
}

fn build_demo_graph<P: AsRef<Path>>(output_dir: &P) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    let plot_path = output_dir.join("demo.svg");
    log::info!("Plotting to {}", plot_path.display());

    let root = SVGBackend::new(&plot_path, (640, 480)).into_drawing_area();
    root.fill(&colors::WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("demo", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(90)
        .build_cartesian_2d(0..10, 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Index")
        .y_desc("Value")
        .y_label_formatter(&|&bps| {
            format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
        })
        .draw()?;

    let size = 10;
    chart.draw_series(
        [(1, 0.5)].map(|coord| EmptyElement::at(coord) + Circle::new((0, 0), size, colors::BLACK)),
    )?;

    chart.draw_series([(2, 0.5)].map(|coord| {
        EmptyElement::at(coord) + Rectangle::new([(-size, -size), (size, size)], colors::BLACK)
    }))?;

    chart.draw_series(
        [(3, 0.5)].map(|coord| {
            EmptyElement::at(coord) + TriangleMarker::new((0, 0), size, colors::BLACK)
        }),
    )?;

    chart.draw_series(
        [(3, 0.25)].map(|coord| EmptyElement::at(coord) + Cross::new((0, 0), size, colors::BLACK)),
    )?;

    fn up_triangle<DB: DrawingBackend, S: Into<ShapeStyle>>(
        coord: (i32, i32),
        size: i32,
        style: S,
    ) -> DynElement<'static, DB, (i32, i32)> {
        let (x, y) = coord;
        let s = size;
        Polygon::new(
            vec![(x, y - s), (x - s, y + s), (x + s, y + s)],
            style.into(),
        )
        .into_dyn()
    }
    chart.draw_series(
        [(5, 0.5)].map(|coord| EmptyElement::at(coord) + up_triangle((0, 0), size, &colors::BLACK)),
    )?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
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
        ("bpe_backtrack", colors::GREY_A400.filled()),
        ("buffer_sweep", colors::GREEN_A400.filled()),
        ("merge_heap", colors::BLUE_A400.filled()),
        ("priority_merge", colors::PINK_A700.filled()),
        ("tail_sweep", colors::DEEPPURPLE_A400.filled()),
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

    let mut plot_series: Vec<Series> = Default::default();
    for (span, style) in span_map.iter() {
        if let Some(series_data) = data.select_series(&span_key(span)) {
            let style = if accel { style.filled() } else { *style };

            plot_series.push(Series {
                name: span.to_string(),
                glyph: "X".to_string(),
                style,
                points: as_points(&series_data, |_, br| median_bps(br)),
            })
        }
    }

    // Normalize the points to the max value.
    let mut baseline: BTreeMap<u32, f64> = Default::default();
    for series in plot_series.iter() {
        for point in series.points.iter() {
            let entry = baseline.entry(point.threads).or_default();
            *entry = fmax(*entry, point.value);
        }
    }
    for series in plot_series.iter_mut() {
        series.points.iter_mut().for_each(|point| {
            point.value /= baseline[&point.threads];
        })
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
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_threads..max_threads).log_scale().base(2.0),
            min_bps..max_bps,
        )?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Relative Median Throughput")
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
        ("bpe_backtrack", colors::GREY_A400.filled()),
        ("buffer_sweep", colors::GREEN_A400.filled()),
        ("merge_heap", colors::BLUE_A400.filled()),
        ("priority_merge", colors::PINK_A700.filled()),
        ("tail_sweep", colors::DEEPPURPLE_A400.filled()),
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
                glyph: "X".to_string(),
                style,
                points: as_points(&series_data, |_, br| median_bps(br)),
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
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_threads..max_threads).log_scale().base(2.0),
            min_bps..max_bps,
        )?;

    chart
        .configure_mesh()
        .x_desc("Thread Count")
        .y_desc("Median Throughput")
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
    span_encoder: &str,
    output_dir: &P,
    data: &ParBenchData,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();

    let series_names = data
        .series_names()
        .into_iter()
        .filter(|name| name.contains(model))
        .collect::<Vec<_>>();

    let mut brandx_group: Vec<Series> = Default::default();
    for (ext, (glyph, style)) in external_styles().iter() {
        if let Some(name) = series_names.iter().find(|name| name.contains(ext)) {
            let series_data = data.select_series(name).unwrap();

            brandx_group.push(Series {
                name: ext.to_string(),
                glyph: glyph.to_string(),
                style: *style,
                points: as_points(&series_data, |_, br| median_bps(br)),
            })
        }
    }

    let regex_series = Series {
        name: "wordchipper:regex".to_string(),
        glyph: "✦".to_string(),
        style: colors::GREEN_200.filled(),
        points: as_points(
            &data
                .select_series(&format!(
                    "encoding_parallel::wordchipper::{span_encoder}::{model}"
                ))
                .expect("Failed to select regex series"),
            |_, br| median_bps(br),
        ),
    };

    let logos_series = Series {
        name: "wordchipper:logos".to_string(),
        glyph: "★".to_string(),
        style: colors::LIGHTBLUE_200.filled(),
        points: as_points(
            &data
                .select_series(&format!(
                    "encoding_parallel::wordchipper::{span_encoder}::{model}_fast"
                ))
                .expect("Failed to select regex series"),
            |_, br| median_bps(br),
        ),
    };

    let min_threads = brandx_group.iter().map(|s| s.min_threads()).min().unwrap();
    let max_threads = brandx_group.iter().map(|s| s.max_threads()).max().unwrap();

    let min_bps = fmin(
        brandx_group
            .iter()
            .map(|s| s.min_bps())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        regex_series.min_bps(),
    );
    let max_bps = fmax(
        brandx_group
            .iter()
            .map(|s| s.max_bps())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        regex_series.max_bps(),
    );

    for include_logos in [false, true] {
        let chart_name = if include_logos { "logos" } else { "regex" };

        let plot_path = output_dir.join(format!("wc_{chart_name}_vrs_brandx.rust.{model}.svg"));
        log::info!("Plotting to {}", plot_path.display());

        let min_bps = if include_logos {
            fmin(min_bps, logos_series.min_bps())
        } else {
            min_bps
        };
        let max_bps = if include_logos {
            fmax(max_bps, logos_series.max_bps())
        } else {
            max_bps
        };

        let root = SVGBackend::new(&plot_path, (640, 480)).into_drawing_area();
        root.fill(&colors::WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("wordchipper:{chart_name} vrs brandx, rust, model: \"{model}\"",),
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
            .y_desc("Median Throughput")
            .y_label_formatter(&|&bps| {
                format!("{}/s", humansize::format_size_i(bps, humansize::BINARY))
            })
            .draw()?;

        let glyph_size = 24;

        for pseries in brandx_group.iter() {
            let name = &pseries.name;
            let style = pseries.style;
            let points: Vec<(u32, f64)> = pseries
                .points
                .iter()
                .map(|p| (p.threads, p.value))
                .collect();

            chart.draw_series(LineSeries::new(
                pseries.points.iter().map(|p| (p.threads, p.value)),
                style.stroke_width(4),
            ))?;

            chart
                .draw_series(points.iter().map(|coord| {
                    EmptyElement::at(*coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                }))?
                .label(name)
                .legend(move |coord| {
                    let glyph_size = glyph_size * 2 / 3;

                    EmptyElement::at(coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                });
        }

        for pseries in if include_logos {
            vec![&logos_series, &regex_series]
        } else {
            vec![&regex_series]
        } {
            let name = &pseries.name;
            let style = pseries.style;
            let points: Vec<(u32, f64)> = pseries
                .points
                .iter()
                .map(|p| (p.threads, p.value))
                .collect();

            chart.draw_series(LineSeries::new(
                pseries.points.iter().map(|p| (p.threads, p.value)),
                style.stroke_width(4),
            ))?;

            chart
                .draw_series(points.iter().map(|coord| {
                    EmptyElement::at(*coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                }))?
                .label(name)
                .legend(move |coord| {
                    let glyph_size = glyph_size * 2 / 3;

                    EmptyElement::at(coord)
                        + Text::new(
                            pseries.glyph.clone(),
                            (glyph_size * -2 / 5, glyph_size * -1 / 3),
                            ("sans-serif", glyph_size as f64)
                                .into_font()
                                .color(&colors::BLACK),
                        )
                });
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}
