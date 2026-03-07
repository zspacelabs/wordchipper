use std::{
    f64::consts::FRAC_1_SQRT_2,
    path::Path,
};

use plotters::{
    backend::SVGBackend,
    chart::{
        ChartBuilder,
        SeriesLabelPosition,
    },
    drawing::IntoDrawingArea,
    element::{
        Drawable,
        PointCollection,
    },
    prelude,
    prelude::{
        Color,
        IntoFont,
        ShapeStyle,
        WHITE,
        full_palette,
        full_palette::BLACK,
    },
    style::SizeDesc,
};
use plotters_backend::{
    BackendCoord,
    DrawingBackend,
    DrawingErrorKind,
};

pub const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64;
const ORIGIN: (f64, f64) = (0.0, 0.0);
const SQ_STEP: f64 = FRAC_1_SQRT_2;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerType {
    #[default]
    Circle,
    Diamond,
    Square,
    TriDown,
    TriUp,
    CrossCircle,
    CrossDiamond,
    CrossSquare,
    CrossTriDown,
    CrossTriUp,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerLevel {
    #[default]
    Hypo,
    Para,
    Meta,
    Hyper,
}

impl From<Option<MarkerLevel>> for MarkerLevel {
    fn from(value: Option<MarkerLevel>) -> Self {
        value.unwrap_or_default()
    }
}

impl MarkerLevel {
    pub fn level(&self) -> u32 {
        use MarkerLevel::*;

        match self {
            Hypo => 0,
            Para => 1,
            Meta => 2,
            Hyper => 3,
        }
    }
}

impl MarkerType {
    fn poly_path(&self) -> Option<&[(f64, f64)]> {
        use MarkerType::*;

        match self {
            Circle => None,
            CrossCircle => Some(&[
                (0.0, -1.0),
                ORIGIN,
                (1.0, 0.0),
                ORIGIN,
                (-1.0, 0.0),
                ORIGIN,
                (0.0, 1.0),
                ORIGIN,
            ]),
            Square => Some(&[
                (-SQ_STEP, -SQ_STEP),
                (SQ_STEP, -SQ_STEP),
                (SQ_STEP, SQ_STEP),
                (-SQ_STEP, SQ_STEP),
            ]),
            Diamond => Some(&[(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]),
            CrossSquare => Some(&[
                (-SQ_STEP, -SQ_STEP),
                (0.0, -SQ_STEP),
                ORIGIN,
                (0.0, -SQ_STEP),
                (SQ_STEP, -SQ_STEP),
                (SQ_STEP, 0.0),
                ORIGIN,
                (SQ_STEP, 0.0),
                (SQ_STEP, SQ_STEP),
                (0.0, SQ_STEP),
                ORIGIN,
                (0.0, SQ_STEP),
                (-SQ_STEP, SQ_STEP),
                (-SQ_STEP, 0.0),
                ORIGIN,
                (-SQ_STEP, 0.0),
            ]),
            CrossDiamond => Some(&[
                (0.0, -1.0),
                (0.5, -0.5),
                ORIGIN,
                (0.5, -0.5),
                (1.0, 0.0),
                (0.5, 0.5),
                ORIGIN,
                (0.5, 0.5),
                (0.0, 1.0),
                (-0.5, 0.5),
                ORIGIN,
                (-0.5, 0.5),
                (-1.0, 0.0),
                (-0.5, -0.5),
                ORIGIN,
                (-0.5, -0.5),
            ]),
            TriUp => Some(&[
                (0.0, -1.0),
                (1.5 / SQRT_3, 1.0 / 2.0),
                (-1.5 / SQRT_3, 1.0 / 2.0),
            ]),
            TriDown => Some(&[
                (0.0, 1.0),
                (1.5 / SQRT_3, -1.0 / 2.0),
                (-1.5 / SQRT_3, -1.0 / 2.0),
            ]),
            CrossTriUp => Some(&[
                (0.0, -1.0),
                (1.5 / SQRT_3 / 2.0, -0.25),
                ORIGIN,
                (1.5 / SQRT_3 / 2.0, -0.25),
                (1.5 / SQRT_3, 1.0 / 2.0),
                (0.0, 1.0 / 2.0),
                ORIGIN,
                (0.0, 1.0 / 2.0),
                (-1.5 / SQRT_3, 1.0 / 2.0),
                (-1.5 / SQRT_3 / 2.0, -0.25),
                ORIGIN,
                (-1.5 / SQRT_3 / 2.0, -0.25),
            ]),
            CrossTriDown => Some(&[
                (0.0, 1.0),
                (1.5 / SQRT_3 / 2.0, 0.25),
                ORIGIN,
                (1.5 / SQRT_3 / 2.0, 0.25),
                (1.5 / SQRT_3, -1.0 / 2.0),
                (0.0, -1.0 / 2.0),
                ORIGIN,
                (0.0, -1.0 / 2.0),
                (-1.5 / SQRT_3, -1.0 / 2.0),
                (-1.5 / SQRT_3 / 2.0, 0.25),
                ORIGIN,
                (-1.5 / SQRT_3 / 2.0, 0.25),
            ]),
        }
    }

    #[allow(unused)]
    fn points(
        &self,
        size: u32,
        coord: BackendCoord,
    ) -> Vec<BackendCoord> {
        let (ax, ay) = coord;
        self.poly_path()
            .unwrap()
            .iter()
            .map(|(x, y)| {
                let x = (x * size as f64) as i32;
                let y = (y * size as f64) as i32;
                (ax + x, ay + y)
            })
            .collect()
    }
}

pub struct GraphMarker<Coord, Size: SizeDesc> {
    coord: Coord,
    size: Size,
    style: MarkerStyle,
}

#[derive(Clone, Copy, Debug)]
pub struct MarkerStyle {
    pub marker_type: MarkerType,
    pub marker_level: MarkerLevel,
    pub stroke_style: ShapeStyle,
    pub fill_style: Option<ShapeStyle>,
}

impl Default for MarkerStyle {
    fn default() -> Self {
        Self {
            marker_type: Default::default(),
            marker_level: Default::default(),
            stroke_style: BLACK.into(),
            fill_style: None,
        }
    }
}

impl MarkerStyle {
    pub fn new(marker_type: MarkerType) -> Self {
        Self {
            marker_type,
            marker_level: MarkerLevel::Hypo,
            stroke_style: BLACK.into(),
            fill_style: None,
        }
    }

    pub fn line_style(&self) -> ShapeStyle {
        if let Some(fill) = &self.fill_style {
            *fill
        } else {
            self.stroke_style
        }
    }

    pub fn with_marker_type(
        mut self,
        marker_type: MarkerType,
    ) -> Self {
        self.marker_type = marker_type;
        self
    }

    pub fn with_marker_level<E>(
        mut self,
        marker_level: E,
    ) -> Self
    where
        E: Into<MarkerLevel>,
    {
        self.marker_level = marker_level.into();
        self
    }

    pub fn with_stroke_style<S: Into<ShapeStyle>>(
        mut self,
        stroke_style: S,
    ) -> Self {
        self.stroke_style = stroke_style.into();
        self.stroke_style.filled = false;
        self
    }

    pub fn with_fill_style<S: Into<Option<ShapeStyle>>>(
        mut self,
        fill_style: S,
    ) -> Self {
        self.fill_style = fill_style.into();
        if let Some(fill) = self.fill_style.as_mut() {
            fill.filled = true;
        }
        self
    }

    pub fn marker<Coord, Size: SizeDesc>(
        &self,
        coord: Coord,
        size: Size,
    ) -> GraphMarker<Coord, Size> {
        GraphMarker::new(coord, size, *self)
    }
}

impl<Coord, Size: SizeDesc> GraphMarker<Coord, Size> {
    pub fn new(
        coord: Coord,
        size: Size,
        style: MarkerStyle,
    ) -> Self {
        Self { coord, size, style }
    }

    pub fn legacy<S, F, E>(
        coord: Coord,
        size: Size,
        marker_type: MarkerType,
        marker_level: E,
        stroke_style: S,
        fill_style: F,
    ) -> Self
    where
        S: Into<ShapeStyle>,
        F: Into<Option<ShapeStyle>>,
        E: Into<MarkerLevel>,
    {
        let style = MarkerStyle::default()
            .with_marker_type(marker_type)
            .with_marker_level(marker_level)
            .with_stroke_style(stroke_style)
            .with_fill_style(fill_style);

        Self::new(coord, size, style)
    }
}

impl<'a, Coord, Size: SizeDesc> PointCollection<'a, Coord> for &'a GraphMarker<Coord, Size> {
    type IntoIter = std::iter::Once<&'a Coord>;
    type Point = &'a Coord;

    fn point_iter(self) -> std::iter::Once<&'a Coord> {
        std::iter::once(&self.coord)
    }
}

impl<DB: DrawingBackend, Coord, Size: SizeDesc> Drawable<DB> for GraphMarker<Coord, Size> {
    fn draw<I: Iterator<Item = BackendCoord>>(
        &self,
        mut pos: I,
        backend: &mut DB,
        parent_dim: (u32, u32),
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        let Some((ax, ay)) = pos.next() else {
            return Ok(());
        };
        let size = self.size.in_pixels(&parent_dim).max(0) as u32;

        let extra_level = self.style.marker_level.level();

        if extra_level > 0 {
            backend.draw_circle((ax, ay), size * 3 / 2, &self.style.stroke_style, false)?;
        }
        if extra_level > 1 {
            let shift = (size * 3 / 2) as i32;
            backend.draw_path(
                vec![
                    (ax - shift, ay - shift),
                    (ax + shift, ay - shift),
                    (ax + shift, ay + shift),
                    (ax - shift, ay + shift),
                    (ax - shift, ay - shift),
                    (ax + shift, ay - shift),
                ],
                &self.style.stroke_style,
            )?;
        }
        if extra_level > 2 {
            let shift = ((size * 3 / 2) as f64 * std::f64::consts::SQRT_2) as i32;
            backend.draw_path(
                vec![
                    (ax, ay - shift),
                    (ax + shift, ay),
                    (ax, ay + shift),
                    (ax - shift, ay),
                    (ax, ay - shift),
                    (ax + shift, ay),
                ],
                &self.style.stroke_style,
            )?;
        }

        use MarkerType::*;

        match self.style.marker_type {
            CrossCircle | Circle => {
                if let Some(style) = &self.style.fill_style {
                    backend.draw_circle((ax, ay), size, style, style.filled)?;
                }
                backend.draw_circle((ax, ay), size, &self.style.stroke_style, false)?;

                if self.style.marker_type == CrossCircle {
                    let mut points: Vec<BackendCoord> =
                        self.style.marker_type.points(size, (ax, ay));
                    points.extend_from_within(0..2);

                    backend.draw_path(points, &self.style.stroke_style)?;
                }
            }
            _ => {
                let mut points: Vec<BackendCoord> = self.style.marker_type.points(size, (ax, ay));

                if let Some(style) = &self.style.fill_style {
                    backend.fill_polygon(points.clone(), style)?;
                }

                points.extend_from_within(0..2);
                backend.draw_path(points, &self.style.stroke_style)?;
            }
        }
        Ok(())
    }
}

pub fn build_demo_graph<P: AsRef<Path>>(output_dir: &P) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    let plot_path = output_dir.join("demo.svg");
    log::info!("Plotting to {}", plot_path.display());

    let root = SVGBackend::new(&plot_path, (640, 480)).into_drawing_area();
    root.fill(&full_palette::WHITE)?;
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

    let mut col = 0;
    for style in [
        MarkerStyle::default().with_stroke_style(full_palette::RED.stroke_width(2)),
        MarkerStyle::default()
            .with_stroke_style(full_palette::BLACK.stroke_width(2))
            .with_fill_style(full_palette::RED.filled()),
    ] {
        for level in [
            MarkerLevel::Hypo,
            MarkerLevel::Para,
            MarkerLevel::Meta,
            MarkerLevel::Hyper,
        ] {
            col += 1;

            let style = style.with_marker_level(level);

            chart.draw_series([(col, 0.10)].map(|coord| {
                style
                    .with_marker_type(MarkerType::Circle)
                    .marker(coord, size)
            }))?;
            chart.draw_series([(col, 0.20)].map(|coord| {
                style
                    .with_marker_type(MarkerType::CrossCircle)
                    .marker(coord, size)
            }))?;

            chart.draw_series([(col, 0.30)].map(|coord| {
                style
                    .with_marker_type(MarkerType::Square)
                    .marker(coord, size)
            }))?;
            chart.draw_series([(col, 0.40)].map(|coord| {
                style
                    .with_marker_type(MarkerType::CrossSquare)
                    .marker(coord, size)
            }))?;

            chart.draw_series([(col, 0.50)].map(|coord| {
                style
                    .with_marker_type(MarkerType::Diamond)
                    .marker(coord, size)
            }))?;
            chart.draw_series([(col, 0.60)].map(|coord| {
                style
                    .with_marker_type(MarkerType::CrossDiamond)
                    .marker(coord, size)
            }))?;

            chart.draw_series([(col, 0.70)].map(|coord| {
                style
                    .with_marker_type(MarkerType::TriUp)
                    .marker(coord, size)
            }))?;
            chart.draw_series([(col, 0.80)].map(|coord| {
                style
                    .with_marker_type(MarkerType::CrossTriUp)
                    .marker(coord, size)
            }))?;

            chart.draw_series([(col, 0.90)].map(|coord| {
                style
                    .with_marker_type(MarkerType::TriDown)
                    .marker(coord, size)
            }))?;
            chart.draw_series([(col, 1.0)].map(|coord| {
                style
                    .with_marker_type(MarkerType::CrossTriDown)
                    .marker(coord, size)
            }))?;
        }
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(prelude::BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
