use std::f64::consts::FRAC_1_SQRT_2;

use plotters::{
    element::{
        Drawable,
        DynElement,
        IntoDynElement,
        PointCollection,
    },
    prelude::{
        ShapeStyle,
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

    pub fn new<'b, DB, Coord, Size, S, F, E: Into<MarkerLevel>>(
        &self,
        coord: Coord,
        size: Size,
        stroke: S,
        fill: F,
        extra: E,
    ) -> DynElement<'b, DB, Coord>
    where
        Size: SizeDesc,
        S: Into<ShapeStyle>,
        F: Into<Option<ShapeStyle>>,
        GraphMarker<Coord, Size>: Drawable<DB> + 'b,
        for<'a> &'a GraphMarker<Coord, Size>: PointCollection<'a, Coord>,
        Coord: Clone,
        DB: DrawingBackend,
    {
        GraphMarker::legacy(coord, size, *self, extra, stroke, fill).into_dyn()
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
        self.fill_style.as_mut().map(|s| s.filled = true);
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
