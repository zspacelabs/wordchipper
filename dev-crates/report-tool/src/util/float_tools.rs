use std::ops::Range;

pub fn fmin(
    a: f64,
    b: f64,
) -> f64 {
    match a.partial_cmp(&b).unwrap() {
        std::cmp::Ordering::Less => a,
        std::cmp::Ordering::Equal => a,
        std::cmp::Ordering::Greater => b,
    }
}

pub fn fmax(
    a: f64,
    b: f64,
) -> f64 {
    match a.partial_cmp(&b).unwrap() {
        std::cmp::Ordering::Less => b,
        std::cmp::Ordering::Equal => a,
        std::cmp::Ordering::Greater => a,
    }
}

pub fn fiter_min_max<'a>(iter: impl IntoIterator<Item = &'a f64>) -> Option<(f64, f64)> {
    let mut acc = None;
    for &b in iter {
        acc = Some(match acc {
            Some((a, _)) => (fmin(a, b), fmax(a, b)),
            None => (b, b),
        })
    }
    acc
}

pub fn fiter_range<'a>(iter: impl IntoIterator<Item = &'a f64>) -> Option<Range<f64>> {
    fiter_min_max(iter).map(|(a, b)| a..b)
}

pub fn fiter_max<'a>(iter: impl IntoIterator<Item = &'a f64>) -> Option<f64> {
    let mut acc = None;
    for &b in iter {
        acc = Some(match acc {
            Some(a) => fmax(a, b),
            None => b,
        })
    }
    acc
}

pub fn fiter_min<'a>(iter: impl IntoIterator<Item = &'a f64>) -> Option<f64> {
    let mut acc = None;
    for &b in iter {
        acc = Some(match acc {
            Some(a) => fmin(a, b),
            None => b,
        })
    }
    acc
}
