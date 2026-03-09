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

pub fn fiter_min_max(iter: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let mut acc = None;
    for b in iter {
        acc = Some(match acc {
            Some((low, high)) => (fmin(low, b), fmax(high, b)),
            None => (b, b),
        })
    }
    acc
}

pub fn iter_frange(iter: impl IntoIterator<Item = f64>) -> Option<Range<f64>> {
    fiter_min_max(iter).map(|(a, b)| a..b)
}

pub fn fiter_max(iter: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut acc = None;
    for b in iter {
        acc = Some(match acc {
            Some(a) => fmax(a, b),
            None => b,
        })
    }
    acc
}

pub fn fiter_min(iter: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut acc = None;
    for b in iter {
        acc = Some(match acc {
            Some(a) => fmin(a, b),
            None => b,
        })
    }
    acc
}

pub fn iter_min_max<T: Copy + Ord>(iter: impl Iterator<Item = T>) -> Option<(T, T)> {
    iter.fold(None, |acc, x| match acc {
        None => Some((x, x)),
        Some((low, high)) => Some((std::cmp::min(low, x), std::cmp::max(high, x))),
    })
}

pub fn iter_range<T: Copy + Ord>(iter: impl Iterator<Item = T>) -> Option<Range<T>> {
    iter_min_max(iter).map(|(low, high)| low..high)
}
