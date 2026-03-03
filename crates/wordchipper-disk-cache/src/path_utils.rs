//! # Path Utilities

use std::path::{
    Path,
    PathBuf,
};

/// Extend a path with a context and filename.
///
/// * Does not check that the path exists.
/// * Does not initialize the containing directories.
///
/// # Arguments
/// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
/// * `file` - the final file name.
pub fn extend_path<P, S, F>(
    path: P,
    context: &[S],
    filename: F,
) -> PathBuf
where
    P: AsRef<Path>,
    S: AsRef<Path>,
    F: AsRef<Path>,
{
    let mut path = path.as_ref().to_path_buf();
    path.extend(context.iter().map(|s| s.as_ref()));
    path.push(filename.as_ref());
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extend_path() {
        let path = extend_path("/tmp/wordchipper", &["cache", "data"], "file.txt");
        assert_eq!(path, PathBuf::from("/tmp/wordchipper/cache/data/file.txt"));
    }
}
