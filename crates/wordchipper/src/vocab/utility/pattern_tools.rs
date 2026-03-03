//! # Pattern Tools

/// A macro to concatenate multiple string literals with a specified separator.
///
/// # Examples
///
/// ```rust
/// use wordchipper::join_strs;
///
/// // Concatenating string literals with a comma as the separator
/// let result = join_strs!(",", ("Hello", "World", "Rust"));
/// assert_eq!(result, "Hello,World,Rust");
///
/// // Concatenating string literals with a dash as the separator
/// let result = join_strs!("-", ("A", "B", "C"));
/// assert_eq!(result, "A-B-C");
///
/// // Concatenating a single string literal without a separator
/// let result = join_strs!(";", ("OnlyOne"));
/// assert_eq!(result, "OnlyOne");
/// ```
///
/// # Parameters
///
/// - `$sep`: A string literal used as a separator between the provided string
///   literals.
/// - `($first $(, $rest)*)`: A tuple of at least one string literal. The first
///   string literal is mandatory, and the rest are optional. A trailing comma
///   is allowed but not required.
#[macro_export]
macro_rules! join_strs {
    ($sep:literal, ($first:literal $(, $rest:literal)* $(,)?)) => {
        concat!($first $(, $sep, $rest)*)
    };
}

/// An extension of [`join_strs!()`] which uses the "|" as the seperator.
#[macro_export]
macro_rules! join_patterns {
    ($($e:expr),* $(,)?) => { $crate::join_strs!("|", ($($e),*)) };
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_join_patterns() {
        assert_eq!(join_patterns!("a", "b", "c"), "a|b|c");
    }

    #[test]
    fn test_join_strs() {
        assert_eq!(join_strs!("+", ("a", "b", "c")), "a+b+c");
    }
}
