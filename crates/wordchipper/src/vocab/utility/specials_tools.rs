//! # Special Tokens Tools

use crate::alloc::{
    format,
    string::String,
};

/// Generate a "<|$name|>" string literal.
#[macro_export]
macro_rules! carrot_str {
    ($value:literal) => {
        concat!("<|", $value, "|>")
    };
}

/// Generate a "<|$value|>" String.
pub fn format_carrot<S: AsRef<str>>(value: S) -> String {
    format!("<|{}|>", value.as_ref())
}

/// Generate a "<|reserved_{$value}|>" string literal."
#[macro_export]
macro_rules! reserved_carrot_str {
    ($value:literal) => {
        concat!("<|reserved_", stringify!($value), "|>")
    };
}

/// Generate a "<|reserved_{$value}|>" String.
pub fn format_reserved_carrot(value: usize) -> String {
    format!("<|reserved_{}|>", value)
}

/// Declare a special token constant with [`carrot_str!()`].
///
/// Declare 1:
/// - ``declare_special!(CONST_NAME, const_value);``
///
/// Declare many:
/// - ``declare_special!( (N1, V1), (N2, V2), ... );``
#[macro_export]
macro_rules! declare_carrot_special {
    ($name:ident, $value:literal $(,)?) => {
        #[doc = concat!("Special token: ", $value) ]
        pub const $name: &str = $crate::carrot_str!($value);
    };

    (($name:ident, $value:literal) $(,)?) => {
        declare_carrot_special!($name, $value);
    };

    (($name:ident, $value:literal), $($rest:tt)*) => {
        declare_carrot_special!($name, $value);
        declare_carrot_special!($($rest)*);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specials_tools() {
        assert_eq!(carrot_str!("test_value"), "<|test_value|>");

        assert_eq!(
            format_carrot("test_value").as_str(),
            carrot_str!("test_value")
        );
    }

    #[test]
    fn test_format_reserved_carrot() {
        assert_eq!(reserved_carrot_str!(123), "<|reserved_123|>");

        assert_eq!(
            format_reserved_carrot(123).as_str(),
            reserved_carrot_str!(123)
        );
    }

    #[test]
    fn test_declare_special() {
        declare_carrot_special!((FOO, "foo"), (BAR, "bar"));

        assert_eq!(FOO, "<|foo|>");
        assert_eq!(BAR, "<|bar|>");
    }
}
