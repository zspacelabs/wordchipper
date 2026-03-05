//! Compile-time DFA word scanners using [`logos`](https://docs.rs/logos).
//!
//! Each lexer module defines a `#[derive(Logos)]` token enum, maps variants
//! to [`gpt2_family::Gpt2FamilyTokenRole`], and uses the `logos_lexer!` macro
//! to generate the `SpanLexer` impl.

/// Generate a `SpanLexer` struct + `inventory` registration from a logos
/// token enum and a pattern constant.
macro_rules! logos_lexer {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident;
        token = $token:ty;
        pattern = $pattern:expr;
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug)]
        $vis struct $name;

        inventory::submit! {
            crate::spanners::span_lexers::accelerators::RegexAcceleratorHook::new(
                $pattern,
                || alloc::sync::Arc::new($name),
            )
        }

        impl crate::spanners::span_lexers::SpanLexer for $name {
            fn find_span_iter<'a>(
                &'a self,
                text: &'a str,
            ) -> alloc::boxed::Box<
                dyn Iterator<Item = core::ops::Range<usize>> + 'a,
            > {
                alloc::boxed::Box::new(
                    super::gpt2_family::logos_span_iter(
                        text,
                        <$token as logos::Logos>::lexer(text).spanned(),
                    ),
                )
            }
        }
    };
}

pub mod cl100k;
pub mod gpt2_family;
pub mod o200k;
pub mod r50k;

#[cfg(any(test, feature = "testing"))]
pub mod testutil;
