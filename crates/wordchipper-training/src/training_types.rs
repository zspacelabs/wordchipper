//! # Training Types
use core::{
    fmt::{
        Debug,
        Display,
    },
    hash::Hash,
    ops::{
        AddAssign,
        SubAssign,
    },
};

use num_traits::{
    FromPrimitive,
    PrimInt,
    ToPrimitive,
};

/// A type that can be used as a string key.
pub trait StringChunkType:
    for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

impl<T> StringChunkType for T where
    T: for<'a> From<&'a str> + AsRef<str> + Debug + Clone + Send + Sync + Eq + Hash + Ord
{
}

/// A type that can be used as a word count.
pub trait CountType:
    'static
    + PrimInt
    + FromPrimitive
    + ToPrimitive
    + Hash
    + Default
    + Debug
    + Display
    + Send
    + Sync
    + AddAssign
    + SubAssign
{
}

impl<T> CountType for T where
    T: 'static
        + PrimInt
        + FromPrimitive
        + ToPrimitive
        + Hash
        + Default
        + Debug
        + Display
        + Send
        + Sync
        + AddAssign
        + SubAssign
{
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use compact_str::CompactString;

    use super::*;

    #[test]
    fn test_common_count_types() {
        struct IsCount<T: CountType>(PhantomData<T>);

        let _: IsCount<u16>;
        let _: IsCount<u32>;
        let _: IsCount<u64>;
        let _: IsCount<usize>;
    }

    #[test]
    fn test_common_string_chunk_types() {
        struct IsStringChunk<T: StringChunkType>(PhantomData<T>);

        let _: IsStringChunk<String>;
        let _: IsStringChunk<CompactString>;
    }
}
