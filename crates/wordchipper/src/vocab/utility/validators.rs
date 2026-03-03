//! Validators for various configuration options.
use crate::{
    TokenType,
    WCError,
    WCResult,
};

/// The size of the u8 space.
pub const U8_SIZE: usize = u8::MAX as usize + 1;

/// Validates and returns the vocabulary size, ensuring it's at least the size
/// of the u8 space.
pub fn try_vocab_size<T: TokenType>(vocab_size: usize) -> WCResult<usize> {
    if T::from_usize(vocab_size - 1).is_none() {
        Err(WCError::VocabSizeOverflow { size: vocab_size })
    } else if vocab_size < U8_SIZE {
        Err(WCError::VocabSizeTooSmall { size: vocab_size })
    } else {
        Ok(vocab_size)
    }
}

/// Validates and returns the vocab size, panicking if it's too small.
pub fn expect_vocab_size<T: TokenType>(vocab_size: usize) -> usize {
    try_vocab_size::<T>(vocab_size).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size() {
        assert_eq!(expect_vocab_size::<u16>(256), 256);

        assert!(try_vocab_size::<u16>(100).is_err());

        assert_eq!(
            expect_vocab_size::<u16>(u16::MAX as usize),
            u16::MAX as usize
        );

        assert_eq!(
            expect_vocab_size::<u16>(u16::MAX as usize + 1),
            u16::MAX as usize + 1
        );
        assert!(try_vocab_size::<u16>(u16::MAX as usize + 2).is_err());

        assert_eq!(expect_vocab_size::<u8>(256), 256);
        assert!(try_vocab_size::<u8>(257).is_err());
    }
}
