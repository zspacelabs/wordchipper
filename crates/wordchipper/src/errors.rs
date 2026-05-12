//! # Error Types

use crate::alloc::string::String;

/// Errors from wordchipper operations.
#[derive(Debug, thiserror::Error)]
pub enum WCError {
    /// Not Implemented Error.
    #[error("Not Implemented: {0}")]
    NotImplemented(String),

    /// Resource not found.
    #[error("Resource Not Found: {0}")]
    ResourceNotFound(String),

    /// The resource is a duplicate.
    #[error("Duplicate: {0}")]
    DuplicatedResource(String),

    /// Vocab size exceeds the capacity of the target token type.
    #[error("vocab size ({size}) exceeds token type capacity")]
    VocabSizeOverflow {
        /// The vocab size that exceeded the capacity.
        size: usize,
    },

    /// Vocab size is below the minimum (256, the u8 space).
    #[error("vocab size ({size}) must be >= 256")]
    VocabSizeTooSmall {
        /// The vocab size that was too small.
        size: usize,
    },

    /// Vocabulary data is inconsistent.
    #[error("Vocab Conflict: {0}")]
    VocabConflict(String),

    /// Token value out of range for the target type.
    #[error("token out of range")]
    TokenOutOfRange,

    /// Decoding did not consume all tokens.
    #[error("incomplete decode: {remaining} remaining tokens")]
    IncompleteDecode {
        /// The number of remaining tokens.
        remaining: usize,
    },

    /// I/O error.
    #[cfg(feature = "std")]
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Parse error (base64, integer, etc.)
    #[error("parse error: {0}")]
    Parse(String),

    /// Error from an external component.
    #[error("{0}")]
    External(String),
}

/// Result type for wordchipper operations.
pub type WCResult<T> = core::result::Result<T, WCError>;
