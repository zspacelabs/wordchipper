//! # Token List Utility

use crate::{
    TokenType,
    alloc::{
        string::{
            String,
            ToString,
        },
        vec::Vec,
    },
};

/// A utility trait to generate a listing of tokens.
pub trait ToTokenList {
    /// Convert the special tokens into a vector of `(String, T)` pairs.
    fn to_token_list<T: TokenType>(&self) -> Vec<(String, T)>;
}

impl ToTokenList for &[(&'static str, usize)] {
    fn to_token_list<T: TokenType>(&self) -> Vec<(String, T)> {
        self.iter()
            .map(|(k, v)| (k.to_string(), T::from_usize(*v).unwrap()))
            .collect()
    }
}
