//! # Vocabulary Types

use crate::{
    alloc::vec::Vec,
    types::{
        Pair,
        WCHashMap,
    },
};

/// `{ Pair<T> -> T}` map.
///
/// ## Style Hints
/// Instance names should prefer `pair_map`, or `pair_token_map`.
pub type PairTokenMap<T> = WCHashMap<Pair<T>, T>;

/// `{ T -> Pair<T> }` map.
///
/// ## Style Hints
/// Instance names should prefer `taken_pairs`, or `token_pair_map`.
pub type TokenPairMap<T> = WCHashMap<T, Pair<T>>;

/// `{ Vec<u8> -> T }` map.
///
/// ## Style Hints
/// Instance names should prefer `span_map`, or `span_token_map`.
pub type SpanTokenMap<T> = WCHashMap<Vec<u8>, T>;

/// `{ T -> Vec<u8> }` map.
///
/// ## Style Hints
/// Instance names should prefer `token_spans`, or `token_span_map`.
pub type TokenSpanMap<T> = WCHashMap<T, Vec<u8>>;

/// `{ T -> u8 }` map.
///
/// ## Style Hints
/// Instance names should prefer `token_bytes`, or `token_byte_map`.
pub type TokenByteMap<T> = WCHashMap<T, u8>;

/// `{ u8 -> T }` map.
///
/// ## Style Hints
/// Instance names should prefer `byte_tokens`, or `byte_token_map`.
pub type ByteTokenMap<T> = WCHashMap<u8, T>;

/// `[T; 256]` array.
///
/// ## Style Hints
/// Instance names should prefer `byte_tokens`, or `byte_token_array`.
pub type ByteTokenArray<T> = [T; 256];
