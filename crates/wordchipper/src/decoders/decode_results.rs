//! # `TokenDecoder` Result Types

use core::fmt::Debug;

use crate::{
    WCResult,
    alloc::vec::Vec,
};

/// The result of decoding tokens into bytes.
#[derive(Debug)]
pub struct DecodeResult<V>
where
    V: Debug,
{
    /// The remaining token count.
    pub remaining: Option<usize>,

    /// The decoded result.
    pub value: V,
}

impl<V: PartialEq> PartialEq for DecodeResult<V>
where
    V: Debug,
{
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.remaining == other.remaining && self.value == other.value
    }
}

impl<V: Clone> Clone for DecodeResult<V>
where
    V: Debug,
{
    fn clone(&self) -> Self {
        Self {
            remaining: self.remaining,
            value: self.value.clone(),
        }
    }
}

impl<V> DecodeResult<V>
where
    V: Debug,
{
    /// Construct a new result.
    pub fn new(
        value: V,
        remaining: Option<usize>,
    ) -> Self {
        let remaining = remaining.filter(|&r| r > 0);
        Self { value, remaining }
    }

    /// Try to unwrap the result, returning an error if the decoding is
    /// incomplete.
    pub fn try_result(self) -> WCResult<V> {
        if let Some(remaining) = self.remaining
            && remaining > 0
        {
            return Err(crate::WCError::IncompleteDecode { remaining });
        }
        Ok(self.value)
    }

    /// Unwrap the result, panicking if the decoding is incomplete.
    pub fn unwrap(self) -> V {
        self.try_result().unwrap()
    }

    /// Returns `true` if the decoding is complete.
    pub fn is_complete(&self) -> bool {
        if let Some(remaining) = self.remaining
            && remaining > 0
        {
            return false;
        }
        true
    }

    /// Convert the result using a conversion function.
    pub fn convert<F, U>(
        self,
        f: F,
    ) -> DecodeResult<U>
    where
        F: Fn(V) -> U,
        U: Debug,
    {
        DecodeResult {
            remaining: self.remaining,
            value: f(self.value),
        }
    }
}

/// The result of decoding a batch of tokens into bytes.
#[derive(Debug)]
pub struct BatchDecodeResult<V>
where
    V: Debug,
{
    /// The per-item results.
    pub results: Vec<DecodeResult<V>>,
}

impl<V> From<Vec<DecodeResult<V>>> for BatchDecodeResult<V>
where
    V: Debug,
{
    fn from(results: Vec<DecodeResult<V>>) -> Self {
        Self { results }
    }
}

impl<V> BatchDecodeResult<V>
where
    V: Debug,
{
    /// Is the decoding complete for all items?
    pub fn is_complete(&self) -> bool {
        self.results.iter().all(|r| r.is_complete())
    }

    /// Try to unwrap the results, returning an error if any decoding is
    /// incomplete.
    pub fn try_results(self) -> WCResult<Vec<V>> {
        self.results.into_iter().map(|r| r.try_result()).collect()
    }

    /// Unwrap the results, panicking if any decoding is incomplete.
    pub fn unwrap(self) -> Vec<V> {
        self.try_results().unwrap()
    }

    /// Convert the results using a conversion function.
    pub fn convert<F, U>(
        self,
        f: &F,
    ) -> BatchDecodeResult<U>
    where
        F: Fn(V) -> U,
        U: Debug,
    {
        BatchDecodeResult {
            results: self.results.into_iter().map(|r| r.convert(f)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{
        string::ToString,
        vec,
    };

    #[test]
    fn test_decode_result_new() {
        let result = DecodeResult::new(42, Some(0));
        assert_eq!(result.value, 42);
        assert_eq!(result.remaining, None);

        let result = DecodeResult::new("test", Some(5));
        assert_eq!(result.value, "test");
        assert_eq!(result.remaining, Some(5));
    }

    #[test]
    fn test_decode_result_try_result() {
        let result = DecodeResult::new(42, None);
        assert_eq!(result.try_result().unwrap(), 42);

        let result = DecodeResult::new(42, Some(5));
        assert!(result.try_result().is_err());
    }

    #[test]
    fn test_decode_result_unwrap() {
        let result = DecodeResult::new(42, None);
        assert_eq!(result.unwrap(), 42);

        let result = DecodeResult::new("test", Some(0));
        assert_eq!(result.unwrap(), "test");
    }

    #[test]
    #[should_panic]
    fn test_decode_result_unwrap_panic() {
        let result = DecodeResult::new(42, Some(5));
        result.unwrap();
    }

    #[test]
    fn test_decode_result_is_complete() {
        let result = DecodeResult::new(42, None);
        assert!(result.is_complete());

        let result = DecodeResult::new(42, Some(0));
        assert!(result.is_complete());

        let result = DecodeResult::new(42, Some(5));
        assert!(!result.is_complete());
    }

    #[test]
    fn test_decode_result_convert() {
        let result: DecodeResult<i32> = DecodeResult::new(42, Some(5));
        let converted = result.convert(|x| x.to_string());

        assert_eq!(converted.value, "42");
        assert_eq!(converted.remaining, Some(5));
    }

    #[test]
    fn test_batch_decode_result_from() {
        let results = vec![DecodeResult::new(1, None), DecodeResult::new(2, Some(5))];
        let batch: BatchDecodeResult<i32> = results.clone().into();
        assert_eq!(&batch.results, &results);
    }

    #[test]
    fn test_batch_decode_result_is_complete() {
        let complete = BatchDecodeResult {
            results: vec![DecodeResult::new(1, None), DecodeResult::new(2, None)],
        };
        assert!(complete.is_complete());

        let incomplete = BatchDecodeResult {
            results: vec![DecodeResult::new(1, None), DecodeResult::new(2, Some(5))],
        };
        assert!(!incomplete.is_complete());
    }

    #[test]
    fn test_batch_decode_result_try_results() {
        let complete = BatchDecodeResult {
            results: vec![DecodeResult::new(1, None), DecodeResult::new(2, None)],
        };
        assert_eq!(complete.try_results().unwrap(), vec![1, 2]);

        let incomplete = BatchDecodeResult {
            results: vec![DecodeResult::new(1, None), DecodeResult::new(2, Some(5))],
        };
        assert!(incomplete.try_results().is_err());
    }

    #[test]
    fn test_batch_decode_result_unwrap() {
        let complete = BatchDecodeResult {
            results: vec![DecodeResult::new(1, None), DecodeResult::new(2, None)],
        };
        assert_eq!(complete.unwrap(), vec![1, 2]);
    }

    #[test]
    #[should_panic]
    fn test_batch_decode_result_unwrap_panic() {
        let incomplete = BatchDecodeResult {
            results: vec![DecodeResult::new(1, None), DecodeResult::new(2, Some(5))],
        };
        incomplete.unwrap();
    }

    #[test]
    fn test_batch_decode_result_convert() {
        let batch = BatchDecodeResult {
            results: vec![DecodeResult::new(1, None), DecodeResult::new(2, Some(5))],
        };
        let converted = batch.convert(&|x| x.to_string());
        assert_eq!(converted.results[0].value, "1");
        assert_eq!(converted.results[1].value, "2");
        assert_eq!(converted.results[1].remaining, Some(5));
    }
}
