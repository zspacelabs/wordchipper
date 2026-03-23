//! # Thread Pool Toy

use core::{
    fmt::Debug,
    ops::Deref,
};
use std::num::NonZeroUsize;

use crate::{
    prelude::*,
    support::concurrency::threads::{
        resolve_max_pool,
        unstable_current_thread_id_hash,
    },
};

/// Current Thread -> T Pool.
///
/// This struct provides a thread-id hashed pool of items.
/// Rather than pure thread-local storage, the pool is
/// initialized with a vector of items, and the current
/// thread's ID is hashed to get the item.
///
/// `PoolToy<T>` implements `Deref<Target = T>`, allowing it to be used
/// transparently with any trait methods on `T` through Rust's deref coercion.
///
/// ## Style Hints for AI
///
/// Instance names should prefer `${T-name}_pool`,
/// for example, `regex_pool`, `cache_pool`, etc.
pub struct PoolToy<T>
where
    T: Send,
{
    pool: Vec<T>,
}

impl<T> Deref for PoolToy<T>
where
    T: Send,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> AsRef<T> for PoolToy<T>
where
    T: Send,
{
    fn as_ref(&self) -> &T {
        self.get()
    }
}

impl<T> PoolToy<T>
where
    T: Clone + Send,
{
    /// Initialize a new thread-local pool with the given item and maximum pool
    /// size.
    ///
    /// ## Arguments
    /// * `pool` - the pool of items.
    /// * `max_pool` - override the maximum pool size, see [`resolve_max_pool`].
    pub fn new(
        item: T,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let max_pool = resolve_max_pool(max_pool);

        Self::from_pool(vec![item; max_pool])
    }
}

impl<T> PoolToy<T>
where
    T: Send,
{
    /// Create a new thread-local pool with the given vector of items.
    pub fn from_pool(pool: Vec<T>) -> Self {
        assert!(!pool.is_empty());
        Self { pool }
    }

    /// Get a reference to the item for the current thread.
    pub fn get(&self) -> &T {
        let tid = unstable_current_thread_id_hash();
        &self.pool[tid % self.pool.len()]
    }

    /// Get the length of the pool.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Is this empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Clone for PoolToy<T>
where
    T: Clone + Send,
{
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

impl<T> Debug for PoolToy<T>
where
    T: Send + Debug,
{
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("PoolToy")
            .field("item", &self.pool[0])
            .field("len", &self.pool.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::concurrency::threads::resolve_max_pool;

    #[test]
    fn test_pool_toy() {
        let max_pool = Some(NonZeroUsize::new(128).unwrap());
        let pool = PoolToy::new(10, max_pool);

        // The pool should not be empty.
        assert!(!pool.is_empty());

        // This will be different sizes on different systems.
        let size = resolve_max_pool(max_pool);

        assert_eq!(pool.len(), size);
        assert_eq!(&pool.pool, vec![10; size].as_slice());

        assert_eq!(pool.get(), &10);
        assert_eq!(pool.as_ref(), &10);
        assert_eq!(*pool, 10);

        assert_eq!(
            format!("{:?}", pool),
            format!("PoolToy {{ item: 10, len: {size} }}")
        );

        let clone = pool.clone();
        assert_eq!(&clone.pool, &pool.pool);
    }
}
