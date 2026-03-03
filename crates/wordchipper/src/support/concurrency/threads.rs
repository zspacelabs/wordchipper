//! # Thread Utilities

use core::num::{
    NonZeroU64,
    NonZeroUsize,
};
use std::thread;

/// Current Thread -> u64 Pool.
///
/// ``thread::current().id().as_u64()`` is unstable.
pub fn unstable_current_thread_id_hash() -> usize {
    // c/o `tiktoken`:
    // It's easier to use unsafe than to use nightly. Rust has this nice u64 thread
    // id counter that works great for our use case of avoiding collisions in
    // our array. Unfortunately, it's private. However, there are only so many
    // ways you can layout a u64, so just transmute https://github.com/rust-lang/rust/issues/67939

    struct FakeThreadId(NonZeroU64);
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let val = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(val) as usize
}

/// The search list of environment variables that Rayon uses to control
/// parallelism.
#[cfg(feature = "parallel")]
const RAYON_VARS: &[&str] = &["RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"];

/// Get the max parallelism available.
///
/// When `parallel` is enabled, will scan over `RAYON_VARS`.
pub fn est_max_parallelism() -> usize {
    let default = || {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    };

    #[cfg(feature = "parallel")]
    for name in RAYON_VARS {
        use core::str::FromStr;
        use std::env;
        if let Some(x @ 1..) = env::var(name).ok().and_then(|s| usize::from_str(&s).ok()) {
            return x;
        }
    }

    default()
}

/// Resolve the max pool size.
///
/// ``min(max_pool, thread::available_parallelism() || MAX_POOL,
/// env::var("RAYON_NUM_THREADS"))``
pub fn resolve_max_pool(max_pool: Option<NonZeroUsize>) -> usize {
    let sys_max = est_max_parallelism();

    let max_pool = max_pool.map(|x| x.get()).unwrap_or(sys_max);

    core::cmp::min(max_pool, sys_max)
}

#[cfg(test)]
mod tests {
    use std::env;

    use serial_test::serial;

    use super::*;
    use crate::{
        prelude::*,
        types::WCHashMap,
    };

    #[test]
    #[serial]
    fn test_est_max_parallelism() {
        #[allow(unused_mut)]
        let mut orig_env: WCHashMap<String, Option<String>> = Default::default();

        #[cfg(feature = "parallel")]
        for name in RAYON_VARS {
            orig_env.insert(name.to_string(), env::var(name).ok());
            unsafe { env::remove_var(name) };
        }

        let base = est_max_parallelism();

        #[cfg(feature = "parallel")]
        for name in ["RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"] {
            let orig = env::var(name);

            unsafe { env::set_var(name, format!("{}", base + 12)) };

            assert_eq!(est_max_parallelism(), base + 12);

            match orig {
                Ok(s) => unsafe { env::set_var(name, s) },
                Err(_) => unsafe { env::remove_var(name) },
            }
        }

        assert_eq!(est_max_parallelism(), base);

        for (name, val) in orig_env {
            match val {
                Some(s) => unsafe { env::set_var(name, s) },
                None => unsafe { env::remove_var(name) },
            }
        }
    }
}
