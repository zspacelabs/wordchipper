//! # Wordchipper Disk Cache
#![warn(missing_docs)]

mod disk_cache;
mod path_resolver;
mod path_utils;

#[doc(inline)]
pub use disk_cache::WordchipperDiskCache;
#[doc(inline)]
pub use disk_cache::WordchipperDiskCacheOptions;
#[doc(inline)]
pub use path_resolver::PathResolver;
#[doc(inline)]
pub use path_utils::extend_path;

/// Environment variable key to override the default cache directory.
pub const WORDCHIPPER_CACHE_DIR: &str = "WORDCHIPPER_CACHE_DIR";
/// Environment variable key to override the default data directory.
pub const WORDCHIPPER_DATA_DIR: &str = "WORDCHIPPER_DATA_DIR";

/// Default [`PathResolver`] for wordchipper.
pub const WORDCHIPPER_CACHE_CONFIG: PathResolver = PathResolver {
    qualifier: "io.crates.wordchipper",
    organization: "",
    application: "wordchipper",
    cache_env_vars: &[WORDCHIPPER_CACHE_DIR],
    data_env_vars: &[WORDCHIPPER_DATA_DIR],
};
