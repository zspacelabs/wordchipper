//! # Wordchipper Disk Cache

use std::{
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use downloader::{
    Download,
    Downloader,
};

use crate::{
    WORDCHIPPER_CACHE_CONFIG,
    path_utils,
};

/// Options for [`WordchipperDiskCache`].
#[derive(Clone, Default, Debug)]
pub struct WordchipperDiskCacheOptions {
    /// Optional path to the cache directory.
    pub cache_dir: Option<PathBuf>,

    /// Optional path to the data directory.
    pub data_dir: Option<PathBuf>,

    /// Optional [`Downloader`] builder.
    pub downloader: Option<fn() -> Downloader>,
}

impl WordchipperDiskCacheOptions {
    /// Set the cache directory.
    pub fn with_cache_dir<P: AsRef<Path>>(
        mut self,
        cache_dir: Option<P>,
    ) -> Self {
        self.cache_dir = cache_dir.map(|p| p.as_ref().to_path_buf());
        self
    }

    /// Set the data directory.
    pub fn with_data_dir<P: AsRef<Path>>(
        mut self,
        data_dir: Option<P>,
    ) -> Self {
        self.data_dir = data_dir.map(|p| p.as_ref().to_path_buf());
        self
    }

    /// Set the downloader builder.
    pub fn with_downloader(
        mut self,
        downloader: Option<fn() -> Downloader>,
    ) -> Self {
        self.downloader = downloader;
        self
    }
}

/// Disk cache for downloaded files.
///
/// Leverages [`Downloader`] for downloading files,
/// and [`PathResolver`](`crate::PathResolver`) for resolving cache and data
/// paths appropriate for a user/system combo, and any environment overrides.
pub struct WordchipperDiskCache {
    /// Cache directory.
    cache_dir: PathBuf,

    /// Data directory.
    data_dir: PathBuf,

    /// Connection pool for downloading files.
    downloader: Downloader,
}

impl Default for WordchipperDiskCache {
    fn default() -> Self {
        Self::new(WordchipperDiskCacheOptions::default()).unwrap()
    }
}

impl WordchipperDiskCache {
    /// Construct a new [`WordchipperDiskCache`].
    pub fn new(options: WordchipperDiskCacheOptions) -> Result<Self, Box<dyn std::error::Error>> {
        let cache_dir = WORDCHIPPER_CACHE_CONFIG
            .resolve_cache_dir(options.cache_dir)
            .ok_or("failed to resolve cache directory")?;

        let data_dir = WORDCHIPPER_CACHE_CONFIG
            .resolve_data_dir(options.data_dir)
            .ok_or("failed to resolve data directory")?;

        let downloader = match options.downloader {
            Some(builder) => builder(),
            None => Downloader::builder().build()?,
        };

        Ok(Self {
            cache_dir,
            data_dir,
            downloader,
        })
    }

    /// Get the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get the data directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Get the downloader.
    pub fn downloader(&self) -> &Downloader {
        &self.downloader
    }

    /// Get the cache path for the given key.
    ///
    /// * Does not check that the path exists.
    /// * Does not initialize the containing directories.
    ///
    /// # Arguments
    /// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
    /// * `file` - the final file name.
    pub fn cache_path<C, F>(
        &self,
        context: &[C],
        file: F,
    ) -> PathBuf
    where
        C: AsRef<Path>,
        F: AsRef<Path>,
    {
        path_utils::extend_path(&self.cache_dir, context, file)
    }

    /// Loads a cached file from a specified path or downloads it if it does not
    /// exist.
    ///
    /// # Arguments
    /// * `context`: A slice of `C` containing path-related context used in
    ///   determining the cache location. These paths are combined to build the
    ///   cached file's location.
    /// * `urls`: A slice of string references specifying the URLs to download
    ///   the file from if it is not already cached.
    /// * `download`: A boolean flag indicating whether to attempt downloading
    ///   the file from the provided URLs if it does not already exist in the
    ///   cache.
    ///
    /// # Returns
    /// * Returns a [`PathBuf`] pointing to the cached file if it exists or is
    ///   successfully downloaded.
    /// * Returns an error if the file is not found in the cache and downloading
    ///   is not allowed or fails.
    ///
    /// # Errors
    /// * Returns an error if the cached file does not exist and `download` is
    ///   `false`.
    /// * Returns an error if the downloading process fails.
    pub fn load_cached_path<C, S>(
        &mut self,
        context: &[C],
        urls: &[S],
        download: bool,
        // TODO: hash: Option<&str>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>>
    where
        C: AsRef<Path>,
        S: AsRef<str>,
    {
        let urls: Vec<_> = urls.iter().map(|s| s.as_ref()).collect();
        let mut dl = Download::new_mirrored(&urls);
        let file_name = dl.file_name.clone();
        let path = self.cache_path(context, &file_name);
        dl.file_name = path.clone();

        if path.exists() {
            return Ok(path);
        }

        if !download {
            return Err(format!("cached file not found: {}", path.display()).into());
        }

        fs::create_dir_all(path.parent().unwrap())?;

        self.downloader.download(&[dl])?;

        Ok(path)
    }

    /// Get the data path for the given key.
    ///
    /// * Does not check that the path exists.
    /// * Does not initialize the containing directories.
    ///
    /// # Arguments
    /// * `context` - prefix dirs, inserted between `self.cache_dir` and `file`.
    /// * `file` - the final file name.
    pub fn data_path<C, F>(
        &self,
        context: &[C],
        file: F,
    ) -> PathBuf
    where
        C: AsRef<Path>,
        F: AsRef<Path>,
    {
        path_utils::extend_path(&self.data_dir, context, file)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        path::PathBuf,
    };

    use serial_test::serial;

    use crate::{
        WORDCHIPPER_CACHE_CONFIG,
        WORDCHIPPER_CACHE_DIR,
        WORDCHIPPER_DATA_DIR,
        disk_cache::{
            WordchipperDiskCache,
            WordchipperDiskCacheOptions,
        },
    };

    #[test]
    #[serial]
    fn test_resolve_dirs() {
        let orig_cache_dir = env::var(WORDCHIPPER_CACHE_DIR);
        let orig_data_dir = env::var(WORDCHIPPER_CACHE_DIR);

        let pds = WORDCHIPPER_CACHE_CONFIG
            .project_dirs()
            .expect("failed to get project dirs");

        let user_cache_dir = PathBuf::from("/tmp/wordchipper/cache");
        let user_data_dir = PathBuf::from("/tmp/wordchipper/data");

        let env_cache_dir = PathBuf::from("/tmp/wordchipper/env_cache");
        let env_data_dir = PathBuf::from("/tmp/wordchipper/env_data");

        // No env vars
        unsafe {
            env::remove_var(WORDCHIPPER_CACHE_DIR);
            env::remove_var(WORDCHIPPER_DATA_DIR);
        }

        let cache = WordchipperDiskCache::new(
            WordchipperDiskCacheOptions::default()
                .with_cache_dir(Some(user_cache_dir.clone()))
                .with_data_dir(Some(user_data_dir.clone())),
        )
        .unwrap();
        assert_eq!(&cache.cache_dir(), &user_cache_dir);
        assert_eq!(&cache.data_dir(), &user_data_dir);

        let cache = WordchipperDiskCache::new(WordchipperDiskCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir(), &pds.cache_dir().to_path_buf());
        assert_eq!(&cache.data_dir(), &pds.data_dir().to_path_buf());

        // With env var.
        unsafe {
            env::set_var(WORDCHIPPER_CACHE_DIR, env_cache_dir.to_str().unwrap());
            env::set_var(WORDCHIPPER_DATA_DIR, env_data_dir.to_str().unwrap());
        }

        let cache = WordchipperDiskCache::new(
            WordchipperDiskCacheOptions::default()
                .with_cache_dir(Some(user_cache_dir.clone()))
                .with_data_dir(Some(user_data_dir.clone())),
        )
        .unwrap();
        assert_eq!(&cache.cache_dir(), &user_cache_dir);
        assert_eq!(&cache.data_dir(), &user_data_dir);

        let cache = WordchipperDiskCache::new(WordchipperDiskCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir(), &env_cache_dir);
        assert_eq!(&cache.data_dir(), &env_data_dir);

        // restore original env var.
        match orig_cache_dir {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_CACHE_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_CACHE_DIR) },
        }
        match orig_data_dir {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_DATA_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_DATA_DIR) },
        }
    }

    #[test]
    fn test_data_path() {
        let cache = WordchipperDiskCache::new(WordchipperDiskCacheOptions::default()).unwrap();
        let path = cache.data_path(&["prefix"], "file.txt");
        assert_eq!(path, cache.data_dir.join("prefix").join("file.txt"));
    }

    #[test]
    fn test_cache_path() {
        let cache = WordchipperDiskCache::new(WordchipperDiskCacheOptions::default()).unwrap();
        let path = cache.cache_path(&["prefix"], "file.txt");
        assert_eq!(path, cache.cache_dir.join("prefix").join("file.txt"));
    }
}
