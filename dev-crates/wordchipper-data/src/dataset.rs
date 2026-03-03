//! # Nanochat Dataset Loader

use std::{
    fs,
    fs::File,
    path::PathBuf,
};

use downloader::{
    Download,
    Downloader,
};
use parquet::arrow::arrow_reader::{
    ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The upstream dataset URL.
pub const NANOCHAT_TRAIN_BASE_URL: &str =
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main";

/// The shard template.
pub const NANOCHAT_TRAIN_SHARD_TEMPLATE: &str = "shard_{index}.parquet";

/// The number of shards in the dataset.
pub const NANOCHAT_TRAIN_MAX_SHARD: usize = 1822;

/// Dataset Source Configuration.
#[derive(Debug, Clone)]
pub struct DatasetSource {
    /// The upstream dataset URL.
    pub base_url: String,

    /// The number of shards in the dataset.
    pub max_shard: usize,

    /// The 0-pad width of the shard index.
    pub index_pad_width: usize,

    /// The shard template.
    pub shard_template: String,
}

impl Default for DatasetSource {
    fn default() -> Self {
        DatasetSource {
            base_url: NANOCHAT_TRAIN_BASE_URL.to_string(),
            max_shard: NANOCHAT_TRAIN_MAX_SHARD,
            index_pad_width: 5,
            shard_template: NANOCHAT_TRAIN_SHARD_TEMPLATE.to_string(),
        }
    }
}

impl DatasetSource {
    /// Format a shard index with 0-padding.
    pub fn format_index(
        &self,
        index: usize,
    ) -> String {
        format!("{index:0width$}", width = self.index_pad_width)
    }

    /// Construct a shard filename.
    ///
    /// Substitutes the ``Self::format_index(index)`` result in for `"{index}"`
    /// in the [`Self::shard_template`].
    pub fn format_shard_filename(
        &self,
        index: usize,
    ) -> String {
        self.shard_template
            .replace("{index}", &self.format_index(index))
    }
}

/// Config for [`DatasetCache`].
#[derive(Debug, Clone)]
pub struct DatasetCacheConfig {
    /// The dataset cache directory.
    pub cache_dir: String,

    /// The dataset source configuration.
    pub source: DatasetSource,
}

impl Default for DatasetCacheConfig {
    fn default() -> Self {
        DatasetCacheConfig {
            cache_dir: "~/.cache/brn-nanochat/dataset/".to_string(),
            source: DatasetSource::default(),
        }
    }
}

impl DatasetCacheConfig {
    pub fn init(self) -> Result<DatasetCache> {
        let cache_dir = shellexpand::full(&self.cache_dir)?.to_string();

        fs::create_dir_all(&cache_dir)?;

        let downloader = Downloader::builder().parallel_requests(8).build()?;

        Ok(DatasetCache {
            cache_dir: PathBuf::from(cache_dir).canonicalize()?,
            source: self.source.clone(),
            downloader,
        })
    }

    pub fn with_cache_dir(
        mut self,
        cache_dir: String,
    ) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    pub fn with_source(
        mut self,
        source: DatasetSource,
    ) -> Self {
        self.source = source;
        self
    }
}

/// Dataset Cache.
pub struct DatasetCache {
    cache_dir: PathBuf,
    source: DatasetSource,
    downloader: Downloader,
}

impl Default for DatasetCache {
    fn default() -> Self {
        DatasetCacheConfig::default().init().unwrap()
    }
}

impl DatasetCache {
    /// Construct a shard path.
    pub fn format_shard_path(
        &self,
        index: usize,
    ) -> PathBuf {
        let path: PathBuf = self.cache_dir.clone();
        path.join(self.source.format_shard_filename(index))
    }

    /// Get a shard path, or an error.
    pub fn try_shard_path(
        &self,
        index: usize,
    ) -> Result<PathBuf> {
        let path = self.format_shard_path(index);
        if path.exists() {
            Ok(path)
        } else {
            Err(format!("shard {} not found", index).into())
        }
    }

    /// Check if a shard is cached.
    pub fn has_shard(
        &self,
        index: usize,
    ) -> bool {
        let path = self.format_shard_path(index);
        path.exists()
    }

    /// List all parquet files in the cache.
    pub fn list_cached_shard_paths(&self) -> Result<Vec<PathBuf>> {
        const EXTENSION: &str = "parquet";

        let mut paths = Vec::new();
        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_file() && path.extension().unwrap_or_default() == EXTENSION {
                paths.push(path);
            }
        }

        paths.sort();
        Ok(paths)
    }

    /// List the ids of all cached shards.
    pub fn list_cached_shard_ids(&self) -> Result<Vec<usize>> {
        let (pre, post) = self.source.shard_template.split_once("{index}").unwrap();

        Ok(self
            .list_cached_shard_paths()?
            .into_iter()
            .map(|path| {
                let name = path.file_name().unwrap().to_str().unwrap();
                let index = name.strip_prefix(pre).unwrap().strip_suffix(post).unwrap();
                index.parse::<usize>().unwrap()
            })
            .collect::<Vec<_>>())
    }

    /// Load a shard (download if not cached).
    pub fn load_shard(
        &mut self,
        index: usize,
    ) -> Result<PathBuf> {
        let path = self.format_shard_path(index);
        if path.exists() {
            return Ok(path);
        }

        let url = format!(
            "{}/{}",
            self.source.base_url,
            self.source.format_shard_filename(index)
        );

        self.downloader
            .download(&[Download::new(&url).file_name(path.as_ref())])?;

        Ok(path)
    }

    /// Load multiple shards (download if not cached).
    pub fn load_shards(
        &mut self,
        shards: &[usize],
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::with_capacity(shards.len());
        let mut downloads = Vec::new();
        for &shard in shards {
            let path = self.format_shard_path(shard);
            paths.push(path.clone());

            if path.exists() {
                continue;
            }

            let url = format!(
                "{}/{}",
                self.source.base_url,
                self.source.format_shard_filename(shard)
            );

            downloads.push(Download::new(&url).file_name(path.as_ref()));
        }

        if !downloads.is_empty() {
            self.downloader.download(&downloads)?;
        }

        Ok(paths)
    }

    /// Get a shard path.
    ///
    /// # Arguments
    /// * `shard` - the shard index.
    /// * `download` - Whether to download the shard if not cached.
    ///
    /// # Returns
    /// An `Result<PathBuf>` containing the shard path.
    pub fn get_shard(
        &mut self,
        shard: usize,
        download: bool,
    ) -> Result<PathBuf> {
        if download {
            self.load_shard(shard)
        } else {
            self.try_shard_path(shard)
        }
    }

    /// Read a shard as a parquet reader.
    ///
    /// # Arguments
    /// * `shard` - The shard index.
    /// * `download` - Whether to download the shard if not cached.
    ///
    /// # Returns
    ///
    /// An `Iterator<Item=Result<RecordBatch, ArrowError>>` reader.
    pub fn read_batches(
        &mut self,
        shard: usize,
        download: bool,
    ) -> Result<ParquetRecordBatchReader> {
        if download {
            let _ = self.load_shard(shard)?;
        }
        self.read_cached_batches(shard)
    }

    /// Read a (pre-cached) shard as a parquet reader.
    ///
    /// # Arguments
    /// * `shard` - The shard index.
    ///
    /// # Returns
    ///
    /// An `Iterator<Item=Result<RecordBatch, ArrowError>>` reader.
    pub fn read_cached_batches(
        &self,
        shard: usize,
    ) -> Result<ParquetRecordBatchReader> {
        let path = self.format_shard_path(shard);
        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        Ok(reader)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_dataset_source_config() {
        let config = DatasetSource::default();

        assert_eq!(config.base_url, NANOCHAT_TRAIN_BASE_URL.to_string());
        assert_eq!(config.max_shard, NANOCHAT_TRAIN_MAX_SHARD);
        assert_eq!(config.index_pad_width, 5);
        assert_eq!(config.shard_template, "shard_{index}.parquet".to_string());

        assert_eq!(config.format_index(0), "00000");
        assert_eq!(config.format_index(312), "00312");

        assert_eq!(config.format_shard_filename(0), "shard_00000.parquet");
    }

    #[test]
    fn test_dataset_cache() -> Result<()> {
        let tmpdir = TempDir::new("brn-nanochat-test")?;
        let base_dir = tmpdir.path();

        let cache = DatasetCacheConfig::default()
            .with_cache_dir(base_dir.to_string_lossy().to_string())
            .init()?;

        let shards = vec![0, 12, 312, 1821];
        let mut expected_paths = Vec::new();
        for &idx in &shards {
            let path = cache.format_shard_path(idx);
            expected_paths.push(path.clone());
            File::create(path)?;
        }

        for idx in 0..cache.source.max_shard {
            assert_eq!(cache.has_shard(idx), shards.contains(&idx));
        }

        assert_eq!(&cache.list_cached_shard_paths()?, &expected_paths);
        assert_eq!(&cache.list_cached_shard_ids()?, &shards);

        Ok(())
    }
}
