use wordchipper::disk_cache::{WordchipperDiskCache, WordchipperDiskCacheOptions};

/// Disk cache argument group.
#[derive(clap::Args, Debug)]
pub struct DiskCacheArgs {
    /// Cache directory.
    #[arg(long, default_value = None)]
    cache_dir: Option<String>,
}

impl DiskCacheArgs {
    /// Initialize the disk cache.
    pub fn init_disk_cache(&self) -> Result<WordchipperDiskCache, Box<dyn std::error::Error>> {
        let mut options = WordchipperDiskCacheOptions::default();

        if let Some(cache_dir) = &self.cache_dir {
            options = options.with_cache_dir(Some(cache_dir.clone()));
        }

        WordchipperDiskCache::new(options)
    }
}
