use std::io::{BufRead, BufReader};

use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::commands::train_cmd::FileFormat;

/// Args for batched input.
#[derive(clap::Args, Debug)]
pub struct BatchedInputArgs {
    /// Input files.
    files: Vec<String>,

    /// The input shard file format.
    #[arg(long)]
    input_format: FileFormat,

    /// The input batch size.
    #[arg(long, default_value = "100")]
    input_batch_size: usize,
}

impl BatchedInputArgs {
    /// Run the function for each batch.
    pub fn for_each_batch<F>(
        &self,
        f: &mut F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(&[String]) -> Result<bool, Box<dyn std::error::Error>>,
    {
        InputBatcher::new(self.input_format, self.files.clone())
            .with_batch_size(self.input_batch_size)
            .for_each_batch(f)
    }
}

/// Batcher for input files.
pub struct InputBatcher {
    pub format: FileFormat,
    pub files: Vec<String>,
    pub batch_size: usize,
}

impl InputBatcher {
    /// Create a new input batcher.
    pub fn new(
        format: FileFormat,
        files: Vec<String>,
    ) -> Self {
        Self {
            format,
            files,
            batch_size: 1000,
        }
    }

    /// Set the batch size.
    pub fn with_batch_size(
        mut self,
        batch_size: usize,
    ) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Run the function for each batch.
    pub fn for_each_batch<F>(
        &self,
        f: &mut F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(&[String]) -> Result<bool, Box<dyn std::error::Error>>,
    {
        let mut buffer: Vec<String> = Vec::with_capacity(self.batch_size);
        for (idx, path) in self.files.iter().enumerate() {
            log::info!("{idx}: {path}");

            if !self.for_each_path_item(path, &mut |item| {
                buffer.push(item.to_string());

                if buffer.len() >= self.batch_size {
                    let chunk: Vec<String> = buffer.drain(..self.batch_size).collect();
                    return f(&chunk);
                }

                Ok(true)
            })? {
                return Ok(());
            }
        }
        if !buffer.is_empty() {
            f(&buffer)?;
        }
        Ok(())
    }

    fn for_each_path_item<F>(
        &self,
        path: &str,
        f: &mut F,
    ) -> Result<bool, Box<dyn std::error::Error>>
    where
        F: FnMut(&String) -> Result<bool, Box<dyn std::error::Error>>,
    {
        match self.format {
            FileFormat::Text => self.for_each_text_item(path, f),
            FileFormat::Parquet => self.for_each_parquet_item(path, f),
        }
    }

    fn for_each_text_item<F>(
        &self,
        path: &str,
        f: &mut F,
    ) -> Result<bool, Box<dyn std::error::Error>>
    where
        F: FnMut(&String) -> Result<bool, Box<dyn std::error::Error>>,
    {
        let reader = BufReader::new(std::fs::File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if !f(&line)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn for_each_parquet_item<F>(
        &self,
        path: &str,
        f: &mut F,
    ) -> Result<bool, Box<dyn std::error::Error>>
    where
        F: FnMut(&String) -> Result<bool, Box<dyn std::error::Error>>,
    {
        let file = std::fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        for batch in reader {
            let batch = batch?;

            let samples = batch
                .column_by_name("text")
                .expect("failed to find 'text' column in batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .filter_map(|s| s.map(|s| s.to_string()));

            for sample in samples {
                if !f(&sample)? {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}
