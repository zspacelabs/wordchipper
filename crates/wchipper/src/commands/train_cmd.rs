use std::{
    io::{BufRead, BufReader},
    sync::Arc,
};

use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use wordchipper::{
    UnifiedTokenVocab,
    VocabIndex,
    pretrained::openai::OA_R50K_BASE_PATTERN,
    vocab::io::write_base64_span_map,
};
use wordchipper_training::{BPETRainerOptions, BPETrainer};

use crate::util::{input_output::OutputArgs, logging::LogArgs};

/// File formats for the train command.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum FileFormat {
    /// Simple text files.
    Text,

    /// Parquet files.
    Parquet,
}

/// Args for the train command.
#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Input files.
    files: Vec<String>,

    #[clap(flatten)]
    pub logging: LogArgs,

    #[arg(long, default_value = "text")]
    input_format: FileFormat,

    /// Max vocab size.
    #[arg(long, default_value = "50281")]
    vocab_size: usize,

    /// Word span regex.
    #[arg(long, default_value_t = OA_R50K_BASE_PATTERN.as_str().to_string())]
    regex: String,

    #[command(flatten)]
    output: OutputArgs,
}

impl TrainArgs {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;

        let mut trainer = BPETRainerOptions::new(self.regex.clone(), self.vocab_size).init();

        log::info!("Reading shards:");
        for (idx, path) in self.files.iter().enumerate() {
            log::info!("{idx}: {path}");
            match self.input_format {
                FileFormat::Text => {
                    self.read_text_file(&mut trainer, path)?;
                }
                FileFormat::Parquet => {
                    self.read_parquet_file(&mut trainer, path)?;
                }
            }
        }

        log::info!("Training Tokenizer...");
        let vocab: Arc<UnifiedTokenVocab<u32>> = trainer
            .train(Default::default())
            .expect("training failed")
            .into();

        log::info!("Vocabulary Size: {:?}", vocab.max_token().unwrap());

        if let Some(path) = &self.output.output {
            log::info!("output: {}", path);
        }
        let mut writer = self.output.open_writer()?;
        write_base64_span_map(vocab.span_vocab().span_map(), &mut writer)?;

        Ok(())
    }

    fn read_text_file(
        &self,
        trainer: &mut BPETrainer,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = BufReader::new(std::fs::File::open(path)?);
        for line in reader.lines() {
            trainer.update_from_samples(vec![line?.to_string()]);
        }
        Ok(())
    }

    fn read_parquet_file(
        &self,
        trainer: &mut BPETrainer,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

            trainer.update_from_samples(samples);
        }

        Ok(())
    }
}
