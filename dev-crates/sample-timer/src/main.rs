extern crate core;

use std::{
    io,
    io::IsTerminal,
    iter::Iterator,
    sync::Arc,
    time::Duration,
};

use arrow::array::{
    Array,
    StringArray,
};
use batch_stats::{
    BatchStats,
    EngineBatchTimes,
};
use clap::{
    Parser,
    ValueEnum,
};
use engines::{
    BoxError,
    EncDecEngine,
};
use indicatif::ProgressBar;
use rand::prelude::SliceRandom;
use similar::TextDiff;
use tiktoken_rs::{
    CoreBPE,
    Rank,
};
use tiktoken_support::TiktokenRsEngine;
use wordchipper::{
    TokenEncoderOptions,
    TokenType,
    Tokenizer,
    TokenizerOptions,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    support::{
        slices::{
            inner_slice_view,
            inner_str_view,
        },
        timers::timeit,
    },
};
use wordchipper_data::dataset::{
    DatasetCache,
    DatasetCacheConfig,
};
use wordchipper_support::WordchipperEngine;

mod engines;

mod batch_stats;
mod tiktoken_support;
mod wordchipper_support;

#[cfg(feature = "tokenizers")]
mod tokenizers_support;
#[cfg(feature = "tokenizers")]
use tokenizers_support::TokenizersEngine;
#[cfg(feature = "tokenizers")]
use tokenizers_support::load_tokenizers_tok;
use wordchipper::spanners::span_lexers::accelerators::get_regex_accelerator;

/// Wordchipper Encode/Decode Side-by-Side Benchmarks.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to sample-shard dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// The shards to use for timing.
    ///
    /// Caches local nanochat training shards.
    #[arg(long, default_values_t = vec![0, 1])]
    pub shards: Vec<usize>,

    /// The batch size to use for timing.
    #[arg(long, default_value_t = 1024)]
    pub batch_size: usize,

    /// The pretrained model to compare.
    #[arg(long, default_value = "openai:o200k_harmony")]
    pub model: ModelSelector,

    /// Ignore missing models?
    // hack: 3-state boolean
    #[arg(long, num_args = 0..=1, default_value_t = true, default_missing_value = "true")]
    pub ignore_missing: bool,

    /// Test against tiktoken-rs?
    // hack: 3-state boolean
    #[arg(long, num_args = 0..=1, default_value_t = true, default_missing_value = "true")]
    pub tiktoken: bool,

    #[cfg(feature = "tokenizers")]
    /// Test against HF tokenizers?
    // hack: 3-state boolean
    #[arg(long, num_args = 0..=1, default_value_t = true, default_missing_value = "true")]
    pub tokenizers: bool,

    /// Time decoding as well.
    // hack: 3-state boolean
    #[arg(long, num_args = 0..=1, default_value_t = false, default_missing_value = "true")]
    pub decode: bool,

    /// Validate encoders against each other, decoders against input.
    // hack: 3-state boolean
    #[arg(long, num_args = 0..=1, default_value_t = true, default_missing_value = "true")]
    pub validate: bool,

    /// Re-span text when verifying?
    /// Slower, but works around span configs which leave gaps in the text.
    // hack: 3-state boolean
    #[arg(long,
    num_args = 0..=1,
    default_value_t = false,
      default_missing_value = "false")]
    pub respan_input_for_decode_check: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, strum::EnumString, strum::Display)]
pub enum ModelSelector {
    /// Select "`openai::gpt2`" model.
    #[value(name = "openai:gpt2")]
    #[strum(serialize = "openai:gpt2")]
    OpenaiGpt2,

    /// Select "`openai::r50k_base`" model.
    #[value(name = "openai:r50k_base")]
    #[strum(serialize = "openai:r50k_base")]
    OpenaiR50kBase,

    /// Select "`openai::p50k_base`" model.
    #[value(name = "openai:p50k_base")]
    #[strum(serialize = "openai:p50k_base")]
    OpenaiP50kBase,

    /// Select "`openai::p50k_edit`" model.
    #[value(name = "openai:p50k_edit")]
    #[strum(serialize = "openai:p50k_edit")]
    OpenaiP50kEdit,

    /// Select "`openai::cl100k_base`" model.
    #[value(name = "openai:cl100k_base")]
    #[strum(serialize = "openai:cl100k_base")]
    OpenaiCl100kBase,

    /// Select "`openai::o200k_base`" model.
    #[value(name = "openai:o200k_base")]
    #[strum(serialize = "openai:o200k_base")]
    OpenaiO200kBase,

    /// Select "`openai::o200k_harmony`" model.
    #[value(name = "openai:o200k_harmony")]
    #[strum(serialize = "openai:o200k_harmony")]
    OpenaiO200kHarmony,
}

impl ModelSelector {
    pub fn model(&self) -> String {
        self.to_string()
    }

    pub fn load_vocab<T: TokenType>(
        &self,
        disk_cache: &mut WordchipperDiskCache,
    ) -> Result<Arc<UnifiedTokenVocab<T>>, BoxError> {
        let vocab = wordchipper::load_vocab(&self.model(), disk_cache)?
            .vocab()
            .to_token_type()?
            .into();
        Ok(vocab)
    }

    pub fn load_tiktoken_bpe(&self) -> Result<(String, Arc<CoreBPE>), BoxError> {
        tiktoken_support::load_tiktoken_bpe(&self.model())
    }

    #[cfg(feature = "tokenizers")]
    pub fn load_tokenizers_tokenizer(
        &self
    ) -> Result<(String, Arc<tokenizers::tokenizer::Tokenizer>), BoxError> {
        load_tokenizers_tok(&self.model())
    }
}

#[allow(unused)]
#[allow(clippy::vec_init_then_push)]
fn main() -> Result<(), BoxError> {
    let args = Args::parse();
    let display_progress = io::stdout().is_terminal();

    let mut shard_data_cache = DatasetCacheConfig::default()
        .with_cache_dir(args.dataset_dir.clone())
        .init()?;

    println!("{:#?}", args);

    let mut disk_cache = WordchipperDiskCache::default();
    // println!("Loading wordchipper...");
    let loaded = wordchipper::load_vocab(args.model.to_string().as_str(), &mut disk_cache)?;

    let vocab = loaded.vocab().clone();

    // TODO: complete batch-observer inversion of control for additional tokenizer
    // wrappers.

    let mut candidate_engines: Vec<Arc<dyn EncDecEngine<Rank>>> = Vec::new();

    let wc_engine = Arc::new(WordchipperEngine::<Rank>::new(
        loaded.description().id().to_string(),
        TokenizerOptions::default()
            .with_parallel(true)
            .with_accelerated_lexers(false)
            .build(vocab.clone()),
    ));
    candidate_engines.push(wc_engine.clone());

    if get_regex_accelerator(vocab.spanning().pattern().as_str()).is_some() {
        let encoder = TokenEncoderOptions::default()
            .with_parallel(true)
            .with_accelerated_lexers(true)
            .build(vocab.clone());

        candidate_engines.push(Arc::new(WordchipperEngine::<Rank>::new(
            format!("{}/accel", loaded.description().id(),),
            Tokenizer::new(
                vocab.clone(),
                encoder,
                wc_engine.tokenizer().decoder().clone(),
            )
            .into(),
        )));
    }

    if args.tiktoken {
        // println!("Loading tiktoken...");
        match args.model.load_tiktoken_bpe() {
            Ok((name, bpe)) => candidate_engines.push(Arc::new(TiktokenRsEngine::new(name, bpe))),
            Err(e) => {
                if args.ignore_missing {
                    println!("Unable to load tiktoken model");
                } else {
                    return Err(format!("Unable to load tiktoken model: {}", e).into());
                }
            }
        }
    }

    #[cfg(feature = "tokenizers")]
    if args.tokenizers {
        // println!("Loading tokenizers...");
        match args.model.load_tokenizers_tokenizer() {
            Ok((name, tok)) => {
                candidate_engines.push(Arc::new(TokenizersEngine::new(
                    name.clone(),
                    tok.clone(),
                    true,
                )));
                candidate_engines.push(Arc::new(TokenizersEngine::new(name, tok, false)));
            }
            Err(e) => {
                if args.ignore_missing {
                    println!("Unable to load HuggingFace tokenizer");
                } else {
                    return Err(format!("Unable to load HuggingFace tokenizer: {}", e).into());
                }
            }
        }
    }

    println!("Loaded:");
    for eng in candidate_engines.iter() {
        println!("- \"{}\"", eng.name());
    }

    let mut stats = Vec::new();

    for_each_batch(
        display_progress,
        &args.shards,
        args.batch_size,
        &mut shard_data_cache,
        &mut |str_batch: &[&str]| -> Result<(), BoxError> {
            let degapped_input =
                if args.decode && args.validate && args.respan_input_for_decode_check {
                    Some(wc_engine.spanner().batch_remove_gaps(str_batch))
                } else {
                    None
                };
            let degapped_slice_view = degapped_input.as_ref().map(|s| inner_str_view(s));
            let expected_decode = match &degapped_slice_view {
                Some(expected) => expected,
                None => str_batch,
            };

            let mut batch_stats: BatchStats = Default::default();
            batch_stats
                .sample_bytes
                .extend(str_batch.iter().map(|s| s.len()));

            let mut token_reference: Option<(String, Vec<Vec<Rank>>)> = None;
            let mut token_counts: Option<Vec<usize>> = None;

            // We shuffle the order, to counteract cacheline bias.
            let mut shuffle_order = candidate_engines.clone();
            shuffle_order.shuffle(&mut rand::rng());

            for eng in shuffle_order.iter() {
                let name = eng.name();
                let mut batch_times: EngineBatchTimes = Default::default();

                let (encode_duration, tokens) = timeit(|| eng.expect_encode_batch(str_batch));
                batch_times.encode = encode_duration;

                let token_slices: Vec<&[Rank]> = inner_slice_view(&tokens);

                let decode_batch = if args.decode {
                    let (decode_duration, decode_batch) =
                        timeit(|| eng.expect_decode_batch(&token_slices));
                    batch_times.decode = decode_duration;
                    Some(decode_batch)
                } else {
                    None
                };

                // We want to show encode validation errors first.
                if args.validate
                    && let Some((ref_name, ref_tokens)) = &token_reference
                {
                    verify_encode(str_batch, name, &tokens, ref_name, ref_tokens)?;
                }

                if args.validate
                    && let Some(decode_batch) = decode_batch
                {
                    verify_decode(name, &token_slices, &decode_batch, expected_decode)?;
                }

                if batch_stats.token_counts.is_empty() {
                    batch_stats
                        .token_counts
                        .extend(tokens.iter().map(|t| t.len()));
                }

                if args.validate && token_reference.is_none() {
                    token_reference = Some((name.to_string(), tokens));
                }

                batch_stats.timings.insert(name.to_string(), batch_times);
            }

            stats.push(batch_stats);

            Ok(())
        },
    )?;

    println!();
    println!("Samples Summary:");
    let num_batches = stats.len();
    println!("- num batches: {}", num_batches);

    let avg_batch_bytes = stats.iter().map(|s| s.batch_bytes()).sum::<usize>() / num_batches;
    let avg_sample_bytes = stats.iter().map(|s| s.avg_sample_bytes()).sum::<usize>() / num_batches;
    println!("- avg bytes/sample: {avg_sample_bytes}");

    let total_bytes = stats.iter().map(|s| s.batch_bytes()).sum::<usize>();
    let total_tokens = stats.iter().map(|s| s.total_tokens()).sum::<usize>();
    let avg_bytes_per_token = total_bytes as f64 / total_tokens as f64;
    println!("- avg bytes/token: {avg_bytes_per_token:.1}");

    fn print_timing(
        name: &str,
        batch_time: Duration,
        batch_bytes: usize,
        batch_size: usize,
    ) {
        println!("- \"{name}\"");
        println!("  - batch:  {:>10.1?}", batch_time);
        println!("  - sample: {:>10.1?}", batch_time / batch_size as u32);
        println!("  - bps:    {:>10}", format_bps(batch_bytes, batch_time));
    }

    println!();
    println!("Encoder Batch Timing:");
    for w in candidate_engines.iter() {
        let name = w.name();
        let total_duration = stats
            .iter()
            .map(|s| s.timings[name].encode)
            .sum::<Duration>();
        let mean_duration = total_duration / num_batches as u32;
        print_timing(name, mean_duration, avg_batch_bytes, args.batch_size);
    }

    if args.decode {
        println!();
        println!("Decoder Batch Timing:");
        for w in candidate_engines.iter() {
            let name = w.name();
            let total_duration = stats
                .iter()
                .map(|s| s.timings[name].decode)
                .sum::<Duration>();
            let mean_duration = total_duration / num_batches as u32;
            print_timing(name, mean_duration, avg_batch_bytes, args.batch_size);
        }
    }

    Ok(())
}

fn for_each_batch(
    display_progress: bool,
    shards: &[usize],
    batch_size: usize,
    shard_data_cache: &mut DatasetCache,
    observe_batch: &mut dyn FnMut(&[&str]) -> Result<(), BoxError>,
) -> Result<(), BoxError> {
    let progress_bar = if display_progress {
        ProgressBar::new_spinner()
    } else {
        ProgressBar::hidden()
    };

    let mut batch_count = 0;
    let mut sample_buffer = Vec::new();
    let num_shards = shards.len();
    for &shard in shards {
        progress_bar.set_message(format!("Loading shard: {}", shard));
        shard_data_cache.get_shard(shard, true)?;

        for batch in shard_data_cache.read_batches(shard, false)? {
            let batch = batch?;
            let column = batch
                .column_by_name("text")
                .expect("failed to find 'text' column in batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for val in column {
                let val = val.unwrap().to_string();
                sample_buffer.push(val);
            }

            while sample_buffer.len() >= batch_size {
                let batch = sample_buffer.drain(..batch_size).collect::<Vec<_>>();
                let str_batch = inner_str_view(&batch);

                batch_count += 1;
                progress_bar.set_message(format!(
                    "Timing shard: {shard}/{num_shards}, batch: {}",
                    batch_count + 1
                ));
                progress_bar.tick();

                observe_batch(&str_batch)?;
            }
        }
    }

    Ok(())
}

pub fn verify_encode(
    source_batch: &[&str],
    actual_name: &str,
    actual_batch: &[Vec<Rank>],
    expected_name: &str,
    expected_batch: &[Vec<Rank>],
) -> Result<(), BoxError> {
    assert_eq!(source_batch.len(), actual_batch.len());
    assert_eq!(source_batch.len(), expected_batch.len());
    for (i, source) in source_batch.iter().enumerate() {
        let actual_tokens = &actual_batch[i];
        let expected_tokens = &expected_batch[i];

        if actual_tokens == expected_tokens {
            continue;
        }

        // Find first divergence index.
        let div = actual_tokens
            .iter()
            .zip(expected_tokens.iter())
            .position(|(a, e)| a != e)
            .unwrap_or(actual_tokens.len().min(expected_tokens.len()));

        // Show a window of tokens around the divergence.
        let window = 5;
        let lo = div.saturating_sub(window);
        let hi_a = (div + window).min(actual_tokens.len());
        let hi_e = (div + window).min(expected_tokens.len());

        let preview = &source[..source.floor_char_boundary(500)];

        return Err(format!(
            "ENCODER MISMATCH: {actual_name} != {expected_name}\n\
             First diff at token index {div} (of {} vs {})\n\
             ACTUAL[{lo}..{hi_a}]: {:?}\n\
             EXPECTED[{lo}..{hi_e}]: {:?}\n\
             SOURCE (first 500 chars): {:?}",
            actual_tokens.len(),
            expected_tokens.len(),
            &actual_tokens[lo..hi_a],
            &expected_tokens[lo..hi_e],
            preview,
        )
        .into());
    }
    Ok(())
}

pub fn verify_decode(
    decoder_name: &str,
    batch_tokens: &[&[Rank]],
    actual_decode: &[String],
    expected_decode: &[&str],
) -> Result<(), BoxError> {
    assert_eq!(batch_tokens.len(), expected_decode.len());
    assert_eq!(batch_tokens.len(), actual_decode.len());

    for (i, &expected_str) in expected_decode.iter().enumerate() {
        let actual_str = &actual_decode[i];

        if actual_str == expected_str {
            continue;
        }
        let diff = TextDiff::from_lines(expected_str, actual_str.as_str());
        let mut udiff = diff.unified_diff();
        udiff.header("expected", decoder_name);

        return Err(format!("DECODER MISMATCH: {decoder_name}\n{}", udiff).into());
    }

    Ok(())
}

/// Format a bytes/sec string.
pub fn format_bps(
    bytes: usize,
    duration: Duration,
) -> String {
    let bps = bytes as f64 / duration.as_secs_f64();
    format!(r"{}/s", humansize::format_size_i(bps, humansize::DECIMAL))
}
