use std::{
    io::{
        BufRead,
        Write,
    },
    sync::Arc,
};

use wordchipper::{
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
};
use wordchipper_cli_util::io::input_output::{
    InputArgs,
    OutputArgs,
};

use crate::util::{
    disk_cache,
    model_selector::ModelSelectorArgs,
    tokenizer_mode::{
        TokenizerMode,
        TokenizerModeArgs,
    },
};

/// Args for the cat command.
#[derive(clap::Args, Debug)]
pub struct CatArgs {
    #[command(flatten)]
    model_selector: ModelSelectorArgs,

    #[command(flatten)]
    tokenizer_mode: TokenizerModeArgs,

    #[command(flatten)]
    input: InputArgs,

    #[command(flatten)]
    output: OutputArgs,

    #[command(flatten)]
    disk_cache: disk_cache::DiskCacheArgs,
}

impl CatArgs {
    /// Run the cat command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut disk_cache = self.disk_cache.init_disk_cache()?;
        let tokenizer = self.model_selector.load_tokenizer(&mut disk_cache)?;

        let mut reader = self.input.open_reader()?;
        let mut writer = self.output.open_writer()?;

        match self.tokenizer_mode.mode() {
            TokenizerMode::Encode => run_cat_encode(&mut reader, &mut writer, tokenizer)?,
            TokenizerMode::Decode => run_cat_decode(&mut reader, &mut writer, tokenizer)?,
        }

        Ok(())
    }
}

fn run_cat_encode(
    reader: &mut dyn BufRead,
    writer: &mut dyn Write,
    tokenizer: Arc<Tokenizer<u32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This could probably be sped up with a non-blocking buffer accumulation;
    // but that's a bit more complex to get right.

    // Read lines, but keep the end-of-line characters.
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let tokens = tokenizer.try_encode(&line, None)?;

        for (idx, token) in tokens.iter().enumerate() {
            write!(writer, "{}{}", if idx == 0 { "" } else { " " }, token)?;
        }
        writeln!(writer)?;
        writer.flush()?;
    }
    Ok(())
}

fn run_cat_decode(
    reader: &mut dyn BufRead,
    writer: &mut dyn Write,
    tokenizer: Arc<Tokenizer<u32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // non-block reading + buffering is complicated on rust.
    // we'd be able to get more speed out of this with a bit of muckery here.
    // We're also not handling the partial utf-8 boundary splitting yet.

    for line in reader.lines() {
        let tokens = line?
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect::<Vec<u32>>();

        let text = tokenizer.try_decode_to_string(&tokens)?.unwrap();

        write!(writer, "{}", text)?;
        writer.flush()?;
    }
    Ok(())
}
