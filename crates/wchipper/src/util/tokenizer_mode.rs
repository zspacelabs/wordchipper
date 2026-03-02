/// The tokenizer mode.
#[derive(Debug, Clone, Copy)]
pub enum TokenizerMode {
    /// Encode from text to tokens.
    Encode,

    /// Decode from tokens to text.
    Decode,
}

/// Tokenizer mode argument group.
#[derive(clap::Args, Debug)]
#[group(required = true, multiple = false)]
pub struct TokenizerModeArgs {
    /// Encode from text to tokens.
    #[arg(long, action=clap::ArgAction::SetTrue)]
    encode: bool,

    /// Decode from tokens to text.
    #[arg(long, action=clap::ArgAction::SetTrue)]
    decode: bool,
}

impl TokenizerModeArgs {
    /// Get the tokenizer mode.
    pub fn mode(&self) -> TokenizerMode {
        if self.encode {
            TokenizerMode::Encode
        } else if self.decode {
            TokenizerMode::Decode
        } else {
            panic!("No tokenizer mode specified.");
        }
    }
}
