use wordchipper::{
    spanners::span_lexers::accelerators::get_regex_accelerator,
    support::regex::{RegexPattern, RegexWrapper},
};

use crate::commands::lexers::LexerSelectorArgs;

#[derive(clap::Args, Debug)]
pub struct StressLexerArgs {
    #[command(flatten)]
    selector: LexerSelectorArgs,
}

impl StressLexerArgs {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let pattern = self.selector.resolve()?;

        let accel_lexer = get_regex_accelerator(&pattern);
        if accel_lexer.is_none() {
            return Err(format!("No regex accelerator found for pattern: {pattern}").into());
        }
        let _regex_lexer: RegexWrapper = RegexPattern::Adaptive(pattern).into();

        Ok(())
    }
}
