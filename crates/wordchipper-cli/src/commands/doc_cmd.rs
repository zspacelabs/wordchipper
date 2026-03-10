use crate::Args;

/// Args for the train command.
#[derive(clap::Args, Debug)]
pub struct DocArgs {}

impl DocArgs {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        clap_markdown::print_help_markdown::<Args>();

        Ok(())
    }
}
