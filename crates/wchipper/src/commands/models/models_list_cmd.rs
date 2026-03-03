#[derive(clap::Args, Debug)]
pub struct ListModelsArgs {}

impl ListModelsArgs {
    /// Run the model listing command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let vocabs = wordchipper::list_vocabs();

        for listing in &vocabs {
            println!("\"{}\" - {}", listing.provider(), listing.description());

            for desc in listing.vocabs() {
                println!("* \"{}\"", desc.id());
                println!("  {}", desc.description());
            }
        }

        Ok(())
    }
}
