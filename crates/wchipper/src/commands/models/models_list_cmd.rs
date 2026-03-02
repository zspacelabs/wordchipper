#[derive(clap::Args, Debug)]
pub struct ListModelsArgs {}

impl ListModelsArgs {
    /// Run the model listing command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let vocabs = wordchipper::list_vocabs();

        for listing in &vocabs {
            println!("\"{}\" - {}", listing.source, listing.description);

            for vocab in &listing.vocabs {
                println!("  * \"{}:{}\"", listing.source, vocab.id);
                println!("    {}", vocab.description);
            }
        }

        Ok(())
    }
}
