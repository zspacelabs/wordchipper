use crate::util::lexers::LexerInventory;

#[derive(clap::Args, Debug)]
pub struct ListLexersArgs {
    /// Display the patterns.
    #[clap(long, short = 'p')]
    pub patterns: bool,
}

impl ListLexersArgs {
    /// Run the lexer listing command.
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let inventory = LexerInventory::build();

        for lexer in &inventory.lexers {
            println!(
                "* \"{}\"{} - {}",
                lexer.id(),
                if lexer.accelerated {
                    " (accelerated)"
                } else {
                    ""
                },
                lexer.description
            );
            println!("  {:?}", lexer.aliases);
            if self.patterns {
                println!("{}", lexer.pattern);
                println!();
            }
        }

        Ok(())
    }
}
