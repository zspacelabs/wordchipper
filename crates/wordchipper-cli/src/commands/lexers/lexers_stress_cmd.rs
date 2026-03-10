use std::{
    cmp::{
        max,
        min,
    },
    sync::Arc,
};

use rayon::prelude::*;
use wordchipper::{
    spanners::span_lexers::{
        SpanLexer,
        accelerators::get_regex_accelerator,
    },
    support::regex::{
        RegexPattern,
        RegexWrapper,
    },
};
use wordchipper_cli_util::io::input_batcher::BatchedInputArgs;

use crate::util::{
    lexers::LexerSelectorArgs,
    logging::LogArgs,
};

#[derive(clap::Args, Debug)]
pub struct StressLexerArgs {
    #[command(flatten)]
    batched_input: BatchedInputArgs,

    #[clap(flatten)]
    pub logging: LogArgs,

    #[command(flatten)]
    selector: LexerSelectorArgs,

    /// Span context before error.
    #[clap(long, default_value_t = 8)]
    pub pre_context: usize,

    /// Span context after error.
    #[clap(long, default_value_t = 8)]
    pub post_context: usize,
}

impl StressLexerArgs {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.logging.setup_logging(3)?;
        let pattern = self.selector.get_pattern()?;

        let ref_lexer: RegexWrapper = RegexPattern::Adaptive(pattern.clone()).into();
        let ref_lexer: Arc<dyn SpanLexer> = Arc::new(ref_lexer);

        let accel_lexer: Arc<dyn SpanLexer> = match get_regex_accelerator(&pattern) {
            Some(lexer) => lexer,
            None => {
                return Err(format!("No accelerator found for pattern: {}", pattern).into());
            }
        };

        self.batched_input.for_each_batch(&mut |samples| {
            let failures = samples
                .par_iter()
                .filter_map(|s| {
                    let expected = ref_lexer.find_span_iter(s).collect::<Vec<_>>();
                    let test = accel_lexer.find_span_iter(s).collect::<Vec<_>>();
                    if test != expected {
                        return Some(s);
                    }
                    None
                })
                .collect::<Vec<_>>();

            for failure in failures {
                self.check(failure, &ref_lexer, &accel_lexer)?;
            }
            Ok(true)
        })?;

        Ok(())
    }

    fn check(
        &self,
        sample: &str,
        ref_lexer: &dyn SpanLexer,
        test_lexer: &dyn SpanLexer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected_spans = ref_lexer.find_span_iter(sample).collect::<Vec<_>>();
        let observed_spans = test_lexer.find_span_iter(sample).collect::<Vec<_>>();

        if observed_spans == expected_spans {
            return Ok(());
        }

        let mut first_diff = None;
        for i in 0..max(observed_spans.len(), expected_spans.len()) {
            if observed_spans[i] != expected_spans[i] {
                first_diff = Some(i);
                break;
            }
        }
        let first_diff = first_diff.unwrap();
        let mut start = first_diff;
        for _ in 0..self.pre_context {
            start -= 1;
        }
        let end = first_diff + self.post_context;

        let expected_ctx = &expected_spans[start..end];
        let observed_ctx = &observed_spans[start..end];

        let sample_start = min(
            expected_ctx.first().unwrap().start,
            observed_ctx.first().unwrap().start,
        );
        let sample_end = max(
            expected_ctx.last().unwrap().end,
            observed_ctx.last().unwrap().end,
        );

        let marker = first_diff - start;

        let sample_ctx = &sample[sample_start..sample_end];

        log::error!("Accelerated lexer failed to match reference lexer.");
        log::error!("sample: <<<{}>>>", sample_ctx);
        log::error!("expected: {:?}", expected_ctx);
        for (i, span) in expected_ctx.iter().enumerate() {
            log::error!(
                " {}{}: {:?}: <<<{}>>>",
                if marker == i { "*" } else { " " },
                i,
                span,
                &sample[span.clone()]
            );
        }
        log::error!("observed: {:?}", observed_ctx);
        for (i, span) in observed_ctx.iter().enumerate() {
            log::error!(
                " {}{}: {:?}: <<<{}>>>",
                if marker == i { "*" } else { " " },
                i,
                span,
                &sample[span.clone()]
            );
        }

        Err("Accelerated lexer failed to match reference lexer."
            .to_string()
            .into())
    }
}
