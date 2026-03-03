//! # Trainer Implementation Utilities

mod pair_span_index;
mod text_span_counter;
mod token_span_buffer;

#[doc(inline)]
pub use pair_span_index::PairCountMap;
#[doc(inline)]
pub use pair_span_index::PairIndexMap;
#[doc(inline)]
pub use pair_span_index::PairSpanIndex;
#[doc(inline)]
pub use text_span_counter::TextSpanCounter;
#[doc(inline)]
pub use text_span_counter::TextSpanCounterOptions;
#[doc(inline)]
pub use token_span_buffer::TokenSpanBuf;
