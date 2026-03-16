//! # [`SpanEncoder`] Implementations

mod bpe_backtrack_encoder;
mod buffer_sweep_encoder;
mod merge_buffer_heap_encoder;
mod merge_tail_heap_encoder;
mod priority_merge_encoder;
mod span_encoder;
mod span_encoder_selector;
mod tail_sweep_encoder;

#[doc(inline)]
pub use bpe_backtrack_encoder::*;
#[doc(inline)]
pub use buffer_sweep_encoder::*;
#[doc(inline)]
pub use merge_buffer_heap_encoder::*;
#[doc(inline)]
pub use merge_tail_heap_encoder::*;
#[doc(inline)]
pub use priority_merge_encoder::*;
#[doc(inline)]
pub use span_encoder::*;
#[doc(inline)]
pub use span_encoder_selector::*;
#[doc(inline)]
pub use tail_sweep_encoder::*;
