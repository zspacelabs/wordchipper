use pyo3::{prelude::*, pymodule};

pub(crate) mod wc {
    pub use wordchipper::{
        Tokenizer,
        TokenizerOptions,
        disk_cache::WordchipperDiskCache,
        support::{
            slices::{inner_slice_view, inner_str_view},
            strings::string_from_utf8_lossy,
        },
        vocab::io::save_base64_span_map_path,
    };
}

mod support;
mod tokenizer;

#[pymodule]
fn _wordchipper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tokenizer::TokenizerOptions>()?;
    m.add_class::<tokenizer::Tokenizer>()?;
    Ok(())
}
