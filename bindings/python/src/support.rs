use pyo3::{
    PyErr,
    exceptions::{PyIOError, PyValueError},
};

pub fn to_pyerr(err: wordchipper::WCError) -> PyErr {
    match err {
        wordchipper::WCError::Io(e) => PyIOError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}
