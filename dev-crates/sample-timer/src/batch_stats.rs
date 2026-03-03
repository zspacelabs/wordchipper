use std::{
    collections::HashMap,
    time::Duration,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct EngineBatchTimes {
    pub encode: Duration,
    pub decode: Duration,
}

#[derive(Debug, Default)]
pub struct BatchStats {
    pub sample_bytes: Vec<usize>,
    pub token_counts: Vec<usize>,

    pub timings: HashMap<String, EngineBatchTimes>,
}

impl BatchStats {
    pub fn len(&self) -> usize {
        self.sample_bytes.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.token_counts.iter().sum::<usize>()
    }

    pub fn batch_bytes(&self) -> usize {
        self.sample_bytes.iter().sum::<usize>()
    }

    pub fn avg_sample_bytes(&self) -> usize {
        self.batch_bytes() / self.len()
    }
}
