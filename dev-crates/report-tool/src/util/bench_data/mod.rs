use divan_parser::BenchResult;

pub mod par_bench;

pub fn median_bps(br: &BenchResult) -> f64 {
    br.throughput_bps.as_ref().unwrap().median.unwrap()
}

pub fn alloc_count(br: &BenchResult) -> Option<usize> {
    if let Some(allocs) = br.allocs.as_ref()
        && let Some(alloc_count) = allocs.get("alloc_count")
    {
        return Some(alloc_count.median.unwrap() as usize);
    }
    None
}

pub fn dealloc_count(br: &BenchResult) -> Option<usize> {
    if let Some(allocs) = br.allocs.as_ref()
        && let Some(dealloc_count) = allocs.get("dealloc_count")
    {
        return Some(dealloc_count.median.unwrap() as usize);
    }
    None
}
