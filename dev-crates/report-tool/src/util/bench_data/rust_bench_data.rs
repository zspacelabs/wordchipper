use std::{
    collections::{
        BTreeMap,
        BTreeSet,
    },
    path::Path,
};

use divan_parser::BenchResult;
use regex::Regex;

pub struct RustParBenchData {
    pub data: BTreeMap<u32, Vec<BenchResult>>,
}

impl RustParBenchData {
    pub fn new(data: BTreeMap<u32, Vec<BenchResult>>) -> Self {
        Self { data }
    }

    pub fn load_data<P: AsRef<Path>>(dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = dir.as_ref();
        log::info!("Loading RustParBenchData from: {}", dir.display());

        let pattern = r"^encoding_parallel\.(?<threads>\d+)\.json$";
        let re = Regex::new(pattern)?;

        let mut res: BTreeMap<u32, Vec<BenchResult>> = Default::default();

        for entry in dir.read_dir()?.filter_map(Result::ok) {
            let filename = entry.file_name().to_string_lossy().to_string();
            if let Some(caps) = re.captures(&filename) {
                let threads = caps.name("threads").unwrap().as_str().parse::<u32>()?;
                log::info!("loading [{}]: {}", threads, filename);

                let file = std::fs::File::open(entry.path())?;
                let reader = std::io::BufReader::new(file);
                let data = serde_json::from_reader(reader)?;

                res.insert(threads, data);
            }
        }

        Ok(Self::new(res))
    }

    pub fn series_names(&self) -> BTreeSet<String> {
        self.data
            .iter()
            .flat_map(|(_, results)| results.iter())
            .map(|result| result.name.clone())
            .collect()
    }

    pub fn select_series(
        &self,
        name: &str,
    ) -> Option<Vec<(u32, BenchResult)>> {
        let res = self
            .data
            .iter()
            .filter_map(|(&threads, results)| {
                results.iter().find_map(|result| {
                    if result.name == name {
                        Some((threads, result.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<(u32, BenchResult)>>();

        if res.is_empty() { None } else { Some(res) }
    }

    pub fn try_select_series(
        &self,
        name: &str,
    ) -> Result<Vec<(u32, BenchResult)>, Box<dyn std::error::Error>> {
        if let Some(series) = self.select_series(name) {
            Ok(series)
        } else {
            Err(format!("No series named \"{}\"", name).into())
        }
    }
}

pub fn rust_bench_median_bps(br: &BenchResult) -> f64 {
    br.throughput_bps.as_ref().unwrap().median.unwrap()
}

pub fn rust_bench_alloc_count(br: &BenchResult) -> Option<usize> {
    if let Some(allocs) = br.allocs.as_ref()
        && let Some(alloc_count) = allocs.get("alloc_count")
    {
        return Some(alloc_count.median.unwrap() as usize);
    }
    None
}

pub fn rust_bench_dealloc_count(br: &BenchResult) -> Option<usize> {
    if let Some(allocs) = br.allocs.as_ref()
        && let Some(dealloc_count) = allocs.get("dealloc_count")
    {
        return Some(dealloc_count.median.unwrap() as usize);
    }
    None
}
