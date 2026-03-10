use std::collections::{
    BTreeMap,
    BTreeSet,
};

use regex::Regex;
use serde_json::Value;

pub struct PythonParBenchData {
    pub data: BTreeMap<u32, Value>,
}

pub fn py_bench_input_bytes(benchmark: &Value) -> Option<u64> {
    benchmark
        .as_object()
        .unwrap()
        .get("extra_info")
        .unwrap()
        .as_object()
        .unwrap()
        .get("input_bytes")
        .unwrap()
        .as_u64()
}

pub fn py_bench_median_bps(br: &Value) -> f64 {
    let input_bytes = py_bench_input_bytes(br).unwrap();

    let median_delay = br
        .as_object()
        .unwrap()
        .get("stats")
        .unwrap()
        .as_object()
        .unwrap()
        .get("median")
        .unwrap()
        .as_f64()
        .unwrap();

    input_bytes as f64 / median_delay
}

impl PythonParBenchData {
    pub fn new(data: BTreeMap<u32, Value>) -> Self {
        Self { data }
    }

    pub fn load_data<P: AsRef<std::path::Path>>(
        dir: P
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = dir.as_ref();
        log::info!("Loading PythonParBenchData from: {}", dir.display());

        let pattern = r"^encoding_parallel\.(?<threads>\d+)\.json$";
        let re = Regex::new(pattern)?;

        let mut res: BTreeMap<u32, Value> = Default::default();

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

    pub fn rewrite_name(name: &str) -> String {
        name.rsplit_once("::test_").unwrap().1.to_string()
    }

    pub fn series_names(&self) -> BTreeSet<String> {
        self.data
            .iter()
            .flat_map(|(_, v)| {
                v.as_object()
                    .unwrap()
                    .get("benchmarks")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        v.as_object()
                            .unwrap()
                            .get("fullname")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_string()
                    })
            })
            .map(|name| Self::rewrite_name(&name))
            .collect()
    }

    pub fn select_series(
        &self,
        name: &str,
    ) -> Option<Vec<(u32, Value)>> {
        let res = self
            .data
            .iter()
            .filter_map(|(&threads, summary)| {
                summary
                    .as_object()
                    .unwrap()
                    .get("benchmarks")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .find_map(|v| {
                        let bm = v.as_object().unwrap();
                        if Self::rewrite_name(bm.get("fullname").unwrap().as_str().unwrap()) == name
                        {
                            Some((threads, v.clone()))
                        } else {
                            None
                        }
                    })
            })
            .collect::<Vec<(u32, Value)>>();

        if res.is_empty() { None } else { Some(res) }
    }

    pub fn try_select_series(
        &self,
        name: &str,
    ) -> Result<Vec<(u32, Value)>, Box<dyn std::error::Error>> {
        if let Some(series) = self.select_series(name) {
            Ok(series)
        } else {
            Err(format!("No series named \"{}\"", name).into())
        }
    }
}
