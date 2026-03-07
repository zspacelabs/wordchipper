use std::{
    collections::{
        BTreeMap,
        BTreeSet,
    },
    path::Path,
};

use divan_parser::BenchResult;
use regex::Regex;

pub struct ParBenchData {
    pub data: BTreeMap<u32, Vec<BenchResult>>,
}

impl ParBenchData {
    pub fn new(data: BTreeMap<u32, Vec<BenchResult>>) -> Self {
        Self { data }
    }

    pub fn load_data<P: AsRef<Path>>(dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = dir.as_ref();
        log::info!("Loading ParBenchData from: {}", dir.display());

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

    pub fn threads(&self) -> Vec<u32> {
        self.data.keys().copied().collect()
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
