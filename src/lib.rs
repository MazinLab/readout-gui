#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::ClickThrough;

use serde::{Deserialize, Serialize};

use ndarray::Array2;
use num_complex::Complex;

use ndarray_npy::NpzReader;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Waveform {
    pub freqs: Vec<f64>,
    pub amps: Vec<f64>,
    pub phases: Vec<f64>,
    pub n_samples: u64,
    _sample_rate: f64,
    pub allow_sat: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SweepConfig {
    pub steps: Vec<f64>,
    pub waveform: Waveform,
    pub lo_center: f64,
    pub average: u64,
    pub attens: Option<(f64, f64)>,
    pub tap: String,
    pub rmses: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PowerSweepConfig {
    pub attens: Vec<(f64, f64)>,
    pub sweep_config: SweepConfig,
}

pub type ComplexPSweep = Vec<((f64, f64), Array2<Complex<f32>>)>;

pub struct PowerSweepValues {
    pub iq: ComplexPSweep,
    pub iqs: Option<ComplexPSweep>,
}

impl PowerSweepValues {
    pub fn from_reader<T: std::io::Read + std::io::Seek>(
        reader: &mut NpzReader<T>,
    ) -> PowerSweepValues {
        let mut psv = PowerSweepValues {
            iq: Vec::new(),
            iqs: None,
        };

        for file in reader.names().unwrap().iter() {
            if &file[0..1] == "o" && &file[file.len() - 2..] == "iq" {
                let parts: Vec<_> = file[1..file.len() - 2].split_terminator("d").collect();
                assert_eq!(parts.len(), 2);
                let a = (parts[0].parse().unwrap(), parts[1].parse().unwrap());
                psv.iq.push((a, reader.by_name(file).unwrap()));
            }
        }

        psv
    }
}

#[cfg(test)]
mod test {
    use std::io::Read;

    use super::*;

    #[test]
    fn deserialize_waveform() {
        let blah = r#"
        {
            "freqs": [0.0, 0.1],
            "amps": [0.0, 0.1],
            "phases": [0.0, 0.1],
            "n_samples": 10,
            "_sample_rate": 0.1,
            "allow_sat": false
        }
        "#;
        let _wav: Waveform = serde_json::from_str(blah).unwrap();
    }

    #[test]
    fn deserialize_sweep() {
        let blah = r#"
        {
            "steps": [0.0, 0.1],
            "waveform": {
                "freqs": [0.0, 0.1],
                "amps": [0.0, 0.1],
                "phases": [0.0, 0.1],
                "n_samples": 10,
                "_sample_rate": 0.1,
                "allow_sat": false
            },
            "lo_center": 6000.0,
            "average": 10,
            "attens": [0, 1],
            "tap": "ddciq",
            "rmses": true
        }
        "#;
        let _conf: SweepConfig = serde_json::from_str(blah).unwrap();
    }

    #[test]
    fn deserialize_psweep() {
        let blah = r#"
        {
            "attens": [[0, 1], [1, 2]],
            "sweep_config": {
                "steps": [0.0, 0.1],
                "waveform": {
                    "freqs": [0.0, 0.1],
                    "amps": [0.0, 0.1],
                    "phases": [0.0, 0.1],
                    "n_samples": 10,
                    "_sample_rate": 0.1,
                    "allow_sat": false
                },
                "lo_center": 6000.0,
                "average": 10,
                "attens": [0, 1],
                "tap": "ddciq",
                "rmses": true
            }
        }
        "#;
        let _conf: PowerSweepConfig = serde_json::from_str(blah).unwrap();
    }

    #[test]
    fn deserialize_file() {
        let mut blah = String::new();
        std::fs::File::open("./psweepconfig.json")
            .unwrap()
            .read_to_string(&mut blah)
            .unwrap();
        let _conf: PowerSweepConfig = serde_json::from_str(&blah).unwrap();
    }

    #[test]
    fn load_file() {
        let sweep = PowerSweepValues::from_reader(
            &mut NpzReader::new(std::fs::File::open("./psweep.npz").unwrap()).unwrap(),
        );
        assert!(sweep.iq.len() > 1);
    }
}
