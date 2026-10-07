//! Results as CSV files: one row per run, per validation and per probe, plus one file
//! of gradient curves per run. Every row starts with the run key.

use crate::config::{Config, KEY_COLUMNS, KEY_FIELDS};
use crate::training::Outcome;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct Output {
    runs: File,
    history: File,
    probes: File,
    curves: PathBuf,
}

impl Output {
    pub fn open(dir: &Path) -> io::Result<Output> {
        let curves = dir.join("curves");
        fs::create_dir_all(&curves)?;
        Ok(Output {
            runs: append(
                &dir.join("runs.csv"),
                "solved_at,updates,test_loss,test_metric,seconds",
            )?,
            history: append(&dir.join("history.csv"), "update,loss,metric")?,
            probes: append(&dir.join("probes.csv"), "update,mode,task_1,task_2,task_3")?,
            curves,
        })
    }

    /// Keys of the runs already recorded in `dir`.
    pub fn completed(dir: &Path) -> io::Result<HashSet<String>> {
        let Ok(file) = File::open(dir.join("runs.csv")) else {
            return Ok(HashSet::new());
        };
        BufReader::new(file)
            .lines()
            .skip(1)
            .map(|line| {
                Ok(line?
                    .split(',')
                    .take(KEY_FIELDS)
                    .collect::<Vec<_>>()
                    .join(","))
            })
            .collect()
    }

    pub fn write(&mut self, o: &Outcome) -> io::Result<()> {
        let key = o.config.key();
        let solved = o.solved_at.map_or(String::new(), |s| s.to_string());
        writeln!(
            self.runs,
            "{key},{solved},{},{},{},{:.1}",
            o.updates, o.test_loss, o.test_metric, o.seconds
        )?;
        for e in &o.history {
            writeln!(self.history, "{key},{},{},{}", e.update, e.loss, e.metric)?;
        }
        for p in &o.probes {
            let mut values = p.task.clone();
            values.resize(3, f64::NAN);
            writeln!(
                self.probes,
                "{key},{},{},{},{},{}",
                p.update,
                mode(p.training_mode),
                values[0],
                values[1],
                values[2]
            )?;
        }
        self.write_curves(o)?;
        for f in [&mut self.runs, &mut self.history, &mut self.probes] {
            f.flush()?;
        }
        Ok(())
    }

    fn write_curves(&self, o: &Outcome) -> io::Result<()> {
        let path = self.curves.join(curve_file(&o.config));
        let mut f = BufWriter::new(File::create(path)?);
        writeln!(
            f,
            "update,mode,lag,loss_mean,loss_log_mean,loss_zeros,state_mean,state_log_mean,state_zeros"
        )?;
        for p in &o.probes {
            for lag in 0..p.loss.mean.len() {
                writeln!(
                    f,
                    "{},{},{lag},{:e},{:e},{},{:e},{:e},{}",
                    p.update,
                    mode(p.training_mode),
                    p.loss.mean[lag],
                    p.loss.log_mean[lag],
                    p.loss.zeros[lag],
                    p.state.mean[lag],
                    p.state.log_mean[lag],
                    p.state.zeros[lag]
                )?;
            }
        }
        f.flush()
    }
}

fn mode(training_mode: bool) -> &'static str {
    if training_mode {
        "train"
    } else {
        "eval"
    }
}

fn curve_file(config: &Config) -> String {
    format!("{}.csv", config.key().replace(',', "_"))
}

fn append(path: &Path, columns: &str) -> io::Result<File> {
    let fresh = !path.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if fresh {
        writeln!(file, "{KEY_COLUMNS},{columns}")?;
    }
    Ok(file)
}
