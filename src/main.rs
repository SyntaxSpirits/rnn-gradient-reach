#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod config;
mod network;
mod output;
mod plans;
mod probe;
mod runner;
mod tasks;
mod training;

use config::Config;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  rnn-gradient-reach plan <calibrate|calibrate_long|main> [--jobs N] [--out DIR]
  rnn-gradient-reach run <key> [--out DIR]
  rnn-gradient-reach count <plan>

A key lists arch,task,length,init,regulariser,rate,seed,hidden,max_updates as in the
first columns of runs.csv, e.g. lstm,recall,30,chrono,zoneout_cell,0.15,0,64,5000.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let option = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
    };
    let out = |default: &str| {
        option("--out").map_or_else(|| PathBuf::from("results").join(default), PathBuf::from)
    };

    let result = match (args.first().map(String::as_str), args.get(1)) {
        (Some("plan"), Some(name)) => match plans::plan(name) {
            Some(configs) => {
                let jobs = option("--jobs").map_or(Ok(8), |j| j.parse());
                match jobs {
                    Ok(jobs) => runner::run_all(configs, &out(name), jobs),
                    Err(_) => return usage(),
                }
            }
            None => return usage(),
        },
        (Some("run"), Some(key)) => match Config::from_key(key) {
            Some(config) => runner::run_all(vec![config], &out("single"), 1),
            None => return usage(),
        },
        (Some("count"), Some(name)) => match plans::plan(name) {
            Some(configs) => {
                println!("{}", configs.len());
                Ok(())
            }
            None => return usage(),
        },
        _ => return usage(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}\n\nplans: {}", plans::NAMES.join(", "));
    ExitCode::from(2)
}
