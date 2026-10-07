#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod experiment;
mod model;
mod tasks;

use experiment::{run, Config, Outcome};
use model::{Arch, Init, Regulariser};
use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tasks::Task;

const KEY: &str = "arch,task,length,init,regulariser,rate,seed,hidden,max_updates";
const KEY_FIELDS: usize = 9;

/// Lengths chosen from the calibration runs: one length that most conditions solve,
/// one in the transition and one that most fail.
const MAIN_LENGTHS: [(Task, [usize; 3]); 2] =
    [(Task::Adding, [50, 100, 200]), (Task::Recall, [15, 20, 30])];

fn config(
    arch: Arch,
    task: Task,
    length: usize,
    init: Init,
    regulariser: Regulariser,
    seed: u64,
) -> Config {
    Config {
        arch,
        task,
        length,
        init,
        regulariser,
        rate: match regulariser {
            Regulariser::None => 0.0,
            r if r.is_zoneout() => 0.15,
            _ => 0.25,
        },
        seed,
        hidden: 64,
        batch: 32,
        max_updates: 5000,
        learning_rate: 1e-3,
        clip: 1.0,
    }
}

fn plan(name: &str) -> Vec<Config> {
    let archs = [Arch::Lstm, Arch::Gru];
    let inits = [Init::Conventional, Init::Chrono];
    let mut configs = Vec::new();
    match name {
        "calibrate" => {
            let lengths = [
                (Task::Adding, vec![50, 100, 200, 400]),
                (Task::Recall, vec![10, 15, 20, 30, 45, 60]),
            ];
            for seed in 0..2 {
                for arch in archs {
                    for init in inits {
                        for (task, ls) in &lengths {
                            for &length in ls {
                                configs.push(config(
                                    arch,
                                    *task,
                                    length,
                                    init,
                                    Regulariser::None,
                                    seed,
                                ));
                            }
                        }
                    }
                }
            }
        }
        "main" => {
            for seed in 0..5 {
                for arch in archs {
                    for init in inits {
                        for (task, ls) in MAIN_LENGTHS {
                            for length in ls {
                                for regulariser in Regulariser::ALL {
                                    if arch.supports(regulariser) {
                                        configs.push(config(
                                            arch,
                                            task,
                                            length,
                                            init,
                                            regulariser,
                                            seed,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        other => panic!("unknown plan {other}"),
    }
    configs
}

fn key(c: &Config) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{}",
        c.arch.name(),
        c.task.name(),
        c.length,
        c.init.name(),
        c.regulariser.name(),
        c.rate,
        c.seed,
        c.hidden,
        c.max_updates
    )
}

fn open_csv(path: &Path, columns: &str) -> fs::File {
    let fresh = !path.exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    if fresh {
        writeln!(file, "{KEY},{columns}").unwrap();
    }
    file
}

fn completed(path: &Path) -> HashSet<String> {
    let Ok(file) = fs::File::open(path) else {
        return HashSet::new();
    };
    BufReader::new(file)
        .lines()
        .skip(1)
        .map(|l| {
            l.unwrap()
                .split(',')
                .take(KEY_FIELDS)
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

fn write_curves(dir: &Path, o: &Outcome) {
    let path = dir.join(format!("{}.csv", key(&o.config).replace(',', "_")));
    let mut f = BufWriter::new(fs::File::create(path).unwrap());
    writeln!(
        f,
        "update,mode,lag,loss_mean,loss_log_mean,loss_zeros,state_mean,state_log_mean,state_zeros"
    )
    .unwrap();
    for p in &o.probes {
        let mode = if p.training_mode { "train" } else { "eval" };
        for lag in 0..p.loss.mean.len() {
            writeln!(
                f,
                "{},{mode},{lag},{:e},{:e},{},{:e},{:e},{}",
                p.update,
                p.loss.mean[lag],
                p.loss.log_mean[lag],
                p.loss.zeros[lag],
                p.state.mean[lag],
                p.state.log_mean[lag],
                p.state.zeros[lag]
            )
            .unwrap();
        }
    }
}

fn write_outcome(files: &mut HashMap<&str, fs::File>, curves: &Path, o: &Outcome) {
    let k = key(&o.config);
    let solved = o.solved_at.map_or(String::new(), |s| s.to_string());
    writeln!(
        files.get_mut("runs").unwrap(),
        "{k},{solved},{},{},{},{:.1}",
        o.updates,
        o.test_loss,
        o.test_metric,
        o.seconds
    )
    .unwrap();
    for e in &o.history {
        writeln!(
            files.get_mut("history").unwrap(),
            "{k},{},{},{}",
            e.update,
            e.loss,
            e.metric
        )
        .unwrap();
    }
    for p in &o.probes {
        let mode = if p.training_mode { "train" } else { "eval" };
        let mut values = p.task.clone();
        values.resize(3, f64::NAN);
        writeln!(
            files.get_mut("probes").unwrap(),
            "{k},{},{mode},{},{},{}",
            p.update,
            values[0],
            values[1],
            values[2]
        )
        .unwrap();
    }
    write_curves(curves, o);
    for f in files.values_mut() {
        f.flush().unwrap();
    }
}

fn run_plan(name: &str, jobs: usize, out: PathBuf) {
    let curves = out.join("curves");
    fs::create_dir_all(&curves).unwrap();
    let runs_path = out.join("runs.csv");
    let done = completed(&runs_path);
    let todo: Vec<Config> = plan(name)
        .into_iter()
        .filter(|c| !done.contains(&key(c)))
        .collect();
    eprintln!(
        "{name}: {} runs to do, {} already done",
        todo.len(),
        done.len()
    );

    let files = Mutex::new(HashMap::from([
        (
            "runs",
            open_csv(
                &runs_path,
                "solved_at,updates,test_loss,test_metric,seconds",
            ),
        ),
        (
            "history",
            open_csv(&out.join("history.csv"), "update,loss,metric"),
        ),
        (
            "probes",
            open_csv(&out.join("probes.csv"), "update,mode,task_1,task_2,task_3"),
        ),
    ]));
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(config) = todo.get(i) else { break };
                let outcome = run(config);
                eprintln!(
                    "[{}/{}] {} solved_at={:?} test={:.4} {:.0}s",
                    i + 1,
                    todo.len(),
                    key(config),
                    outcome.solved_at,
                    outcome.test_metric,
                    outcome.seconds
                );
                write_outcome(&mut files.lock().unwrap(), &curves, &outcome);
            });
        }
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .map(|i| args[i + 1].clone())
    };
    match args.get(1).map(String::as_str) {
        Some("plan") => {
            let name = value("--name").expect("--name calibrate|main");
            let jobs = value("--jobs").map_or(8, |j| j.parse().unwrap());
            let out = value("--out").map_or_else(|| PathBuf::from("results").join(&name), PathBuf::from);
            run_plan(&name, jobs, out);
        }
        Some("count") => {
            let name = value("--name").expect("--name calibrate|main");
            println!("{}", plan(&name).len());
        }
        _ => eprintln!("usage: rnn-gradient-reach plan --name calibrate|main [--jobs 8] [--out DIR]\n       rnn-gradient-reach count --name calibrate|main"),
    }
}
