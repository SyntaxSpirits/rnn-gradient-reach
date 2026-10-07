use crate::config::Config;
use crate::output::Output;
use crate::training::run;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Runs every configuration not yet recorded in `dir` on `jobs` threads, appending
/// each result as soon as it is complete, so an interrupted plan can be resumed.
pub fn run_all(configs: Vec<Config>, dir: &Path, jobs: usize) -> io::Result<()> {
    let done = Output::completed(dir)?;
    let todo: Vec<Config> = configs
        .into_iter()
        .filter(|c| !done.contains(&c.key()))
        .collect();
    eprintln!("{} runs to do, {} already done", todo.len(), done.len());

    let output = Mutex::new(Output::open(dir)?);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs)
            .map(|_| {
                scope.spawn(|| -> io::Result<()> {
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        let Some(config) = todo.get(i) else {
                            return Ok(());
                        };
                        let outcome = run(config);
                        eprintln!(
                            "[{}/{}] {} solved_at={:?} test={:.4} {:.0}s",
                            i + 1,
                            todo.len(),
                            config.key(),
                            outcome.solved_at,
                            outcome.test_metric,
                            outcome.seconds
                        );
                        output.lock().unwrap().write(&outcome)?;
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .try_for_each(|w| w.join().expect("worker panicked"))
    })
}
