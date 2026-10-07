use crate::config::{Arch, Config, Init, Regulariser};
use crate::tasks::Task;

/// Lengths chosen from the calibration plans: adding 100 is solved by every
/// unregularised network, 200 and 400 by all but the conventionally initialised LSTM;
/// recall 30 is solved slowly with conventional initialisation, 45 and 250 only with
/// chrono initialisation.
const MAIN_LENGTHS: [(Task, [usize; 3]); 2] = [
    (Task::Adding, [100, 200, 400]),
    (Task::Recall, [30, 45, 250]),
];

pub const NAMES: [&str; 3] = ["calibrate", "calibrate_long", "main"];

/// The runs of a named plan; `None` for an unknown name.
pub fn plan(name: &str) -> Option<Vec<Config>> {
    let configs = match name {
        "calibrate" => unregularised(
            0..2,
            &[
                (Task::Adding, &[50, 100, 200, 400]),
                (Task::Recall, &[10, 15, 20, 30, 45, 60]),
            ],
        ),
        "calibrate_long" => unregularised(0..2, &[(Task::Recall, &[120, 250, 500])]),
        "main" => {
            let mut configs = Vec::new();
            for seed in 0..5 {
                for arch in Arch::ALL {
                    for init in Init::ALL {
                        for (task, lengths) in MAIN_LENGTHS {
                            for length in lengths {
                                for regulariser in Regulariser::ALL {
                                    if arch.supports(regulariser) {
                                        configs.push(Config::new(
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
            configs
        }
        _ => return None,
    };
    Some(configs)
}

fn unregularised(seeds: std::ops::Range<u64>, lengths: &[(Task, &[usize])]) -> Vec<Config> {
    let mut configs = Vec::new();
    for seed in seeds {
        for arch in Arch::ALL {
            for init in Init::ALL {
                for (task, ls) in lengths {
                    for &length in *ls {
                        configs.push(Config::new(
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
    configs
}
