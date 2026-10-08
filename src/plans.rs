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

/// Conventionally initialised conditions that solved fewer than 5 of 5 runs within
/// 5000 updates in the main grid, at every length except the longest of each task.
const BELOW_CEILING: [(Task, usize, Arch, &[Regulariser]); 7] = [
    (
        Task::Adding,
        100,
        Arch::Lstm,
        &[
            Regulariser::RecurrentDropout,
            Regulariser::VariationalDropout,
            Regulariser::UpdateDropout,
            Regulariser::ZoneoutCell,
            Regulariser::ZoneoutBoth,
        ],
    ),
    (
        Task::Adding,
        200,
        Arch::Gru,
        &[Regulariser::VariationalDropout],
    ),
    (Task::Adding, 200, Arch::Lstm, &Regulariser::ALL),
    (
        Task::Recall,
        30,
        Arch::Gru,
        &[
            Regulariser::None,
            Regulariser::RecurrentDropout,
            Regulariser::VariationalDropout,
            Regulariser::UpdateDropout,
        ],
    ),
    (
        Task::Recall,
        30,
        Arch::Lstm,
        &[
            Regulariser::RecurrentDropout,
            Regulariser::VariationalDropout,
            Regulariser::UpdateDropout,
        ],
    ),
    (
        Task::Recall,
        45,
        Arch::Gru,
        &[
            Regulariser::None,
            Regulariser::RecurrentDropout,
            Regulariser::VariationalDropout,
            Regulariser::UpdateDropout,
            Regulariser::ZoneoutHidden,
        ],
    ),
    (Task::Recall, 45, Arch::Lstm, &Regulariser::ALL),
];

/// Conditions of the main grid in which some but not all of the five runs solved the
/// task; all are conventionally initialised.
const MIXED: [(Task, usize, Arch, Regulariser); 13] = [
    (Task::Adding, 100, Arch::Lstm, Regulariser::RecurrentDropout),
    (Task::Adding, 100, Arch::Lstm, Regulariser::UpdateDropout),
    (Task::Adding, 100, Arch::Lstm, Regulariser::ZoneoutBoth),
    (Task::Adding, 100, Arch::Lstm, Regulariser::ZoneoutCell),
    (
        Task::Adding,
        200,
        Arch::Gru,
        Regulariser::VariationalDropout,
    ),
    (Task::Recall, 30, Arch::Gru, Regulariser::None),
    (Task::Recall, 30, Arch::Lstm, Regulariser::RecurrentDropout),
    (Task::Recall, 30, Arch::Lstm, Regulariser::UpdateDropout),
    (
        Task::Recall,
        30,
        Arch::Lstm,
        Regulariser::VariationalDropout,
    ),
    (Task::Recall, 45, Arch::Lstm, Regulariser::UpdateDropout),
    (
        Task::Recall,
        45,
        Arch::Lstm,
        Regulariser::VariationalDropout,
    ),
    (Task::Recall, 45, Arch::Lstm, Regulariser::ZoneoutBoth),
    (Task::Recall, 45, Arch::Lstm, Regulariser::ZoneoutHidden),
];

pub const NAMES: [&str; 6] = [
    "calibrate",
    "calibrate_long",
    "main",
    "long_budget",
    "chrono_mismatch",
    "more_seeds",
];

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
        "long_budget" => {
            let mut configs = Vec::new();
            for seed in 0..5 {
                for (task, length, arch, regularisers) in BELOW_CEILING {
                    for &regulariser in regularisers {
                        if arch.supports(regulariser) {
                            let mut c = Config::new(
                                arch,
                                task,
                                length,
                                Init::Conventional,
                                regulariser,
                                seed,
                            );
                            c.max_updates = 20_000;
                            configs.push(c);
                        }
                    }
                }
            }
            configs
        }
        "chrono_mismatch" => {
            let mut configs = Vec::new();
            for seed in 0..5 {
                for arch in Arch::ALL {
                    for init in Init::MISMATCHED {
                        for (task, lengths) in MAIN_LENGTHS {
                            for length in lengths {
                                configs.push(Config::new(
                                    arch,
                                    task,
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
        "more_seeds" => {
            let mut configs = Vec::new();
            for seed in 5..20 {
                for (task, length, arch, regulariser) in MIXED {
                    configs.push(Config::new(
                        arch,
                        task,
                        length,
                        Init::Conventional,
                        regulariser,
                        seed,
                    ));
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
