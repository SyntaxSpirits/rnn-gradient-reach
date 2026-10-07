use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const RECALL_SYMBOLS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// Sum of the two values marked in a sequence of uniform noise (Le et al., 2015).
    Adding,
    /// Recall a symbol shown at the first step when queried at the last step; the
    /// inputs in between are zero.
    Recall,
}

pub struct Batch {
    pub inputs: Vec<Array2<f64>>,
    pub targets: Array2<f64>,
}

impl Task {
    pub fn name(self) -> &'static str {
        match self {
            Task::Adding => "adding",
            Task::Recall => "recall",
        }
    }

    pub fn input_size(self) -> usize {
        match self {
            Task::Adding => 2,
            Task::Recall => RECALL_SYMBOLS + 1,
        }
    }

    pub fn output_size(self) -> usize {
        match self {
            Task::Adding => 1,
            Task::Recall => RECALL_SYMBOLS,
        }
    }

    pub fn batch(self, rng: &mut StdRng, length: usize, size: usize) -> Batch {
        match self {
            Task::Adding => adding(rng, length, size),
            Task::Recall => {
                let symbols: Vec<usize> = (0..size)
                    .map(|_| rng.gen_range(0..RECALL_SYMBOLS))
                    .collect();
                recall(&symbols, length)
            }
        }
    }

    /// Fixed sequences for deciding when a run has solved the task. Recall has only
    /// eight distinct sequences per length, so all eight are used.
    pub fn validation(self, length: usize) -> Batch {
        match self {
            Task::Adding => adding(&mut fixed_rng(1, length), length, 1024),
            Task::Recall => recall(&(0..RECALL_SYMBOLS).collect::<Vec<_>>(), length),
        }
    }

    /// Fresh sequences for the final score of a run.
    pub fn test(self, length: usize) -> Batch {
        match self {
            Task::Adding => adding(&mut fixed_rng(2, length), length, 4096),
            Task::Recall => self.validation(length),
        }
    }

    /// Sequences on which gradients are probed; recall repeats each symbol eight
    /// times so that training-mode probes see several masks per symbol.
    pub fn probe(self, length: usize) -> Batch {
        match self {
            Task::Adding => adding(&mut fixed_rng(3, length), length, 64),
            Task::Recall => recall(
                &(0..64).map(|i| i % RECALL_SYMBOLS).collect::<Vec<_>>(),
                length,
            ),
        }
    }

    /// Validation metric that counts as solving the task: mean squared error below
    /// 0.01 for adding (predicting 1 gives 1/6), all eight symbols correct for recall.
    pub fn solved(self, metric: f64) -> bool {
        match self {
            Task::Adding => metric < 0.01,
            Task::Recall => metric >= 1.0,
        }
    }
}

fn fixed_rng(purpose: u64, length: usize) -> StdRng {
    StdRng::seed_from_u64(0x7a5c_0000 + purpose * 100_000 + length as u64)
}

fn adding(rng: &mut StdRng, length: usize, size: usize) -> Batch {
    let mut inputs = vec![Array2::zeros((2, size)); length];
    let mut targets = Array2::zeros((1, size));
    for b in 0..size {
        let first = rng.gen_range(0..length / 2);
        let second = rng.gen_range(length / 2..length);
        for (t, x) in inputs.iter_mut().enumerate() {
            x[[0, b]] = rng.gen::<f64>();
            x[[1, b]] = if t == first || t == second { 1.0 } else { 0.0 };
        }
        targets[[0, b]] = inputs[first][[0, b]] + inputs[second][[0, b]];
    }
    Batch { inputs, targets }
}

fn recall(symbols: &[usize], length: usize) -> Batch {
    let size = symbols.len();
    let mut inputs = vec![Array2::zeros((RECALL_SYMBOLS + 1, size)); length];
    let mut targets = Array2::zeros((RECALL_SYMBOLS, size));
    for (b, &symbol) in symbols.iter().enumerate() {
        inputs[0][[symbol, b]] = 1.0;
        inputs[length - 1][[RECALL_SYMBOLS, b]] = 1.0;
        targets[[symbol, b]] = 1.0;
    }
    Batch { inputs, targets }
}
