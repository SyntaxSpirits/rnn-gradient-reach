//! Gradients of a trained network with respect to its inputs, as a function of lag.

use crate::config::{mix, Config};
use crate::network::{Model, RnnCaches};
use crate::tasks::{Batch, Task, RECALL_SYMBOLS};
use ndarray::{Array2, ArrayView1, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TRAINING_MODE_DRAWS: u64 = 8;
const STATE_DIRECTIONS: usize = 4;

/// Per-sequence gradient norms at every lag, summarised over the probe batch: the
/// arithmetic mean, the mean of logarithms over non-zero norms, and the number of
/// zero norms. Index k is the input k steps before the last one.
#[derive(Clone, Debug, Default)]
pub struct Curve {
    pub mean: Vec<f64>,
    pub log_mean: Vec<f64>,
    pub zeros: Vec<f64>,
}

/// Gradients measured at one checkpoint. In training mode every quantity is averaged
/// over several draws of the dropout and zoneout masks.
#[derive(Clone, Debug)]
pub struct Probe {
    pub update: usize,
    pub training_mode: bool,
    /// Norm of the derivative of each sequence's own loss.
    pub loss: Curve,
    /// Norm of the derivative of u · h_T for fixed unit vectors u; independent of the
    /// loss and of the output layer.
    pub state: Curve,
    /// Adding: mean derivative of the prediction with respect to the first and the
    /// second marked value (1 for a perfect model) and mean absolute derivative with
    /// respect to unmarked values (0). Recall: mean derivative of the true-class logit
    /// with respect to the symbol input at the first step.
    pub task: Vec<f64>,
}

/// Fixed probe inputs and output directions shared by every checkpoint of a run.
pub struct Prober {
    batch: Batch,
    directions: Vec<Array2<f64>>,
}

impl Prober {
    pub fn new(config: &Config) -> Prober {
        Prober {
            batch: config.task.probe(config.length),
            directions: unit_directions(config.hidden),
        }
    }

    pub fn measure(
        &self,
        model: &mut Model,
        config: &Config,
        update: usize,
        training_mode: bool,
    ) -> Probe {
        let draws = if training_mode {
            TRAINING_MODE_DRAWS
        } else {
            1
        };
        if !training_mode {
            model.rnn.eval();
        }
        let mut losses = Vec::new();
        let mut states = Vec::new();
        let mut tasks: Vec<Vec<f64>> = Vec::new();
        for draw in 0..draws {
            rust_lstm::seed(config.stream(&[2, update as u64, draw]));
            let (loss, state, task) = self.measure_once(model, config.task);
            losses.push(loss);
            states.push(state);
            tasks.push(task);
        }
        model.rnn.train();
        let task = (0..tasks[0].len())
            .map(|i| tasks.iter().map(|t| t[i]).sum::<f64>() / tasks.len() as f64)
            .collect();
        Probe {
            update,
            training_mode,
            loss: average(&losses),
            state: average(&states),
            task,
        }
    }

    fn measure_once(&self, model: &mut Model, task: Task) -> (Curve, Curve, Vec<f64>) {
        let steps = self.batch.inputs.len();
        let size = self.batch.targets.ncols();
        let (prediction, caches) = model.forward(&self.batch.inputs);
        let input_gradients = |model: &Model, caches: &RnnCaches, d_last| {
            model.backward_from_last(caches, steps, d_last).1
        };

        let (_, d_prediction) = task.loss_and_gradient(&prediction, &self.batch.targets);
        let (_, d_last) = model.head.backward(&(d_prediction * size as f64));
        let loss = curve(&input_gradients(model, &caches, d_last));

        let state_curves: Vec<Curve> = self
            .directions
            .iter()
            .map(|u| {
                let d_last = u.broadcast((u.nrows(), size)).unwrap().to_owned();
                curve(&input_gradients(model, &caches, d_last))
            })
            .collect();
        let state = average(&state_curves);

        let inputs = &self.batch.inputs;
        let task_values = match task {
            Task::Adding => {
                let (_, d_last) = model.head.backward(&Array2::ones((1, size)));
                let d = input_gradients(model, &caches, d_last);
                let (mut first, mut second, mut unmarked, mut count) = (0.0, 0.0, 0.0, 0.0);
                for b in 0..size {
                    let marked: Vec<usize> =
                        (0..steps).filter(|&t| inputs[t][[1, b]] == 1.0).collect();
                    first += d[marked[0]][[0, b]];
                    second += d[marked[1]][[0, b]];
                    for t in (0..steps).filter(|t| !marked.contains(t)) {
                        unmarked += d[t][[0, b]].abs();
                        count += 1.0;
                    }
                }
                vec![first / size as f64, second / size as f64, unmarked / count]
            }
            Task::Recall => {
                let (_, d_last) = model.head.backward(&self.batch.targets);
                let d = input_gradients(model, &caches, d_last);
                let total: f64 = (0..size)
                    .map(|b| {
                        let symbol = (0..RECALL_SYMBOLS)
                            .find(|&s| inputs[0][[s, b]] == 1.0)
                            .unwrap();
                        d[0][[symbol, b]]
                    })
                    .sum();
                vec![total / size as f64]
            }
        };
        (loss, state, task_values)
    }
}

fn unit_directions(hidden: usize) -> Vec<Array2<f64>> {
    let mut rng = StdRng::seed_from_u64(mix(&[hidden as u64, 3]));
    (0..STATE_DIRECTIONS)
        .map(|_| {
            let v = Array2::from_shape_fn((hidden, 1), |_| rng.gen::<f64>() - 0.5);
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v / n
        })
        .collect()
}

/// Euclidean norm computed on scaled values so that tiny gradients do not underflow.
fn norm(values: ArrayView1<f64>) -> f64 {
    let scale = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    scale
        * values
            .iter()
            .map(|v| (v / scale).powi(2))
            .sum::<f64>()
            .sqrt()
}

fn curve(d_inputs: &[Array2<f64>]) -> Curve {
    let mut c = Curve::default();
    for d in d_inputs.iter().rev() {
        let norms: Vec<f64> = d.axis_iter(Axis(1)).map(norm).collect();
        let positive: Vec<f64> = norms.iter().copied().filter(|&n| n > 0.0).collect();
        c.mean.push(norms.iter().sum::<f64>() / norms.len() as f64);
        c.log_mean.push(if positive.is_empty() {
            f64::NAN
        } else {
            positive.iter().map(|n| n.ln()).sum::<f64>() / positive.len() as f64
        });
        c.zeros.push((norms.len() - positive.len()) as f64);
    }
    c
}

fn average(curves: &[Curve]) -> Curve {
    let n = curves.len() as f64;
    let avg = |pick: fn(&Curve) -> &Vec<f64>| -> Vec<f64> {
        (0..pick(&curves[0]).len())
            .map(|k| curves.iter().map(|c| pick(c)[k]).sum::<f64>() / n)
            .collect()
    };
    Curve {
        mean: avg(|c| &c.mean),
        log_mean: avg(|c| &c.log_mean),
        zeros: avg(|c| &c.zeros),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_survives_values_whose_squares_underflow() {
        let v = ndarray::arr1(&[3e-200, 4e-200]);
        assert!((norm(v.view()) / 5e-200 - 1.0).abs() < 1e-12);
    }
}
