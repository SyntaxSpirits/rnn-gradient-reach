use crate::model::{clip_jointly, Arch, Init, Model, Regulariser, Rnn};
use crate::tasks::{Batch, Task, RECALL_SYMBOLS};
use ndarray::{Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rust_lstm::{Adam, CrossEntropyLoss, LinearLayer, LossFunction, MSELoss};

pub const CHECKPOINTS: [usize; 6] = [0, 100, 250, 500, 1000, 2000];
const EVAL_EVERY: usize = 100;
const TRAINING_PROBE_DRAWS: u64 = 8;
const STATE_DIRECTIONS: usize = 4;

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Arch,
    pub task: Task,
    pub length: usize,
    pub init: Init,
    pub regulariser: Regulariser,
    pub rate: f64,
    pub seed: u64,
    pub hidden: usize,
    pub batch: usize,
    pub max_updates: usize,
    pub learning_rate: f64,
    pub clip: f64,
}

/// Summary over the sequences of the probe batch of one per-sequence gradient norm at
/// every lag: arithmetic mean, mean of logarithms over non-zero norms, and the number
/// of zero norms.
#[derive(Clone, Debug, Default)]
pub struct Curve {
    pub mean: Vec<f64>,
    pub log_mean: Vec<f64>,
    pub zeros: Vec<f64>,
}

/// Gradients measured at one checkpoint. Lag k refers to the input k steps before the
/// last one. In training mode every quantity is averaged over several mask draws.
#[derive(Clone, Debug)]
pub struct Probe {
    pub update: usize,
    pub training_mode: bool,
    /// Per-sequence norm of the derivative of that sequence's loss.
    pub loss: Curve,
    /// Per-sequence norm of the derivative of u · h_T for fixed unit vectors u,
    /// independent of the loss and of the output layer.
    pub state: Curve,
    /// Adding: mean derivative of the prediction with respect to the first and the
    /// second marked value (both 1 for a perfect model) and mean absolute derivative
    /// with respect to unmarked values (0). Recall: mean derivative of the true-class
    /// logit with respect to the symbol input at the first step.
    pub task: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub update: usize,
    pub loss: f64,
    pub metric: f64,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub config: Config,
    pub solved_at: Option<usize>,
    pub updates: usize,
    pub history: Vec<Evaluation>,
    pub test_loss: f64,
    pub test_metric: f64,
    pub probes: Vec<Probe>,
    pub seconds: f64,
}

/// Mixes a run seed with further identifiers into an independent seed.
fn mix(parts: &[u64]) -> u64 {
    parts.iter().fold(0xcbf2_9ce4_8422_2325, |h, &p| {
        (h ^ p).wrapping_mul(0x0100_0000_01b3).rotate_left(29)
    })
}

fn loss_and_gradient(
    task: Task,
    prediction: &Array2<f64>,
    target: &Array2<f64>,
) -> (f64, Array2<f64>) {
    match task {
        Task::Adding => (
            MSELoss.compute_batch_loss(prediction, target),
            MSELoss.compute_batch_gradient(prediction, target),
        ),
        Task::Recall => (
            CrossEntropyLoss.compute_loss(prediction, target),
            CrossEntropyLoss.compute_gradient(prediction, target),
        ),
    }
}

/// Mean squared error for adding, accuracy for recall.
fn metric(task: Task, prediction: &Array2<f64>, target: &Array2<f64>) -> f64 {
    match task {
        Task::Adding => MSELoss.compute_batch_loss(prediction, target),
        Task::Recall => {
            let argmax = |m: &Array2<f64>, b: usize| {
                m.column(b)
                    .iter()
                    .enumerate()
                    .max_by(|a, c| a.1.total_cmp(c.1))
                    .map(|(i, _)| i)
                    .unwrap()
            };
            let hits = (0..target.ncols())
                .filter(|&b| argmax(prediction, b) == argmax(target, b))
                .count();
            hits as f64 / target.ncols() as f64
        }
    }
}

/// Euclidean norm computed on scaled values so that tiny gradients do not underflow.
fn norm(values: ndarray::ArrayView1<f64>) -> f64 {
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

fn average_curves(curves: &[Curve]) -> Curve {
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

impl Model {
    fn new(config: &Config) -> Model {
        let mut init_rng = StdRng::seed_from_u64(mix(&[config.seed, 1]));
        let rnn = Rnn::new(
            config.arch,
            config.task.input_size(),
            config.hidden,
            config.regulariser,
            config.rate,
            config.init,
            config.length,
            &mut init_rng,
        );
        Model {
            rnn,
            head: LinearLayer::new(config.hidden, config.task.output_size()),
        }
    }

    fn evaluate(&mut self, task: Task, data: &Batch) -> (f64, f64) {
        self.rnn.eval();
        let (outputs, _) = self.rnn.forward(&data.inputs);
        let prediction = self.head.forward(outputs.last().unwrap());
        self.rnn.train();
        let (loss, _) = loss_and_gradient(task, &prediction, &data.targets);
        (loss, metric(task, &prediction, &data.targets))
    }

    /// Input gradients of the network for the gradient `d_last` placed on the last
    /// hidden state.
    fn input_gradients(
        &self,
        caches: &crate::model::RnnCaches,
        steps: usize,
        d_last: Array2<f64>,
    ) -> Vec<Array2<f64>> {
        let mut d_outputs: Vec<Array2<f64>> = (0..steps)
            .map(|_| Array2::zeros(d_last.raw_dim()))
            .collect();
        d_outputs[steps - 1] = d_last;
        self.rnn.backward(&d_outputs, caches).1
    }

    fn probe_once(
        &mut self,
        task: Task,
        probe: &Batch,
        directions: &[Array2<f64>],
    ) -> (Curve, Curve, Vec<f64>) {
        let steps = probe.inputs.len();
        let size = probe.targets.ncols();
        let (outputs, caches) = self.rnn.forward(&probe.inputs);
        let prediction = self.head.forward(outputs.last().unwrap());

        let (_, d_prediction) = loss_and_gradient(task, &prediction, &probe.targets);
        let (_, d_last) = self.head.backward(&(d_prediction * size as f64));
        let loss = curve(&self.input_gradients(&caches, steps, d_last));

        let state_curves: Vec<Curve> = directions
            .iter()
            .map(|u| {
                let d_last = u.broadcast((u.nrows(), size)).unwrap().to_owned();
                curve(&self.input_gradients(&caches, steps, d_last))
            })
            .collect();
        let state = average_curves(&state_curves);

        let task_values = match task {
            Task::Adding => {
                let (_, d_last) = self.head.backward(&Array2::ones((1, size)));
                let d = self.input_gradients(&caches, steps, d_last);
                let (mut first, mut second, mut unmarked, mut count) = (0.0, 0.0, 0.0, 0.0);
                for b in 0..size {
                    let marked: Vec<usize> = (0..steps)
                        .filter(|&t| probe.inputs[t][[1, b]] == 1.0)
                        .collect();
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
                let (_, d_last) = self.head.backward(&probe.targets);
                let d = self.input_gradients(&caches, steps, d_last);
                let total: f64 = (0..size)
                    .map(|b| {
                        let symbol = (0..RECALL_SYMBOLS)
                            .find(|&s| probe.inputs[0][[s, b]] == 1.0)
                            .unwrap();
                        d[0][[symbol, b]]
                    })
                    .sum();
                vec![total / size as f64]
            }
        };
        (loss, state, task_values)
    }

    fn probe(
        &mut self,
        config: &Config,
        probe: &Batch,
        directions: &[Array2<f64>],
        update: usize,
        training_mode: bool,
    ) -> Probe {
        let draws = if training_mode {
            TRAINING_PROBE_DRAWS
        } else {
            1
        };
        if !training_mode {
            self.rnn.eval();
        }
        let mut losses = Vec::new();
        let mut states = Vec::new();
        let mut tasks: Vec<Vec<f64>> = Vec::new();
        for draw in 0..draws {
            rust_lstm::seed(mix(&[config.seed, 2, update as u64, draw]));
            let (l, s, t) = self.probe_once(config.task, probe, directions);
            losses.push(l);
            states.push(s);
            tasks.push(t);
        }
        self.rnn.train();
        let task = (0..tasks[0].len())
            .map(|i| tasks.iter().map(|t| t[i]).sum::<f64>() / tasks.len() as f64)
            .collect();
        Probe {
            update,
            training_mode,
            loss: average_curves(&losses),
            state: average_curves(&states),
            task,
        }
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

pub fn run(config: &Config) -> Outcome {
    let start = std::time::Instant::now();
    rust_lstm::seed(mix(&[config.seed, 0]));
    let mut data_rng = StdRng::seed_from_u64(mix(&[config.seed, 4]));
    let probe = config.task.probe(config.length);
    let validation = config.task.validation(config.length);
    let directions = unit_directions(config.hidden);

    let mut model = Model::new(config);
    model.rnn.train();
    let mut optimizer = Adam::new(config.learning_rate);
    let mut probes = Vec::new();
    let mut history = Vec::new();
    let mut solved_at = None;

    let mut update = 0;
    loop {
        if update % EVAL_EVERY == 0 {
            let (loss, metric) = model.evaluate(config.task, &validation);
            history.push(Evaluation {
                update,
                loss,
                metric,
            });
            if update > 0 && config.task.solved(metric) {
                solved_at = Some(update);
            }
        }
        if CHECKPOINTS.contains(&update) && solved_at.is_none() {
            probes.push(model.probe(config, &probe, &directions, update, true));
            probes.push(model.probe(config, &probe, &directions, update, false));
            rust_lstm::seed(mix(&[config.seed, 5, update as u64]));
        }
        if update == config.max_updates || solved_at.is_some() {
            break;
        }

        let batch = config
            .task
            .batch(&mut data_rng, config.length, config.batch);
        let (outputs, caches) = model.rnn.forward(&batch.inputs);
        let prediction = model.head.forward(outputs.last().unwrap());
        let (_, d_prediction) = loss_and_gradient(config.task, &prediction, &batch.targets);
        let (mut head_gradients, d_last) = model.head.backward(&d_prediction);
        let mut d_outputs: Vec<Array2<f64>> =
            outputs.iter().map(|o| Array2::zeros(o.raw_dim())).collect();
        *d_outputs.last_mut().unwrap() = d_last;
        let (mut rnn_gradients, _) = model.rnn.backward(&d_outputs, &caches);
        clip_jointly(&mut rnn_gradients, &mut head_gradients, config.clip);
        model.rnn.update(&rnn_gradients, &mut optimizer);
        model
            .head
            .update_parameters(&head_gradients, &mut optimizer, "head");
        update += 1;
    }

    let (test_loss, test_metric) = model.evaluate(config.task, &config.task.test(config.length));
    Outcome {
        config: config.clone(),
        solved_at,
        updates: update,
        history,
        test_loss,
        test_metric,
        probes,
        seconds: start.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(arch: Arch, task: Task, init: Init, regulariser: Regulariser) -> Config {
        Config {
            arch,
            task,
            length: 12,
            init,
            regulariser,
            rate: 0.2,
            seed: 3,
            hidden: 8,
            batch: 4,
            max_updates: 120,
            learning_rate: 1e-3,
            clip: 1.0,
        }
    }

    #[test]
    fn norm_survives_values_whose_squares_underflow() {
        let v = ndarray::arr1(&[3e-200, 4e-200]);
        assert!((norm(v.view()) / 5e-200 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn runs_are_reproducible() {
        let c = small(
            Arch::Lstm,
            Task::Adding,
            Init::Chrono,
            Regulariser::ZoneoutBoth,
        );
        let a = run(&c);
        let b = run(&c);
        assert_eq!(
            a.history.last().unwrap().loss,
            b.history.last().unwrap().loss
        );
        assert_eq!(a.probes[2].loss.mean, b.probes[2].loss.mean);
    }

    #[test]
    fn chrono_biases_follow_tallec_and_ollivier() {
        let c = small(Arch::Lstm, Task::Recall, Init::Chrono, Regulariser::None);
        let model = Model::new(&c);
        let Rnn::Lstm(net) = &model.rnn else {
            unreachable!()
        };
        let b = &net.get_cells()[0].b_ih;
        let h = c.hidden;
        for j in 0..h {
            let log_u = b[[h + j, 0]];
            assert!(log_u >= 0.0 && log_u <= ((c.length - 1) as f64).ln());
            assert_eq!(b[[j, 0]], -log_u);
        }
        let g = small(Arch::Gru, Task::Recall, Init::Chrono, Regulariser::None);
        let Rnn::Gru(net) = &Model::new(&g).rnn else {
            unreachable!()
        };
        assert!(net.get_cells()[0].b_iz.iter().all(|&v| v <= 0.0));
    }

    #[test]
    fn every_condition_runs_and_probes_both_modes() {
        for arch in [Arch::Lstm, Arch::Gru] {
            for regulariser in Regulariser::ALL.into_iter().filter(|r| arch.supports(*r)) {
                for task in [Task::Adding, Task::Recall] {
                    let o = run(&small(arch, task, Init::Conventional, regulariser));
                    assert_eq!(o.probes[0].loss.mean.len(), 12);
                    assert!(o.probes.iter().any(|p| p.training_mode));
                    assert!(o.probes.iter().any(|p| !p.training_mode));
                    assert!(o
                        .probes
                        .iter()
                        .all(|p| p.loss.mean.iter().all(|v| v.is_finite())));
                }
            }
        }
    }
}
