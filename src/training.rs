use crate::config::Config;
use crate::network::{clip_jointly, Model};
use crate::probe::{Probe, Prober};
use crate::tasks::Batch;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rust_lstm::Adam;

/// Updates at which gradients are probed, unless the run has already solved the task.
pub const CHECKPOINTS: [usize; 6] = [0, 100, 250, 500, 1000, 2000];
const EVAL_EVERY: usize = 100;

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

fn evaluate(model: &mut Model, config: &Config, data: &Batch) -> (f64, f64) {
    model.rnn.eval();
    let (prediction, _) = model.forward(&data.inputs);
    model.rnn.train();
    let (loss, _) = config.task.loss_and_gradient(&prediction, &data.targets);
    (loss, config.task.metric(&prediction, &data.targets))
}

fn step(model: &mut Model, config: &Config, optimizer: &mut Adam, batch: &Batch) {
    let (prediction, caches) = model.forward(&batch.inputs);
    let (_, d_prediction) = config.task.loss_and_gradient(&prediction, &batch.targets);
    let (mut head_gradients, d_last) = model.head.backward(&d_prediction);
    let (mut rnn_gradients, _) = model.backward_from_last(&caches, batch.inputs.len(), d_last);
    clip_jointly(&mut rnn_gradients, &mut head_gradients, config.clip);
    model.rnn.update(&rnn_gradients, optimizer);
    model
        .head
        .update_parameters(&head_gradients, optimizer, "head");
}

/// Trains one network with Adam until the validation metric meets the task's
/// threshold or the update budget runs out, probing gradients at the checkpoints.
pub fn run(config: &Config) -> Outcome {
    let start = std::time::Instant::now();
    rust_lstm::seed(config.stream(&[0]));
    let mut data_rng = StdRng::seed_from_u64(config.stream(&[4]));
    let validation = config.task.validation(config.length);
    let prober = Prober::new(config);

    let mut model = Model::new(config);
    model.rnn.train();
    let mut optimizer = Adam::new(config.learning_rate);
    let mut probes = Vec::new();
    let mut history = Vec::new();
    let mut solved_at = None;

    let mut update = 0;
    loop {
        if update % EVAL_EVERY == 0 {
            let (loss, metric) = evaluate(&mut model, config, &validation);
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
            probes.push(prober.measure(&mut model, config, update, true));
            probes.push(prober.measure(&mut model, config, update, false));
            rust_lstm::seed(config.stream(&[5, update as u64]));
        }
        if update == config.max_updates || solved_at.is_some() {
            break;
        }
        let batch = config
            .task
            .batch(&mut data_rng, config.length, config.batch);
        step(&mut model, config, &mut optimizer, &batch);
        update += 1;
    }

    let (test_loss, test_metric) = evaluate(&mut model, config, &config.task.test(config.length));
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
    use crate::config::{Arch, Init, Regulariser};
    use crate::tasks::Task;

    fn small(arch: Arch, task: Task, init: Init, regulariser: Regulariser) -> Config {
        let mut c = Config::new(arch, task, 12, init, regulariser, 3);
        c.rate = 0.2;
        c.hidden = 8;
        c.batch = 4;
        c.max_updates = 120;
        c
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
    fn every_condition_runs_and_probes_both_modes() {
        for arch in Arch::ALL {
            for regulariser in Regulariser::ALL.into_iter().filter(|r| arch.supports(*r)) {
                for task in Task::ALL {
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
