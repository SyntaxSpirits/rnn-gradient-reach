use crate::config::{Arch, Config, Init, Regulariser};
use ndarray::{s, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rust_lstm::optimizers::Optimizer;
use rust_lstm::{
    GRUCellGradients, GRUNetwork, GRUNetworkCache, LSTMCellGradients, LSTMNetwork,
    LSTMNetworkCache, LinearGradients, LinearLayer,
};

pub enum Rnn {
    Lstm(LSTMNetwork),
    Gru(GRUNetwork),
}

pub enum RnnCaches {
    Lstm(Vec<LSTMNetworkCache>),
    Gru(Vec<GRUNetworkCache>),
}

pub enum RnnGradients {
    Lstm(Vec<LSTMCellGradients>),
    Gru(Vec<GRUCellGradients>),
}

impl Rnn {
    /// Builds a one-layer network. Biases are set on `b_ih`; `b_hh` stays zero, so the
    /// stated value is the effective bias.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        arch: Arch,
        input: usize,
        hidden: usize,
        regulariser: Regulariser,
        rate: f64,
        init: Init,
        t_max: usize,
        rng: &mut StdRng,
    ) -> Rnn {
        let mut timescales = || -> Vec<f64> {
            (0..hidden)
                .map(|_| rng.gen_range(1.0..(t_max - 1) as f64).ln())
                .collect()
        };
        match arch {
            Arch::Lstm => {
                let mut net = LSTMNetwork::new(input, hidden, 1);
                let b = &mut net.get_cells_mut()[0].b_ih;
                match init {
                    Init::Conventional => b.slice_mut(s![hidden..2 * hidden, ..]).fill(1.0),
                    Init::Chrono => {
                        for (j, log_u) in timescales().into_iter().enumerate() {
                            b[[j, 0]] = -log_u;
                            b[[hidden + j, 0]] = log_u;
                        }
                    }
                }
                Rnn::Lstm(match regulariser {
                    Regulariser::None => net,
                    Regulariser::RecurrentDropout => net.with_recurrent_dropout(rate, false),
                    Regulariser::VariationalDropout => net.with_recurrent_dropout(rate, true),
                    Regulariser::UpdateDropout => net.with_cell_update_dropout(rate, false),
                    Regulariser::ZoneoutCell => net.with_zoneout(rate, 0.0),
                    Regulariser::ZoneoutHidden => net.with_zoneout(0.0, rate),
                    Regulariser::ZoneoutBoth => net.with_zoneout(rate, rate),
                })
            }
            Arch::Gru => {
                let mut net = GRUNetwork::new(input, hidden, 1);
                if init == Init::Chrono {
                    let b = &mut net.get_cells_mut()[0].b_iz;
                    for (j, log_u) in timescales().into_iter().enumerate() {
                        b[[j, 0]] = -log_u;
                    }
                }
                Rnn::Gru(match regulariser {
                    Regulariser::None => net,
                    Regulariser::RecurrentDropout => net.with_recurrent_dropout(rate, false),
                    Regulariser::VariationalDropout => net.with_recurrent_dropout(rate, true),
                    Regulariser::UpdateDropout => net.with_candidate_dropout(rate, false),
                    Regulariser::ZoneoutHidden => net.with_zoneout(rate),
                    Regulariser::ZoneoutCell | Regulariser::ZoneoutBoth => {
                        panic!("GRU has no cell state to zone out")
                    }
                })
            }
        }
    }

    pub fn train(&mut self) {
        match self {
            Rnn::Lstm(n) => n.train(),
            Rnn::Gru(n) => n.train(),
        }
    }

    pub fn eval(&mut self) {
        match self {
            Rnn::Lstm(n) => n.eval(),
            Rnn::Gru(n) => n.eval(),
        }
    }

    /// Top-layer outputs at every step and the caches for `backward`.
    pub fn forward(&mut self, inputs: &[Array2<f64>]) -> (Vec<Array2<f64>>, RnnCaches) {
        match self {
            Rnn::Lstm(n) => {
                let (outs, caches) = n.forward_sequence_with_cache(inputs);
                (
                    outs.into_iter().map(|(h, _)| h).collect(),
                    RnnCaches::Lstm(caches),
                )
            }
            Rnn::Gru(n) => {
                let (outs, caches) = n.forward_sequence_with_cache(inputs);
                (
                    outs.into_iter().map(|(h, _)| h).collect(),
                    RnnCaches::Gru(caches),
                )
            }
        }
    }

    pub fn backward(
        &self,
        d_outputs: &[Array2<f64>],
        caches: &RnnCaches,
    ) -> (RnnGradients, Vec<Array2<f64>>) {
        match (self, caches) {
            (Rnn::Lstm(n), RnnCaches::Lstm(c)) => {
                let (g, dx) = n.backward_sequence(d_outputs, c);
                (RnnGradients::Lstm(g), dx)
            }
            (Rnn::Gru(n), RnnCaches::Gru(c)) => {
                let (g, dx) = n.backward_sequence(d_outputs, c);
                (RnnGradients::Gru(g), dx)
            }
            _ => unreachable!("caches come from the same network"),
        }
    }

    pub fn update<O: Optimizer>(&mut self, gradients: &RnnGradients, optimizer: &mut O) {
        match (self, gradients) {
            (Rnn::Lstm(n), RnnGradients::Lstm(g)) => n.update_parameters(g, optimizer),
            (Rnn::Gru(n), RnnGradients::Gru(g)) => n.update_parameters(g, optimizer),
            _ => unreachable!("gradients come from the same network"),
        }
    }
}

impl RnnGradients {
    fn matrices_mut(&mut self) -> Vec<&mut Array2<f64>> {
        match self {
            RnnGradients::Lstm(gs) => gs
                .iter_mut()
                .flat_map(|g| [&mut g.w_ih, &mut g.w_hh, &mut g.b_ih, &mut g.b_hh])
                .collect(),
            RnnGradients::Gru(gs) => gs
                .iter_mut()
                .flat_map(|g| {
                    [
                        &mut g.w_ir,
                        &mut g.w_hr,
                        &mut g.b_ir,
                        &mut g.b_hr,
                        &mut g.w_iz,
                        &mut g.w_hz,
                        &mut g.b_iz,
                        &mut g.b_hz,
                        &mut g.w_ih,
                        &mut g.w_hh,
                        &mut g.b_ih,
                        &mut g.b_hh,
                    ]
                })
                .collect(),
        }
    }
}

/// Rescales the recurrent and head gradients together to a joint L2 norm of at most
/// `max_norm`.
pub fn clip_jointly(rnn: &mut RnnGradients, head: &mut LinearGradients, max_norm: f64) {
    let mut matrices = rnn.matrices_mut();
    matrices.push(&mut head.weight);
    matrices.push(&mut head.bias);
    let norm = matrices
        .iter()
        .map(|m| m.iter().map(|v| v * v).sum::<f64>())
        .sum::<f64>()
        .sqrt();
    if norm > max_norm {
        for m in matrices {
            m.mapv_inplace(|v| v * max_norm / norm);
        }
    }
}

/// A one-layer recurrent network with a linear output layer on its last state.
pub struct Model {
    pub rnn: Rnn,
    pub head: LinearLayer,
}

impl Model {
    pub fn new(config: &Config) -> Model {
        let mut init_rng = StdRng::seed_from_u64(config.stream(&[1]));
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

    /// Output-layer prediction from the last hidden state, with the network caches.
    pub fn forward(&mut self, inputs: &[Array2<f64>]) -> (Array2<f64>, RnnCaches) {
        let (outputs, caches) = self.rnn.forward(inputs);
        (self.head.forward(outputs.last().unwrap()), caches)
    }

    /// Input gradients for a gradient `d_last` on the last hidden state, all other
    /// hidden states receiving none.
    pub fn backward_from_last(
        &self,
        caches: &RnnCaches,
        steps: usize,
        d_last: Array2<f64>,
    ) -> (RnnGradients, Vec<Array2<f64>>) {
        let mut d_outputs: Vec<Array2<f64>> = (0..steps)
            .map(|_| Array2::zeros(d_last.raw_dim()))
            .collect();
        d_outputs[steps - 1] = d_last;
        self.rnn.backward(&d_outputs, caches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::Task;

    #[test]
    fn chrono_biases_follow_tallec_and_ollivier() {
        let mut c = Config::new(
            Arch::Lstm,
            Task::Recall,
            12,
            Init::Chrono,
            Regulariser::None,
            3,
        );
        c.hidden = 8;
        let Rnn::Lstm(net) = &Model::new(&c).rnn else {
            unreachable!()
        };
        let b = &net.get_cells()[0].b_ih;
        for j in 0..c.hidden {
            let log_u = b[[c.hidden + j, 0]];
            assert!(log_u >= 0.0 && log_u <= ((c.length - 1) as f64).ln());
            assert_eq!(b[[j, 0]], -log_u);
        }
        c.arch = Arch::Gru;
        let Rnn::Gru(net) = &Model::new(&c).rnn else {
            unreachable!()
        };
        assert!(net.get_cells()[0].b_iz.iter().all(|&v| v <= 0.0));
    }
}
