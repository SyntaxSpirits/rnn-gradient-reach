use ndarray::{s, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rust_lstm::optimizers::Optimizer;
use rust_lstm::{
    GRUCellGradients, GRUNetwork, GRUNetworkCache, LSTMCellGradients, LSTMNetwork,
    LSTMNetworkCache, LinearGradients, LinearLayer,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Lstm,
    Gru,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Init {
    /// LSTM forget-gate bias 1 (Gers et al., 2000; Jozefowicz et al., 2015), all other
    /// biases 0; GRU biases 0, so the update gate starts at 0.5.
    Conventional,
    /// Gate biases drawn so that unit j keeps its memory for about u_j steps,
    /// u_j ~ U[1, T_max - 1] (Tallec & Ollivier, 2018).
    Chrono,
}

impl Init {
    pub fn name(self) -> &'static str {
        match self {
            Init::Conventional => "conventional",
            Init::Chrono => "chrono",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regulariser {
    None,
    /// Dropout on the recurrent input of the gates, new mask at every step.
    RecurrentDropout,
    /// The same placement with one mask per sequence (Gal & Ghahramani, 2016).
    VariationalDropout,
    /// Dropout on the candidate update (Semeniuta et al., 2016).
    UpdateDropout,
    /// Zoneout of the LSTM cell state (Krueger et al., 2017).
    ZoneoutCell,
    /// Zoneout of the hidden state.
    ZoneoutHidden,
    ZoneoutBoth,
}

impl Arch {
    pub fn name(self) -> &'static str {
        match self {
            Arch::Lstm => "lstm",
            Arch::Gru => "gru",
        }
    }

    pub fn supports(self, regulariser: Regulariser) -> bool {
        self == Arch::Lstm
            || !matches!(
                regulariser,
                Regulariser::ZoneoutCell | Regulariser::ZoneoutBoth
            )
    }
}

impl Regulariser {
    pub const ALL: [Regulariser; 7] = [
        Regulariser::None,
        Regulariser::RecurrentDropout,
        Regulariser::VariationalDropout,
        Regulariser::UpdateDropout,
        Regulariser::ZoneoutCell,
        Regulariser::ZoneoutHidden,
        Regulariser::ZoneoutBoth,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Regulariser::None => "none",
            Regulariser::RecurrentDropout => "recurrent_dropout",
            Regulariser::VariationalDropout => "variational_dropout",
            Regulariser::UpdateDropout => "update_dropout",
            Regulariser::ZoneoutCell => "zoneout_cell",
            Regulariser::ZoneoutHidden => "zoneout_hidden",
            Regulariser::ZoneoutBoth => "zoneout_both",
        }
    }

    pub fn is_zoneout(self) -> bool {
        matches!(
            self,
            Regulariser::ZoneoutCell | Regulariser::ZoneoutHidden | Regulariser::ZoneoutBoth
        )
    }
}

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

pub struct Model {
    pub rnn: Rnn,
    pub head: LinearLayer,
}
