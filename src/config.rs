use crate::tasks::Task;

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

#[derive(Clone, Debug, PartialEq)]
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

/// Column names of the fields that identify a run in every output file.
pub const KEY_COLUMNS: &str = "arch,task,length,init,regulariser,rate,seed,hidden,max_updates";
pub const KEY_FIELDS: usize = 9;

impl Arch {
    pub const ALL: [Arch; 2] = [Arch::Lstm, Arch::Gru];

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

impl Init {
    pub const ALL: [Init; 2] = [Init::Conventional, Init::Chrono];

    pub fn name(self) -> &'static str {
        match self {
            Init::Conventional => "conventional",
            Init::Chrono => "chrono",
        }
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

    /// Rate used throughout the study: 0.25 for dropout, 0.15 for zoneout.
    pub fn default_rate(self) -> f64 {
        match self {
            Regulariser::None => 0.0,
            r if r.is_zoneout() => 0.15,
            _ => 0.25,
        }
    }
}

impl Config {
    pub fn new(
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
            rate: regulariser.default_rate(),
            seed,
            hidden: 64,
            batch: 32,
            max_updates: 5000,
            learning_rate: 1e-3,
            clip: 1.0,
        }
    }

    /// Values of `KEY_COLUMNS`, comma-separated.
    pub fn key(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{}",
            self.arch.name(),
            self.task.name(),
            self.length,
            self.init.name(),
            self.regulariser.name(),
            self.rate,
            self.seed,
            self.hidden,
            self.max_updates
        )
    }

    /// Inverse of `key`; the remaining settings take the study defaults.
    pub fn from_key(key: &str) -> Option<Config> {
        let f: Vec<&str> = key.split(',').collect();
        if f.len() != KEY_FIELDS {
            return None;
        }
        let arch = Arch::ALL.into_iter().find(|a| a.name() == f[0])?;
        let task = Task::ALL.into_iter().find(|t| t.name() == f[1])?;
        let init = Init::ALL.into_iter().find(|i| i.name() == f[3])?;
        let regulariser = Regulariser::ALL.into_iter().find(|r| r.name() == f[4])?;
        let mut config = Config::new(
            arch,
            task,
            f[2].parse().ok()?,
            init,
            regulariser,
            f[6].parse().ok()?,
        );
        config.rate = f[5].parse().ok()?;
        config.hidden = f[7].parse().ok()?;
        config.max_updates = f[8].parse().ok()?;
        Some(config)
    }

    /// Mixes the run seed with further identifiers into the seed of an independent
    /// random stream.
    pub fn stream(&self, parts: &[u64]) -> u64 {
        mix(&[&[self.seed], parts].concat())
    }
}

pub fn mix(parts: &[u64]) -> u64 {
    parts.iter().fold(0xcbf2_9ce4_8422_2325, |h, &p| {
        (h ^ p).wrapping_mul(0x0100_0000_01b3).rotate_left(29)
    })
}
