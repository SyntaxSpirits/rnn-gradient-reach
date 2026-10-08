# rnn-gradient-reach

How dropout and zoneout change how far gradients reach back in time in LSTM and GRU
networks, and whether that reach, measured early in training, predicts which runs learn
long-range dependencies.

Networks are trained with [rust-lstm](https://github.com/SyntaxSpirits/rust-lstm)
0.10.0 ([doi:10.5281/zenodo.23200142](https://doi.org/10.5281/zenodo.23200142)), whose
backpropagation through time is checked against finite differences and PyTorch.

## Design

| Factor | Levels |
|---|---|
| Architecture | one-layer LSTM, GRU; 64 hidden units |
| Gate initialisation | conventional (LSTM forget bias 1), chrono (Tallec & Ollivier, 2018) |
| Regulariser | none; per-step and variational dropout on the recurrent input; dropout on the candidate update; zoneout of the cell, hidden or both states (LSTM only for cell) |
| Rate | 0.25 for dropout, 0.15 for zoneout |
| Task | adding problem (T = 100, 200, 400); recall of a symbol after T zero inputs (T = 30, 45, 250) |
| Seeds | 5 |

Training uses Adam (learning rate 1e-3), batches of 32, joint gradient clipping at 1.0
and a budget of 5000 updates; a run counts as solved when the validation mean squared
error drops below 0.01 (adding) or all eight symbols are recalled (recall). At updates
0, 100, 250, 500, 1000 and 2000 the input gradients of a fixed probe batch are recorded
at every lag, in evaluation mode and averaged over eight mask draws in training mode.

## Follow-up runs

Planned on 2026-10-08, after the main grid, before any of these runs:

- `more_seeds` — seeds 5–19 for the 13 conditions of the main grid with mixed outcomes.
  Confirmatory test of an exploratory main-grid observation: within these conditions,
  the evaluation-mode state-gradient log-slope (lags 1..T/2) at update 100 ranks runs
  that reach the validation criterion above runs that do not (concordance above 0.5,
  pooled over conditions, 95% interval from resampling conditions), whereas the log
  validation loss at update 100 does not. Only seeds 5–19 enter the test.
- `long_budget` — the 32 conventionally initialised conditions below 5/5 at 5000
  updates (all lengths but the longest of each task) rerun with 20 000 updates, to
  separate slow learning from failure within the budget. Descriptive.
- `chrono_mismatch` — unregularised networks with chrono horizon T/4 or 4T instead of
  T, to see how much of the advantage of chrono initialisation depends on knowing the
  task length. Descriptive.

## Reproducing

```bash
cargo run --release -- plan calibrate          # lengths, unregularised networks
cargo run --release -- plan calibrate_long
cargo run --release -- plan main               # 720 runs, about four hours on 8 cores
cargo run --release -- plan more_seeds
cargo run --release -- plan long_budget
cargo run --release -- plan chrono_mismatch
cargo run --release -- run lstm,recall,30,chrono,zoneout_cell,0.15,0,64,5000

uv venv .venv && uv pip install -r analysis/requirements.txt
.venv/bin/python -m analysis results/main
```

Every run is seeded from its key, so any row of `results/*/runs.csv` can be rerun
exactly. Plans resume where they stopped.

## Results layout

`results/<plan>/runs.csv`, `history.csv` and `probes.csv` hold one row per run, per
validation and per probe; `results/main/PROVENANCE.txt` records the code revision,
toolchain and machine. The per-run gradient curves (`curves/`, 140 MB) are archived
separately. The analysis writes tables and figures to `results/<plan>/analysis/`.

## Licence

MIT
