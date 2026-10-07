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

## Reproducing

```bash
cargo run --release -- plan calibrate          # lengths, unregularised networks
cargo run --release -- plan calibrate_long
cargo run --release -- plan main               # 720 runs, about four hours on 8 cores
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
