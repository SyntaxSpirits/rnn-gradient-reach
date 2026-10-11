"""Replication of the initialisation contrast of the main grid with PyTorch's own LSTM and GRU modules.

Unregularised one-layer networks, 64 hidden units, conventional and chrono gate initialisation, the adding problem
(T = 100, 200, 400) and delayed recall (T = 30, 45, 250), seeds 0-4: the same tasks, optimiser, batch size, clipping,
budget, validation schedule and success criteria as the Rust runs, but PyTorch's default weight initialisation,
single precision and independently drawn data.

    python replicate.py results/pytorch           # all 120 runs, resumes
    python replicate.py results/pytorch --workers 7
"""
import argparse
import csv
import itertools
import math
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

ARCHS = ("lstm", "gru")
INITS = ("conventional", "chrono")
TASKS = (("adding", 100), ("adding", 200), ("adding", 400), ("recall", 30), ("recall", 45), ("recall", 250))
SEEDS = range(5)
HIDDEN, BATCH, MAX_UPDATES, LR, CLIP, EVAL_EVERY = 64, 32, 5000, 1e-3, 1.0, 100
SYMBOLS = 8
FIELDS = ["arch", "task", "length", "init", "seed", "solved_at", "updates", "final_metric", "seconds", "torch"]


def adding(rng, length, size):
    x = np.zeros((length, size, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((length, size))
    first = rng.integers(0, length // 2, size)
    second = rng.integers(length // 2, length, size)
    x[first, np.arange(size), 1] = 1.0
    x[second, np.arange(size), 1] = 1.0
    y = x[first, np.arange(size), 0] + x[second, np.arange(size), 0]
    return torch.from_numpy(x), torch.from_numpy(y[:, None])


def recall(symbols, length):
    size = len(symbols)
    x = np.zeros((length, size, SYMBOLS + 1), dtype=np.float32)
    x[0, np.arange(size), symbols] = 1.0
    x[length - 1, :, SYMBOLS] = 1.0
    return torch.from_numpy(x), torch.from_numpy(np.asarray(symbols, dtype=np.int64))


def batch(task, rng, length, size):
    if task == "adding":
        return adding(rng, length, size)
    return recall(rng.integers(0, SYMBOLS, size), length)


def validation(task, length):
    if task == "adding":
        return adding(np.random.default_rng(1_000_000 + length), length, 1024)
    return recall(np.arange(SYMBOLS), length)


class Model(nn.Module):
    def __init__(self, arch, task, length, init, rng):
        super().__init__()
        inputs, outputs = (2, 1) if task == "adding" else (SYMBOLS + 1, SYMBOLS)
        self.rnn = (nn.LSTM if arch == "lstm" else nn.GRU)(inputs, HIDDEN)
        self.head = nn.Linear(HIDDEN, outputs)
        with torch.no_grad():
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
            b = self.rnn.bias_ih_l0
            log_u = torch.from_numpy(np.log(rng.uniform(1.0, length - 1, HIDDEN)).astype(np.float32))
            if arch == "lstm":
                # gate order i, f, g, o
                if init == "chrono":
                    b[:HIDDEN] = -log_u
                    b[HIDDEN:2 * HIDDEN] = log_u
                else:
                    b[HIDDEN:2 * HIDDEN] = 1.0
            elif init == "chrono":
                # gate order r, z, n; PyTorch keeps the old state with weight z: h' = (1 - z) n + z h
                b[HIDDEN:2 * HIDDEN] = log_u

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[-1])


def metric(task, prediction, target):
    if task == "adding":
        return nn.functional.mse_loss(prediction, target).item()
    return (prediction.argmax(1) == target).float().mean().item()


def solved(task, value):
    return value < 0.01 if task == "adding" else value >= 1.0


def run(key):
    arch, task, length, init, seed = key
    torch.set_num_threads(1)
    start = time.time()
    stream = zlib.crc32(f"{arch},{task},{length},{init},{seed}".encode())
    torch.manual_seed(stream)
    rng = np.random.default_rng(stream)
    model = Model(arch, task, length, init, np.random.default_rng(stream + 1))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss() if task == "adding" else nn.CrossEntropyLoss()
    x_val, y_val = validation(task, length)
    solved_at, value = None, math.nan
    for update in range(MAX_UPDATES + 1):
        if update % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                value = metric(task, model(x_val), y_val)
            model.train()
            if update > 0 and solved(task, value):
                solved_at = update
                break
        if update == MAX_UPDATES:
            break
        x, y = batch(task, rng, length, BATCH)
        optimiser.zero_grad()
        loss_fn(model(x), y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimiser.step()
    return dict(zip(FIELDS, (arch, task, length, init, seed, solved_at or "", update, value,
                             round(time.time() - start, 1), torch.__version__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "runs.csv"
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["arch"], r["task"], int(r["length"]), r["init"], int(r["seed"])) for r in csv.DictReader(f)}
    keys = [(a, t, l, i, s) for a, (t, l), i, s in itertools.product(ARCHS, TASKS, INITS, SEEDS)
            if (a, t, l, i, s) not in done]
    keys.sort(key=lambda k: -k[2])
    new = not path.exists()
    with path.open("a", newline="") as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        for row in pool.imap_unordered(run, keys):
            writer.writerow(row)
            f.flush()
            print(*row.values(), flush=True)


if __name__ == "__main__":
    main()
