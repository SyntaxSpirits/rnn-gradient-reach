import glob
import os

import pandas as pd

KEY = ["arch", "task", "length", "init", "regulariser", "rate", "seed", "hidden", "max_updates"]
CONDITION = ["task", "length", "arch", "init", "regulariser"]
FAMILY = {
    "none": "none",
    "recurrent_dropout": "dropout",
    "variational_dropout": "dropout",
    "update_dropout": "dropout",
    "zoneout_cell": "zoneout",
    "zoneout_hidden": "zoneout",
    "zoneout_both": "zoneout",
}
TEST_THRESHOLD = {"adding": lambda mse: mse < 0.01, "recall": lambda accuracy: accuracy >= 1.0}


def key_from_file_name(name):
    """Curve files are named by the run key joined with underscores; regulariser names
    contain underscores themselves, so the fields are recovered from both ends."""
    fields = name.split("_")
    arch, task, length, init = fields[:4]
    rate, seed, hidden, max_updates = fields[-4:]
    regulariser = "_".join(fields[4:-4])
    return dict(
        arch=arch,
        task=task,
        length=int(length),
        init=init,
        regulariser=regulariser,
        rate=float(rate),
        seed=int(seed),
        hidden=int(hidden),
        max_updates=int(max_updates),
    )


def load_runs(root):
    runs = pd.read_csv(os.path.join(root, "runs.csv"))
    runs["solved"] = runs["solved_at"].notna()
    passes_test = [TEST_THRESHOLD[t](m) for t, m in zip(runs["task"], runs["test_metric"])]
    runs["solved_on_test"] = runs["solved"] & pd.Series(passes_test, index=runs.index)
    runs["family"] = runs["regulariser"].map(FAMILY)
    return runs


def load(root):
    """Runs, validation history and gradient curves of one results directory. A run
    interrupted and rerun on resume may repeat identical history rows; they are dropped."""
    runs = load_runs(root)
    history = pd.read_csv(os.path.join(root, "history.csv")).drop_duplicates()
    curves = []
    for path in sorted(glob.glob(os.path.join(root, "curves", "*.csv"))):
        frame = pd.read_csv(path)
        for column, value in key_from_file_name(os.path.basename(path)[: -len(".csv")]).items():
            frame[column] = value
        curves.append(frame)
    curves = pd.concat(curves, ignore_index=True)
    return runs, history, curves


def load_probes(root):
    return pd.read_csv(os.path.join(root, "probes.csv")).drop_duplicates()


def condition_label(frame):
    return frame[CONDITION].astype(str).agg("|".join, axis=1)


def censored_time(runs):
    return runs["solved_at"].fillna(runs["max_updates"]).astype(float)
