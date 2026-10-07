import glob
import os

import numpy as np
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


def load(root):
    runs = pd.read_csv(os.path.join(root, "runs.csv"))
    history = pd.read_csv(os.path.join(root, "history.csv"))
    curves = []
    for path in sorted(glob.glob(os.path.join(root, "curves", "*.csv"))):
        frame = pd.read_csv(path)
        for column, value in key_from_file_name(os.path.basename(path)[: -len(".csv")]).items():
            frame[column] = value
        curves.append(frame)
    curves = pd.concat(curves, ignore_index=True)
    runs["solved"] = runs["solved_at"].notna()
    runs["family"] = runs["regulariser"].map(FAMILY)
    return runs, history, curves


def condition_label(frame):
    return frame[CONDITION].astype(str).agg("|".join, axis=1)


def censored_time(runs):
    return runs["solved_at"].fillna(runs["max_updates"]).astype(float)


def log_or_nan(values):
    values = np.asarray(values, float)
    return np.where(values > 0, np.log(np.where(values > 0, values, 1.0)), np.nan)
