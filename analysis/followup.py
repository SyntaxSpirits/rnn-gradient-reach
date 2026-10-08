"""Analyses of the follow-up plans, specified in README.md before the runs.

- more_seeds: confirmatory test, on seeds 5–19 only, that within mixed conditions the
  update-100 state slope ranks runs reaching the validation criterion above runs that
  do not, while the update-100 validation loss does not.
- long_budget: how many of the conditions below the 5000-update ceiling reach the
  criterion within 20 000 updates.
- chrono_mismatch: success and time to solve with chrono horizon T/4 and 4T against
  the matched horizon T and conventional initialisation.
"""

import os

import pandas as pd

from . import prediction, reach
from .data import CONDITION, censored_time, load, load_runs


def confirmatory(root, rng):
    runs, history, curves = load(os.path.join(root, "more_seeds"))
    summaries = reach.summarise(curves)
    data = prediction.landmark_frame(runs, history, summaries)
    data["negative_log_landmark_loss"] = -data["log_landmark_loss"]
    rows = {}
    for predictor in (prediction.PRIMARY, "negative_log_landmark_loss"):
        rows[predictor] = prediction.within_condition_auc(data, predictor, rng)
    solved = data.groupby(CONDITION)["solved"].agg(["sum", "size"]).reset_index()
    return pd.DataFrame(rows).T, solved, summaries


def long_budget(root):
    long = load_runs(os.path.join(root, "long_budget"))
    main = load_runs(os.path.join(root, "main"))
    long["solved_by_5000"] = long["solved_at"].le(5000)
    table = long.groupby(CONDITION).agg(
        runs=("solved", "size"),
        solved_by_5000=("solved_by_5000", "sum"),
        solved_by_20000=("solved", "sum"),
        median_solved_at=("solved_at", "median"),
    )
    reference = main.groupby(CONDITION)["solved"].sum().rename("main_grid_solved")
    return table.join(reference).reset_index()


def chrono_mismatch(root):
    mismatch = load_runs(os.path.join(root, "chrono_mismatch"))
    main = load_runs(os.path.join(root, "main"))
    runs = pd.concat([mismatch, main[main["regulariser"] == "none"]], ignore_index=True)
    runs["time"] = censored_time(runs)
    table = runs.groupby(["task", "length", "arch", "init"]).agg(
        runs=("solved", "size"),
        solved=("solved", "sum"),
        restricted_mean_time=("time", "mean"),
    )
    order = {"conventional": 0, "chrono-short": 1, "chrono": 2, "chrono-long": 3}
    table = table.reset_index().assign(order=lambda t: t["init"].map(order))
    return table.sort_values(["task", "length", "arch", "order"]).drop(columns="order")


def available(root, plan):
    return os.path.exists(os.path.join(root, plan, "runs.csv"))


def run_all(root, out, rng):
    if available(root, "more_seeds"):
        auc, solved, _ = confirmatory(root, rng)
        auc.to_csv(os.path.join(out, "confirmatory_within_condition.csv"))
        solved.to_csv(os.path.join(out, "more_seeds_solved.csv"), index=False)
        print("confirmatory within-condition concordance, seeds 5-19")
        print(auc.round(3).to_string())
    if available(root, "long_budget"):
        table = long_budget(root)
        table.to_csv(os.path.join(out, "long_budget.csv"), index=False)
        print(table.to_string(index=False))
    if available(root, "chrono_mismatch"):
        table = chrono_mismatch(root)
        table.to_csv(os.path.join(out, "chrono_mismatch.csv"), index=False)
        print(table.round(0).to_string(index=False))
