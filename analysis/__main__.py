"""Tables and figures for one results directory.

    .venv/bin/python -m analysis results/main
"""

import os
import sys

import numpy as np
import pandas as pd

from . import figures, prediction, reach, survival
from .data import CONDITION, censored_time, load

SEED = 20261007
SECONDARY_PREDICTORS = {
    "state_slope_quarter": "eval",
    "state_slope_full": "eval",
    "state_tail_share": "eval",
    "loss_slope": "eval",
}


def outcome_table(runs):
    runs = runs.assign(time=censored_time(runs))
    table = runs.groupby(CONDITION).agg(
        runs=("solved", "size"),
        solved=("solved", "sum"),
        median_solved_at=("solved_at", "median"),
        restricted_mean_time=("time", "mean"),
    )
    return table.reset_index()


def main(root):
    out = os.path.join(root, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    runs, history, curves = load(root)

    summaries = reach.summarise(curves)
    summaries.to_csv(os.path.join(out, "reach.csv"), index=False)
    outcomes = outcome_table(runs)
    outcomes.to_csv(os.path.join(out, "outcomes.csv"), index=False)

    effects = pd.concat(
        [
            reach.regulariser_effects(summaries, update, mode)
            for update in (0, prediction.LANDMARK)
            for mode in ("eval", "train")
        ]
    )
    effects.to_csv(os.path.join(out, "regulariser_effects.csv"), index=False)

    data = prediction.landmark_frame(runs, history, summaries)
    data["task_length"] = data["task"] + "_" + data["length"].astype(str)
    primary, predictions = prediction.compare(data, prediction.PRIMARY, "condition", rng, folds=10)
    rows = {"primary: held-out conditions": primary}
    for name, groups in (("seed", "seed"), ("regulariser family", "family"), ("task length", "task_length")):
        rows[f"held-out {name}"] = prediction.compare(data, prediction.PRIMARY, groups, rng)[0]
    training_mode = prediction.landmark_frame(runs, history, summaries, mode="train")
    rows["predictor in training mode"] = prediction.compare(training_mode, prediction.PRIMARY, "condition", rng, folds=10)[0]
    for predictor in SECONDARY_PREDICTORS:
        rows[f"predictor {predictor}"] = prediction.compare(data, predictor, "condition", rng, folds=10)[0]
    pd.DataFrame(rows).T.to_csv(os.path.join(out, "prediction.csv"))
    predictions.to_csv(os.path.join(out, "predictions.csv"), index=False)

    within = prediction.within_condition_auc(data, prediction.PRIMARY, rng)
    hazard = survival.hazard_effect(data, prediction.PRIMARY, rng)
    pd.Series({**{f"within_{k}": v for k, v in within.items()}, **{f"hazard_{k}": v for k, v in hazard.items()}}).to_csv(
        os.path.join(out, "secondary.csv"), header=["value"]
    )

    figures.outcomes(outcomes, os.path.join(out, "outcomes.png"))
    figures.slope_vs_time(data, os.path.join(out, "slope_vs_time.png"))
    figures.prediction(predictions, os.path.join(out, "prediction.png"))
    for task, length in runs[["task", "length"]].drop_duplicates().itertuples(index=False):
        for update in (0, prediction.LANDMARK):
            figures.curves(curves, os.path.join(out, f"curves_{task}_{length}_update{update}.png"), task, length, update)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).T[["runs", "events", "baseline_log_loss", "full_log_loss", "improvement", "improvement_low", "improvement_high", "baseline_auc", "full_auc"]].round(3))
        print(pd.Series(within).round(3).to_string())
        print(pd.Series(hazard).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/main")
