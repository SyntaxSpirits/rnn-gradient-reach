"""Tables and figures for the main grid and, when present, the follow-up plans.

    .venv/bin/python -m analysis results

Outputs go to results/analysis.
"""

import os
import sys

import numpy as np
import pandas as pd

from . import figures, followup, prediction, reach, survival
from .data import CONDITION, censored_time, load, load_probes

SEED = 20261007
ALTERNATIVE_PREDICTORS = ["state_slope_quarter", "state_slope_full", "state_tail_share", "loss_slope"]
WITHIN_PREDICTORS = [
    "state_slope",
    "state_slope_full",
    "state_tail_share",
    "loss_slope",
    "negative_log_landmark_loss",
]


def outcome_table(runs):
    runs = runs.assign(time=censored_time(runs))
    table = runs.groupby(CONDITION).agg(
        runs=("solved", "size"),
        solved=("solved", "sum"),
        solved_on_test=("solved_on_test", "sum"),
        median_solved_at=("solved_at", "median"),
        restricted_mean_time=("time", "mean"),
    )
    return table.reset_index()


def task_probe_table(root):
    """Derivatives with respect to the task-relevant inputs at the landmark, evaluation mode."""
    probes = load_probes(root)
    p = probes[(probes["update"] == prediction.LANDMARK) & (probes["mode"] == "eval")]
    names = {
        "adding": ("first_marked", "second_marked", "unmarked_abs"),
        "recall": ("symbol_logit", None, None),
    }
    frames = []
    for task, columns in names.items():
        t = p[p["task"] == task].rename(columns={f"task_{i + 1}": c for i, c in enumerate(columns) if c})
        keep = [c for c in columns if c]
        frames.append(t.groupby(["task", "arch", "init", "regulariser"])[keep].mean().reset_index())
    return pd.concat(frames, ignore_index=True)


def prediction_tables(runs, history, summaries, rng):
    data = prediction.landmark_frame(runs, history, summaries)
    training = prediction.landmark_frame(runs, history, summaries, mode="train")
    primary, predictions = prediction.compare(
        data, prediction.PRIMARY, "condition", rng, folds=10, crossed=True
    )
    rows = {"additive baseline, held-out conditions": primary}
    rows["flexible baseline, held-out conditions"] = prediction.compare(
        data, prediction.PRIMARY, "condition", rng, folds=10, flexible=True, crossed=True
    )[0]
    for name, groups in (("seed", "seed"), ("regulariser family", "family"), ("task length", "task_length")):
        rows[f"additive baseline, held-out {name}"] = prediction.compare(data, prediction.PRIMARY, groups, rng)[0]
        rows[f"flexible baseline, held-out {name}"] = prediction.compare(
            data, prediction.PRIMARY, groups, rng, flexible=True
        )[0]
    rows["additive baseline, test endpoint"] = prediction.compare(
        data, prediction.PRIMARY, "condition", rng, folds=10, outcome="solved_on_test"
    )[0]
    fresh = data[data["seed"] >= 2].reset_index(drop=True)
    rows["additive baseline, seeds 2-4"] = prediction.compare(fresh, prediction.PRIMARY, "condition", rng, folds=10)[0]
    rows["flexible baseline, seeds 2-4"] = prediction.compare(
        fresh, prediction.PRIMARY, "condition", rng, folds=10, flexible=True
    )[0]
    rows["additive baseline, training-mode predictor"] = prediction.compare(
        training, prediction.PRIMARY, "condition", rng, folds=10
    )[0]
    for predictor in ALTERNATIVE_PREDICTORS:
        rows[f"additive baseline, predictor {predictor}"] = prediction.compare(
            data, predictor, "condition", rng, folds=10
        )[0]

    data["negative_log_landmark_loss"] = -data["log_landmark_loss"]
    within = {p: prediction.within_condition_auc(data, p, rng) for p in WITHIN_PREDICTORS}
    return pd.DataFrame(rows).T, predictions, pd.DataFrame(within).T, data


def main(root):
    out = os.path.join(root, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    main_dir = os.path.join(root, "main")
    runs, history, curves = load(main_dir)

    summaries = reach.summarise(curves)
    summaries.to_csv(os.path.join(out, "reach.csv"), index=False)
    outcomes = outcome_table(runs)
    outcomes.to_csv(os.path.join(out, "outcomes.csv"), index=False)
    pd.concat(
        [reach.regulariser_effects(summaries, u, m) for u in (0, prediction.LANDMARK) for m in ("eval", "train")]
    ).to_csv(os.path.join(out, "regulariser_effects.csv"), index=False)
    pd.concat([reach.fit_quality(summaries, u) for u in (0, prediction.LANDMARK)]).to_csv(
        os.path.join(out, "fit_quality.csv"), index=False
    )
    task_probe_table(main_dir).to_csv(os.path.join(out, "task_probes.csv"), index=False)

    table, predictions, within, data = prediction_tables(runs, history, summaries, rng)
    table.to_csv(os.path.join(out, "prediction.csv"))
    predictions.to_csv(os.path.join(out, "predictions.csv"), index=False)
    within.to_csv(os.path.join(out, "within_condition.csv"))
    hazard = survival.hazard_effect(data, prediction.PRIMARY, rng)
    pd.Series(hazard).to_csv(os.path.join(out, "hazard.csv"), header=["value"])

    figures.outcomes(outcomes, os.path.join(out, "outcomes.png"))
    figures.slope_vs_time(data, os.path.join(out, "slope_vs_time.png"))
    figures.prediction(predictions, os.path.join(out, "prediction.png"))
    for task, length in runs[["task", "length"]].drop_duplicates().itertuples(index=False):
        for update in (0, prediction.LANDMARK):
            figures.curves(curves, os.path.join(out, f"curves_{task}_{length}_update{update}.png"), task, length, update)

    columns = ["runs", "events", "baseline_log_loss", "full_log_loss", "gain", "gain_low", "gain_high"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table[columns + ["crossed_low", "crossed_high", "groups_with_gain", "groups"]].round(3))
        print(within[["conditions", "runs", "auc", "auc_low", "auc_high"]].round(3))
        print(pd.Series(hazard).round(3).to_string())
    followup.run_all(root, out, rng)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
