"""Does early gradient reach improve the prediction of whether a run will solve its task?

Landmark design: runs not solved by update 100 are followed to the end of their budget.
The baseline model uses what is known at the landmark without probing gradients: task,
log length, architecture, initialisation, regulariser and the log validation loss at
update 100. The full model adds one gradient summary measured at update 100.

Models are L2-penalised logistic regressions (C = 1) on standardised features, which
keeps coefficients finite under the separation this grid produces. Predictions are
made out of sample, with all runs of a condition held out together (primary), or a
whole seed, regulariser family or task length held out (secondary). Uncertainty in the
log-loss improvement comes from resampling conditions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .data import CONDITION, KEY, condition_label

LANDMARK = 100
PRIMARY = "state_slope"
CATEGORICAL = ["task", "arch", "init", "regulariser"]
BASELINE_NUMERIC = ["log_length", "log_landmark_loss"]


def landmark_frame(runs, history, reach, mode="eval"):
    loss = history[history["update"] == LANDMARK][KEY + ["loss"]]
    loss = loss.rename(columns={"loss": "landmark_loss"})
    summaries = reach[(reach["update"] == LANDMARK) & (reach["mode"] == mode)]
    summaries = summaries.drop(columns=["update", "mode"])
    at_risk = runs[runs["solved_at"].isna() | (runs["solved_at"] > LANDMARK)]
    data = at_risk.merge(loss, on=KEY).merge(summaries, on=KEY)
    data["log_length"] = np.log(data["length"])
    data["log_landmark_loss"] = np.log(data["landmark_loss"])
    data["condition"] = condition_label(data)
    return data.reset_index(drop=True)


def features(data, predictor):
    columns = BASELINE_NUMERIC + ([predictor] if predictor else [])
    dummies = pd.get_dummies(data[CATEGORICAL], drop_first=True).astype(float)
    return pd.concat([dummies, data[columns]], axis=1).to_numpy(float)


def fit_predict(x_train, y_train, x_test):
    if len(set(y_train)) < 2:
        return np.full(len(x_test), y_train.mean())
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=1.0, max_iter=10_000)
    model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def out_of_sample(data, predictor, groups, folds=None):
    """Predictions for every run from a model fitted without its group."""
    usable = data[predictor].notna() if predictor else np.ones(len(data), bool)
    data = data[usable].reset_index(drop=True)
    x = features(data, predictor)
    y = data["solved"].to_numpy(int)
    g = data[groups].to_numpy() if isinstance(groups, str) else groups[usable]
    splitter = GroupKFold(n_splits=folds) if folds else None
    splits = (
        splitter.split(x, y, g)
        if splitter
        else ((np.flatnonzero(g != v), np.flatnonzero(g == v)) for v in np.unique(g))
    )
    p = np.zeros(len(y))
    for train, test in splits:
        p[test] = fit_predict(x[train], y[train], x[test])
    return data, p


def scores(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p) if len(set(y)) == 2 else np.nan,
    }


def compare(data, predictor, groups, rng, folds=None, resamples=2000):
    """Baseline against baseline plus `predictor` on the same runs."""
    usable = data[data[predictor].notna()].reset_index(drop=True)
    _, base = out_of_sample(usable, None, groups, folds)
    _, full = out_of_sample(usable, predictor, groups, folds)
    y = usable["solved"].to_numpy(int)
    result = {"runs": len(y), "events": int(y.sum())}
    result.update({f"baseline_{k}": v for k, v in scores(y, base).items()})
    result.update({f"full_{k}": v for k, v in scores(y, full).items()})
    result["improvement"] = result["baseline_log_loss"] - result["full_log_loss"]

    conditions = usable["condition"].to_numpy()
    members = {c: np.flatnonzero(conditions == c) for c in np.unique(conditions)}
    names = np.array(list(members))
    draws = []
    for _ in range(resamples):
        pick = np.concatenate([members[c] for c in rng.choice(names, len(names))])
        draws.append(scores(y[pick], base[pick])["log_loss"] - scores(y[pick], full[pick])["log_loss"])
    result["improvement_low"], result["improvement_high"] = np.percentile(draws, [2.5, 97.5])
    return result, usable.assign(baseline=base, full=full)


def within_condition_auc(data, predictor, rng, resamples=2000):
    """Concordance between the predictor and success among runs of the same condition,
    using only conditions with both outcomes."""
    pairs = []
    for _, g in data.groupby(CONDITION):
        solved = g.loc[g["solved"], predictor].to_numpy()
        failed = g.loc[~g["solved"], predictor].to_numpy()
        if len(solved) and len(failed):
            pairs.append((solved[:, None] > failed[None, :]).astype(float) + 0.5 * (solved[:, None] == failed[None, :]))
    if not pairs:
        return {"conditions": 0}
    estimate = np.concatenate([p.ravel() for p in pairs]).mean()
    draws = [
        np.concatenate([pairs[i].ravel() for i in rng.integers(0, len(pairs), len(pairs))]).mean()
        for _ in range(resamples)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"conditions": len(pairs), "auc": estimate, "auc_low": low, "auc_high": high}
