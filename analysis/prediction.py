"""Does early gradient reach help predict whether a run will reach the validation
criterion?

Landmark design: runs not solved by update 100 are followed to the end of their budget.
A gradient-free baseline uses what is known at the landmark: task, length,
architecture, initialisation, regulariser and the validation loss at update 100. The
full model adds one gradient summary measured at update 100.

Two baselines are compared. The additive baseline (prespecified before the main grid)
enters every factor as a main effect. The flexible baseline (added after review) gives
every task × architecture × initialisation stratum its own intercept and its own
effects of log length and log landmark loss, plus regulariser main effects.

Models are L2-penalised logistic regressions (C = 1) on standardised features, which
keeps coefficients finite under the separation this grid produces. Predictions are made
out of sample, with all runs of a condition held out together (primary), or a whole
seed, regulariser family or task length held out. Intervals for the log-loss gain
resample conditions, or conditions and seeds crossed, holding the out-of-sample
predictions fixed; they do not include refitting variability.
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
    data["stratum"] = data["task"] + "|" + data["arch"] + "|" + data["init"]
    data["task_length"] = data["task"] + "_" + data["length"].astype(str)
    return data.reset_index(drop=True)


def features(data, predictor, flexible):
    if flexible:
        strata = pd.get_dummies(data["stratum"]).astype(float)
        parts = [strata, pd.get_dummies(data["regulariser"], drop_first=True).astype(float)]
        for column in ("log_length", "log_landmark_loss"):
            parts.append(strata.mul(data[column], axis=0).add_suffix(f"*{column}"))
    else:
        categorical = ["task", "arch", "init", "regulariser"]
        parts = [
            pd.get_dummies(data[categorical], drop_first=True).astype(float),
            data[["log_length", "log_landmark_loss"]],
        ]
    if predictor:
        parts.append(data[[predictor]])
    return pd.concat(parts, axis=1).to_numpy(float)


def fit_predict(x_train, y_train, x_test):
    if len(set(y_train)) < 2:
        return np.full(len(x_test), y_train.mean())
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=1.0, max_iter=10_000)
    model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def splits(groups, folds):
    if folds:
        return GroupKFold(n_splits=folds).split(np.zeros(len(groups)), groups=groups)
    return ((np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups))


def out_of_sample(data, predictor, groups, folds, flexible, outcome):
    x = features(data, predictor, flexible)
    y = data[outcome].to_numpy(int)
    p = np.zeros(len(y))
    for train, test in splits(data[groups].to_numpy(), folds):
        p[test] = fit_predict(x[train], y[train], x[test])
    return p


def scores(y, p, weights=None):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "log_loss": log_loss(y, p, labels=[0, 1], sample_weight=weights),
        "brier": brier_score_loss(y, p, sample_weight=weights),
        "auc": roc_auc_score(y, p, sample_weight=weights) if len(set(y)) == 2 else np.nan,
    }


def gain(y, base, full, weights=None):
    return scores(y, base, weights)["log_loss"] - scores(y, full, weights)["log_loss"]


def condition_interval(frame, y, base, full, rng, resamples):
    conditions = frame["condition"].to_numpy()
    members = {c: np.flatnonzero(conditions == c) for c in np.unique(conditions)}
    names = np.array(list(members))
    draws = []
    for _ in range(resamples):
        pick = np.concatenate([members[c] for c in rng.choice(names, len(names))])
        draws.append(gain(y[pick], base[pick], full[pick]))
    return np.percentile(draws, [2.5, 97.5])


def crossed_interval(frame, y, base, full, rng, resamples):
    """Resamples conditions and seeds independently; a run enters as often as its
    condition and its seed were drawn."""
    conditions = frame["condition"].to_numpy()
    seeds = frame["seed"].to_numpy()
    names, seed_ids = np.unique(conditions), np.unique(seeds)
    condition_index = np.searchsorted(names, conditions)
    seed_index = np.searchsorted(seed_ids, seeds)
    draws = []
    for _ in range(resamples):
        c = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        s = np.bincount(rng.integers(0, len(seed_ids), len(seed_ids)), minlength=len(seed_ids))
        weights = c[condition_index] * s[seed_index]
        keep = weights > 0
        if len(set(y[keep])) == 2:
            draws.append(gain(y[keep], base[keep], full[keep], weights[keep]))
    return np.percentile(draws, [2.5, 97.5])


def compare(
    data,
    predictor,
    groups,
    rng,
    folds=None,
    flexible=False,
    outcome="solved",
    crossed=False,
    resamples=2000,
):
    """Gradient-free baseline against baseline plus `predictor`, on the same runs."""
    frame = data[data[predictor].notna()].reset_index(drop=True)
    y = frame[outcome].to_numpy(int)
    base = out_of_sample(frame, None, groups, folds, flexible, outcome)
    full = out_of_sample(frame, predictor, groups, folds, flexible, outcome)
    result = {"runs": len(y), "events": int(y.sum())}
    result.update({f"baseline_{k}": v for k, v in scores(y, base).items()})
    result.update({f"full_{k}": v for k, v in scores(y, full).items()})
    result["gain"] = gain(y, base, full)
    result["gain_low"], result["gain_high"] = condition_interval(frame, y, base, full, rng, resamples)
    if crossed:
        result["crossed_low"], result["crossed_high"] = crossed_interval(frame, y, base, full, rng, resamples)
    if not folds:
        per_group = [
            gain(y[m], base[m], full[m])
            for m in (frame[groups].to_numpy() == g for g in frame[groups].unique())
        ]
        result["groups"] = len(per_group)
        result["groups_with_gain"] = int(np.sum(np.array(per_group) > 0))
        result["worst_group_gain"] = min(per_group)
    return result, frame.assign(baseline=base, full=full)


def within_condition_pairs(data, predictor):
    """Every pair of a run that reached the criterion and one that did not, within the
    same condition, scored 1 if the first has the larger predictor, 1/2 for a tie."""
    rows = []
    for condition, g in data.groupby(CONDITION):
        solved, failed = g[g["solved"]], g[~g["solved"]]
        for _, a in solved.iterrows():
            for _, b in failed.iterrows():
                score = float(a[predictor] > b[predictor]) + 0.5 * float(a[predictor] == b[predictor])
                rows.append(("|".join(map(str, condition)), a["seed"], b["seed"], score))
    return pd.DataFrame(rows, columns=["condition", "seed_solved", "seed_failed", "score"])


def pair_intervals(pairs, values, rng, resamples):
    """95 % intervals for the weighted mean of `values` over pairs, resampling
    conditions, seeds (a pair counts once per draw of each of its two seeds) and both."""
    conditions = pairs["condition"].to_numpy()
    names = np.unique(conditions)
    condition_index = np.searchsorted(names, conditions)
    seeds = np.unique(np.concatenate([pairs["seed_solved"], pairs["seed_failed"]]))
    first = np.searchsorted(seeds, pairs["seed_solved"].to_numpy())
    second = np.searchsorted(seeds, pairs["seed_failed"].to_numpy())
    draws = {"condition": [], "seed": [], "crossed": []}
    for _ in range(resamples):
        c = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[condition_index]
        w = np.bincount(rng.integers(0, len(seeds), len(seeds)), minlength=len(seeds))
        s = w[first] * w[second]
        for name, weight in (("condition", c), ("seed", s), ("crossed", c * s)):
            if weight.sum() > 0:
                draws[name].append(np.average(values, weights=weight))
    return {name: np.percentile(d, [2.5, 97.5]) for name, d in draws.items()}


def within_condition_auc(data, predictor, rng, resamples=2000):
    """Concordance between `predictor` and reaching the criterion among runs of the same
    condition (higher predictor, more likely solved), over conditions with both
    outcomes. Intervals resample conditions (prespecified), seeds, and both crossed."""
    pairs = within_condition_pairs(data, predictor)
    if pairs.empty:
        return {"conditions": 0}
    intervals = pair_intervals(pairs, pairs["score"].to_numpy(), rng, resamples)
    runs = data[data.groupby(CONDITION)["solved"].transform(lambda v: 0 < v.sum() < len(v))]
    result = {
        "conditions": pairs["condition"].nunique(),
        "runs": len(runs),
        "pairs": len(pairs),
        "auc": pairs["score"].mean(),
    }
    for name, (low, high) in intervals.items():
        result[f"{name}_low"], result[f"{name}_high"] = low, high
    return result


def paired_difference(data, first, second, rng, resamples=2000):
    """Difference in within-condition concordance between two predictors on the same pairs."""
    a = within_condition_pairs(data, first)
    b = within_condition_pairs(data, second)
    difference = a["score"].to_numpy() - b["score"].to_numpy()
    intervals = pair_intervals(a, difference, rng, resamples)
    result = {"difference": difference.mean()}
    for name, (low, high) in intervals.items():
        result[f"{name}_low"], result[f"{name}_high"] = low, high
    return result


def per_condition_auc(data, predictor):
    pairs = within_condition_pairs(data, predictor)
    table = pairs.groupby("condition")["score"].agg(auc="mean", pairs="size").reset_index()
    return table
