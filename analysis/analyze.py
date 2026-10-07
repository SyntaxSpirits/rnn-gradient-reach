"""Summaries, gradient-reach statistics and the landmark prediction analysis.

    .venv/bin/python analysis/analyze.py results/main

Prespecified choices
- Reach is the least-squares slope of the log geometric-mean gradient norm against
  lag over lags 1..T/2 (lag 0 is excluded as a boundary point). The e-folding lag
  R = -1/slope is reported only for decaying curves.
- Primary predictor: slope of the loss-independent state curve, evaluation mode, at
  update 100.
- Landmark analysis: runs not solved by update 100; outcome solved by the budget.
  Baseline model: task, log length, architecture, initialisation, regulariser and the
  log validation loss at update 100. Full model: baseline plus the primary predictor.
  L2-penalised logistic regression (C = 1) on standardised features, evaluated with
  10-fold cross-validation grouped by condition.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

KEY = ["arch", "task", "length", "init", "regulariser", "rate", "seed", "hidden", "max_updates"]
CONDITION = ["task", "length", "arch", "init", "regulariser"]
LANDMARK = 100
FOLDS = 10
BOOTSTRAP = 2000


def load(root):
    runs = pd.read_csv(os.path.join(root, "runs.csv"))
    history = pd.read_csv(os.path.join(root, "history.csv"))
    curves = []
    for path in glob.glob(os.path.join(root, "curves", "*.csv")):
        values = os.path.basename(path)[: -len(".csv")].split("_")
        c = pd.read_csv(path)
        key = dict(zip(KEY, values_to_key(values)))
        for k, v in key.items():
            c[k] = v
        curves.append(c)
    curves = pd.concat(curves, ignore_index=True)
    for frame in (runs, history, curves):
        frame["length"] = frame["length"].astype(int)
        frame["seed"] = frame["seed"].astype(int)
    runs["solved"] = runs["solved_at"].notna()
    return runs, history, curves


def values_to_key(values):
    """File names join the key with underscores; regulariser names contain them too."""
    arch, task, length, init = values[:4]
    tail = values[4:]
    rate, seed, hidden, max_updates = tail[-4:]
    regulariser = "_".join(tail[:-4])
    return [arch, task, int(length), init, regulariser, float(rate), int(seed), int(hidden), int(max_updates)]


def fit_slope(lags, log_norms):
    ok = np.isfinite(log_norms)
    x, y = lags[ok], log_norms[ok]
    if len(x) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    r2 = 1 - residual.var() / y.var() if y.var() > 0 else np.nan
    return slope, r2


def reach_table(curves):
    rows = []
    for group, c in curves.groupby(KEY + ["update", "mode"]):
        key, update, mode = group[: len(KEY)], group[-2], group[-1]
        c = c.sort_values("lag")
        length = key[KEY.index("length")]
        window = c[(c["lag"] >= 1) & (c["lag"] <= length // 2)]
        row = dict(zip(KEY, key), update=update, mode=mode)
        for curve in ("state", "loss"):
            slope, r2 = fit_slope(window["lag"].to_numpy(float), window[f"{curve}_log_mean"].to_numpy(float))
            mean = c[f"{curve}_mean"].to_numpy(float)
            tail = c["lag"].to_numpy() >= length // 4
            row[f"{curve}_slope"] = slope
            row[f"{curve}_r2"] = r2
            row[f"{curve}_reach"] = -1 / slope if slope < 0 else np.nan
            row[f"{curve}_tail_share"] = mean[tail].sum() / mean.sum() if mean.sum() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_table(runs):
    def summary(g):
        budget = g["max_updates"].iloc[0]
        time = g["solved_at"].fillna(budget)
        return pd.Series(
            {
                "runs": len(g),
                "solved": int(g["solved"].sum()),
                "median_solved_at": g.loc[g["solved"], "solved_at"].median(),
                "restricted_mean_time": time.mean(),
            }
        )

    return runs.groupby(CONDITION).apply(summary, include_groups=False).reset_index()


def landmark_data(runs, history, reach):
    loss = history[history["update"] == LANDMARK][KEY + ["loss"]].rename(columns={"loss": "landmark_loss"})
    predictor = reach[(reach["update"] == LANDMARK) & (reach["mode"] == "eval")][KEY + ["state_slope"]]
    at_risk = runs[runs["solved_at"].isna() | (runs["solved_at"] > LANDMARK)]
    data = at_risk.merge(loss, on=KEY).merge(predictor, on=KEY)
    data = data[np.isfinite(data["state_slope"])].copy()
    data["log_length"] = np.log(data["length"])
    data["log_landmark_loss"] = np.log(data["landmark_loss"])
    data["condition"] = data[CONDITION].astype(str).agg("|".join, axis=1)
    return data


def design(data, with_predictor):
    categorical = pd.get_dummies(data[["task", "arch", "init", "regulariser"]], drop_first=True).astype(float)
    numeric = data[["log_length", "log_landmark_loss"] + (["state_slope"] if with_predictor else [])]
    return pd.concat([categorical, numeric], axis=1).to_numpy(float)


def cross_validated(data, with_predictor):
    x = design(data, with_predictor)
    y = data["solved"].to_numpy(int)
    groups = data["condition"].to_numpy()
    p = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=min(FOLDS, len(set(groups)))).split(x, y, groups):
        if len(set(y[train])) < 2:
            p[test] = y[train].mean()
            continue
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(x[train]), y[train])
        p[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
    return p


def scores(y, p):
    out = {"log_loss": log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1]), "brier": brier_score_loss(y, p)}
    out["auc"] = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    return out


def landmark_analysis(data, rng):
    y = data["solved"].to_numpy(int)
    base = cross_validated(data, False)
    full = cross_validated(data, True)
    result = {"runs_at_risk": len(y), "events": int(y.sum())}
    result.update({f"baseline_{k}": v for k, v in scores(y, base).items()})
    result.update({f"full_{k}": v for k, v in scores(y, full).items()})
    conditions = data["condition"].to_numpy()
    unique = np.unique(conditions)
    index = {c: np.flatnonzero(conditions == c) for c in unique}
    deltas = []
    for _ in range(BOOTSTRAP):
        pick = np.concatenate([index[c] for c in rng.choice(unique, len(unique))])
        yb = y[pick]
        deltas.append(
            log_loss(yb, np.clip(base[pick], 1e-9, 1 - 1e-9), labels=[0, 1])
            - log_loss(yb, np.clip(full[pick], 1e-9, 1 - 1e-9), labels=[0, 1])
        )
    result["log_loss_improvement"] = result["baseline_log_loss"] - result["full_log_loss"]
    result["log_loss_improvement_ci_low"], result["log_loss_improvement_ci_high"] = np.percentile(deltas, [2.5, 97.5])
    return result


def regulariser_effects(reach, curve="state", update=0):
    """Change in reach slope against the unregularised network with the same seed."""
    r = reach[(reach["mode"] == "eval") & (reach["update"] == update)]
    base = r[r["regulariser"] == "none"][["task", "length", "arch", "init", "seed", f"{curve}_slope"]]
    merged = r.merge(base, on=["task", "length", "arch", "init", "seed"], suffixes=("", "_none"))
    merged["delta"] = merged[f"{curve}_slope"] - merged[f"{curve}_slope_none"]
    return (
        merged[merged["regulariser"] != "none"]
        .groupby(["arch", "init", "regulariser"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def plot_outcomes(outcomes, path):
    tasks = outcomes["task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        o = outcomes[outcomes["task"] == task].copy()
        o["column"] = o["arch"] + " / " + o["init"]
        o["row"] = o["regulariser"]
        table = o.pivot_table(index="row", columns=["column", "length"], values="solved", aggfunc="sum")
        image = ax.imshow(table.to_numpy(float), cmap="viridis", vmin=0, vmax=o["runs"].max(), aspect="auto")
        ax.set_yticks(range(len(table.index)), table.index)
        ax.set_xticks(range(len(table.columns)), [f"{c}\nT={t}" for c, t in table.columns], rotation=90, fontsize=8)
        ax.set_title(f"{task}: runs solved")
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(curves, path, task, length, update=100):
    c = curves[(curves["task"] == task) & (curves["length"] == length) & (curves["mode"] == "eval") & (curves["update"] == update)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (arch, init) in zip(axes.flat, [(a, i) for a in ("lstm", "gru") for i in ("conventional", "chrono")]):
        g = c[(c["arch"] == arch) & (c["init"] == init)]
        for regulariser, r in g.groupby("regulariser"):
            mean = r.groupby("lag")["state_log_mean"].mean()
            ax.plot(mean.index, mean.to_numpy() / np.log(10), label=regulariser, lw=1.2)
        ax.set_title(f"{arch}, {init}")
        ax.set_xlabel("lag k")
        ax.set_ylabel("log10 state-gradient norm")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{task}, T = {length}, update {update}, evaluation mode")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(root):
    out = os.path.join(root, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261007)
    runs, history, curves = load(root)
    reach = reach_table(curves)
    reach.to_csv(os.path.join(out, "reach.csv"), index=False)
    outcomes = outcome_table(runs)
    outcomes.to_csv(os.path.join(out, "outcomes.csv"), index=False)
    plot_outcomes(outcomes, os.path.join(out, "outcomes.png"))

    data = landmark_data(runs, history, reach)
    landmark = landmark_analysis(data, rng) if data["solved"].nunique() == 2 else {"runs_at_risk": len(data)}
    pd.Series(landmark).to_csv(os.path.join(out, "landmark.csv"), header=["value"])

    for update in (0, LANDMARK):
        regulariser_effects(reach, update=update).to_csv(
            os.path.join(out, f"regulariser_effects_update{update}.csv"), index=False
        )
    for task, length in runs[["task", "length"]].drop_duplicates().itertuples(index=False):
        plot_curves(curves, os.path.join(out, f"curves_{task}_{length}.png"), task, length)

    print(outcomes.to_string(index=False))
    print()
    for k, v in landmark.items():
        print(f"{k:32} {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/main")
