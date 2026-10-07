"""Summaries of gradient-norm curves.

The primary summary is the least-squares slope of the log geometric-mean norm against
lag over lags 1..T/2; lag 0 is a boundary point and is excluded. R = -1/slope (the
e-folding lag) is reported only for decaying curves.
"""

import numpy as np
import pandas as pd

from .data import KEY

WINDOWS = {
    "half": lambda length: (1, length // 2),
    "quarter": lambda length: (1, length // 4),
    "full": lambda length: (1, length - 1),
}


def fit(lags, log_norms):
    ok = np.isfinite(log_norms)
    x, y = lags[ok], log_norms[ok]
    if len(x) < 3 or y.var() == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return slope, 1 - residual.var() / y.var()


def summarise(curves):
    rows = []
    for group, c in curves.groupby(KEY + ["update", "mode"], sort=False):
        c = c.sort_values("lag")
        length = group[KEY.index("length")]
        lag = c["lag"].to_numpy(float)
        row = dict(zip(KEY + ["update", "mode"], group))
        for curve in ("state", "loss"):
            log_norm = c[f"{curve}_log_mean"].to_numpy(float)
            for window, bounds in WINDOWS.items():
                low, high = bounds(length)
                inside = (lag >= low) & (lag <= high)
                slope, r2 = fit(lag[inside], log_norm[inside])
                suffix = "" if window == "half" else f"_{window}"
                row[f"{curve}_slope{suffix}"] = slope
                row[f"{curve}_r2{suffix}"] = r2
            row[f"{curve}_reach"] = -1 / row[f"{curve}_slope"] if row[f"{curve}_slope"] < 0 else np.nan
            mean = c[f"{curve}_mean"].to_numpy(float)
            total = mean.sum()
            row[f"{curve}_tail_share"] = mean[lag >= length // 4].sum() / total if total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def regulariser_effects(reach, update, mode, curve="state"):
    """Mean change in slope against the unregularised network with the same seed,
    architecture, initialisation, task and length."""
    r = reach[(reach["update"] == update) & (reach["mode"] == mode)]
    pair = ["task", "length", "arch", "init", "seed"]
    base = r[r["regulariser"] == "none"][pair + [f"{curve}_slope"]]
    merged = r[r["regulariser"] != "none"].merge(base, on=pair, suffixes=("", "_none"))
    merged["change"] = merged[f"{curve}_slope"] - merged[f"{curve}_slope_none"]
    summary = merged.groupby(["arch", "init", "regulariser"])["change"].agg(["mean", "std", "count"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    return summary.reset_index().assign(update=update, mode=mode)
