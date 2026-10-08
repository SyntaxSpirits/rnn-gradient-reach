"""Time to solve after the landmark, as a discrete-time hazard model.

Validation happens every 100 updates, so the event is observed in intervals. Each run at
risk contributes one row per interval from update 200 until it solves the task or its
budget ends (right-censored). The hazard of solving in an interval is modelled by an
L2-penalised logistic regression on the log of the interval's update, the landmark
baseline features and the gradient summary. The coefficient is per across-run standard
deviation of the summary, an adjusted association rather than a causal effect, with an
interval from refitting on resampled conditions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .prediction import LANDMARK, features

STEP = 100


def person_period(data):
    rows = []
    for i, run in data.iterrows():
        last = run["solved_at"] if run["solved"] else run["max_updates"]
        for update in range(LANDMARK + STEP, int(last) + 1, STEP):
            rows.append((i, update, int(run["solved"] and update == run["solved_at"])))
    periods = pd.DataFrame(rows, columns=["run", "update", "event"])
    return periods


def hazard_coefficient(data, predictor, periods):
    """Coefficient of the predictor, standardised across runs, in the interval hazard."""
    x_runs = features(data, predictor, flexible=False)
    x_runs = (x_runs - x_runs.mean(axis=0)) / np.where(x_runs.std(axis=0) > 0, x_runs.std(axis=0), 1.0)
    log_update = np.log(periods["update"].to_numpy(float))
    x = np.column_stack([x_runs[periods["run"]], (log_update - log_update.mean()) / log_update.std()])
    model = LogisticRegression(C=1.0, max_iter=10_000)
    model.fit(x, periods["event"].to_numpy(int))
    return model.coef_[0][x_runs.shape[1] - 1]


def hazard_effect(data, predictor, rng, resamples=500):
    data = data[data[predictor].notna()].reset_index(drop=True)
    periods = person_period(data)
    estimate = hazard_coefficient(data, predictor, periods)
    members = {c: np.flatnonzero(data["condition"].to_numpy() == c) for c in data["condition"].unique()}
    names = np.array(list(members))
    draws = []
    for _ in range(resamples):
        pick = np.concatenate([members[c] for c in rng.choice(names, len(names))])
        sample = data.iloc[pick].reset_index(drop=True)
        draws.append(hazard_coefficient(sample, predictor, person_period(sample)))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "runs": len(data),
        "intervals": len(periods),
        "events": int(periods["event"].sum()),
        "coefficient": estimate,
        "coefficient_low": low,
        "coefficient_high": high,
    }
