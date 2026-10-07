import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

REGULARISERS = [
    "none",
    "recurrent_dropout",
    "variational_dropout",
    "update_dropout",
    "zoneout_cell",
    "zoneout_hidden",
    "zoneout_both",
]
PANELS = [(a, i) for a in ("lstm", "gru") for i in ("conventional", "chrono")]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "savefig.bbox": "tight"})


def outcomes(table, path):
    tasks = list(table["task"].unique())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 3.6 * len(tasks)), squeeze=False)
    for ax, task in zip(axes[:, 0], tasks):
        t = table[table["task"] == task]
        lengths = sorted(t["length"].unique())
        columns = [(a, i, l) for a, i in PANELS for l in lengths]
        grid = np.full((len(REGULARISERS), len(columns)), np.nan)
        for _, row in t.iterrows():
            r = REGULARISERS.index(row["regulariser"])
            c = columns.index((row["arch"], row["init"], row["length"]))
            grid[r, c] = row["solved"] / row["runs"]
            ax.text(
                c,
                r,
                f"{int(row['solved'])}/{int(row['runs'])}\n{int(round(row['restricted_mean_time'], -2))}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if grid[r, c] > 0.6 else "black",
            )
        ax.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(REGULARISERS)), REGULARISERS)
        ax.set_xticks(
            range(len(columns)),
            [f"{a.upper()}\n{'conv.' if i == 'conventional' else 'chrono'}\nT={l}" for a, i, l in columns],
            fontsize=7,
        )
        for boundary in range(len(lengths), len(columns), len(lengths)):
            ax.axvline(boundary - 0.5, color="white", lw=2)
        ax.set_title(f"{task}: runs solved and restricted mean updates to solve (budget 5000)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def curves(frame, path, task, length, update, mode="eval"):
    c = frame[
        (frame["task"] == task)
        & (frame["length"] == length)
        & (frame["update"] == update)
        & (frame["mode"] == mode)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
    for ax, (arch, init) in zip(axes.flat, PANELS):
        g = c[(c["arch"] == arch) & (c["init"] == init)]
        for regulariser in REGULARISERS:
            r = g[g["regulariser"] == regulariser]
            if r.empty:
                continue
            mean = r.groupby("lag")["state_log_mean"].mean() / np.log(10)
            ax.plot(mean.index, mean.to_numpy(), label=regulariser, lw=1.1)
        ax.set_title(f"{arch.upper()}, {init} initialisation")
        ax.set_xlabel("lag k")
        ax.set_ylabel("log10 state-gradient norm")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{task}, T = {length}, update {update}, {mode} mode, mean over seeds")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def slope_vs_time(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, ("adding", "recall")):
        d = data[data["task"] == task]
        for init, marker in (("conventional", "o"), ("chrono", "^")):
            for solved, face in ((True, None), (False, "none")):
                g = d[(d["init"] == init) & (d["solved"] == solved)]
                time = g["solved_at"].fillna(g["max_updates"])
                ax.scatter(
                    g["state_slope"],
                    time,
                    marker=marker,
                    s=14,
                    facecolors=face,
                    edgecolors="C0" if init == "conventional" else "C1",
                    label=f"{init}, {'solved' if solved else 'not solved'}",
                )
        ax.set_title(task)
        ax.set_xlabel("state-gradient log-slope per step at update 100")
    axes[0].set_ylabel("updates to solve (budget if not solved)")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def prediction(predictions, path):
    y = predictions["solved"].to_numpy(int)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for column, label in (("baseline", "baseline"), ("full", "baseline + gradient slope")):
        fpr, tpr, _ = roc_curve(y, predictions[column])
        axes[0].plot(fpr, tpr, label=label)
        observed, predicted = calibration_curve(y, predictions[column], n_bins=10, strategy="quantile")
        axes[1].plot(predicted, observed, marker="o", ms=3, label=label)
    axes[0].plot([0, 1], [0, 1], color="grey", lw=0.8)
    axes[1].plot([0, 1], [0, 1], color="grey", lw=0.8)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC, held-out conditions")
    axes[1].set(xlabel="predicted probability", ylabel="observed frequency", title="Calibration")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)
