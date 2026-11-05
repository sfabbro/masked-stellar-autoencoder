from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
p = (
    root
    / "experiments/xp/results-stellar-improvement/multisurvey_flexible_calibration_20261105_v1/mixture-cal.npz"
)
out = root / "docs/reports/msa-nongaussian-morning-20261105"
with np.load(p, allow_pickle=False) as f:
    d = dict(f)
fig, axes = plt.subplots(
    2, 3, figsize=(13, 8), sharex=True, sharey=True, layout="constrained"
)
for s, name in enumerate(["APOGEE", "DESI DR1 RVS", "GALAH"]):
    vals = []
    inds = []
    for i in range(len(d["source_id"])):
        v = []
        for target in range(3):
            j = np.flatnonzero(
                d["mask"][i]
                & (d["source_index"][i] == s)
                & (np.arange(9) % 3 == target)
            )
            v.append(d["reference"][i, j[0]] if len(j) else np.nan)
        if np.isfinite(v).all():
            vals.append(v)
            inds.append(i)
    v = np.asarray(vals)
    c = d["canonical_median"][inds]
    for row, a in enumerate((v, c)):
        ax = axes[row, s]
        inside = (
            (a[:, 0] >= 3500) & (a[:, 0] <= 8500) & (a[:, 1] >= 0) & (a[:, 1] <= 5.5)
        )
        im = ax.scatter(
            a[:, 0],
            a[:, 1],
            c=a[:, 2],
            vmin=-2.5,
            vmax=0.5,
            cmap="viridis",
            s=7,
            alpha=0.65,
            linewidths=0,
        )
        ax.set_title(
            f"{name}: {len(v)} matched stars\n"
            + (
                "Native catalogue labels"
                if row == 0
                else "Shared-reference predictive medians"
            )
        )
        ax.text(
            0.03,
            0.96,
            f"{(~inside).sum()} outside displayed window",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.grid(alpha=0.15)
        ax.set_xlim(8500, 3500)
        ax.set_ylim(5.5, 0)
        if row == 1:
            ax.set_xlabel("Effective temperature [K]")
        if s == 0:
            ax.set_ylabel("log g [dex]")
fig.colorbar(
    im,
    ax=axes,
    label="[Fe/H] [dex]; colour saturates outside −2.5 to +0.5",
    extend="both",
    shrink=0.8,
)
fig.suptitle(
    "Kiel diagram preview — 2,048 CAL-role stars, held out from density-head fitting\nPilot mixture model; common reference is provisional. This is not the devtest set.",
    fontsize=13,
)
fig.savefig(out / "kiel-cal-preview.png", dpi=170)
print(out / "kiel-cal-preview.png")
