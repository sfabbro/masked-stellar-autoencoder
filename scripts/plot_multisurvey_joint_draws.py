"""Intrinsic joint-mixture draws from saved prediction arrays, no network execution."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msa-joint-mpl")
import matplotlib

matplotlib.use("Agg")
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
    description="Compare saved predictive means and intrinsic mixture draws in the Kiel plane"
)
parser.add_argument("predictions", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
with h5py.File(args.predictions) as h:
    w = h["weights"][:]
    mu = h["component_mean"][:]
    lower = h["component_lower"][:]
    mean = h["mean"][:]
rng = np.random.default_rng(20261105)
component = (rng.random(len(w))[:, None] > w.cumsum(1)).sum(1).clip(0, 3)
index = np.arange(len(w))
draw = mu[index, component] + np.einsum(
    "nij,nj->ni", lower[index, component], rng.standard_normal((len(w), 3))
)
fig, axes = plt.subplots(
    2, 3, figsize=(13, 8), sharex=True, sharey=True, layout="constrained"
)
for col, (lo, hi) in enumerate(((-np.inf, -1), (-1, -0.5), (-0.5, np.inf))):
    keep = (mean[:, 2] >= lo) & (mean[:, 2] < hi)
    for row, values in enumerate((mean, draw)):
        a = values[keep]
        inside = (
            (a[:, 0] >= 3500) & (a[:, 0] <= 8500) & (a[:, 1] >= 0) & (a[:, 1] <= 5.5)
        )
        ax = axes[row, col]
        ax.hexbin(
            a[inside, 0],
            a[inside, 1],
            gridsize=110,
            extent=(3500, 8500, 0, 5.5),
            bins="log",
            mincnt=1,
            cmap="magma",
        )
        ax.set(
            xlim=(8500, 3500),
            ylim=(5.5, 0),
            title=f"{lo} ≤ predicted mean [Fe/H] < {hi}\n"
            + ("Predictive means" if row == 0 else "One intrinsic joint draw per star"),
        )
        ax.text(
            0.02,
            0.97,
            f"{keep.sum():,} stars; {(~inside).sum():,} outside window",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        if row == 1:
            ax.set_xlabel("Effective temperature [K]")
        if col == 0:
            ax.set_ylabel("log g [dex]")
fig.suptitle(
    "DEVTEST — predictive-mean sharpness versus full mixture uncertainty\nSame stars and metallicity groups; no measurement-noise draw added; independent log-count colour scales"
)
fig.savefig(out / "kiel-devtest-distribution.png", dpi=170)
print(
    "Saved distribution diagnostic; nonpositive Teff draws:",
    int((draw[:, 0] <= 0).sum()),
)
