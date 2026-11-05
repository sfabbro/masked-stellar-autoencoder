"""Matched native/common Kiel views from a full-role distribution export."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msa-joint-mpl")
import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot(path, output):
    with h5py.File(path) as h5:
        mean = h5["mean"][:]
        values = h5["native_value"][:]
        mask = h5["native_observed"][:]
        role = h5.attrs["role"]
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8), sharex=True, sharey=True, layout="constrained"
    )
    for s, name in enumerate(("APOGEE", "DESI DR1 RVS", "GALAH")):
        keep = mask[:, s, :2].all(1)
        for r, a in enumerate((values[keep, s], mean[keep])):
            ax = axes[r, s]
            finite = np.isfinite(a[:, :2]).all(1)
            inside = (
                finite
                & (a[:, 0] >= 3500)
                & (a[:, 0] <= 8500)
                & (a[:, 1] >= 0)
                & (a[:, 1] <= 5.5)
            )
            if inside.any():
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
                title=f"{name}: {keep.sum():,} matched stars\n"
                + (
                    "Native survey labels"
                    if r == 0
                    else "Common-reference predictive mean"
                ),
            )
            ax.text(
                0.02,
                0.97,
                f"{(~inside).sum():,} outside window",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            if r == 1:
                ax.set_xlabel("Effective temperature [K]")
            if s == 0:
                ax.set_ylabel("log g [dex]")
    fig.suptitle(
        f"{role.upper()} — native versus shared-reference Kiel diagrams\nSame stars within each column; log-count density, independent panel colour scales; natural inputs, no artificial masking"
    )
    fig.savefig(output / f"kiel-{role}.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (lo, hi) in zip(axes, [(-np.inf, -1), (-1, -0.5), (-0.5, np.inf)]):
        keep = (mean[:, 2] >= lo) & (mean[:, 2] < hi) & np.isfinite(mean).all(1)
        a = mean[keep]
        inside = (
            (a[:, 0] >= 3500) & (a[:, 0] <= 8500) & (a[:, 1] >= 0) & (a[:, 1] <= 5.5)
        )
        if inside.any():
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
            title=f"{lo} ≤ predicted [Fe/H] < {hi}\n{keep.sum():,} stars; {(~inside).sum():,} outside window",
            xlabel="Effective temperature [K]",
        )
    axes[0].set_ylabel("log g [dex]")
    fig.suptitle(
        f"{role.upper()} — all common-reference predictions, by predicted metallicity"
    )
    fig.savefig(output / f"kiel-common-{role}.png", dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    for role in ("fit", "devtest"):
        plot(a.run / f"predictions-{role}.h5", a.run)


if __name__ == "__main__":
    main()
