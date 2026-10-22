#!/usr/bin/env python3
"""Standard stellar-parameter residual panel for a saved Multilabel prediction file.

Reads a `predictions.npz` produced by `scripts/run_multilabel_screen.py` and the
matching paired-cohort `.npz`, inverse-transforms the standardized test means, and
writes a 3x3 residual diagnostic (residual vs truth, residual histogram, and binned
median residual vs true [Fe/H]) plus a JSON metrics summary.

Usage:
  pixi run python scripts/plot_stellar_residuals.py \
    --predictions experiments/.../predictions.npz \
    --cohort experiments/.../paired-cohort.npz \
    --output docs/reports/residuals-<name>.png \
    --title "MSA long noise Gaussian (selected 99M)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ("T_eff", "log g", "[Fe/H]")
UNITS = ("K", "dex", "dex")
FEH_BINS = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.6])


def metrics(residual: np.ndarray) -> dict:
    finite = residual[np.isfinite(residual)]
    if not len(finite):
        return {"n": 0, "mae": None, "bias": None, "nmad": None, "rmse": None}
    return {
        "n": int(len(finite)),
        "mae": float(np.mean(np.abs(finite))),
        "bias": float(np.mean(finite)),
        "nmad": float(1.4826 * np.median(np.abs(finite - np.median(finite)))),
        "rmse": float(np.sqrt(np.mean(finite**2))),
    }


def load_pair(predictions: Path, cohort_path: Path):
    cohort = np.load(cohort_path, allow_pickle=False)
    rows = cohort["test_rows"]
    truth = cohort["value"][rows]
    center = cohort["label_center"]
    scale = cohort["label_scale"]
    pred = np.load(predictions, allow_pickle=False)
    if "test_source_id" in pred:
        expected = cohort["source_id"][rows]
        if not np.array_equal(pred["test_source_id"], expected):
            raise ValueError("prediction and cohort test rows differ")
    expected_rows = cohort["test_rows"]
    if "test_rows" in pred and not np.array_equal(pred["test_rows"], expected_rows):
        raise ValueError("prediction test_rows differ from cohort test_rows")
    predicted = np.asarray(pred["test_mean"], dtype="f8") * scale + center
    return truth, predicted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="MSA residuals")
    parser.add_argument("--tail-feh", type=float, default=-2.0)
    args = parser.parse_args(argv)

    truth, predicted = load_pair(args.predictions, args.cohort)
    residual = predicted - truth
    tail = truth[:, 2] < args.tail_feh
    summary = {
        "predictions": str(args.predictions),
        "cohort": str(args.cohort),
        "n": int(len(truth)),
        "tail_feh": float(args.tail_feh),
        "tail_n": int(tail.sum()),
        "global": {LABELS[j]: metrics(residual[:, j]) for j in range(3)},
        "tail": {LABELS[j]: metrics(residual[tail, j]) for j in range(3)},
    }

    fig, axes = plt.subplots(3, 3, figsize=(16.8, 13.4), constrained_layout=True)
    fig.suptitle(
        f"{args.title}\n"
        f"test n={len(truth):,}; tail [Fe/H]<{args.tail_feh:g} n={int(tail.sum())} "
        "(local preliminary scoring, not a certificate)",
        fontsize=12,
    )
    clip = ((None, None), (-600.0, 600.0), (-1.0, 1.0), (-1.5, 1.5))
    for row, label in enumerate(["residual", "histogram", "vs [Fe/H]"]):
        for col in range(3):
            ax = axes[row, col]
            values = residual[:, col]
            finite = np.isfinite(values)
            if row == 0:
                lo, hi = clip[col + 1]
                keep = finite & (values >= lo) & (values <= hi)
                ax.hexbin(
                    truth[keep, col], values[keep], gridsize=55, bins="log",
                    cmap="viridis", mincnt=1, rasterized=True,
                )
                ax.axhline(0, color="0.3", lw=0.8)
                stats = metrics(values)
                ax.set_title(f"{LABELS[col]}: residual vs truth", fontsize=10)
                ax.text(
                    0.02, 0.97,
                    f"MAE={stats['mae']:.3g} {UNITS[col]}\n"
                    f"bias={stats['bias']:+.3g} {UNITS[col]}\n"
                    f"NMAD={stats['nmad']:.3g} {UNITS[col]}",
                    transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.78, lw=0),
                )
                ax.set_xlabel(f"true {LABELS[col]} [{UNITS[col]}]")
                ax.set_ylabel(f"pred − true [{UNITS[col]}]")
                if keep.sum() < finite.sum():
                    ax.text(
                        0.01, 0.02, f"display clipped; {finite.sum() - keep.sum()} points",
                        transform=ax.transAxes, fontsize=7, color="0.4",
                    )
            elif row == 1:
                q = np.nanpercentile(np.abs(values), [0.1, 99.9])
                ax.hist(
                    values[finite], bins=70, range=(-q[1], q[1]),
                    color="#4c72b0", alpha=0.85,
                )
                ax.axvline(0, color="0.3", lw=0.8)
                ax.set_yscale("log")
                ax.set_xlabel(f"pred − true [{UNITS[col]}]")
                ax.set_ylabel("count")
                ax.set_title(f"{LABELS[col]}: residual distribution", fontsize=10)
            else:
                medians, counts, centers = [], [], []
                for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:]):
                    bin_mask = finite & (truth[:, 2] >= lo) & (truth[:, 2] < hi)
                    counts.append(int(bin_mask.sum()))
                    centers.append(0.5 * (lo + hi))
                    medians.append(
                        np.median(values[bin_mask]) if bin_mask.sum() >= 5 else np.nan
                    )
                ax.plot(centers, medians, "o-", color="#c44e52", lw=1.2)
                ax.axhline(0, color="0.3", lw=0.8)
                ax.axvline(args.tail_feh, color="0.55", ls=":", lw=1.0)
                ax.set_xlabel("true [Fe/H]")
                ax.set_ylabel(f"median {LABELS[col]} residual [{UNITS[col]}]")
                ax.set_title(f"{LABELS[col]}: median residual vs [Fe/H]", fontsize=10)
                for x, y, nbin in zip(centers, medians, counts):
                    if np.isfinite(y):
                        ax.annotate(
                            f"{nbin}", (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7,
                        )
                tail_stats = summary["tail"][LABELS[col]]
                ax.text(
                    0.01, 0.98,
                    f"tail n={tail_stats['n']}  MAE={tail_stats['mae']:.3g}  "
                    f"bias={tail_stats['bias']:+.3g}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.75, lw=0),
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, dpi=200)
    plt.close(fig)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": "complete", "png": str(args.output), "pdf": str(pdf), "json": str(summary_path)}))


if __name__ == "__main__":
    main()
