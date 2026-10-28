"""Static figure from independently verified all-star distance-pilot metrics."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = (
    Path(__file__).resolve().parents[1]
    / "results-stellar-improvement/distance_value_frozen_v3/verified-comparison"
)
rows = list(csv.DictReader((root / "conditional-metrics.csv").open()))
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
colors = {
    "parallax_free": "#667788",
    "parallax_assisted": "#377eb8",
    "raw_gaia": "#d95f02",
}
labels = {
    "parallax_free": "Free",
    "parallax_assisted": "Assisted",
    "raw_gaia": "Raw Gaia",
}
for ax, kind, metric, title, unit in [
    (
        axes[0],
        "native_kpc",
        "mae",
        "Native StarHorse agreement\n22,932 stars",
        "MAE (kpc)",
    ),
    (
        axes[1],
        "reference_mas",
        "mae",
        "Hipparcos agreement\n4,166 stars",
        "Parallax MAE (mas)",
    ),
    (
        axes[2],
        "reference_mas",
        "rmse",
        "Hipparcos agreement\n4,166 stars",
        "Parallax RMSE (mas)",
    ),
]:
    selected = [r for r in rows if r["kind"] == kind and r["group"] == "all"]
    names = [r["variant"] for r in selected]
    values = [float(r[metric]) for r in selected]
    ax.bar(
        [labels[n] for n in names], values, color=[colors[n] for n in names], width=0.65
    )
    if kind == "reference_mas":
        ax.set_yscale("log")
        ax.set_ylim(0.3, max(values) * 2.5)
    else:
        ax.set_ylim(0, max(values) * 1.25)
    for i, v in enumerate(values):
        ax.text(i, v * 1.07, f"{v:.3g}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(unit)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
fig.suptitle(
    "Bounded frozen distance pilot: native and Hipparcos agreement", fontsize=13
)
fig.supxlabel(
    "All reference measurements retained; 159 Hipparcos and 4 Gaia parallaxes are nonpositive. No clipping or SNR cut.",
    fontsize=9,
)
fig.savefig(root / "distance-value.png", dpi=180)
fig.savefig(root / "distance-value.pdf")
