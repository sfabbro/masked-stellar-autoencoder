"""Static scientific figure from independently verified paired age metrics."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = (
    Path(__file__).resolve().parents[1]
    / "results-stellar-improvement/native-age-comparison"
)
rows = list(csv.DictReader((root / "conditional-metrics.csv").open()))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
colors = ["#667788", "#377eb8", "#d95f02"]
names = ["frozen", "sft", "recon"]
labels = ["Frozen encoder", "SFT", "SFT + reconstruction"]
for ax, kind, groups, title in [
    (
        axes[0],
        "native",
        ["all", "apogee17", "galahdr3"],
        "Native StarHorse agreement (22,830 test stars)",
    ),
    (
        axes[1],
        "seismic",
        ["AST", "BASTA", "C2kSMO", "GOE", "V&A", "YMCM"],
        "LEGACY reference agreement (same 50 stars)",
    ),
]:
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        values = [
            float(
                next(
                    r
                    for r in rows
                    if r["kind"] == kind
                    and r["group"] == g
                    and r["run"] == "native_age_strict_" + name + "_v1"
                )["mae_gyr"]
            )
            for g in groups
        ]
        ax.bar(
            np.arange(len(groups)) + (i - 1) * 0.24,
            values,
            width=0.23,
            color=color,
            label=label,
        )
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel("Mean absolute age difference (Gyr)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(
    "Bounded strict age controls: matched fit IDs, initial states and updates",
    fontsize=13,
)
fig.supxlabel(
    "Single seed; model-dependent labels/references; six seismic pipelines are correlated",
    fontsize=10,
)
fig.savefig(root / "native-age.png", dpi=180)
fig.savefig(root / "native-age.pdf")
