"""Visualize predeclared metallicity strata from the saved paired audit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/xp/science-review-20261015"
audit = json.loads((OUT / "metallicity-tails.json").read_text())
assert audit["status"] == "complete"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
keys = [
    "[-inf,-2.5)",
    "[-2.5,-2.0)",
    "[-2.0,-1.5)",
    "[-1.5,-1.0)",
    "[-1.0,-0.5)",
    "[-0.5,0.0)",
    "[0.0,inf)",
]
labels = [
    "<−2.5",
    "−2.5 to −2",
    "−2 to −1.5",
    "−1.5 to −1",
    "−1 to −0.5",
    "−0.5 to 0",
    "≥0",
]
models = [
    ("probe_new_control_s42", "Control continuation"),
    ("probe_new_spectral_s42", "Spectral 0.1 continuation"),
    ("latest_standard57", "Standard epoch 57 (more pretraining)"),
]
for ax, split in zip(axes, ["identity", "sky"]):
    ss = audit["splits"][split]
    for model, label in models:
        bins = ss["models"][model]["methods"]["frozen_head"]["strata"]
        ax.plot(
            range(len(keys)),
            [bins[k]["mae_dex"] for k in keys],
            marker="o",
            label=label,
        )
    ax.set_xticks(
        range(len(keys)),
        [f"{label}\nn={ss['stratum_counts'][key]}" for label, key in zip(labels, keys)],
        rotation=35,
        ha="right",
    )
    ax.set(title=f"{split.capitalize()} holdout", xlabel="Catalogue [Fe/H] bin (dex)")
    ax.grid(alpha=0.18)
maximum = max(
    audit["splits"][split]["models"][model]["methods"]["frozen_head"]["strata"][key][
        "mae_dex"
    ]
    for split in ["identity", "sky"]
    for model, _ in models
    for key in keys
)
axes[0].set_ylim(0, maximum * 1.08)
axes[0].set_ylabel("Metallicity prediction MAE (dex)")
fig.suptitle(
    "Frozen nonlinear probes: bulk accuracy hides metal-poor failure", fontsize=14
)
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(OUT / "metallicity-by-bin.png", dpi=180)
fig.savefig(OUT / "metallicity-by-bin.pdf")
