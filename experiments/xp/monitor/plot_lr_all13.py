import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("experiments/xp/results-stellar-improvement")
verified = json.loads((root / "lr-comparison.json").read_text())
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey="row")
colors = ["#0072B2", "#444444", "#D55E00", "#009E73", "#CC79A7"]
styles = ["--", "-", "-.", ":", "--"]
for col, batch in enumerate([32768, 8192, 2048]):
    for index, factor in enumerate(
        ["05", "1", "2"] if batch == 32768 else ["05", "1", "2", "4", "12"]
    ):
        name = f"mae_b{batch}_lr{factor}_s42"
        path = root / name / name / "metrics.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        x = np.array([r["rows_seen_total"] / 1e6 for r in rows])
        train = np.array(
            [np.nan if r["train_loss"] is None else r["train_loss"] for r in rows]
        )
        xp = np.array([r["xp_only_mae"] for r in rows])
        lr = np.array([r["learning_rate"] * 1e6 for r in rows])
        kwargs = dict(
            color=colors[index],
            linestyle=styles[index],
            linewidth=1.6,
            marker="o",
            markersize=3,
            label={
                "05": "0.5× LR",
                "1": "1× LR",
                "2": "2× LR",
                "4": "4× LR",
                "12": "12× LR",
            }[factor],
        )
        axes[0, col].plot(x, train, **kwargs)
        axes[1, col].plot(x, xp, **kwargs)
        axes[2, col].plot(x, lr, **kwargs)
    steps = {32768: 612, 8192: 2443, 2048: 9768}[batch]
    axes[0, col].set_title(f"Batch {batch:,} · {steps:,} updates", fontsize=11)
    for row in range(3):
        axes[row, col].grid(alpha=0.18)
        axes[row, col].set_xlim(0, 20)
        axes[row, col].set_xticks([0, 5, 10, 15, 20])
        axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[2, col].set_xlabel("Training presentations (millions)")
axes[0, 0].set_ylabel("Interval training MAE\n(scaled feature units)")
axes[1, 0].set_ylabel("Validation XP MAE\n(scaled coefficient units)")
axes[2, 0].set_ylabel("Absolute learning rate\n(× 10⁻⁶, held fixed)")
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.945),
    ncol=5,
    frameon=False,
    fontsize=10,
)
fig.suptitle(
    "All 13 LR / batch experiments — fixed 20M presentations",
    x=0.065,
    y=0.98,
    ha="left",
    fontsize=14,
)
fig.text(
    0.065,
    0.025,
    "Source: saved CANFAR metrics, 15 Oct 2026 · epoch-57 parent · 9M train / 1M validation · seed 42.\nTraining MAE averages preceding ≈5M presentations; validation uses all 1M stars. No curve smoothing.\nActual row/mask hashes and initial predictions match; reloads passed. All 13 experiments complete; no checkpoint selection.\nBatch 2,048 makes 9,768 updates because the final pass contains a shuffled short cache block; all runs see exactly 20M presentations.",
    fontsize=9,
    ha="left",
    va="bottom",
)
fig.tight_layout(rect=[0.02, 0.13, 0.99, 0.91])
fig.savefig(root / "lr-loss-and-LR.png", dpi=180)
fig.savefig(root / "lr-loss-and-LR.pdf")
plt.close(fig)
with (root / "lr-final-metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=verified["runs"][0].keys())
    writer.writeheader()
    writer.writerows(verified["runs"])
print(root / "lr-loss-and-LR.png")
