import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("experiments/xp/results-stellar-improvement")
d = json.loads((root / "delayed-comparison.json").read_text())
rows = [
    r
    for r in d["photometry"]
    if r["subset"] == "all_six_groups_complete" and r["weighting"] == "unweighted"
]
fig, ax = plt.subplots(figsize=(8, 4.6))
names = [
    r["run_id"]
    .replace("photo_value_no_", "")
    .replace("skymapper", "SkyMapper")
    .replace("2mass", "2MASS")
    .upper()
    for r in rows
]
delta = [r["delta_mae_dex"] for r in rows]
ax.barh(names, delta, color=["#D55E00" if v > 0 else "#0072B2" for v in delta])
ax.axvline(0, color="#555", lw=0.8)
ax.invert_yaxis()
ax.set_xlabel("Metallicity MAE after removal − full-input MAE (dex)")
ax.set_title("Value of existing inputs on the same 1,157 stars", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)
for i, v in enumerate(delta):
    ax.text(
        v + (0.0005 if v >= 0 else -0.0005),
        i,
        f"{v:+.4f}",
        ha="left" if v >= 0 else "right",
        va="center",
        fontsize=9,
    )
ax.set_xlim(-0.009, 0.039)
fig.text(
    0.02,
    0.02,
    "All six groups observed on every test star; full-input MAE = 0.1423 dex.\nFrozen encoder; head refit per ablation; one seed; provisional catalogue labels.\nPositive means removal hurts. Small negative changes do not establish that a survey is useless.",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.15, 1, 1])
fig.savefig(root / "photometry-common-footprint.png", dpi=180)
fig.savefig(root / "photometry-common-footprint.pdf")
