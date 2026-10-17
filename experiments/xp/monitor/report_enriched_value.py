"""Recompute source-separated paired errors for the independently verified pilot."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = (
    Path(__file__).resolve().parents[1]
    / "results-stellar-improvement/enriched_value_frozen_v1"
)
verified = json.loads((BASE / "independent-verification.json").read_text())
assert verified["independent_augmented_lstsq_all_predictions_verified"]
OUT = BASE / "verified-comparison"
OUT.mkdir(exist_ok=True)
rows = []
summary = {}
with (
    np.load(BASE / "paired-cohort.npz") as data,
    np.load(BASE / "predictions.npz") as pred,
):
    for layer in ("native", "PIGS", "Li2025_CaT_silver"):
        raw = data[layer + "__measurements"]
        position = data[layer + "__used_row"]
        truth = raw["value"]
        bins = {
            "all": np.ones(len(raw), dtype=bool),
            "below_minus2": truth < -2,
            "below_minus3": truth < -3,
            "minus3_to_minus2": (truth >= -3) & (truth < -2),
            "minus2_to_minus1": (truth >= -2) & (truth < -1),
            "at_least_minus1": truth >= -1,
        }
        if layer == "native":
            for survey in np.unique(raw["survey"]):
                bins["survey_" + str(survey)] = raw["survey"] == survey
        summary[layer] = {}
        for arm in ("native_only", "canonical_augmented"):
            residual = pred[arm][position] - truth
            summary[layer][arm] = {}
            for name, mask in bins.items():
                n = int(mask.sum())
                item = dict(
                    layer=layer,
                    arm=arm,
                    bin=name,
                    count=n,
                    mae=float(np.abs(residual[mask]).mean()) if n else None,
                    rmse=float(np.sqrt(np.mean(residual[mask] ** 2))) if n else None,
                    bias=float(residual[mask].mean()) if n else None,
                )
                rows.append(item)
                summary[layer][arm][name] = item
with (OUT / "conditional-metrics.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
layers = ["native", "PIGS", "Li2025_CaT_silver"]
for ax, selected, title in zip(
    axes,
    ["all", "below_minus2"],
    ["All source-layer test stars", "Reported [Fe/H] < -2 test stars"],
    strict=True,
):
    for offset, arm, label, color in [
        (-0.19, "native_only", "Native-only fits", "#2864a0"),
        (0.19, "canonical_augmented", "+ PIGS / Li fits", "#db7726"),
    ]:
        values = [summary[layer][arm][selected]["mae"] for layer in layers]
        bars = ax.bar(
            np.arange(3) + offset, values, width=0.36, label=label, color=color
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    counts = [summary[layer]["native_only"][selected]["count"] for layer in layers]
    ax.set_xticks(
        np.arange(3),
        [
            f"{name}\nn={n:,}"
            for name, n in zip(["Native", "PIGS", "Li CaT silver"], counts, strict=True)
        ],
    )
    ax.set_ylabel("MAE to reported FeH (dex)")
    ax.set_title(title)
    ax.set_ylim(0, 1.6)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle(
    "Frozen epoch57 ridge pilot: added labels improve tails, worsen native overall MAE",
    fontsize=12,
)
fig.savefig(OUT / "enriched-value.png", dpi=170)
fig.savefig(OUT / "enriched-value.pdf")
plt.close(fig)
n = summary["native"]
text = f"""# Verified enriched-label comparison\n\nTwo fixed-alpha10 ridge heads share one frozen epoch57 encoding, native-fit-only scaling and zero initial heads. Native-only19859fits versus canonical20267fits, adding219PIGS+189Li unique stars by fixed precedence. This changes the fit population/count, not a matched training-presentation objective.\n\nOn100620native test stars, MAE worsens from{n["native_only"]["all"]["mae"]:.6f} to{n["canonical_augmented"]["all"]["mae"]:.6f}dex. On185native stars reported below-2, MAE improves from{n["native_only"]["below_minus2"]["mae"]:.6f} to{n["canonical_augmented"]["below_minus2"]["mae"]:.6f}dex. The14native below-3 stars improve from2.106990 to1.578088dex, with positive bias still1.578088dex.\n\nSeparate PIGS test124 MAE0.846808 to0.427381dex; separate LiCaT test129 MAE1.110577 to0.739791dex. Their below-2 counts are28 and128 respectively. They remain distinct silver/spectroscopic calibrations, potentially overlapping source layers; do not pool them into independent truth. Augmented all-layer biases remain+0.272396PIGS/+0.700982Li dex.\n\nDecision: additional labels have useful metal-poor-tail information but do not justify a universal merge or a claim of globally better native predictions. Keep the new prediction-loss/head wave native-only to isolate its variables. No extra acquisition or further wave is inferred.\n\nIndependent verification checked every source/cohort/rawmeasurement/input/fit-test mask, shared scaling and initial heads; all ridge coefficients/predictions against augmented least squares; all121098inputs/latents/predictions against the completed actual-parent CPU reference; and actualGPUencoder tensor bytes exactly against that parent. CPUreference is reused, not a newly rerun encoder. Production real128matchedXP unit check passed. Max latent CPU/GPU difference3.409386e-5; max prediction differences8.328880e-6/1.163615e-5dex. Original sourceH5 retained with metadata; NPZstructuredfield bytes checked without claiming HDF5attrs retention. Single seed/frozen head, teacher/calibration dependencies, sample selection and unsupervised pretraining overlap remain limitations. This is not SOTA, calibrated-NLL gold truth or a result for the later epoch72 heads.\n"""
(OUT / "comparison.md").write_text(text)
print(
    json.dumps(
        {
            layer: {
                arm: {b: summary[layer][arm][b] for b in ["all", "below_minus2"]}
                for arm in summary[layer]
            }
            for layer in summary
        },
        indent=2,
    )
)
