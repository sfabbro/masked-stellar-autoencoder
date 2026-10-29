"""Render saved NSF prediction and corner arrays; never loads a stellar model."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "experiments/xp/results-stellar-improvement/remediation-wave-20261026/nsf-energy-evaluation-recovery-preservation-20261029"
)
OUT = ROOT / "docs/reports/msa-nsf6-energy-20261029"
MODELS = [
    ("Frozen", "remediation_nsf6_energy_frozen_eval_recovery_v1"),
    ("Fine-tuned", "remediation_nsf6_energy_ft_eval_recovery_v1"),
    ("Fine-tuned + recon", "remediation_nsf6_energy_ft_recon003_eval_recovery_v1"),
]
LABELS = [
    "Teff (K)",
    "log g (dex)",
    "[Fe/H] (dex)",
    "alpha (source dex)",
    "Age (Gyr)",
    "Distance (kpc)",
]
OUT.mkdir(parents=True, exist_ok=True)
blocks = []
for label, run in MODELS:
    p = DATA / run
    d = np.load(p / "development-predictions.npz", allow_pickle=False)
    c = np.load(p / "whole-vector-corners.npz", allow_pickle=False)
    norm = np.load(p / "normalization.npz", allow_pickle=False)
    m = json.loads((p / "manifest.json").read_text())
    blocks.append((label, run, d, c, norm, m))
for label, run, d, c, norm, m in blocks:
    short = run.replace("remediation_nsf6_energy_", "").replace("_eval_recovery_v1", "")
    ref = d["value"]
    pred = d["native_median"]
    mask = d["observed"]
    mp = mask[:, 2] & (ref[:, 2] < -2)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        keep = mask[:, j] & np.isfinite(ref[:, j]) & np.isfinite(pred[:, j])
        x = ref[keep, j]
        y = pred[keep, j]
        ax.scatter(x, y, s=5, alpha=0.22, rasterized=True)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.set(
            xlabel="Reference " + LABELS[j],
            ylabel="Predicted " + LABELS[j],
            title=f"{LABELS[j]}   n={len(x)}",
        )
    fig.suptitle(label + " — previously used development panel")
    fig.savefig(OUT / (short + "-prediction.png"), dpi=145)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        keep = mp & mask[:, j] & np.isfinite(ref[:, j]) & np.isfinite(pred[:, j])
        x = ref[keep, j]
        v = pred[keep, j] - x
        ax.scatter(x, v, s=25)
        ax.axhline(0, color="black", lw=0.8) if len(x) else None
        ax.set(
            xlabel="Reference " + LABELS[j],
            ylabel="Prediction − reference",
            title=f"{LABELS[j]}   n={len(x)}",
        )
        if len(x) == 0:
            ax.text(
                0.5,
                0.5,
                "No reference labels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    fig.suptitle(label + " — metal-poor diagnostic ([Fe/H] < −2), only 5 stars")
    fig.savefig(OUT / (short + "-metalpoor.png"), dpi=145)
    plt.close(fig)
    for kind, key in [
        ("intrinsic", "empirical_intrinsic_pit"),
        ("observed", "empirical_observed_pit"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        pit = d[key]
        for j, ax in enumerate(axes.flat):
            keep = mask[:, j] & np.isfinite(pit[:, j])
            vals = pit[keep, j]
            if len(vals):
                ax.hist(
                    vals,
                    bins=np.linspace(0, 1, 21),
                    density=True,
                    color="#4274a4",
                    edgecolor="white",
                )
                ax.axhline(1, color="black", lw=0.8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Unavailable: quoted errors unknown",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set(
                xlim=(0, 1),
                xlabel="Empirical PIT",
                ylabel="Density",
                title=f"{LABELS[j]}   n={len(vals)}",
            )
        fig.suptitle(label + " — " + kind + " PIT (512 samples)")
        fig.savefig(OUT / (short + "-pit-" + kind + ".png"), dpi=145)
        plt.close(fig)
# Identical stars and common per-star native axes across all three NSF models.
reference_ids = blocks[0][3]["source_id"]
for b in blocks[1:]:
    assert np.array_equal(reference_ids, b[3]["source_id"])
for star in (0, 3):
    all_draws = np.concatenate(
        [b[3]["native_draws"][:, star, :] for b in blocks], axis=0
    )
    mins = all_draws.min(0)
    maxs = all_draws.max(0)
    reference = blocks[0][3]["reference"][star]
    observed = blocks[0][3]["observed"][star]
    mins = np.where(observed, np.minimum(mins, reference), mins)
    maxs = np.where(observed, np.maximum(maxs, reference), maxs)
    margin = np.maximum((maxs - mins) * 0.025, 1e-8)
    limits = list(zip(mins - margin, maxs + margin))
    for label, run, d, c, norm, m in blocks:
        short = run.replace("remediation_nsf6_energy_", "").replace(
            "_eval_recovery_v1", ""
        )
        x = c["native_draws"][:, star, :]
        fig, axes = plt.subplots(6, 6, figsize=(13, 13), constrained_layout=True)
        for i in range(6):
            for j in range(6):
                ax = axes[i, j]
                if j > i:
                    ax.axis("off")
                    continue
                if i == j:
                    ax.hist(x[:, i], bins=48, range=limits[i], color="#4274a4")
                    ax.set_xlim(*limits[i])
                    ax.set_yticks([])
                else:
                    ax.hist2d(
                        x[:, j],
                        x[:, i],
                        bins=48,
                        range=[limits[j], limits[i]],
                        cmap="Blues",
                        cmin=1,
                    )
                    ax.set_xlim(*limits[j])
                    ax.set_ylim(*limits[i])
                if observed[j]:
                    ax.axvline(reference[j], color="crimson", lw=0.8)
                if i != j and observed[i]:
                    ax.axhline(reference[i], color="crimson", lw=0.8)
                if i == 5:
                    ax.set_xlabel(LABELS[j], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(LABELS[i], fontsize=8)
                elif j > 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=6)
        fig.suptitle(
            f"{label} — Gaia {int(reference_ids[star])} — 8192 joint samples; shared axes"
        )
        fig.savefig(
            OUT / (short + "-corner-" + str(int(reference_ids[star])) + ".png"), dpi=115
        )
        plt.close(fig)
print("Wrote plots to", OUT)
