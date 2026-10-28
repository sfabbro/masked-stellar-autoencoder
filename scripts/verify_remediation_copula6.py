#!/usr/bin/env python3
"""Independent NumPy checks and saved-sample corners; no model execution."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("out", type=Path)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
receipt = json.loads((a.source / "receipts.json").read_text())
jobs = {x["run_id"]: x for x in receipt["jobs"]}
reports = []
for record in receipt["artifacts"]:
    path = a.source / record["run_id"] / record["name"]
    assert (
        path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    )
for run, job in jobs.items():
    file = a.source / run / "integration-selected.npz"
    if not file.exists():
        continue
    with np.load(file, allow_pickle=False) as d:
        ess = d["relative_ess"]
        rep = d["independent_replica_log_prob_gap"]
        ref = d["refinement_log_prob_gap"]
        ids = d["source_id"]
        assert len(ess) == len(rep) == len(ref) == len(ids) == 128
        assert (
            np.all(np.isfinite(ess))
            and np.all(np.isfinite(rep))
            and np.all(np.isfinite(ref))
        )
        assert np.all(
            (ess > 0) & (ess <= 1 + 1e-6)
        )  # float32 rounding; retain raw values
        metrics = {
            "relative_ess_median": float(np.median(ess)),
            "relative_ess_min": float(np.min(ess)),
            "replica_gap_p95": float(np.quantile(rep, 0.95)),
            "refinement_gap_p95": float(np.quantile(ref, 0.95)),
        }
        expected = (
            job["selected_nsf_integration"]
            if job["status"] == "remediation_pilot_complete"
            else json.loads(job["error"].split("failed: ", 1)[1])
        )
        for key, value in metrics.items():
            np.testing.assert_allclose(value, expected[key], rtol=1e-6, atol=1e-8)
        passed = (
            metrics["relative_ess_median"] >= 0.1
            and metrics["relative_ess_min"] >= 0.01
            and metrics["replica_gap_p95"] <= 0.1
            and metrics["refinement_gap_p95"] <= 0.1
        )
        assert passed == (job["status"] == "remediation_pilot_complete")
        reports.append(
            {
                "run_id": run,
                "passed": passed,
                "metrics": metrics,
                "max_replica_gap": float(np.max(rep)),
                "max_refinement_gap": float(np.max(ref)),
                "max_replica_gap_source_id": int(ids[np.argmax(rep)]),
                "source_ids_sha256": hashlib.sha256(
                    ids.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
        )
assert len(reports) == 5
runs = ["remediation_copula_nsf6_frozen_v1", "remediation_copula_nsf6_ft_v1"]
bundles = []
for run in runs:
    with np.load(a.source / run / "whole-vector-corners.npz", allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    with np.load(a.source / run / "normalization.npz", allow_pickle=False) as z:
        norm = {k: z[k] for k in z.files}
    std = bundle["standardized_draws"]
    native = bundle["native_draws"]
    assert (
        std.shape == native.shape == (8192, 10, 6)
        and np.isfinite(std).all()
        and np.isfinite(native).all()
    )
    expected = std * norm["scale"] + norm["center"]
    expected[:, :, 4:] = np.exp(expected[:, :, 4:])
    np.testing.assert_allclose(native, expected, rtol=1e-12, atol=1e-12)
    for i in range(10):
        np.testing.assert_allclose(
            bundle["standardized_covariance"][i],
            np.cov(std[:, i, :], rowvar=False),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            bundle["native_covariance"][i],
            np.cov(native[:, i, :], rowvar=False),
            rtol=1e-10,
            atol=1e-10,
        )
    bundles.append(bundle)
assert np.array_equal(bundles[0]["source_id"], bundles[1]["source_id"])
assert np.array_equal(bundles[0]["observed"], bundles[1]["observed"])
assert np.array_equal(bundles[0]["reference"], bundles[1]["reference"], equal_nan=True)
labels = [
    "$T_{eff}$ (K)",
    "$\\log g$ (dex)",
    "[Fe/H] (dex)",
    "alpha (source scale, dex)",
    "Age (Gyr, log)",
    "Distance (kpc, log)",
]
colors = ["#1878a0", "#d54b2d"]
names = ["Frozen encoder", "Fine-tuned encoder"]
figure_files = []
for idx in (0, 2, 8):
    star = int(bundles[0]["source_id"][idx])
    values = [b["native_draws"][:, idx, :] for b in bundles]
    reference = bundles[0]["reference"][idx]
    observed = bundles[0]["observed"][idx]
    ranges = []
    for d in range(6):
        both = np.concatenate([v[:, d] for v in values])
        lo, hi = float(both.min()), float(both.max())
        if observed[d]:
            lo = min(lo, float(reference[d]))
            hi = max(hi, float(reference[d]))
        if d >= 4:
            assert lo > 0
            lo *= 0.9
            hi *= 1.1
        else:
            span = max(hi - lo, 1e-8)
            lo -= 0.04 * span
            hi += 0.04 * span
        ranges.append((lo, hi))
    fig, axes = plt.subplots(6, 6, figsize=(12, 12), squeeze=False)
    for y in range(6):
        for x in range(6):
            ax = axes[y, x]
            if x > y:
                ax.axis("off")
                continue
            for k, v in enumerate(values):
                if x == y:
                    bins = (
                        np.geomspace(*ranges[x], 45)
                        if x >= 4
                        else np.linspace(*ranges[x], 45)
                    )
                    ax.hist(
                        v[:, x],
                        bins=bins,
                        density=True,
                        histtype="step",
                        color=colors[k],
                        lw=1.15,
                        label=names[k],
                    )
                else:
                    chosen = np.arange(
                        0, 8192, 4
                    )  # deterministic display subset; stored draws unchanged
                    ax.scatter(
                        v[chosen, x],
                        v[chosen, y],
                        s=0.5,
                        color=colors[k],
                        alpha=0.12,
                        rasterized=True,
                    )
            if observed[x]:
                ax.axvline(reference[x], color="#323232", lw=0.6, ls="--")
            if x != y and observed[y]:
                ax.axhline(reference[y], color="#323232", lw=0.6, ls="--")
            ax.set_xlim(*ranges[x])
            if x >= 4:
                ax.set_xscale("log")
            if x != y:
                ax.set_ylim(*ranges[y])
                if y >= 4:
                    ax.set_yscale("log")
            if y < 5:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(labels[x], fontsize=8)
            if x > 0:
                ax.tick_params(labelleft=False)
            elif y:
                ax.set_ylabel(labels[y], fontsize=8)
            ax.tick_params(labelsize=6, length=2)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Joint spline draws — Gaia " + str(star), fontsize=13)
    fig.text(
        0.5,
        0.025,
        "8192 saved joint draws/model; 2048 shown per scatter. Dashed: available catalog references only.\nGaussian-copula dependence; missing alpha, age and distance references do not validate the posterior.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = a.out / ("corner-" + str(star) + ".png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figure_files.append(path.name)
summary = {
    "scope": "Saved-array arithmetic and plotted samples only; not model replay, physical-copula certification, an untouched test, or scientific selection.",
    "gate_checks": reports,
    "corner_arrays": "two models x 10 stars x 8192 original full-vector draws, coordinate conversion and covariance recomputed",
    "corner_plots": figure_files,
    "corner_missing_truth": "All ten plotted-source candidates lack alpha, age and distance reference values.",
}
(a.out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
