"""Saved-array population and within-star corners; never invent marginal dependence."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

LABELS = [
    r"$T_{\rm eff}$ [kK]",
    r"$\log g$ [dex]",
    r"[Fe/H] [dex]",
    r"$\alpha$ [dex]",
    r"$\log_{10}$ age [Gyr]",
    r"$\log_{10}$ distance [kpc]",
]


def display_coordinates(x):
    x = np.asarray(x, dtype="f8").copy()
    if x.shape[1] != 6 or not np.isfinite(x).all() or (x[:, 4:] <= 0).any():
        raise ValueError("Expected six finite targets with positive age/distance")
    x[:, 0] /= 1000
    x[:, 4:] = np.log10(x[:, 4:])
    return x


def corner(data, title, path, *, reference=None, observed=None, limits=None):
    x = display_coordinates(data)
    if limits is None:
        limits = np.quantile(x, [0.002, 0.998], axis=0).T
    fig, axes = plt.subplots(6, 6, figsize=(13, 13))
    for i in range(6):
        for j in range(6):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(
                    x[:, i],
                    bins=60,
                    range=limits[i],
                    histtype="stepfilled",
                    alpha=0.7,
                    color="#28788e",
                    density=True,
                )
                if reference is not None and observed[i]:
                    value = display_coordinates(reference[None])[0, i]
                    ax.axvline(value, color="#ce6c36", linewidth=1.5)
                ax.set_yticks([])
            else:
                ax.hist2d(
                    x[:, j],
                    x[:, i],
                    bins=55,
                    range=[limits[j], limits[i]],
                    norm=LogNorm(),
                    cmap="Blues",
                    cmin=1,
                    rasterized=True,
                )
                if reference is not None and observed[j] and observed[i]:
                    value = display_coordinates(reference[None])[0]
                    ax.plot(value[j], value[i], "+", color="#ce6c36", ms=7)
            ax.set_xlim(limits[j])
            if i != j:
                ax.set_ylim(limits[i])
            ax.tick_params(labelsize=7)
            if i == 5:
                ax.set_xlabel(LABELS[j], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[i], fontsize=9)
            elif i != j:
                ax.set_yticklabels([])
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, y=0.985)
    fig.text(
        0.5,
        0.014,
        "Display: central 99.6% per coordinate; full arrays retained. Log age/distance axes. Colours show sample counts, not calibrated confidence contours.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.075, right=0.98, bottom=0.07, top=0.915, wspace=0.12, hspace=0.12
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outside = ((x < limits[:, 0]) | (x > limits[:, 1])).any(1)
    return dict(
        rows=len(x),
        display_limits=limits.tolist(),
        outside_display_any_axis=int(outside.sum()),
        displayed_pearson_correlation=np.corrcoef(x, rowvar=False).tolist(),
        age_tail_fractions={
            str(t): float((data[:, 4] > t).mean()) for t in [13.8, 20, 100]
        },
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--source-id", type=int, default=4185355646116022528)
    p.add_argument("--runs", nargs="+")
    a = p.parse_args()
    output = a.directory
    runs = a.runs or [
        "decision6_gaussian_parallax_20261106_v1",
        "decision6_mdn_parallax_20261106_v1",
    ]
    result = dict(
        status="running",
        model_loaded=False,
        training_performed=False,
        source_id=a.source_id,
        plots={},
        tail_audit={},
    )
    within = {}
    for rid in runs:
        with np.load(output / (rid + "-whole-vector-corners.npz")) as f:
            star = np.flatnonzero(f["source_id"] == a.source_id)
            if len(star) != 1:
                raise ValueError("Requested source missing or duplicated")
            k = int(star[0])
            within[rid] = (f["native_draws"][:, k], f["reference"][k], f["observed"][k])
            draws = f["native_draws"]
            result["tail_audit"][rid] = [
                dict(
                    source_id=int(sid),
                    age_over_13p8=float((draws[:, i, 4] > 13.8).mean()),
                    age_over_20=float((draws[:, i, 4] > 20).mean()),
                    maximum_age_draw=float(draws[:, i, 4].max()),
                )
                for i, sid in enumerate(f["source_id"])
            ]
    # Identical limits across the two models on this star make shapes comparable.
    limits = np.quantile(
        np.concatenate([display_coordinates(v[0]) for v in within.values()]),
        [0.002, 0.998],
        axis=0,
    ).T
    for rid, (draws, reference, observed) in within.items():
        if "_nsf_" in rid:
            family = (
                rid.removeprefix("decision6_")
                .removesuffix("_20261106_v1")
                .replace("_", " ")
            )
        else:
            family = "Mixture" if "_mdn_" in rid else "Gaussian"
        tag = (
            rid.removeprefix("decision6_")
            .removesuffix("_20261106_v1")
            .replace("_", "-")
        )
        path = output / (
            ("mixture-parallax" if "_mdn_parallax_" in rid else tag)
            + "-within-star-corner.png"
        )
        # Missing catalogue references are NaNs; only marked observed entries
        # are used and coordinate conversion below gets a finite placeholder.
        finite_reference = np.where(
            observed, reference, np.array([5000.0, 3.0, 0.0, 0.0, 1.0, 1.0])
        )
        result["plots"][path.name] = corner(
            draws,
            f"{family} · one-star predictive draws (8,192 whole six-vectors)\nGaia DR3 {a.source_id} · orange: available catalogue references; age/distance are teacher-scale predictions",
            path,
            reference=finite_reference,
            observed=observed,
            limits=limits,
        )
    rid = runs[1]
    with np.load(output / (rid + "-development-predictions.npz")) as f:
        data = f["native_median"]
        ids = f["source_id"]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Population IDs duplicated")
        path = output / "mixture-parallax-population-corner.png"
        result["plots"][path.name] = corner(
            data,
            f"Population relations across {len(ids):,} different stars · six predicted medians per star\nMixture + measured parallax · fixed development panel; these are not one-star uncertainty correlations",
            path,
        )
    result["status"] = "complete_saved_array_plots"
    (output / "corner-plot-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                status=result["status"],
                plots=list(result["plots"]),
                source_id=a.source_id,
                tail_audit=result["tail_audit"],
            )
        )
    )


if __name__ == "__main__":
    main()
