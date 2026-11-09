"""Visualize paired seismic accuracy and calibration from completed saved QA."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    d = json.loads((a.directory / "summary.json").read_text())
    runs = {x["run_id"]: x for x in d["runs"]}
    order = [
        ("gaussian_photxp", "Gaussian · no parallax"),
        ("gaussian_parallax", "Gaussian · parallax"),
        ("mdn_photxp", "Mixture · no parallax"),
        ("mdn_parallax", "Mixture · parallax"),
        ("nsf_energy_photxp", "NSF 1e-3 · no parallax"),
        ("nsf_energy_parallax", "NSF 1e-3 · parallax"),
        ("nsf_energy_lr3e4_photxp", "NSF 3e-4 · no parallax"),
        ("nsf_energy_lr3e4_parallax", "NSF 3e-4 · parallax"),
    ]
    data = [runs["decision6_" + k + "_20261106_v1"] for k, label in order]
    colors = [
        "#9aa6b2",
        "#52778d",
        "#9cb6c2",
        "#247c9b",
        "#d9b98c",
        "#b96b24",
        "#ccb5cf",
        "#8b6390",
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1.2]},
        layout="constrained",
    )
    y = np.arange(8)
    mae = [x["seismic_logg_mae"] for x in data]
    cov = [100 * x["seismic_coverage90"] for x in data]
    axes[0].barh(y, mae, color=colors)
    axes[1].barh(y, cov, color=colors)
    for i, (m, c) in enumerate(zip(mae, cov)):
        axes[0].text(m + 0.001, i, f"{m:.3f}", va="center", fontsize=9)
        axes[1].text(c + 1, i, f"{c:.1f}%", va="center", fontsize=9)
    axes[0].set(
        yticks=y,
        yticklabels=[label for k, label in order],
        xlim=(0, 0.12),
        xlabel="log g MAE [dex] · lower is better",
    )
    axes[0].invert_yaxis()
    axes[1].axvline(90, color="#333333", linestyle="--", linewidth=1)
    axes[1].set(xlim=(0, 108), xlabel="Nominal 90% coverage [%] · target = 90")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Same 734 seismic development stars · all eight training arms complete\nQuoted seismic errors included in coverage; calibration pedigree and prior exposure remain",
        fontsize=12,
    )
    fig.savefig(a.directory / "seismic-accuracy-and-coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
