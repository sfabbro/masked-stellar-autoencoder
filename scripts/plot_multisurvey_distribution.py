"""Render stored joint draws and observed-label PIT without loading a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ("Teff [K]", "logg [dex]", "[Fe/H] [dex]")


def render(path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as data:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for d, ax in enumerate(axes):
            for s in np.unique(data["record_source_index"]):
                keep = (data["record_target_index"] == d) & (
                    data["record_source_index"] == s
                )
                ax.hist(
                    data["observed_pit"][keep],
                    bins=np.linspace(0, 1, 11),
                    density=True,
                    histtype="step",
                    label=f"source index {s}",
                )
            ax.axhline(1, color="gray", ls="--")
            ax.set(title=LABELS[d], xlabel="Observed-label PIT", ylabel="Density")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "TUNE diagnostics — declared survey errors and illustrative floors"
        )
        fig.tight_layout()
        fig.savefig(output / "observed-pit.png", dpi=150)
        plt.close(fig)
        for i, source_id in enumerate(data["corner_source_id"][:3]):
            samples = data["joint_draws"][i]
            fig, axes = plt.subplots(3, 3, figsize=(8, 8))
            for row in range(3):
                for col in range(3):
                    ax = axes[row, col]
                    if col > row:
                        ax.axis("off")
                        continue
                    if row == col:
                        ax.hist(
                            samples[:, row],
                            bins=50,
                            density=True,
                            color="#3677ad",
                            alpha=0.7,
                        )
                    else:
                        ax.scatter(
                            samples[::4, col], samples[::4, row], s=1, alpha=0.15
                        )
                    ax.set_xlim(samples[:, col].min(), samples[:, col].max())
                    if row != col:
                        ax.set_ylim(samples[:, row].min(), samples[:, row].max())
                    if row == 2:
                        ax.set_xlabel(LABELS[col])
                    if col == 0 and row:
                        ax.set_ylabel(LABELS[row])
            fig.suptitle(
                f"Gaia {int(source_id)}\nJoint Gaussian draws; relative survey gauge, not physical calibration"
            )
            fig.tight_layout()
            fig.savefig(output / f"corner-{int(source_id)}.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.predictions, args.output)


if __name__ == "__main__":
    main()
