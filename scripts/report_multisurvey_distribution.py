"""Compare SHA-bound saved multi-survey distributions without model execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri

LABELS = ("Teff [K]", "logg [dex]", "[Fe/H] [dex]")
SOURCES = ("APOGEE", "DESI", "GALAH")


def summarize(data, selected):
    residual = data["observed_mean"][selected] - data["reference"][selected]
    std = np.sqrt(data["observed_variance"][selected])
    z = -residual / std
    pit = ndtr(z)
    np.testing.assert_allclose(pit, data["observed_pit"][selected], atol=2e-12)
    crps = std * (
        z * (2 * pit - 1)
        + 2 * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
        - 1 / np.sqrt(np.pi)
    )
    return dict(
        count=int(selected.sum()),
        mae=float(np.abs(residual).mean()),
        bias=float(residual.mean()),
        crps=float(crps.mean()),
        coverage68=float(((pit >= 0.16) & (pit <= 0.84)).mean()),
        coverage90=float(((pit >= 0.05) & (pit <= 0.95)).mean()),
        width90=float((2 * ndtri(0.95) * std).mean()),
    )


def report(root, output):
    output.mkdir(parents=True, exist_ok=True)
    arms, manifests, rows = {}, {}, []
    for arm in ("frozen", "finetune"):
        run = f"multisurvey_gaussian_{arm}_20261104_v1"
        manifest = json.loads((root / (run + "-manifest.json")).read_text())
        if manifest["status"] != "multisurvey_distribution_pilot_complete":
            raise ValueError("Only completed pilots can enter the report")
        manifests[arm] = manifest
        for stage in ("initial", "final"):
            path = root / run / (stage + "-tune.npz")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != manifest["result"][stage]["array_sha256"]
            ):
                raise ValueError("Producer prediction digest differs")
            with np.load(path, allow_pickle=False) as archive:
                data = dict(archive)
            for key in ("source_id", "record_source_id", "corner_source_id"):
                if data[key].dtype != np.int64:
                    raise ValueError("Exact int64 identity arrays required")
            if stage == "initial":
                baseline = data
            else:
                for key in (
                    "source_id",
                    "record_source_id",
                    "record_source_index",
                    "record_target_index",
                    "reference",
                    "corner_source_id",
                ):
                    np.testing.assert_array_equal(baseline[key], data[key])
                arms[arm] = data
                for i, source_id in enumerate(data["corner_source_id"]):
                    position = np.flatnonzero(data["source_id"] == source_id)
                    if len(position) != 1:
                        raise ValueError("Stored corner identity is not unique")
                    j = position[0]
                    draws = data["joint_draws"][i]
                    lower = data["scale_tril"][j]
                    whitened = np.linalg.solve(lower, (draws - data["mean"][j]).T).T
                    np.testing.assert_allclose(whitened.mean(0), 0, atol=0.06)
                    np.testing.assert_allclose(
                        np.cov(whitened, rowvar=False), np.eye(3), atol=0.1
                    )
            for s, source in enumerate(SOURCES):
                for d, label in enumerate(LABELS):
                    keep = (data["record_source_index"] == s) & (
                        data["record_target_index"] == d
                    )
                    if keep.any():
                        rows.append(
                            dict(
                                arm=arm,
                                stage=stage,
                                source=source,
                                target=label,
                                **summarize(data, keep),
                            )
                        )
    for key in (
        "source_id",
        "record_source_id",
        "record_source_index",
        "record_target_index",
        "reference",
        "corner_source_id",
    ):
        np.testing.assert_array_equal(arms["frozen"][key], arms["finetune"][key])
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for d, label in enumerate(LABELS):
        for arm, data in arms.items():
            keep = data["record_target_index"] == d
            pit = data["observed_pit"][keep]
            axes[0, d].hist(
                pit,
                bins=np.linspace(0, 1, 11),
                density=True,
                histtype="step",
                label=arm,
            )
            levels = np.linspace(0.1, 0.98, 23)
            coverage = [
                ((pit >= (1 - p) / 2) & (pit <= (1 + p) / 2)).mean() for p in levels
            ]
            axes[1, d].plot(levels, coverage, label=arm)
        axes[0, d].axhline(1, color="gray", ls="--")
        axes[0, d].set(title=label, xlabel="Observed-label PIT", ylabel="Density")
        axes[1, d].plot([0, 1], [0, 1], color="gray", ls="--")
        axes[1, d].set(
            xlabel="Nominal interval probability", ylabel="Observed-label coverage"
        )
    axes[0, 0].legend()
    fig.suptitle("Matched TUNE stars — survey observations, illustrative error floors")
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)
    corners = []
    for i, source_id in enumerate(arms["frozen"]["corner_source_id"][:3]):
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        all_draws = np.concatenate([data["joint_draws"][i] for data in arms.values()])
        for row in range(3):
            for col in range(3):
                ax = axes[row, col]
                if col > row:
                    ax.axis("off")
                    continue
                for arm, data in arms.items():
                    draws = data["joint_draws"][i]
                    if row == col:
                        ax.hist(
                            draws[:, row],
                            bins=np.linspace(
                                all_draws[:, row].min(), all_draws[:, row].max(), 50
                            ),
                            density=True,
                            histtype="step",
                            label=arm,
                        )
                    else:
                        ax.scatter(
                            draws[::4, col], draws[::4, row], s=1, alpha=0.12, label=arm
                        )
                ax.set_xlim(all_draws[:, col].min(), all_draws[:, col].max())
                if row != col:
                    ax.set_ylim(all_draws[:, row].min(), all_draws[:, row].max())
                if row == 2:
                    ax.set_xlabel(LABELS[col])
                if col == 0 and row:
                    ax.set_ylabel(LABELS[row])
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Gaia {int(source_id)} — 8192 stored joint draws per model\nRelative survey gauge; no physical-calibration claim"
        )
        fig.tight_layout()
        name = f"corner-{int(source_id)}.png"
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        corners.append(name)
    summary = dict(
        scope="Matched existing TUNE only, three targets, approximate first-order survey map and illustrative floors; not latent physical-truth calibration",
        checks=dict(
            producer_sha256="pass",
            exact_int64_ids="pass",
            paired_references="pass",
            independent_pit="pass",
            saved_joint_draw_moments="pass",
        ),
        joint_nll={
            arm: {
                stage: m["result"][stage]["mean_joint_observation_nll"]
                for stage in ("initial", "final")
            }
            for arm, m in manifests.items()
        },
        metrics=rows,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    text = (
        "# Multi-survey joint distribution pilot results\n\n"
        + summary["scope"]
        + ".\n\n"
    )
    text += "| Encoder | Initial joint observation NLL | Final joint observation NLL |\n|---|---:|---:|\n"
    for arm, scores in summary["joint_nll"].items():
        text += f"| {arm} | {scores['initial']:.4f} | {scores['final']:.4f} |\n"
    text += "\nLower NLL is better on this fixed panel. This pilot does not select between Gaussian and flow heads.\n\n"
    text += "![Observed-label calibration](calibration.png)\n\n"
    for name in corners:
        text += f"![Joint samples]({name})\n\n"
    text += "[Per-survey accuracy, CRPS, coverage and width](metrics.csv) · [Summary and checks](summary.json).\n"
    (output / "index.md").write_text(text)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report(args.root, args.output)


if __name__ == "__main__":
    main()
