"""Compare SHA-bound saved multi-survey distributions without model execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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
    arms, manifests, rows, seen_ids = {}, {}, [], {}
    for arm in ("frozen", "finetune"):
        run = f"multisurvey_gaussian_{arm}_20261104_v1"
        manifest = json.loads((root / (run + "-manifest.json")).read_text())
        if (
            manifest["status"] != "multisurvey_distribution_pilot_complete"
            or manifest["result"].get("checkpoint_reload") != "pass"
            or manifest["result"].get("updates") != 1000
        ):
            raise ValueError("Only completed pilots can enter the report")
        manifests[arm] = manifest
        shutil.copyfile(
            root / (run + "-manifest.json"), output / (arm + "-manifest.json")
        )
        seen_ids[arm] = np.load(
            root / run / "fit-seen-source-ids.npy", allow_pickle=False
        )
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
    np.testing.assert_array_equal(seen_ids["frozen"], seen_ids["finetune"])
    for arm, data in arms.items():
        active_ids = data["source_id"][
            np.isin(data["source_id"], data["record_source_id"])
        ]
        if len(active_ids) != len(data["per_star_nll"]):
            raise ValueError(
                "Per-star likelihoods do not align with observed identities"
            )
        np.testing.assert_allclose(
            data["per_star_nll"].mean(),
            manifests[arm]["result"]["final"]["mean_joint_observation_nll"],
            rtol=0,
            atol=1e-12,
        )
    differences = arms["finetune"]["per_star_nll"] - arms["frozen"]["per_star_nll"]
    rng = np.random.default_rng(20261104)
    bootstrapped = differences[
        rng.integers(0, len(differences), size=(2000, len(differences)))
    ].mean(1)
    paired_nll = dict(
        mean=float(differences.mean()),
        paired_star_bootstrap_95=np.quantile(bootstrapped, [0.025, 0.975]).tolist(),
        stars=len(differences),
        scope="Fine-tuned minus frozen on this existing TUNE panel; uncertainty over stars only, not training seeds, source systematics or new populations",
    )
    tails = {}
    for arm, data in arms.items():
        tails[arm] = {}
        for name, keep in (
            (
                "native_teff_ge_8000K",
                (data["record_target_index"] == 0) & (data["reference"] >= 8000),
            ),
            (
                "native_feh_lt_minus1p5",
                (data["record_target_index"] == 2) & (data["reference"] < -1.5),
            ),
        ):
            tails[arm][name] = summarize(data, keep) if keep.any() else {"count": 0}
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
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for s, source in enumerate(SOURCES):
        for d, label in enumerate(LABELS):
            ax = axes[s, d]
            for arm, data in arms.items():
                keep = (data["record_source_index"] == s) & (
                    data["record_target_index"] == d
                )
                if keep.any():
                    ax.hist(
                        data["observed_pit"][keep],
                        bins=np.linspace(0, 1, 11),
                        density=True,
                        histtype="step",
                        label=arm,
                    )
                count = int(keep.sum())
            ax.axhline(1, color="gray", ls="--")
            ax.set_title(f"{source}: {label}, n={count}")
            ax.set(xlabel="Observed-label PIT", ylabel="Density")
    axes[0, 0].legend()
    fig.suptitle("Survey-specific TUNE diagnostics — illustrative error floors")
    fig.tight_layout()
    fig.savefig(output / "survey-pit.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (arm, data) in enumerate(arms.items()):
        for d, label in enumerate(LABELS):
            ax = axes[row, d]
            for s, source in enumerate(SOURCES):
                keep = (data["record_source_index"] == s) & (
                    data["record_target_index"] == d
                )
                ax.scatter(
                    data["reference"][keep],
                    data["observed_mean"][keep],
                    s=8,
                    alpha=0.4,
                    label=source,
                )
            values = np.concatenate(
                [
                    np.concatenate(
                        [
                            item["reference"][item["record_target_index"] == d],
                            item["observed_mean"][item["record_target_index"] == d],
                        ]
                    )
                    for item in arms.values()
                ]
            )
            limits = (values.min(), values.max())
            ax.plot(limits, limits, color="gray", ls="--")
            ax.set(
                xlim=limits,
                ylim=limits,
                xlabel=f"Native source {label}",
                ylabel=f"Mapped prediction {label}",
                title=arm,
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Source-space predictions — all stored TUNE references, no axis clipping"
    )
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=160)
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
            identical_unique_fit_exposure="pass",
        ),
        joint_nll={
            arm: {
                stage: m["result"][stage]["mean_joint_observation_nll"]
                for stage in ("initial", "final")
            }
            for arm, m in manifests.items()
        },
        metrics=rows,
        paired_final_nll_difference=paired_nll,
        tail_quicklook=tails,
        checkpoints={
            arm: dict(
                path="/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
                + m["run_id"]
                + "/final.pth",
                sha256=m["result"]["checkpoint_sha256"],
                reload=m["result"]["checkpoint_reload"],
            )
            for arm, m in manifests.items()
        },
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
    text += "Initial scores use freshly initialized three-target heads, not the inherited Fe/H head. Joint Gaussian samples are unimodal and do not resolve arbitrary multimodal stellar degeneracies.\n\n"
    low, high = paired_nll["paired_star_bootstrap_95"]
    text += f"Paired final NLL difference (fine-tuned minus frozen): {paired_nll['mean']:.4f}; star-bootstrap 95% interval [{low:.4f}, {high:.4f}]. This interval excludes training-seed and survey-systematic uncertainty.\n\n"
    text += "The full-range prediction panels retain large hot-star and low-metallicity residuals. Small source-reference tail counts and descriptive scores are preserved in `summary.json` (`tail_quicklook`); they do not establish tail calibration.\n\n"
    text += "![Observed-label calibration](calibration.png)\n\n"
    text += "![PIT by survey](survey-pit.png)\n\n"
    text += "![Native source prediction comparison](predictions.png)\n\n"
    for name in corners:
        text += f"![Joint samples]({name})\n\n"
    text += "[Per-survey accuracy, CRPS, coverage and width](metrics.csv) · [Summary and checks](summary.json).\n"
    text += "\n[Original frozen-run manifest](frozen-manifest.json) · [Original fine-tuned-run manifest](finetune-manifest.json). Checkpoint paths and SHA256 hashes are in the summary.\n"
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
