"""Plot recorded epoch metrics and paired pilot curves; no fitted/smoothed losses."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/xp/science-review-20261015"
OUT.mkdir(parents=True, exist_ok=True)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    data = rows(ROOT / "monitor/main-status/metrics-latest.jsonl")
    epochs = [r for r in data if r.get("record_type") == "epoch"]
    x = np.array([r["epoch"] for r in epochs])
    assert len(set(x)) == len(x)
    lr = np.array([r["lr"] for r in epochs])
    # Epoch metrics log scheduler state AFTER scheduler.step(): LR for next epoch.
    expected = []
    for e in x:
        start, length = 0, 10
        while e >= start + length:
            start += length
            length *= 2
        expected.append(
            1e-7 + (1e-5 - 1e-7) * (1 + np.cos(np.pi * (e - start) / length)) / 2
        )
    np.testing.assert_allclose(lr, expected, rtol=1e-10)
    fig, ax = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, layout="constrained", height_ratios=[2, 1]
    )
    for key, label, color in [
        ("train_loss", "Training MAE (training mode)", "#2563eb"),
        ("val_loss", "Full validation MAE (evaluation mode)", "#c2410c"),
    ]:
        ax[0].plot(x, [r[key] for r in epochs], marker=".", label=label, color=color)
    ax[0].set(
        ylabel="Scaled hidden-feature MAE",
        title="Standard MSA: recorded losses and learning rate, epochs 24–62",
    )
    ax[0].legend(frameon=False)
    ax[1].semilogy(x, lr, marker=".", color="#7c3aed")
    ax[1].set(
        xlabel="Completed epoch",
        ylabel="LR after epoch\n(next epoch setting)",
        ylim=(8e-8, 1.3e-5),
    )
    for a in ax:
        a.axvline(30, color="0.5", ls="--", lw=1)
        a.grid(alpha=0.18)
    ax[1].annotate(
        "Cosine restart",
        xy=(30, 1e-5),
        xytext=(34, 6e-6),
        arrowprops={"arrowstyle": "->", "color": "0.5"},
    )
    fig.savefig(OUT / "main-loss-lr.png", dpi=180)
    fig.savefig(OUT / "main-loss-lr.pdf")
    plt.close(fig)
    with (OUT / "main-epoch-metrics.csv").open("w") as f:
        keys = ["epoch", "timestamp_utc", "train_loss", "val_loss", "lr"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in epochs)
    names = [
        ("control_s42", "Ordinary MAE"),
        ("spectral_s42", "Spectral 0.1"),
        ("spectral_lr05_s42", "Spectral 0.1, half LR"),
        ("spectral_lr125_s42", "Spectral 0.1, 1.25× LR"),
    ]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for name, label in names:
        rr = rows(
            ROOT / "experiments/xp/results-followup-e47" / name / name / "metrics.jsonl"
        )
        xx = np.array([r["rows_seen_total"] / 1e6 for r in rr])
        ax[0].plot(xx, [r["xp_only_mae"] for r in rr], marker="o", label=label)
        tt = [r for r in rr if r.get("train_loss") is not None]
        ax[1].plot(
            [r["rows_seen_total"] / 1e6 for r in tt],
            [r["train_loss"] for r in tt],
            marker="o",
            label=label,
        )
        ax[2].plot(xx, [r["learning_rate"] for r in rr], marker="o", label=label)
    ax[0].set(
        ylabel="Validation XP MAE",
        title="Paired epoch-47 pilots: recorded validation, objective, and LR",
    )
    ax[0].legend(frameon=False, ncol=2)
    ax[1].set(
        ylabel="Recorded training objective",
        title="Objectives differ: spectral runs include an auxiliary term",
    )
    ax[2].set(
        xlabel="Training presentations (millions)",
        ylabel="Learning rate",
        ylim=(0, 9e-6),
    )
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for a in ax:
        a.grid(alpha=0.18)
    fig.savefig(OUT / "pilot-loss-lr.png", dpi=180)
    fig.savefig(OUT / "pilot-loss-lr.pdf")
    plt.close(fig)
    summary = {
        "epoch_range": [int(x[0]), int(x[-1])],
        "last_epoch": epochs[-1],
        "last_interval": next(
            r for r in reversed(data) if r.get("record_type") == "interval"
        ),
        "scheduler_verified": True,
        "lr_convention": "Logged after scheduler.step; next epoch setting",
        "next_nominal_restart_epoch": 70,
        "validation_change_percent": 100
        * (epochs[-1]["val_loss"] / epochs[0]["val_loss"] - 1),
        "train_change_percent": 100
        * (epochs[-1]["train_loss"] / epochs[0]["train_loss"] - 1),
        "notes": [
            "Main and pilot datasets/objectives differ; do not compare their loss levels directly.",
            "Pilot logs have five evaluation snapshots; connecting segments do not imply unrecorded measurements.",
            "Training objectives include dropout and changing batches, and cannot be directly compared across different auxiliary objectives.",
        ],
    }
    (OUT / "loss-lr-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "epoch_range",
                    "validation_change_percent",
                    "train_change_percent",
                    "scheduler_verified",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
