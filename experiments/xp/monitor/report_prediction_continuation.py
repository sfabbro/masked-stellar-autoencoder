"""Report the new final CPU-verified continuation; intermediate curves are GPU artifacts."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_prediction_comparison import BASE, digest
from report_verified_mechanisms import (
    conditional_masks,
    reconstruction,
    score_rows,
    write_csv,
)


def main():
    path = BASE / "residual_huber_long_30e_v1"
    verification = json.loads((path / "independent-verification.json").read_text())
    assert verification["status"] == "passed"
    assert verification["final_checkpoint_sha256"] == digest(path / "checkpoint.pth")
    manifest = json.loads((path / "manifest.json").read_text())
    paired = np.load(path / "paired-cohort.npz", allow_pickle=False)
    y = paired["test_measurements"]["value"]
    masks = conditional_masks(paired["test_measurements"])
    center, scale = float(paired["label_center"]), float(paired["label_scale"])
    out = path / "verified-comparison"
    out.mkdir(exist_ok=True)
    curves = []
    for presentations in [0] + list(range(9000000, 270000001, 9000000)):
        z = np.load(path / f"snapshot-{presentations}.npz", allow_pickle=False)
        point = z["member_outputs"][:, :, 0].mean(1) * scale + center
        curves += score_rows(
            "continued residual Huber",
            y,
            point,
            masks,
            presentations=presentations,
            independently_replayed_model=presentations == 270000000,
        )
    final = np.load(path / "predictions.npz", allow_pickle=False)["prediction"]
    initial = (
        np.load(path / "initial_predictions.npz", allow_pickle=False)["member_outputs"][
            :, :, 0
        ].mean(1)
        * scale
        + center
    )
    rows = score_rows("initial verified 10M winner", y, initial, masks) + score_rows(
        "final additional 270M", y, final, masks
    )
    reco = np.load(path / "initial_reconstruction.npz", allow_pickle=False)
    reco_rows = reconstruction(
        "continuation", path, reco, manifest["comparison"]["reconstruction"]
    )
    write_csv(out, "conditional-metrics.csv", rows)
    write_csv(out, "snapshot-metrics.csv", curves)
    write_csv(out, "reconstruction-spectral-metrics.csv", reco_rows)
    (out / "independent-verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, subset, title in [
        (axes[0, 0], "all", "Population"),
        (axes[0, 1], "below_-2", "Measured FeH < −2"),
    ]:
        r = [r for r in curves if r["subset"] == subset]
        ax.plot(
            [x["presentations"] / 1e6 for x in r],
            [x["mae_dex"] for x in r],
            marker=".",
            color="#215d7c",
        )
        ax.set(
            title=title,
            xlabel="Additional reconstruction presentations (million)",
            ylabel="Observed-label MAE (dex)",
        )
        ax.grid(alpha=0.2)
    for ax, key, title in [
        (axes[1, 0], "masked_mae_standardized", "Fixed masked reconstruction"),
        (axes[1, 1], "spectral_fraction_mean", "Fixed XP spectral diagnostic"),
    ]:
        ax.bar(
            ["Initial", "Final"],
            [r[key] for r in reco_rows],
            color=["#8ca6b3", "#215d7c"],
        )
        ax.set(
            title=title,
            ylabel="Standardized feature MAE"
            if key == "masked_mae_standardized"
            else "Integrated absolute fraction",
        )
    fig.suptitle("Residual-Huber continuation: final model independently replayed")
    fig.text(
        0.01,
        0.01,
        "Intermediate curves: saved GPU snapshots, not independent replay of every intermediate model. Same reused test set; no test-snapshot selection.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(out / "continuation-comparison.png", dpi=170)
    fig.savefig(out / "continuation-comparison.pdf")
    plt.close(fig)
    lines = [
        "# Verified final continuation comparison",
        "",
        "Actual winner model/head/both Adam states continued for 270M additional presentations (30 passes of the same 9M TRAIN rows); 8,250 new updates. No fresh unique-star or new-seed-pretraining claim.",
        "",
        "Final 100,620-star head and 10,000-row decoder CPU replay passed. Initial artifacts are byte-identical to the previously verified winner. Raw inputs, label/row streams, optimizer step deltas and all 30 checkpoint receipts audited.",
        "",
        "Intermediate learning curves are saved GPU predictions; they were not used to choose a tail-test checkpoint and their models were not all independently replayed. Native measured labels, survey dependencies, single seed and unsupervised overlap limits remain.",
    ]
    for stage in ["initial verified 10M winner", "final additional 270M"]:
        for subset in ["all", "below_-2"]:
            row = next(r for r in rows if r["arm"] == stage and r["subset"] == subset)
            lines.append(
                f"\n{stage}, {subset}: n={row['count']}, MAE={row['mae_dex']:.8f} dex, bias={row['bias_dex']:.8f} dex."
            )
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
