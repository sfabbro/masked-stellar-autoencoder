"""Paired RC diagnostics after complete independent CPU verification."""

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_prediction_comparison import BASE, digest, score, spectral

RUNS = [
    (BASE / "prediction_e72_huber_10m_v1", "Plain Huber"),
    (BASE / "input-corrections-e72/prediction_e72_rc_shared_xp_10m_v2", "Shared RC"),
    (
        BASE / "input-corrections-e72/prediction_e72_rc_per_star_xp_10m_v2",
        "Per-star RC",
    ),
]
OUT = BASE / "input-corrections-e72/verified-rc-comparison"


def write_csv(name, rows):
    with (OUT / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(kind="rc"):
    global RUNS, OUT
    flow = kind == "flow"
    if flow:
        RUNS = [
            (BASE / "prediction_e72_huber_10m_v1", "Plain Huber"),
            (BASE / "prediction_e72_residual_huber_10m_v1", "Residual Huber"),
            (BASE / "prediction_e72_contrastive_flow_10m_v1", "Contrastive flow"),
        ]
        OUT = BASE / "prediction_e72_contrastive_flow_10m_v1/verified-comparison"
    prefix = "contrastive-flow" if flow else "rc"
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, snapshots, recon, checks = [], [], [], []
    reference = None
    for directory, label in RUNS:
        manifest = json.loads((directory / "manifest.json").read_text())
        report = json.loads((directory / "independent-verification.json").read_text())
        assert manifest["status"] == "complete" and not manifest["smoke"]
        assert report["status"] == "passed"
        assert report["final_checkpoint_sha256"] == digest(directory / "checkpoint.pth")
        assert (
            manifest["presentations"] == 10000000
            and manifest["optimizer_updates"] == 306
        )
        if "input_correction" in manifest["arm"]:
            assert report["corrected_inputs_independently_derived"]
        checks.append(report)
        with np.load(directory / "paired-cohort.npz", allow_pickle=False) as saved:
            arrays = dict(saved)
        with np.load(
            directory / "initial_reconstruction.npz", allow_pickle=False
        ) as saved:
            initial = dict(saved)
        if reference is None:
            reference = manifest, arrays, initial
        else:
            for key in [
                "parent_sha256",
                "cohort_sha256",
                "row_sha256",
                "mask_sha256",
                "supervised_stream_sha256",
            ]:
                assert reference[0][key] == manifest[key], key
            assert set(arrays) == set(reference[1])
            for key in arrays:
                if flow or key not in {"fit_input", "test_input"}:
                    assert (
                        arrays[key].dtype == reference[1][key].dtype
                        and arrays[key].tobytes() == reference[1][key].tobytes()
                    ), key
            for key in initial:
                assert initial[key].tobytes() == reference[2][key].tobytes(), key
        y = arrays["test_measurements"]["value"]
        surveys = arrays["test_measurements"]["survey"].astype(str)
        masks = {"all": np.ones(len(y), bool), "below_-3": y < -3, "below_-2": y < -2}
        for low, high in [(-3, -2), (-2, -1), (-1, 0), (0, np.inf)]:
            masks[f"[{low},{high})"] = (y >= low) & (y < high)
        masks.update({f"survey:{s}": surveys == s for s in np.unique(surveys)})
        masks.update(
            {
                f"survey:{s}:below_-2": (surveys == s) & (y < -2)
                for s in np.unique(surveys)
            }
        )
        apogal = np.isin(surveys, ["apogee", "galah"])
        masks["apogee_galah_only"] = apogal
        masks["apogee_galah_below_-2"] = apogal & (y < -2)
        with np.load(directory / "predictions.npz") as saved:
            prediction = saved["prediction"]
        np.testing.assert_allclose(
            score(y, prediction)["mae_dex"], report["mae_dex"], rtol=1e-12
        )
        for subset, keep in masks.items():
            if keep.any():
                metrics.append(
                    dict(arm=label, subset=subset, **score(y[keep], prediction[keep]))
                )
        files = sorted(
            directory.glob("snapshot-*.npz"), key=lambda f: int(f.stem.split("-")[-1])
        )
        assert len(files) == 5
        for file in files:
            with np.load(file) as saved:
                raw_point = (
                    saved["standardized_mle"]
                    if "standardized_mle" in saved
                    else saved["member_outputs"][:, :, 0].mean(1)
                )
                point = raw_point * arrays["label_scale"] + arrays["label_center"]
            for subset in ["all", "below_-2"]:
                keep = masks[subset]
                snapshots.append(
                    dict(
                        arm=label,
                        subset=subset,
                        presentations=int(file.stem.split("-")[-1]),
                        **score(y[keep], point[keep]),
                    )
                )
        with np.load(directory / "final_reconstruction.npz") as saved:
            final_reco = saved["prediction"]
        for stage, values in [
            ("initial", initial["prediction"]),
            ("final", final_reco),
        ]:
            width = values.shape[1]
            target = initial["target"][:, :width]
            keep = initial["mask"][:, :width] & np.isfinite(target)
            mae = float(abs(values.astype("f8") - target.astype("f8"))[keep].mean())
            spec = spectral(initial, values)
            declared = manifest["comparison"]["reconstruction"][stage]
            np.testing.assert_allclose(mae, declared["masked_mae"], rtol=2e-7)
            np.testing.assert_allclose(
                spec["spectral_fraction_mean"],
                declared["spectral"]["integrated_absolute_fraction_mean"],
                rtol=2e-7,
            )
            recon.append(
                dict(arm=label, stage=stage, masked_mae_standardized=mae, **spec)
            )
    write_csv("conditional-and-survey-metrics.csv", metrics)
    write_csv("snapshot-metrics.csv", snapshots)
    write_csv("reconstruction-and-spectral-metrics.csv", recon)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    labels = [label for _, label in RUNS]
    colors = ["#687684", "#cb7c38", "#167b83"]
    for ax, subset, title in [
        (axs[0, 0], "all", "All 100,620 reported native labels"),
        (axs[0, 1], "below_-2", "Reported FeH < −2: 185 labels"),
    ]:
        vals = [
            next(
                r["mae_dex"]
                for r in metrics
                if r["arm"] == label and r["subset"] == subset
            )
            for label in labels
        ]
        ax.barh(labels, vals, color=colors)
        ax.invert_yaxis()
        ax.set(
            xlabel="Mean absolute error [dex]", title=title, xlim=(0, max(vals) * 1.2)
        )
        for i, value in enumerate(vals):
            ax.text(value, i, f" {value:.4f}", va="center")
    for label, color in zip(labels, colors, strict=True):
        curve = [
            r
            for r in snapshots
            if r["arm"] == label and r["subset"] == "all" and r["presentations"] > 0
        ]
        axs[1, 0].plot(
            [r["presentations"] / 1e6 for r in curve],
            [r["mae_dex"] for r in curve],
            marker="o",
            label=label,
            color=color,
        )
        r = [r for r in recon if r["arm"] == label]
        axs[1, 1].plot(
            [0, 10],
            [r[0]["spectral_fraction_mean"], r[1]["spectral_fraction_mean"]],
            marker="o",
            label=label,
            color=color,
        )
    axs[1, 0].set(
        xlabel="Reconstruction presentations [million]",
        ylabel="Native-label MAE [dex]",
        title="Saved adaptation snapshots",
    )
    axs[1, 1].set(
        xlabel="Reconstruction presentations [million]",
        ylabel="Integrated spectral absolute fraction",
        title="Fixed 10,000-row validation diagnostic",
    )
    axs[1, 0].legend()
    axs[1, 1].legend()
    fig.suptitle(
        "Epoch 72 · contrastive likelihood-grid flow · one seed"
        if flow
        else "Epoch 72 · diagonal XP regression calibration · one seed"
    )
    fig.savefig(OUT / (prefix + "-comparison.png"), dpi=170)
    fig.savefig(OUT / (prefix + "-comparison.pdf"))
    plt.close(fig)
    allrows = [r for r in metrics if r["subset"] == "all"]
    tails = [r for r in metrics if r["subset"] == "below_-2"]
    lines = [
        "# Independently verified RC comparison",
        "",
        "Both RC checkpoints passed independent raw-source/input-correction checks and full CPU replay of all100620 test heads and10000 validation decoder rows. Plain Huber uses its already completed independent verification. Each arm has the same epoch72 parent, fit/test identities, label scaling, reconstruction/label streams and common GPU masks. Source differs because the corrected adapters were published later; corrected XP inputs intentionally differ. Masks were compared across runs, not independently CPU regenerated.",
        "",
        "| Arm | Population MAE [dex] | Below −2 MAE [dex] | Below −2 bias [dex] |",
        "|---|---:|---:|---:|",
    ]
    for row, tail in zip(allrows, tails, strict=True):
        lines.append(
            f"| {row['arm']} | {row['mae_dex']:.8f} | {tail['mae_dex']:.8f} | {tail['bias_dex']:+.8f} |"
        )
    lines += [
        "",
        "Neither RC arm improves population observed-label MAE over plain Huber at this budget. Tail scores and survey-separated diagnostics are descriptive checks against noisy reported labels; they do not establish recovery of latent physical truth or removal of regression-to-mean. The native registry includes the previously retained Li-VMPS layer (39 of185 below−2 test labels); APOGEE/GALAH-only metrics are separately preserved.",
        "",
        "Diagonal XP uncertainty uses raw reported errors divided by G flux normalization and parent scale. Native XP correlations, shared G uncertainty and label-systematic deconvolution are omitted. Gaussian fit-only RC assumptions and finite-budget neural retraining do not guarantee unbiased nonlinear prediction. No new catalogues or silver label merge. Huber variance channels are untrained; no calibrated predictive PDF or frequentist confidence claim. One seed; source-ID supervised holdout does not exclude unsupervised overlap. SIMEX is still running and is excluded from this report. Full17-model CPU verification is required for its later aggregate.",
        "",
        "CSV files preserve all conditional, survey, snapshot, reconstruction and spectral diagnostics. Figure visual inspection pending.",
    ]
    if flow:
        lines[0] = "# Independently verified contrastive flow comparison"
        lines[2] = (
            "The flow passed all100620 initial and final likelihood-grid CPU replays with independently label-major score enumeration and all10000 final decoder rows. Raw source fields, all paired inputs/measurements, fit-only label scaling, row/label streams, parent and shared GPU masks checked. Plain and residual Huber reuse completed independent reports. Supervised holdout does not exclude unsupervised pretraining overlap; one seed."
        )
        lines[-5] = (
            "Conditional/survey scores describe agreement with noisy reported native labels. The native registry includes Li-VMPS (39 of185 below−2 stars); separate APOGEE/GALAH-only diagnostics are saved. No latent physical truth or attenuation-removal claim follows from mean error alone."
        )
        lines[-3] = (
            "This is p(learned8-dimensional summary|FeH) with201-grid likelihood maximization; it is not q(FeH|data), a calibrated physical PDF, or a frequentist confidence interval. The summary is learned jointly, and density estimation, representation quality and grid resolution all affect performance. No new catalogues or silver merge. SIMEX and the incomplete other-arm QA are excluded."
        )
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    (OUT / "independent-verification.json").write_text(
        json.dumps(
            dict(
                status="passed",
                reports=checks,
                paired_noninput_arrays_byte_equal=True,
                initial_reconstruction_byte_equal=True,
                visual_check="pending",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(population=allrows, below_minus_two=tails, output=str(OUT)), indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["rc", "flow"], default="rc")
    main(parser.parse_args().kind)
