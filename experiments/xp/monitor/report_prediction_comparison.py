"""Independent paired reporting after all six production CPU verifiers pass."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

ROOT = Path("/Users/fabbros/src/sfabbro/masked-stellar-autoencoder")
BASE = ROOT / "experiments/xp/results-stellar-improvement"
OUT = BASE / "prediction-comparison-e72/verified-comparison"
ARMS = [
    "huber",
    "gaussian_nll",
    "gaussian_crps",
    "huber_rnc",
    "residual_huber",
    "batchensemble_huber",
]
LABELS = [
    "MLP Huber",
    "Gaussian NLL",
    "Gaussian CRPS",
    "Huber + RnC",
    "Residual Huber",
    "BatchEnsemble Huber",
]


def score(y, p):
    e = p.astype("f8") - y.astype("f8")
    return dict(
        count=len(e),
        mae_dex=float(np.mean(abs(e))),
        bias_dex=float(np.mean(e)),
        rmse_dex=float(np.sqrt(np.mean(e * e))),
        median_absolute_error_dex=float(np.median(abs(e))),
    )


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def spectral(a, pred):
    width = pred.shape[1]
    y = a["target"][:, :width].astype("f8")
    m = a["mask"][:, :width]
    basis = a["spectral_basis"]
    xp = np.any(basis != 0, axis=1)
    valid = np.isfinite(y[:, xp]).all(1) & m[:, xp].any(1)
    y = y[valid][:, xp]
    pred = pred[valid][:, xp].astype("f8")
    m = m[valid][:, xp]
    sc = a["scaler_scale"][:width][xp]
    c = a["scaler_center"][:width][xp]
    wave = a["wavelength_nm"]
    b = basis[xp]
    flux = (y * sc + c) @ b
    residual = (np.where(m, pred - y, 0) * sc) @ b
    denominator = np.trapezoid(abs(flux), wave, axis=1)
    assert np.all(denominator > 0)
    fraction = np.trapezoid(abs(residual), wave, axis=1) / denominator
    return dict(
        spectral_rows=int(valid.sum()),
        spectral_fraction_mean=float(fraction.mean()),
        spectral_fraction_median=float(np.median(fraction)),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    gaussian = []
    reco = []
    checks = []
    snapshots = []
    reference = None
    shared_keys = [
        "source_sha",
        "parent_sha256",
        "cohort_sha256",
        "row_sha256",
        "mask_sha256",
        "supervised_stream_sha256",
    ]
    for arm, label in zip(ARMS, LABELS, strict=True):
        run = f"prediction_e72_{arm}_10m_v1"
        d = BASE / run
        report = json.loads((d / "independent-verification.json").read_text())
        manifest = json.loads((d / "manifest.json").read_text())
        assert (
            report["status"] == "passed"
            and manifest["status"] == "complete"
            and not manifest["smoke"]
        )
        assert (
            manifest["presentations"] == 10000000
            and manifest["optimizer_updates"] == 306
        )
        assert report["final_checkpoint_sha256"] == digest(d / "checkpoint.pth")
        a = dict(np.load(d / "paired-cohort.npz", allow_pickle=False))
        initial = dict(np.load(d / "initial_reconstruction.npz", allow_pickle=False))
        final = dict(np.load(d / "predictions.npz", allow_pickle=False))
        assert np.all(np.load(d / "initial_predictions.npz")["member_outputs"] == 0)
        if reference is None:
            reference = (manifest, a, initial)
        else:
            for k in shared_keys:
                assert reference[0][k] == manifest[k], (run, k)
            for k in a:
                # HDF5 attributes live in the original raw H5, not this NPZ.
                assert (
                    reference[1][k].dtype == a[k].dtype
                    and reference[1][k].tobytes() == a[k].tobytes()
                ), (run, k)
            for k in initial:
                assert (
                    reference[2][k].dtype == initial[k].dtype
                    and reference[2][k].tobytes() == initial[k].tobytes()
                ), (run, k)
        y = a["test_measurements"]["value"]
        pred = final["prediction"]
        assert len(y) == 100620 and np.isfinite(pred).all()
        masks = {
            "all": np.ones(len(y), bool),
            "below_-3": y < -3,
            "[-3,-2)": (y >= -3) & (y < -2),
            "[-2,-1)": (y >= -2) & (y < -1),
            "[-1,0)": (y >= -1) & (y < 0),
            "[0,inf)": y >= 0,
            "below_-2": y < -2,
        }
        surveys = a["test_measurements"]["survey"].astype(str)
        masks["apogee_galah_only"] = np.isin(surveys, ["apogee", "galah"])
        masks["apogee_galah_below_-2"] = masks["apogee_galah_only"] & (y < -2)
        masks.update({f"survey:{s}": surveys == s for s in np.unique(surveys)})
        for name, keep in masks.items():
            if keep.any():
                rows.append(
                    dict(
                        run_id=run, arm=label, subset=name, **score(y[keep], pred[keep])
                    )
                )
        np.testing.assert_allclose(
            score(y, pred)["mae_dex"], report["mae_dex"], rtol=1e-12
        )
        for n in sorted(
            d.glob("snapshot-*.npz"), key=lambda p: int(p.stem.split("-")[-1])
        ):
            saved = np.load(n)
            point = (
                saved["member_outputs"][:, :, 0].mean(1) * a["label_scale"]
                + a["label_center"]
            )
            for name in ["all", "below_-2"]:
                keep = masks[name]
                snapshots.append(
                    dict(
                        arm=label,
                        presentations=int(n.stem.split("-")[-1]),
                        subset=name,
                        **score(y[keep], point[keep]),
                    )
                )
        if arm in ["gaussian_nll", "gaussian_crps"]:
            mu = (
                final["member_outputs"][:, 0, 0].astype("f8") * a["label_scale"]
                + a["label_center"]
            )
            sigma = (
                np.exp(final["member_outputs"][:, 0, 1].astype("f8") / 2)
                * a["label_scale"]
            )
            assert np.isfinite(sigma).all() and np.all(sigma > 0)
            z = (y - mu) / sigma
            phi = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
            crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / np.sqrt(np.pi))
            nll = 0.5 * z * z + np.log(sigma) + 0.5 * np.log(2 * np.pi)
            np.testing.assert_allclose(
                [crps.mean(), nll.mean()],
                [report["gaussian_crps_dex"], report["gaussian_nll_dex"]],
                rtol=1e-10,
            )
            for name, keep in masks.items():
                if not keep.any():
                    continue
                r = dict(
                    arm=label,
                    subset=name,
                    count=int(keep.sum()),
                    crps_dex=float(crps[keep].mean()),
                    nll_density_dex=float(nll[keep].mean()),
                    mean_sigma_dex=float(sigma[keep].mean()),
                )
                for level in [0.68, 0.95]:
                    q = NormalDist().inv_cdf((1 + level) / 2)
                    r[f"coverage_{int(level * 100)}"] = float(
                        (abs(z[keep]) <= q).mean()
                    )
                    r[f"mean_width_{int(level * 100)}_dex"] = float(
                        (2 * q * sigma[keep]).mean()
                    )
                gaussian.append(r)
        for stage, p in [
            ("initial", initial["prediction"]),
            ("final", np.load(d / "final_reconstruction.npz")["prediction"]),
        ]:
            width = p.shape[1]
            yrec = initial["target"][:, :width]
            scored = initial["mask"][:, :width] & np.isfinite(yrec)
            mae = float(np.abs(p.astype("f8") - yrec.astype("f8"))[scored].mean())
            s = spectral(initial, p)
            ref = manifest["comparison"]["reconstruction"][stage]
            np.testing.assert_allclose(mae, ref["masked_mae"], rtol=2e-7)
            # Independent subtraction is float64; production subtracts float32 arrays.
            np.testing.assert_allclose(
                s["spectral_fraction_mean"],
                ref["spectral"]["integrated_absolute_fraction_mean"],
                rtol=2e-7,
            )
            reco.append(
                dict(
                    arm=label,
                    stage=stage,
                    validation_rows=len(p),
                    masked_mae_standardized=mae,
                    spectral_absolute_difference_from_production=abs(
                        s["spectral_fraction_mean"]
                        - ref["spectral"]["integrated_absolute_fraction_mean"]
                    ),
                    **s,
                )
            )
        checks.append(report)
    write_csv("conditional-metrics.csv", rows)
    write_csv("gaussian-diagnostics.csv", gaussian)
    write_csv("reconstruction-metrics.csv", reco)
    write_csv("snapshot-metrics.csv", snapshots)
    (OUT / "independent-verification.json").write_text(
        json.dumps(
            dict(
                status="passed",
                all_six_full_cpu_reports=checks,
                shared_hashes={k: reference[0][k] for k in shared_keys},
                all_paired_arrays_and_initial_reconstructions_byte_equal=True,
                independent_scores_recomputed=True,
                visual_check="pending",
            ),
            indent=2,
        )
        + "\n"
    )
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = ["#6c757d", "#2563a6", "#16836b", "#8b6daf", "#ae7048", "#497d84"]
    for ax, subset, title in [
        (axs[0, 0], "all", "Native FeH: all 100,620 held-out labels"),
        (axs[0, 1], "below_-2", "Native FeH < −2: 185 held-out labels"),
    ]:
        vals = [
            next(
                r["mae_dex"]
                for r in rows
                if r["arm"] == label and r["subset"] == subset
            )
            for label in LABELS
        ]
        ax.barh(LABELS, vals, color=colors)
        ax.invert_yaxis()
        ax.set(
            xlabel="Mean absolute error to reported native FeH [dex]",
            title=title,
            xlim=(0, max(vals) * 1.2),
        )
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8)
    ax = axs[1, 0]
    for label, c in zip(LABELS, colors, strict=True):
        r = [
            r
            for r in snapshots
            if r["arm"] == label and r["subset"] == "all" and r["presentations"] > 0
        ]
        ax.plot(
            [x["presentations"] / 1e6 for x in r],
            [x["mae_dex"] for x in r],
            label=label,
            color=c,
            marker="o",
            markersize=3,
        )
    ax.set(
        xlabel="Reconstruction presentations [million]",
        ylabel="Native-label MAE [dex]",
        title="Saved prediction snapshots (initial constant excluded)",
    )
    ax.legend(fontsize=7, ncol=2)
    ax = axs[1, 1]
    for label, c in [("Gaussian NLL", colors[1]), ("Gaussian CRPS", colors[2])]:
        for subset, style in [("all", "-"), ("below_-2", ":")]:
            r = next(r for r in gaussian if r["arm"] == label and r["subset"] == subset)
            ax.plot(
                [68, 95],
                [100 * r["coverage_68"], 100 * r["coverage_95"]],
                marker="o",
                linestyle=style,
                label=label + (" all" if subset == "all" else " FeH < -2"),
                color=c,
            )
    ax.plot([68, 95], [68, 95], linestyle="--", color="gray", label="Nominal")
    ax.set(
        xlabel="Nominal central coverage [%]",
        ylabel="Observed native-label coverage [%]",
        title="Gaussian intervals: observed labels, not latent truth",
        ylim=(0, 100),
    )
    ax.legend(fontsize=8)
    fig.suptitle("Epoch 72 · six matched 10M FeH pilots · one seed", fontsize=13)
    fig.savefig(OUT / "prediction-comparison.png", dpi=170)
    fig.savefig(OUT / "prediction-comparison.pdf")
    plt.close(fig)
    allrows = [r for r in rows if r["subset"] == "all"]
    tailrows = [r for r in rows if r["subset"] == "below_-2"]
    lines = [
        "# Verified six-arm epoch-72 FeH comparison",
        "",
        "All six completed and passed independent all-100620 head / all-10000 decoder CPU replay at rtol=atol=5e-5. Raw source fields, cohorts, scalers, initial arrays, row/label streams and common GPU mask hashes checked; masks were not independently CPU-regenerated.",
        "",
        "| Arm | All-label MAE [dex] | FeH < -2 MAE [dex], n=185 | All-label bias [dex] |",
        "|---|---:|---:|---:|",
    ]
    for r, t in zip(allrows, tailrows, strict=True):
        lines.append(
            f"| {r['arm']} | {r['mae_dex']:.8f} | {t['mae_dex']:.8f} | {r['bias_dex']:+.8f} |"
        )
    lines += [
        "",
        "## What this pilot supports",
        "",
        "Residual Huber has the lowest observed overall MAE (0.08537557 dex versus 0.08680719 for the plain MLP, 1.65% lower), and the lowest below-minus-2 MAE (0.51578736 versus 0.54448969 dex, 5.27% lower). Its overall bias remains -0.01246196 dex and below-minus-2 positive bias +0.51258954 dex. These small average gains do not establish seed robustness or satisfactory tail accuracy. Huber+RnC has a smaller overall bias (-0.00095690 dex) but does not beat the residual head's MAE. Our four-member BatchEnsemble did not beat the residual head; this does not rank official TabM or TabR.",
        "",
        "Gaussian CRPS improves population MAE and CRPS relative to Gaussian NLL (CRPS 0.06234022 versus 0.06529461 dex). Overall central 68/95% coverage is 74.07/95.49% for CRPS and 76.50/96.66% for NLL. On reported FeH<-2 labels, 95% coverage collapses to 70.81/67.03%, respectively, and point errors are worse than the Huber control. Neither Gaussian arm supplies a calibrated metal-poor physical-parameter PDF. Tail bins are conditioned on noisy reported labels, so this is a conditional diagnostic, not a universal conditional-coverage theorem.",
        "",
        "All six adaptations worsen the fixed-mask original-validation reconstruction relative to the common initial checkpoint: standardized masked MAE starts at 0.69577884 and finishes at 0.70036992–0.70291693. Mean integrated spectral fraction starts at 0.08022473 and finishes at 0.08234721–0.08345064. Reconstruction weight0.01 did not fully prevent degradation in these pilots. The spectral auxiliary was NOT trained in this wave; its earlier positive evidence remains separate.",
        "",
        "Actual registry composition matters: fit contains APOGEE4728/GALAH15127/previously retained Li-VMPS4; test contains APOGEE22771/GALAH77808/Li-VMPS41. Below-minus-2 test contains APOGEE13/GALAH133/Li-VMPS39. Thus 'native' is the audited registry name, not a claim that every label is high-resolution GALAH/APOGEE. No new enriched PIGS/Li2025 source layers were merged in this wave. APOGEE+GALAH-only and corresponding146-star metal-poor metrics are also saved in the CSV; Li-VMPS is not silently pooled as independent high-resolution gold truth.",
        "",
        "## Interpretation limits",
        "",
        "Matched source69df29a / epoch72 parent / 19859 native fit stars / 10M reconstruction presentations / 306 joint updates / 754688 repeated label presentations. All initial reconstruction arrays match byte for byte. Source-ID supervision holdout does not exclude unsupervised pretraining/replay overlap. This is one seed and a bounded budget, not an architecture-wide or SOTA conclusion. FeH bins use reported native labels, not independent latent physical truth; n=14 below -3 is small. Survey layers are separately reported. No silver merge, alpha, age, joint-target or flow training in these runs.",
        "",
        "Gaussian scores and coverage apply only to the NLL/CRPS-trained observed-label distributions; Huber variance outputs are untrained and BatchEnsemble spread is not a calibrated uncertainty. NLL is a density score in dex and can be negative. No deconvolution of label errors/systematics has been performed. Reconstruction/spectral diagnostics cover a fixed 10000-row original-validation prefix with common masks, not the entire benchmark. Spectral flux is G=8.5 normalized, all 55 bases retained in each band.",
        "",
        "Point comparisons are descriptive; no significance/seed robustness claim. Do not tune further on this inspected test and call it untouched. Full metrics, snapshots, uncertainty diagnostics and reconstruction results are in adjacent CSVs. Figure visual check must pass before notifying scientific findings.",
    ]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                point=allrows,
                tail=tailrows,
                gaussian=[r for r in gaussian if r["subset"] == "all"],
                reconstruction=[r for r in reco if r["stage"] == "final"],
                output=str(OUT),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
