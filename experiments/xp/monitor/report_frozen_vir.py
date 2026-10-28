"""Measured-label point and Student-t diagnostics for a verified frozen VIR screen.

No model training or replay. All NIG Student-t scores are independently derived
with SciPy; they are not Gaussian scores or physical-truth calibration claims.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_prediction_comparison import BASE, digest, score
from report_verified_mechanisms import conditional_masks
from scipy.special import gammaln
from scipy.stats import t as student_t

METHODS = [
    ("raw_mse", "MSE"),
    ("unweighted_nig", "Unweighted NIG"),
    ("density_nig", "Density NIG"),
    ("vir", "VIR"),
]


def student_arrays(posterior, truth_dex, center, label_scale):
    """NIG marginal: df=2α; Student-t scale²=β(1+ν)/(αν), in dex."""
    gamma, nu, alpha, beta = np.asarray(posterior, dtype="f8").T
    assert np.isfinite(posterior).all()
    assert (nu > 0).all() and (alpha > 1).all() and (beta > 0).all()
    assert np.isfinite(label_scale) and label_scale > 0
    point = gamma * label_scale + center
    scale = np.sqrt(beta * (1 + nu) / (alpha * nu)) * label_scale
    df = 2 * alpha
    z = (np.asarray(truth_dex, dtype="f8") - point) / scale
    logpdf = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
        - (df + 1) / 2 * np.log1p(z * z / df)
    )
    # Independent library-distribution check of the closed-form density.
    np.testing.assert_allclose(
        logpdf,
        student_t.logpdf(truth_dex, df, loc=point, scale=scale),
        rtol=2e-12,
        atol=2e-12,
    )
    arrays = dict(point=point, nll=-logpdf, std=scale * np.sqrt(df / (df - 2)))
    for nominal in (0.68, 0.95):
        width = student_t.ppf((1 + nominal) / 2, df) * scale
        key = str(round(nominal * 100))
        arrays["covered_" + key] = np.abs(truth_dex - point) <= width
        arrays["width_" + key] = 2 * width
    return arrays


def write_csv(out, name, rows):
    assert rows
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / name).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(directory):
    out = directory / "verified-comparison"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    verification = json.loads((directory / "independent-verification.json").read_text())
    assert manifest["status"] == "complete" and not manifest["smoke"]
    assert verification["status"] == "passed"
    assert verification["source_sha"] == manifest["source_sha"]
    assert manifest["encoder_unchanged"] and manifest["encoder_updates"] == 0
    assert manifest["reconstruction_presentations"] == 0
    assert manifest["native_fit"] == 19859 and manifest["native_test"] == 100620
    assert {m["method"] for m in verification["methods"]} == {m for m, _ in METHODS}
    assert len({m["stream_sha256"] for m in manifest["methods"]}) == 1
    assert all(
        m["updates"] == 3000 and m["label_presentations"] == 768000
        for m in manifest["methods"]
    )
    for name, expected in verification["artifact_sha256"].items():
        assert manifest["artifact_sha256"][name] == expected
        assert digest(directory / name) == expected, name
    with np.load(directory / "paired-cohort.npz", allow_pickle=False) as saved:
        paired = dict(saved)
    y = paired["test_measurements"]["value"]
    yc, ys = float(paired["label_center"]), float(paired["label_scale"])
    assert not np.isin(
        paired["fit_source_id"], paired["test_measurements"]["source_id"]
    ).any()
    assert len(y) == 100620
    np.testing.assert_allclose(
        [yc, ys], [paired["fit_target"].mean(), paired["fit_target"].std()], rtol=1e-14
    )
    masks = conditional_masks(paired["test_measurements"])
    metrics, uncertainty, initial_rows = [], [], []
    for method, label in METHODS:
        with np.load(
            directory / (method + "-predictions.npz"), allow_pickle=False
        ) as saved:
            final, initial = saved["final_standardized"], saved["initial_standardized"]
            posterior = saved.get("nig_posterior", None)
        # Match production's float32 standardized prediction to dex convention.
        prediction = final * ys + yc
        initial_prediction = initial * ys + yc
        distribution = None
        if posterior is not None:
            np.testing.assert_array_equal(final, posterior[:, 0])
            distribution = student_arrays(posterior, y, yc, ys)
        for subset, keep in masks.items():
            if not keep.any():
                continue
            metrics.append(
                dict(
                    method=method,
                    arm=label,
                    subset=subset,
                    **score(y[keep], prediction[keep]),
                )
            )
            initial_rows.append(
                dict(
                    method=method,
                    arm=label,
                    subset=subset,
                    **score(y[keep], initial_prediction[keep]),
                )
            )
            if distribution:
                uncertainty.append(
                    dict(
                        method=method,
                        arm=label,
                        subset=subset,
                        count=int(keep.sum()),
                        student_t_nll_dex=float(distribution["nll"][keep].mean()),
                        mean_predictive_std_dex=float(distribution["std"][keep].mean()),
                        nominal_68_coverage=float(
                            distribution["covered_68"][keep].mean()
                        ),
                        nominal_95_coverage=float(
                            distribution["covered_95"][keep].mean()
                        ),
                        mean_68_interval_width_dex=float(
                            distribution["width_68"][keep].mean()
                        ),
                        mean_95_interval_width_dex=float(
                            distribution["width_95"][keep].mean()
                        ),
                    )
                )
    write_csv(out, "conditional-and-survey-point-metrics.csv", metrics)
    write_csv(out, "conditional-and-survey-student-t-metrics.csv", uncertainty)
    write_csv(out, "initial-conditional-and-survey-point-metrics.csv", initial_rows)
    labels = [label for _, label in METHODS]
    colors = plt.get_cmap("tab10")(np.arange(len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for ax, subset, title in [
        (axes[0, 0], "all", "Population: 100,620 reported labels"),
        (axes[0, 1], "below_-2", "Reported FeH < −2: 185 labels"),
    ]:
        values = [
            next(
                r["mae_dex"]
                for r in metrics
                if r["arm"] == label and r["subset"] == subset
            )
            for label in labels
        ]
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set(xlabel="MAE [dex]", title=title, xlim=(0, max(values) * 1.2))
        for i, value in enumerate(values):
            ax.text(value, i, f" {value:.4f}", va="center")
    for ax, subset, title in [
        (axes[1, 0], "all", "Student-t intervals: population"),
        (axes[1, 1], "below_-2", "Student-t intervals: below −2"),
    ]:
        rows = [
            next(r for r in uncertainty if r["arm"] == label and r["subset"] == subset)
            for label in labels[1:]
        ]
        coverage = [r["nominal_95_coverage"] * 100 for r in rows]
        ax.barh(labels[1:], coverage, color=colors[1:])
        ax.invert_yaxis()
        ax.axvline(95, color="black", linestyle="--", label="Nominal 95%")
        ax.set(
            xlabel="Observed-label interval coverage [%]", title=title, xlim=(0, 105)
        )
        for i, value in enumerate(coverage):
            ax.text(value, i, f" {value:.1f}%", va="center")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Epoch72 frozen encoder · matched VIR head screen · one seed")
    fig.savefig(out / "vir-comparison.png", dpi=170)
    fig.savefig(out / "vir-comparison.pdf")
    plt.close(fig)
    lines = [
        "# Independently verified frozen VIR head comparison",
        "",
        "Same immutable epoch72 encoder and audited native cohort; 19,859 fit and100,620 test labels, fit-only latent and target scaling, 3,000 updates/768,000 label presentations per head and matched label streams. Encoder unchanged, no reconstruction training. All four heads and NIG posterior parameters passed full independent CPU replay using a separately reproduced120,479-row parent encoding. This screen is not matched to earlier encoder-adaptation10M runs.",
        "",
        "| Head | Population MAE [dex] | Below−2 MAE [dex] | Below−2 bias [dex] |",
        "|---|---:|---:|---:|",
    ]
    for method, label in METHODS:
        allrow = next(
            r for r in metrics if r["method"] == method and r["subset"] == "all"
        )
        tail = next(
            r for r in metrics if r["method"] == method and r["subset"] == "below_-2"
        )
        lines.append(
            f"| {label} | {allrow['mae_dex']:.8f} | {tail['mae_dex']:.8f} | {tail['bias_dex']:+.8f} |"
        )
    lines += [
        "",
        "| Head | Population Student-t NLL [dex units] | Population95% coverage | Below−2 95% coverage |",
        "|---|---:|---:|---:|",
    ]
    for method, label in METHODS[1:]:
        allrow = next(
            r for r in uncertainty if r["method"] == method and r["subset"] == "all"
        )
        tail = next(
            r
            for r in uncertainty
            if r["method"] == method and r["subset"] == "below_-2"
        )
        lines.append(
            f"| {label} | {allrow['student_t_nll_dex']:.8f} | {100 * allrow['nominal_95_coverage']:.3f}% | {100 * tail['nominal_95_coverage']:.3f}% |"
        )
    lines += [
        "",
        "NIG scores use the exact Student-t marginal: df=2alpha, location=gamma, scale²=beta(1+nu)/(alpha*nu), transformed to dex. A separate closed-form log density matches scipy.stats.t.logpdf; intervals use SciPy Student-t quantiles. They are predictive intervals for measured noisy native labels, not deconvolved physical metallicity or frequentist confidence intervals. Predictive standard deviation is not the Student-t scale. MSE has no trained distribution and is excluded from uncertainty scoring.",
        "",
        "Conditional/source-separated CSV retains the native Li-VMPS layer (39 of185 below−2 test labels), APOGEE/GALAH-only146tail rows and below−3 support14. One seed, sparse extreme tail, survey systematics and unsupervised pretraining overlap limit conclusions. Full VIR adds density weighting and feature recolouring; the ablations distinguish these changes from the NIG output family. Its predicted-bin inference does not consume test labels. No new data, imputed targets, calibrated physical PDF or attenuation-cured claim. CNF is excluded because its separate strict CPU verification failed.",
        "",
        "PNG visual inspection pending.",
    ]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    (out / "report-verification.json").write_text(
        json.dumps(
            dict(
                status="passed",
                source_cpu_report=str(directory / "independent-verification.json"),
                source_artifact_sha256=verification["artifact_sha256"],
                same_label_stream=True,
                student_t_logpdf_scipy_checked=True,
                interval_quantiles="scipy.stats.t.ppf",
                visual_check="pending",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                output=str(out),
                point=[r for r in metrics if r["subset"] in {"all", "below_-2"}],
                uncertainty=[
                    r for r in uncertainty if r["subset"] in {"all", "below_-2"}
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=BASE / "frozen_e72_vir_screen_v2"
    )
    main(parser.parse_args().directory)
