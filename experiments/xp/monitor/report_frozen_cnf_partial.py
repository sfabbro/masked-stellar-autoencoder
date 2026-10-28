"""Verified-subset point report; the failed three-arm CNF screen stays failed.

Only independently passed raw-MSE and CNF feature methods are shown. The NLL
flow comparator remains excluded, so shaping versus ordinary NLL is unresolved.
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

METHODS = [("raw_mse", "Raw MSE"), ("cnf", "CNF features + MSE")]


def verified_subset(report):
    """Reject a failed requested method; preserve the excluded failure status."""
    by_name = {row["method"]: row for row in report["methods"]}
    for name, _ in METHODS:
        assert by_name[name]["status"] == "passed", name
        assert by_name[name]["all100620cpu_predictions"], name
        assert by_name[name]["tolerance_violations"] == 0, name
    return [by_name[name] for name, _ in METHODS]


def main(directory):
    out = directory / "verified-subset-comparison"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    verification = json.loads((directory / "independent-verification.json").read_text())
    selected = verified_subset(verification)
    assert manifest["status"] == "complete" and not manifest["smoke"]
    assert verification["status"] == "failed"
    assert verification["source_sha"] == manifest["source_sha"]
    assert verification["raw_source_fields_inputs_scalers_verified"]
    assert manifest["encoder_unchanged"] and manifest["encoder_updates"] == 0
    assert manifest["reconstruction_presentations"] == 0
    assert manifest["native_fit"] == 19859 and manifest["native_test"] == 100620
    methods = {m["method"]: m for m in manifest["methods"]}
    assert len({methods[name]["stream_sha256"] for name, _ in METHODS}) == 1
    assert all(
        methods[name]["updates"] == 3000
        and methods[name]["label_presentations"] == 768000
        for name, _ in METHODS
    )
    required = [
        "paired-cohort.npz",
        "raw_mse-head.pth",
        "raw_mse-predictions.npz",
        "cnf-head.pth",
        "cnf-features.pth",
        "cnf-predictions.npz",
    ]
    for name in required:
        expected = verification["artifact_sha256"][name]
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
    metrics = []
    for method, label in METHODS:
        with np.load(
            directory / (method + "-predictions.npz"), allow_pickle=False
        ) as saved:
            final = saved["final_standardized"]
        prediction = final * ys + yc
        for subset, keep in masks.items():
            if keep.any():
                metrics.append(
                    dict(
                        method=method,
                        arm=label,
                        subset=subset,
                        **score(y[keep], prediction[keep]),
                    )
                )
    with (out / "conditional-and-survey-point-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, subset, title in [
        (axes[0], "all", "Population: 100,620 reported labels"),
        (axes[1], "below_-2", "Reported FeH < −2: 185 labels"),
    ]:
        values = [
            next(
                r["mae_dex"]
                for r in metrics
                if r["method"] == method and r["subset"] == subset
            )
            for method, _ in METHODS
        ]
        ax.barh([label for _, label in METHODS], values, color=["#1f77b4", "#d87526"])
        ax.invert_yaxis()
        ax.set(xlabel="MAE [dex]", title=title, xlim=(0, max(values) * 1.22))
        for i, value in enumerate(values):
            ax.text(value, i, f" {value:.5f}", va="center")
    fig.suptitle("Verified subset only · frozen encoder · NLL comparator excluded")
    fig.savefig(out / "cnf-partial-comparison.png", dpi=170)
    fig.savefig(out / "cnf-partial-comparison.pdf")
    plt.close(fig)
    lines = [
        "# Partial frozen CNF comparison: verified subset only",
        "",
        "The complete three-arm screen remains FAILED independent verification. This report includes only the raw MSE and CNF-feature methods, each of which separately passed full100,620-star CPU prediction replay under unchanged rtol=5e-5/atol=5e-5. Source fields, inputs, fit-only scaling and the shared120,479-row CPU parent encoding were verified. The NLL-feature method failed on one test row (10892), with standardized difference0.000235915184; it is excluded from all metrics and figures here. No tolerances were relaxed and no screen status is changed.",
        "",
        "| Verified method | Population MAE [dex] | Below−2 MAE [dex] | Below−2 bias [dex] |",
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
        "Same immutable epoch72 frozen encoder, 19,859 fits/100,620tests, latent/target scaling and3,000head updates/768,000label presentations per method. Raw head uses256latents plus twozero columns; CNF head uses the same256latents plus two fit-standardized density features. CNF additionally trains two PCA8-space flows for1,000updates per flow using fit-only FeH<−1 versus≥−1 regimes. This is binary density shaping used as regression features, not conditional FeH density estimation, grid MLE or a calibrated physical posterior.",
        "",
        "The two-method result cannot isolate the effect of contrastive shaping from ordinary NLL-flow augmentation because the NLL comparator has not passed its gate. The full three-arm scientific comparison therefore remains unresolved. This frozen-head screen is not matched to earlier encoder-adaptation10M experiments. Single seed, reported noisy labels, survey dependencies,185below−2 stars (including39Li-VMPS),14below−3 and possible unsupervised overlap limit interpretation. Source/APOGEE-GALAH-only conditional scores are preserved as CSV. No uncertainty, physical-truth calibration or attenuation-cured claim.",
        "",
        "PNG visual inspection pending.",
    ]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    status = dict(
        status="partial_verified_subset",
        full_screen_status=verification["status"],
        full_three_arm_comparison_resolved=False,
        included_methods=selected,
        excluded_methods=[
            m
            for m in verification["methods"]
            if m["method"] not in {name for name, _ in METHODS}
        ],
        source_report=str(directory / "independent-verification.json"),
        artifact_sha256={
            name: verification["artifact_sha256"][name] for name in required
        },
        tolerances_unchanged=True,
        visual_check="pending",
    )
    (out / "report-verification.json").write_text(json.dumps(status, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(out),
                metrics=[r for r in metrics if r["subset"] in {"all", "below_-2"}],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=BASE / "frozen_e72_cnf_screen_v2"
    )
    main(parser.parse_args().directory)
