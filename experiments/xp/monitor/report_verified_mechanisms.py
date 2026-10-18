"""Paired reporting of verified point mechanisms and the all-model SIMEX aggregate.

Never trains or replays a network. Requires completed independent CPU reports.
SIMEX reconstruction belongs to individual models, not to an aggregate decoder.
"""

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_prediction_comparison import BASE, digest, score, spectral


def independent_extrapolate(lambdas, values):
    """Grid-search affine reciprocal curve with a separate NumPy least square."""
    lam, y = np.asarray(lambdas, dtype="f8"), np.asarray(values, dtype="f8")
    anchor = y[0]
    shifted = y - anchor
    best = np.full(y.shape[1], np.inf)
    result = np.zeros(y.shape[1])
    for c in np.linspace(1.05, 8.0, 400):
        design = np.column_stack([np.ones(len(lam)), 1 / (c + lam)])
        coefficients = np.linalg.lstsq(design, shifted, rcond=None)[0]
        residual = np.sum((design @ coefficients - shifted) ** 2, axis=0)
        take = residual < best
        best[take] = residual[take]
        result[take] = coefficients[0, take] + coefficients[1, take] / (c - 1)
    return result + anchor


def load_json(path):
    return json.loads(path.read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return dict(arrays)


def conditional_masks(measurements):
    y = measurements["value"]
    survey = measurements["survey"].astype(str)
    masks = {"all": np.ones(len(y), dtype=bool), "below_-3": y < -3, "below_-2": y < -2}
    for lo, hi in [(-3, -2), (-2, -1), (-1, 0), (0, np.inf)]:
        masks[f"[{lo},{hi})"] = (y >= lo) & (y < hi)
    for source in np.unique(survey):
        masks[f"survey:{source}"] = survey == source
        masks[f"survey:{source}:below_-2"] = (survey == source) & (y < -2)
    apogal = np.isin(survey, ["apogee", "galah"])
    masks["apogee_galah_only"] = apogal
    masks["apogee_galah_below_-2"] = apogal & (y < -2)
    return masks


def score_rows(label, y, prediction, masks, **fields):
    return [
        dict(arm=label, subset=name, **fields, **score(y[keep], prediction[keep]))
        for name, keep in masks.items()
        if keep.any()
    ]


def pair(reference, manifest, arrays, initial):
    old_manifest, old_arrays, old_initial = reference
    for key in [
        "parent_sha256",
        "cohort_sha256",
        "row_sha256",
        "mask_sha256",
        "supervised_stream_sha256",
    ]:
        assert old_manifest[key] == manifest[key], key
    assert set(arrays) ^ set(old_arrays) <= {"input_note"}
    for key in set(arrays) & set(old_arrays):
        # SIMEX intentionally adds noise to inputs; all source IDs/labels stay paired.
        if key not in {"fit_input", "test_input"}:
            assert arrays[key].dtype == old_arrays[key].dtype, key
            assert arrays[key].tobytes() == old_arrays[key].tobytes(), key
    for key in old_initial:
        assert initial[key].tobytes() == old_initial[key].tobytes(), key


def snapshot_point(path, arrays):
    saved = load_npz(path)
    # Production uses Python scalars, so NumPy keeps the head's float32 dtype.
    # A stored float64 zero-dimensional array would promote to float64 and its
    # tiny conversion difference can be amplified by rational extrapolation.
    return saved["member_outputs"][:, :, 0].mean(1) * float(
        arrays["label_scale"]
    ) + float(arrays["label_center"])


def write_csv(out, filename, rows):
    assert rows
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / filename).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def reconstruction(label, path, initial, declared):
    final = load_npz(path / "final_reconstruction.npz")["prediction"]
    rows = []
    for stage, values in [("initial", initial["prediction"]), ("final", final)]:
        target = initial["target"][:, : values.shape[1]]
        keep = initial["mask"][:, : values.shape[1]] & np.isfinite(target)
        mae = float(abs(values.astype("f8") - target.astype("f8"))[keep].mean())
        spec = spectral(initial, values)
        np.testing.assert_allclose(mae, declared[stage]["masked_mae"], rtol=2e-7)
        np.testing.assert_allclose(
            spec["spectral_fraction_mean"],
            declared[stage]["spectral"]["integrated_absolute_fraction_mean"],
            rtol=2e-7,
        )
        rows.append(dict(arm=label, stage=stage, masked_mae_standardized=mae, **spec))
    return rows


def main(kind):
    out = BASE / (
        "mechanism-v2-verified-comparison"
        if kind == "mechanisms"
        else "input-corrections-e72/verified-simex-comparison"
    )
    out.mkdir(parents=True, exist_ok=True)
    controls = [
        (BASE / "prediction_e72_huber_10m_v1", "Plain Huber"),
        (BASE / "prediction_e72_residual_huber_10m_v1", "Residual Huber"),
        (BASE / "prediction_e72_residual_mse_mechanism_10m_v2", "Residual MSE"),
    ]
    if kind == "mechanisms":
        runs = controls + [
            (BASE / "prediction_e72_corrected_score_xp_10m_v2", "Corrected-score EIV"),
            (BASE / "prediction_e72_residual_group_dro_10m_v2", "Group DRO Huber"),
        ]
    else:
        runs = controls[:2] + [
            (
                BASE / "input-corrections-e72/prediction_e72_simex_xp_10m_v2",
                "SIMEX aggregate",
            )
        ]
    metrics, snapshots, reco, reports, replica_metrics = [], [], [], [], []
    reference = None
    for path, label in runs:
        manifest, report = (
            load_json(path / "manifest.json"),
            load_json(path / "independent-verification.json"),
        )
        assert manifest["status"] == "complete" and not manifest["smoke"]
        assert (
            report["status"] == "passed"
            and report["source_sha"] == manifest["source_sha"]
        )
        arrays, initial = (
            load_npz(path / "paired-cohort.npz"),
            load_npz(path / "initial_reconstruction.npz"),
        )
        if reference is None:
            reference = manifest, arrays, initial
        else:
            pair(reference, manifest, arrays, initial)
        y = arrays["test_measurements"]["value"]
        masks = conditional_masks(arrays["test_measurements"])
        prediction = load_npz(path / "predictions.npz")["prediction"]
        metrics.extend(score_rows(label, y, prediction, masks))
        is_simex = label == "SIMEX aggregate"
        if not is_simex:
            assert report["final_checkpoint_sha256"] == digest(path / "checkpoint.pth")
            np.testing.assert_allclose(
                score(y, prediction)["mae_dex"], report["mae_dex"], rtol=1e-12
            )
            assert (
                manifest["presentations"] == 10_000_000
                and manifest["optimizer_updates"] == 306
            )
            files = sorted(
                path.glob("snapshot-*.npz"), key=lambda p: int(p.stem.split("-")[-1])
            )
            assert len(files) == 5
            for file in files:
                snapshots.extend(
                    score_rows(
                        label,
                        y,
                        snapshot_point(file, arrays),
                        {k: masks[k] for k in ["all", "below_-2"]},
                        presentations=int(file.stem.split("-")[-1]),
                    )
                )
            reco.extend(
                reconstruction(
                    label, path, initial, manifest["comparison"]["reconstruction"]
                )
            )
        else:
            assert (
                report["all_contributing_checkpoints_cpu_replayed"]
                and report["independent_numpy_extrapolation"]
            )
            index = load_json(path / "replicas.json")["replicas"]
            assert len(index) == report["replicas"] == 17
            assert (
                manifest["presentations"] == 170_000_000
                and manifest["optimizer_updates"] == 5202
            )
            levels = [0.0, 0.5, 1.0, 1.5, 2.0]
            per_snapshots = {}
            for record in index:
                directory = path / record["directory"]
                per = load_json(directory / "independent-verification.json")
                assert (
                    per["status"] == "passed"
                    and per["corrected_inputs_independently_derived"]
                )
                assert (
                    digest(directory / "checkpoint.pth")
                    == per["final_checkpoint_sha256"]
                    == report["checkpoint_sha256s"][record["index"]]
                )
                assert (
                    digest(directory / "replica-manifest.json")
                    == record["manifest_sha256"]
                )
                per_arrays = load_npz(directory / "paired-cohort.npz")
                pair(
                    reference,
                    {**manifest, **record},
                    per_arrays,
                    load_npz(directory / "initial_reconstruction.npz"),
                )
                prediction_replica = load_npz(directory / "predictions.npz")[
                    "prediction"
                ]
                replica_metrics.extend(
                    score_rows(
                        record["directory"],
                        y,
                        prediction_replica,
                        masks,
                        lambda_value=record["lambda_value"],
                        simulation=record["simulation"],
                    )
                )
                reco.extend(
                    reconstruction(
                        record["directory"],
                        directory,
                        initial,
                        record["reconstruction"],
                    )
                )
                files = sorted(
                    directory.glob("snapshot-*.npz"),
                    key=lambda p: int(p.stem.split("-")[-1]),
                )
                assert len(files) == 5
                for file in files:
                    presentations = int(file.stem.split("-")[-1])
                    per_snapshots.setdefault(presentations, []).append(
                        (record["lambda_value"], snapshot_point(file, per_arrays))
                    )
            for presentations, pairs in sorted(per_snapshots.items()):
                means = np.stack(
                    [
                        np.mean(
                            [p.astype("f8") for lam, p in pairs if lam == level], axis=0
                        )
                        for level in levels
                    ]
                )
                point = independent_extrapolate(levels, means)
                snapshots.extend(
                    score_rows(
                        label,
                        y,
                        point,
                        {k: masks[k] for k in ["all", "below_-2"]},
                        presentations=presentations,
                    )
                )
                if presentations == 10_000_000:
                    np.testing.assert_allclose(point, prediction, rtol=1e-8, atol=1e-8)
            assembled = load_npz(path / "predictions.npz")
            for level, points in zip(
                assembled["lambda_values"],
                assembled["lambda_mean_prediction"],
                strict=True,
            ):
                replica_metrics.extend(
                    score_rows(
                        "lambda mean",
                        y,
                        points,
                        masks,
                        lambda_value=float(level),
                        simulation=-1,
                    )
                )
        reports.append(report)
    write_csv(out, "conditional-and-survey-metrics.csv", metrics)
    write_csv(out, "snapshot-metrics.csv", snapshots)
    write_csv(out, "reconstruction-and-spectral-metrics.csv", reco)
    if replica_metrics:
        write_csv(out, "simex-replica-and-lambda-metrics.csv", replica_metrics)
    labels = [label for _, label in runs]
    colors = plt.get_cmap("tab10")(np.arange(len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
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
    for label, color in zip(labels, colors, strict=True):
        for ax, subset in [(axes[1, 0], "all"), (axes[1, 1], "below_-2")]:
            rows = [
                r
                for r in snapshots
                if r["arm"] == label
                and r["subset"] == subset
                and r["presentations"] > 0
            ]
            ax.plot(
                [r["presentations"] / 1e6 for r in rows],
                [r["mae_dex"] for r in rows],
                marker="o",
                label=label,
                color=color,
            )
    axes[1, 0].set(
        xlabel="Presentations per contributing model [million]",
        ylabel="Population MAE [dex]",
        title="Saved snapshots",
    )
    axes[1, 1].set(
        xlabel="Presentations per contributing model [million]",
        ylabel="Below −2 MAE [dex]",
        title="Saved tail snapshots",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        "Epoch 72 · "
        + ("point mechanisms" if kind == "mechanisms" else "17-model affine SIMEX")
        + " · one seed"
    )
    fig.savefig(out / (kind + "-comparison.png"), dpi=170)
    fig.savefig(out / (kind + "-comparison.pdf"))
    plt.close(fig)
    selected = [r for r in metrics if r["subset"] in {"all", "below_-2", "below_-3"}]
    lines = [
        "# Independently replayed " + kind + " comparison",
        "",
        "All contributing final heads and fixed validation decoders passed independent CPU replay. Same epoch72 parent, source-ID roles, target scaling and reconstruction/label streams; GPU masks compared across runs, not independently CPU regenerated. All non-input paired fields and initial reconstruction arrays are byte-equal. Corrected-score objective-specific training audit is separate from its completed output replay. Classifier NRE and old contrastive flow are excluded because verification failed.",
        "",
        "| Arm | Population MAE [dex] | Below −2 MAE [dex] | Below −2 bias [dex] |",
        "|---|---:|---:|---:|",
    ]
    for label in labels:
        allrow = next(r for r in metrics if r["arm"] == label and r["subset"] == "all")
        tail = next(
            r for r in metrics if r["arm"] == label and r["subset"] == "below_-2"
        )
        lines.append(
            f"| {label} | {allrow['mae_dex']:.8f} | {tail['mae_dex']:.8f} | {tail['bias_dex']:+.8f} |"
        )
    lines += [
        "",
        "Source-specific, APOGEE/GALAH-only, below−3 (14 labels), snapshots and reconstruction/spectral metrics are preserved as CSV. Below−2 has 185 labels, including 39 retained Li-VMPS stars; these are reported noisy native measurements, not latent physical truth. Source-ID supervised holdout does not exclude unsupervised overlap. One seed, no SOTA or attenuation-cured claim.",
    ]
    if kind == "simex":
        lines += [
            "",
            "SIMEX costs 17 independent fits ×10M =170M reconstruction presentations, versus10M per control. It is an affine rational extrapolation of float64 lambda means, with shared diagonal XP noise and omitted native correlations/common G uncertainty; it is not a calibrated physical predictive PDF. All17 checkpoints are verified; there is no single aggregate model or decoder. Reconstruction/spectral CSV is per replica and must not be interpreted as an aggregate decoder. Snapshot abscissa is per-model presentations; total SIMEX work is17 times larger. Finite noisy curve extrapolation can amplify errors and does not guarantee an unbiased neural estimator.",
        ]
    else:
        lines += [
            "",
            "Residual MSE is the matched squared-error control for corrected-score EIV; residual Huber is the comparator for group-DRO Huber. The latter changes the objective toward target regimes. EIV uses a first-order Jacobian subtraction with reported diagonal XP errors; off-diagonal XP covariance/common G uncertainty and label-systematic deconvolution are omitted. No fabricated error floors or uncertainty calibration claim.",
            "",
            "At the final fixed budget, corrected-score EIV has worse population and below−2 observed-label MAE than residual MSE. Group DRO improves below−2 MAE over residual Huber while worsening population MAE; this is a population/tail tradeoff. Residual Huber retains the lowest population MAE among these arms. Every arm's final fixed reconstruction and spectral diagnostic is worse than its common initial value. These comparisons were specified by the stored final checkpoints, not selected from tail-test snapshots.",
        ]
    lines += ["", "Figure visual inspection pending."]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    (out / "independent-verification.json").write_text(
        json.dumps(
            dict(
                status="passed",
                reports=reports,
                paired_noninput_arrays_byte_equal=True,
                initial_reconstruction_byte_equal=True,
                visual_check="pending",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(output=str(out), metrics=selected), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["mechanisms", "simex"], required=True)
    main(parser.parse_args().kind)
