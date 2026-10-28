"""Verify matched completed photometry experiments and save count-weighted comparisons."""

import csv
import json
from pathlib import Path

import numpy as np

root = Path("experiments/xp/results-photometry-e47")
launch = json.loads(
    Path("experiments/xp/launch-photometry-e47-20261013.json").read_text()
)
data = {}
for job in launch["jobs"]:
    name = job["run_id"]
    folder = root / name
    nested = folder / name
    m = json.loads((folder / "manifest.json").read_text())
    q = json.loads((nested / "residual_latest.json").read_text())
    v = json.loads((nested / "checkpoint_verification.json").read_text())
    policy = json.loads((nested / "optimizer_policy.json").read_text())
    metrics = [
        json.loads(x)
        for x in (nested / "metrics.jsonl").read_text().splitlines()
        if x.strip()
    ]
    final, initial = metrics[-1], metrics[0]
    assert m["status"] == final["phase"] == "complete"
    assert all(x["status"] == "complete" for x in m["experiments"])
    assert final["rows_seen_total"] == 20_000_000 and final["optimizer_steps"] == 612
    assert (
        v["reload_predictions_equal"] and v["parent_sha256"] == m["checkpoint_sha256"]
    )
    assert q["validation_id"] == m["validation_id"] == final["validation_id"]
    assert (
        m["sample"]["train_rows"] == 9_000_000
        and m["sample"]["validation_rows"] == 1_000_000
    )
    assert (
        m["benchmark_source_id_sha256"]
        == "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
    )
    assert m["checkpoint_epoch"] == 47
    assert (
        m["checkpoint_sha256"]
        == "4ee3c98c6db2e75d50aa54611a2beedd5ecbbb70c5af94e4c718d23eae208482"
    )
    expected_sha = job.get("git_sha", launch["git_sha"])
    assert m["git_sha"] == expected_sha, (name, m["git_sha"])
    data[name] = (m, q, final, initial, policy)
control = data["unchanged_control"]
checks = [
    "checkpoint_sha256",
    "checkpoint_epoch",
    "xp_helpers_sha256",
    "benchmark_source_id_sha256",
    "benchmark_manifest_sha256",
    "validation_id",
    "seed",
    "validation_seed",
    "independent_rng",
    "mask_policy",
    "runner_sha256",
    "source_code_sha256",
    "experiment_environment_lock_sha256",
    "spectral_projection",
    "diagnostic_snapshot_policy",
    "photometry_calibration_sha256",
    "photometry_projection_sha256",
    "photometry_helpers_sha256",
    "diagnostic_validation_rows",
    "presentations_per_experiment",
]
for name, (m, q, final, initial, policy) in data.items():
    for key in checks:
        assert m[key] == control[0][key], (name, key)
    assert m["config"]["data"] == control[0]["config"]["data"]
    assert m["config"]["training"] == control[0]["config"]["training"]
    assert initial["per_column_mae"] == control[3]["per_column_mae"]
    assert initial["all_columns_mae"] == control[3]["all_columns_mae"]
    assert initial["xp_only_mae"] == control[3]["xp_only_mae"]
    assert m["xp_helpers_sha256"] == control[0]["xp_helpers_sha256"]
    factor = 1
    assert policy["reset_moments"] is False
    assert policy["source_learning_rates"] == control[4]["source_learning_rates"]
    assert policy["learning_rates"] == [
        factor * lr for lr in policy["source_learning_rates"]
    ]
# Compare actual shared diagnostic predictions and masks, not just aggregate MAE.
with np.load(
    root / "unchanged_control" / "unchanged_control" / "residual_init.npz",
    allow_pickle=False,
) as ref:
    for name in data:
        with np.load(
            root / name / name / "residual_init.npz", allow_pickle=False
        ) as current:
            assert set(current.files) == set(ref.files), name
            for key in ref.files:
                if key.endswith("_prediction"):
                    np.testing.assert_allclose(
                        current[key],
                        ref[key],
                        rtol=1e-6,
                        atol=1e-7,
                        err_msg=f"{name}:{key}",
                    )
                elif key.endswith("_mask") or key in {
                    "validation_row_index",
                    "feature_names",
                }:
                    np.testing.assert_array_equal(current[key], ref[key])

rows, entries, photometry_rows = [], [], []


def delta(value, ref):
    return 100 * (value / ref - 1)


def pooled_photometry(bands, state=None):
    values = [b if state is None else b[state] for b in bands]
    count = sum(b["count"] for b in values)
    value = (
        sum(
            b["reference_flux_absolute_error_mean"] * b["count"]
            for b in values
            if b["count"]
        )
        / count
        if count
        else None
    )
    return {"star_band_pairs": count, "reference_flux_mae": value}


def group(features, band, low, high):
    selected = [
        f
        for f in features
        if f["name"].startswith(band + "_")
        and low <= int(f["name"].split("_")[1]) <= high
    ]
    assert len(selected) == high - low + 1
    count = sum(f["count"] for f in selected)
    return {
        "count": count,
        "mae": sum(f["mae"] * f["count"] for f in selected if f["count"]) / count
        if count
        else None,
    }


for name, (m, q, final, initial, policy) in data.items():
    groups = {}
    for regime, diagnostics in q["regimes"].items():
        baseline = {f["name"]: f for f in control[1]["regimes"][regime]["features"]}
        assert set(baseline) == {f["name"] for f in diagnostics["features"]}
        for f in diagnostics["features"]:
            ref = baseline[f["name"]]
            assert f["count"] == ref["count"], (name, regime, f["name"])
            difference = (
                f["mae"] - ref["mae"]
                if f["mae"] is not None and ref["mae"] is not None
                else None
            )
            rows.append(
                {
                    "experiment": name,
                    "regime": regime,
                    "column": f["name"],
                    "count": f["count"],
                    "mae": f["mae"],
                    "control_mae": ref["mae"],
                    "delta_mae": difference,
                    "delta_percent": delta(f["mae"], ref["mae"])
                    if difference is not None and ref["mae"]
                    else None,
                }
            )
        groups[regime] = {}
        for low, high in [(1, 10), (11, 30), (31, 55)]:
            bp, rp = [
                group(diagnostics["features"], band, low, high) for band in ["bp", "rp"]
            ]
            ref_bp, ref_rp = [
                group(list(baseline.values()), band, low, high) for band in ["bp", "rp"]
            ]
            total = bp["count"] + rp["count"]
            combined = (
                ((bp["mae"] or 0) * bp["count"] + (rp["mae"] or 0) * rp["count"])
                / total
                if total
                else None
            )
            ref_total = ref_bp["count"] + ref_rp["count"]
            ref_combined = (
                (
                    (ref_bp["mae"] or 0) * ref_bp["count"]
                    + (ref_rp["mae"] or 0) * ref_rp["count"]
                )
                / ref_total
                if ref_total
                else None
            )
            groups[regime][f"{low}-{high}"] = {
                "bp": bp,
                "rp": rp,
                "combined_count": total,
                "combined_mae": combined,
                "combined_delta_percent": delta(combined, ref_combined)
                if combined is not None
                else None,
            }
    spectral = q["regimes"]["common"]["spectral"]
    ref_spectral = control[1]["regimes"]["common"]["spectral"]
    assert spectral["rows"] == ref_spectral["rows"]
    entry = {
        "experiment": name,
        "session_id": next(
            j["session_ids"][0] for j in launch["jobs"] if j["run_id"] == name
        ),
        "all_reconstructed_mae": final["all_columns_mae"],
        "xp_only_mae": final["xp_only_mae"],
        "all_mae_delta_percent": delta(
            final["all_columns_mae"], control[2]["all_columns_mae"]
        ),
        "xp_mae_delta_percent": delta(final["xp_only_mae"], control[2]["xp_only_mae"]),
        "xp_mae_delta_from_parent_percent": delta(
            final["xp_only_mae"], initial["xp_only_mae"]
        ),
        "spectral_rows": spectral["rows"],
        "spectral_mean": spectral["integrated_absolute_fraction_mean"],
        "spectral_median": spectral["integrated_absolute_fraction_median"],
        "spectral_mean_delta_percent": delta(
            spectral["integrated_absolute_fraction_mean"],
            ref_spectral["integrated_absolute_fraction_mean"],
        ),
        "spectral_median_delta_percent": delta(
            spectral["integrated_absolute_fraction_median"],
            ref_spectral["integrated_absolute_fraction_median"],
        ),
        "order_groups": groups,
        "optimizer_policy": policy,
    }
    entry["photometry_consistency"] = {
        regime: value["photometry_consistency"]
        for regime, value in q["regimes"].items()
    }
    reference_bands = control[1]["regimes"]["common"]["photometry_consistency"]["bands"]
    for band, reference in zip(
        entry["photometry_consistency"]["common"]["bands"], reference_bands, strict=True
    ):
        assert (
            band["column"] == reference["column"]
            and band["count"] == reference["count"]
        )
        band["flux_error_delta_percent"] = delta(
            band["reference_flux_absolute_error_mean"],
            reference["reference_flux_absolute_error_mean"],
        )
        band["magnitude_error_delta_percent"] = (
            delta(band["magnitude_mae"], reference["magnitude_mae"])
            if band["magnitude_mae"] is not None
            else None
        )
    entry["pooled_photometry"] = {}
    for regime, diagnostics in entry["photometry_consistency"].items():
        bands = diagnostics["bands"]
        ref_bands = control[1]["regimes"][regime]["photometry_consistency"]["bands"]
        entry["pooled_photometry"][regime] = {}
        for state in [None, "hidden", "visible"]:
            pooled = pooled_photometry(bands, state)
            ref_pooled = pooled_photometry(ref_bands, state)
            assert pooled["star_band_pairs"] == ref_pooled["star_band_pairs"]
            pooled["delta_percent"] = (
                delta(pooled["reference_flux_mae"], ref_pooled["reference_flux_mae"])
                if pooled["star_band_pairs"]
                else None
            )
            entry["pooled_photometry"][regime][state or "all"] = pooled
        for band, ref_band in zip(bands, ref_bands, strict=True):
            assert (
                band["column"] == ref_band["column"]
                and band["count"] == ref_band["count"]
            )
            for state in [None, "hidden", "visible"]:
                value, ref_value = (
                    (band, ref_band)
                    if state is None
                    else (band[state], ref_band[state])
                )
                assert value["count"] == ref_value["count"]
                photometry_rows.append(
                    {
                        "experiment": name,
                        "regime": regime,
                        "photometry_input": state or "all",
                        "band": band["column"],
                        "count": value["count"],
                        "reference_flux_mae": value[
                            "reference_flux_absolute_error_mean"
                        ],
                        "control_reference_flux_mae": ref_value[
                            "reference_flux_absolute_error_mean"
                        ],
                        "delta_percent": delta(
                            value["reference_flux_absolute_error_mean"],
                            ref_value["reference_flux_absolute_error_mean"],
                        )
                        if value["count"]
                        else None,
                        "nonpositive_synthetic_count": band[
                            "nonpositive_synthetic_count"
                        ]
                        if state is None
                        else None,
                        "raw_magnitude_mae_positive_flux_only": band["magnitude_mae"]
                        if state is None
                        else None,
                    }
                )
    entry["nonpositive_synthetic_flux_count"] = sum(
        b["nonpositive_synthetic_count"]
        for b in entry["photometry_consistency"]["common"]["bands"]
    )
    entries.append(entry)
summary = {
    "identity_checks": "passed",
    "parent_checkpoint_sha256": control[0]["checkpoint_sha256"],
    "benchmark_source_id_sha256": control[0]["benchmark_source_id_sha256"],
    "validation_id": control[0]["validation_id"],
    "training_stars": 9000000,
    "validation_stars": 1000000,
    "presentations": 20000000,
    "optimizer_steps": 612,
    "known_source_difference": "All runs use identical source and frozen training-only photometry calibration.",
    "experiments": entries,
}
with (root / "per-column-mae.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (root / "per-band-photometry.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(photometry_rows[0]))
    writer.writeheader()
    writer.writerows(photometry_rows)
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
report = [
    "# Epoch-47 XP photometric consistency experiments",
    "",
    "All five completed 20M presentations and 612 optimizer steps on the same 9M training and 1M validation stars. Parent checkpoint, scaler/configuration, benchmark identity, validation IDs/masks, row-order policy, seeds, initial per-column MAE and actual diagnostic predictions, diagnostic projection and checkpoint reload probes pass. All objectives use inherited optimizer state and the same frozen training-only photometric calibration.",
    "",
    "Ordinary MAE scores hidden finite entries in network-scaled coordinates. Spectral diagnostics score 4,996 shared common-mask stars from the deterministic 10,000-row snapshot; their mean/median are integrated absolute error divided by integrated absolute reference flux. Negative percentage changes improve on the matched unchanged control. Photometric flux residuals are divided by the fixed training median flux in each band; pooled MAE is weighted by eligible star-band pair counts. Magnitude MAE uses positive reconstructed flux only, so its changing sample must be interpreted alongside nonpositive counts; flux MAE includes negative predictions. Training calibration baseline and per-band counts are preserved in photometry_calibration.json.",
    "",
]
for e in entries:
    report.append(
        f"- {e['experiment']}: all-column MAE {e['all_reconstructed_mae']:.9f} ({e['all_mae_delta_percent']:+.3f}%); XP MAE {e['xp_only_mae']:.9f} ({e['xp_mae_delta_percent']:+.3f}%); spectral mean {e['spectral_mean']:.9f} ({e['spectral_mean_delta_percent']:+.3f}%), median {e['spectral_median']:.9f} ({e['spectral_median_delta_percent']:+.3f}%)."
    )
    pooled = e["pooled_photometry"]["common"]["all"]
    report.append(
        f"  Photometric reference-flux MAE pooled over {pooled['star_band_pairs']} eligible star-band pairs: {pooled['reference_flux_mae']:.9f} ({pooled['delta_percent']:+.3f}%); nonpositive reconstructed filter flux count {e['nonpositive_synthetic_flux_count']}."
    )
    for state in ["hidden", "visible"]:
        pooled_state = e["pooled_photometry"]["common"][state]
        report.append(
            f"  {state.capitalize()} survey inputs: pooled reference-flux MAE {pooled_state['reference_flux_mae']:.9f} ({pooled_state['delta_percent']:+.3f}%), {pooled_state['star_band_pairs']} pairs."
        )
    for band in e["photometry_consistency"]["common"]["bands"]:
        report.append(
            f"  {band['column']}: bias-corrected reference-flux MAE {band['reference_flux_absolute_error_mean']:.6f} ({band['flux_error_delta_percent']:+.3f}%); raw magnitude MAE {band['magnitude_mae']}; nonpositive synthetic flux {band['nonpositive_synthetic_count']}/{band['count']}."
        )
    for orders, g in e["order_groups"]["common"].items():
        report.append(
            f"  BP/RP orders {orders}: count-weighted BP MAE {g['bp']['mae']:.9f}, RP {g['rp']['mae']:.9f}, combined {g['combined_mae']:.9f} ({g['combined_delta_percent']:+.3f}%)."
        )
report += [
    "",
    "Photometry plus spectral auxiliary gives the lowest XP MAE: 0.758749182, 0.520% below ordinary MAE control. Its additional coefficient gain over spectral-only is just 0.0297%; mean spectral error is 0.0554% higher than spectral-only. The useful new signal is stronger survey agreement: pooled bias-corrected reference-flux MAE improves 6.60% versus control and 5.41% versus spectral-only, with all nine bands improving against control. Photometry alone improves mean spectral error by 3.46%, but coefficient MAE improves only 0.127% against control and remains slightly worse than the starting parent. Adding detached decoder consistency does not improve coefficient or spectral error over the observed-photometry term alone. These point estimates support a photometric constraint for survey consistency, with no meaningful demonstrated extra XP coefficient gain beyond the spectral auxiliary.",
    "",
    "These are single-seed, short continuations. Small final differences need repeat-seed or longer matched confirmation; do not infer significance from one run. The fixed-budget final checkpoint is compared, without choosing the best validation interval. The control ends slightly worse than the starting parent; deltas relative to the parent are retained in comparison.json.",
    "",
    "Raw downloaded artifacts are preserved in each experiment folder. Per-column counts/MAE/deltas for all six validation mask regimes are in per-column-mae.csv; count-weighted order groups for every regime are in comparison.json. Per-band and pooled photometric residuals, with hidden/visible inputs separated, are also preserved; per-band-photometry.csv contains all six mask regimes. Persistent checkpoints remain on ARC.",
]
(root / "comparison.md").write_text("\n".join(report) + "\n")
print(
    json.dumps(
        [
            {
                k: v
                for k, v in e.items()
                if k
                not in ["order_groups", "optimizer_policy", "photometry_consistency"]
            }
            for e in entries
        ],
        indent=2,
    )
)
