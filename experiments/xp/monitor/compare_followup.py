"""Verify fixed-budget follow-up continuations, then compare matched seeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PARENT = "4ee3c98c6db2e75d50aa54611a2beedd5ecbbb70c5af94e4c718d23eae208482"
BENCHMARK = "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
IDENTITY_KEYS = (
    "checkpoint_sha256",
    "checkpoint_epoch",
    "benchmark_source_id_sha256",
    "benchmark_manifest_sha256",
    "validation_id",
    "validation_seed",
    "mask_policy",
    "runner_sha256",
    "source_code_sha256",
    "xp_helpers_sha256",
    "experiment_environment_lock_sha256",
    "spectral_projection",
    "diagnostic_snapshot_policy",
    "diagnostic_validation_rows",
    "photometry_calibration_sha256",
    "photometry_projection_sha256",
    "photometry_helpers_sha256",
    "presentations_per_experiment",
)


def read_json(path):
    return json.loads(path.read_text())


def delta(value, reference):
    return 100 * (value / reference - 1) if value is not None and reference else None


def reference_for(name, settings):
    if name.startswith(("control_s", "spectral_s", "photometry_spectral_s")):
        return f"control_s{settings['seed']}"
    return "spectral_s42"


def weighted(values, key="mae", count="count"):
    total = sum(value[count] for value in values)
    return {
        "count": total,
        "mae": sum(value[key] * value[count] for value in values if value[count])
        / total
        if total
        else None,
    }


def compare_spectral(actual, reference):
    """XP-visible regimes have no scored spectra and intentionally omit stats."""
    assert actual["rows"] == reference["rows"]
    result = {"rows": actual["rows"]}
    for field in (
        "integrated_absolute_fraction_mean",
        "integrated_absolute_fraction_median",
    ):
        if actual["rows"]:
            result[field] = actual[field]
            result[field + "_delta_percent"] = delta(actual[field], reference[field])
        else:
            result[field] = None
            result[field + "_delta_percent"] = None
    return result


def verify_policy(policy, settings, reference_policy):
    assert policy["seed"] == settings["seed"]
    assert policy["independent_rng"] is True
    assert policy["reset_moments"] == bool(settings.get("reset_optimizer", False))
    assert policy["warmup_steps"] == settings.get("warmup_steps", 0)
    assert policy["source_learning_rates"] == reference_policy["source_learning_rates"]
    np.testing.assert_allclose(
        policy["learning_rates"],
        np.asarray(policy["source_learning_rates"]) * settings.get("lr_factor", 1),
        rtol=0,
        atol=0,
    )


def verify_initial(path, reference_path):
    with (
        np.load(path, allow_pickle=False) as actual,
        np.load(reference_path, allow_pickle=False) as reference,
    ):
        assert set(actual.files) == set(reference.files)
        for key in actual.files:
            if key.endswith("_prediction"):
                np.testing.assert_allclose(
                    actual[key], reference[key], rtol=1e-6, atol=1e-7, err_msg=key
                )
            else:
                # Includes actual scaler-derived targets, masks and row provenance.
                np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)


def verify_dropout(actual, reference, settings):
    assert actual.keys() == reference.keys()
    for name, source in reference.items():
        key = "encoder_dropout" if name.startswith("encoder.") else "decoder_dropout"
        expected = settings.get(key, settings.get("dropout", source))
        assert actual[name] == expected, (name, actual[name], expected)


def compare(root, launch):
    jobs = {
        job["run_id"]: job
        for job in launch["jobs"]
        if job["settings"].get("task", "training") == "training"
    }
    data = {}
    for name, job in jobs.items():
        outer, nested = root / name, root / name / name
        manifest = read_json(outer / "manifest.json")
        qa = read_json(nested / "residual_latest.json")
        reload = read_json(nested / "checkpoint_verification.json")
        policy = read_json(nested / "optimizer_policy.json")
        metrics = [
            json.loads(line)
            for line in (nested / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        initial, final = metrics[0], metrics[-1]
        assert manifest["status"] == final["phase"] == "complete", name
        assert all(item["status"] == "complete" for item in manifest["experiments"])
        recorded = next(
            item for item in manifest["experiments"] if item["run_id"] == name
        )
        for key, value in job["settings"].items():
            if key != "task":
                assert recorded[key] == value, (name, key)
        assert (final["rows_seen_total"], final["optimizer_steps"]) == (20_000_000, 612)
        assert manifest["checkpoint_sha256"] == reload["parent_sha256"] == PARENT
        assert manifest["checkpoint_epoch"] == 47 and reload["reload_predictions_equal"]
        assert manifest["benchmark_source_id_sha256"] == BENCHMARK
        assert manifest["git_sha"] == job.get("git_sha", launch["git_sha"])
        assert manifest["sample"]["train_rows"] == 9_000_000
        assert manifest["sample"]["validation_rows"] == 1_000_000
        assert (
            qa["validation_id"] == manifest["validation_id"] == final["validation_id"]
        )
        assert final["seed"] == job["settings"]["seed"]
        data[name] = (manifest, qa, initial, final, policy)

    common = data["control_s42"]
    for name, (manifest, qa, initial, final, policy) in data.items():
        settings = jobs[name]["settings"]
        for key in IDENTITY_KEYS:
            assert manifest[key] == common[0][key], (name, key)
        for key in ("data", "model", "training"):
            assert manifest["config"][key] == common[0]["config"][key], (name, key)
        verify_policy(policy, settings, common[4])
        assert final["learning_rate"] == policy["learning_rates"][0]
        for key in ("per_column_mae", "all_columns_mae", "xp_only_mae"):
            assert initial[key] == common[2][key], (name, key)
        verify_initial(
            root / name / name / "residual_init.npz",
            root / "control_s42" / "control_s42" / "residual_init.npz",
        )
        verify_dropout(
            read_json(root / name / name / "dropout_policy.json"),
            read_json(root / "control_s42" / "control_s42" / "dropout_policy.json"),
            settings,
        )

    entries, columns, bands = [], [], []
    for name, (_, qa, initial, final, policy) in data.items():
        reference = reference_for(name, jobs[name]["settings"])
        ref = data[reference]
        entry = {
            "experiment": name,
            "reference": reference,
            "seed": jobs[name]["settings"]["seed"],
            "session_ids": jobs[name]["session_ids"],
            "settings": jobs[name]["settings"],
            "optimizer_policy": policy,
            "latent_diagnostics": qa.get("latent"),
            "regimes": {},
        }
        for field in ("all_columns_mae", "xp_only_mae"):
            entry[field] = final[field]
            entry[field + "_delta_percent"] = delta(final[field], ref[3][field])
            entry[field + "_parent_delta_percent"] = delta(final[field], initial[field])
        assert qa["regimes"].keys() == ref[1]["regimes"].keys()
        for regime, diagnostics in qa["regimes"].items():
            reference_qa = ref[1]["regimes"][regime]
            reference_features = {
                item["name"]: item for item in reference_qa["features"]
            }
            assert reference_features.keys() == {
                item["name"] for item in diagnostics["features"]
            }
            for item in diagnostics["features"]:
                base = reference_features[item["name"]]
                assert item["count"] == base["count"]
                columns.append(
                    {
                        "experiment": name,
                        "reference": reference,
                        "regime": regime,
                        "column": item["name"],
                        "count": item["count"],
                        "mae": item["mae"],
                        "reference_mae": base["mae"],
                        "delta_percent": delta(item["mae"], base["mae"]),
                    }
                )
            groups = {}
            for low, high in ((1, 10), (11, 30), (31, 55)):
                group = {}
                for band in ("bp", "rp", "combined"):
                    names = {
                        f"{channel}_{order}"
                        for channel in (("bp", "rp") if band == "combined" else (band,))
                        for order in range(low, high + 1)
                    }
                    selected = [
                        item
                        for item in diagnostics["features"]
                        if item["name"] in names
                    ]
                    assert len(selected) == len(names)
                    value = weighted(selected)
                    base = weighted([reference_features[key] for key in names])
                    value["delta_percent"] = delta(value["mae"], base["mae"])
                    group[band] = value
                groups[f"{low}-{high}"] = group
            spectral, base_spectral = diagnostics["spectral"], reference_qa["spectral"]
            spectral_summary = compare_spectral(spectral, base_spectral)
            current_bands = diagnostics["photometry_consistency"]["bands"]
            reference_bands = reference_qa["photometry_consistency"]["bands"]
            pooled = {}
            for state in ("all", "hidden", "visible"):
                current, baseline = [], []
                for item, base in zip(current_bands, reference_bands, strict=True):
                    assert item["column"] == base["column"]
                    value, base_value = (
                        (item, base) if state == "all" else (item[state], base[state])
                    )
                    assert value["count"] == base_value["count"]
                    current.append(value)
                    baseline.append(base_value)
                    bands.append(
                        {
                            "experiment": name,
                            "reference": reference,
                            "regime": regime,
                            "input": state,
                            "band": item["column"],
                            "count": value["count"],
                            "flux_mae": value["reference_flux_absolute_error_mean"],
                            "delta_percent": delta(
                                value["reference_flux_absolute_error_mean"],
                                base_value["reference_flux_absolute_error_mean"],
                            ),
                            "raw_magnitude_mae": item["magnitude_mae"]
                            if state == "all"
                            else None,
                            "nonpositive_flux_count": item[
                                "nonpositive_synthetic_count"
                            ]
                            if state == "all"
                            else None,
                        }
                    )
                value = weighted(current, "reference_flux_absolute_error_mean")
                baseline_value = weighted(
                    baseline, "reference_flux_absolute_error_mean"
                )
                value["delta_percent"] = delta(value["mae"], baseline_value["mae"])
                pooled[state] = value
            entry["regimes"][regime] = {
                "coefficient_orders": groups,
                "spectral": spectral_summary,
                "pooled_photometry": pooled,
                "nonpositive_flux_count": sum(
                    item["nonpositive_synthetic_count"] for item in current_bands
                ),
            }
        entries.append(entry)
    return entries, columns, bands


def seed_spread(entries):
    summary = {}
    for family in ("spectral", "photometry_spectral"):
        selected = [
            next(
                entry for entry in entries if entry["experiment"] == f"{family}_s{seed}"
            )
            for seed in (42, 43, 44)
        ]
        fields = {
            "xp_mae_delta_percent": [
                entry["xp_only_mae_delta_percent"] for entry in selected
            ],
            "spectral_mean_delta_percent": [
                entry["regimes"]["common"]["spectral"][
                    "integrated_absolute_fraction_mean_delta_percent"
                ]
                for entry in selected
            ],
            "photometry_flux_delta_percent": [
                entry["regimes"]["common"]["pooled_photometry"]["all"]["delta_percent"]
                for entry in selected
            ],
        }
        summary[family] = {
            key: {
                "seed_order": [42, 43, 44],
                "paired_values": values,
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "min": min(values),
                "max": max(values),
            }
            for key, values in fields.items()
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/xp/results-followup-e47")
    )
    parser.add_argument(
        "--launch",
        type=Path,
        default=Path("experiments/xp/launch-followup-e47-20261014.json"),
    )
    args = parser.parse_args()
    entries, columns, bands = compare(args.root, read_json(args.launch))
    spread = seed_spread(entries)
    summary = {
        "identity_checks": "passed",
        "parent_sha256": PARENT,
        "benchmark_source_id_sha256": BENCHMARK,
        "presentations": 20_000_000,
        "optimizer_steps": 612,
        "experiments": entries,
        "three_seed_paired_spread": spread,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    for filename, rows in (
        ("per-column-mae.csv", columns),
        ("per-band-photometry.csv", bands),
    ):
        with (args.root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = [
        "# Matched epoch47 follow-up experiments",
        "",
        "All runs pass fixed-budget final checkpoint, parent, benchmark, scaler-derived validation identity, initial prediction/target/mask arrays, model/data/training configuration, projection/calibration and checkpoint reload checks. Seed-specific RNG, LR/reset/warmup, dropout and auxiliary objectives are deliberate differences verified against launch settings.",
        "",
        "Negative percentage changes improve the named reference. Replication families compare with their seed-matched ordinary MAE control; all other variants compare with spectral_s42. Spectral diagnostics use the shared diagnostic snapshot. Photometric flux residuals use frozen training calibration and training reference flux; raw positive-flux-only magnitude errors must be read with nonpositive counts. Final checkpoints are used without validation-interval selection.",
        "",
    ]
    for entry in entries:
        spectral = entry["regimes"]["common"]["spectral"]
        photo = entry["regimes"]["common"]["pooled_photometry"]["all"]
        report.append(
            f"- {entry['experiment']} versus {entry['reference']}: XP MAE {entry['xp_only_mae_delta_percent']:+.3f}%; spectral mean {spectral['integrated_absolute_fraction_mean_delta_percent']:+.3f}%; calibrated photometric flux MAE {photo['delta_percent']:+.3f}%."
        )
    report += [
        "",
        "Three-seed paired percentage changes (mean ± sample SD; descriptive spread, no significance claim):",
        "",
    ]
    for family, fields in spread.items():
        for metric, values in fields.items():
            report.append(
                f"- {family}, {metric}: {values['mean']:+.3f} ± {values['sample_sd']:.3f}; range {values['min']:+.3f} to {values['max']:+.3f}."
            )
    report += [
        "",
        "The spectral auxiliary improves coefficient, spectral and calibrated photometric flux error in all three seeds. Photometry plus spectral improves coefficient MAE in all three, but its physical-consistency gains do not replicate uniformly: seed 43 worsens both spectra and photometric flux error. Keep that instability visible rather than selecting the favourable seed.",
        "",
        "Among single-seed variants, halving the inherited LR improves all three metrics over spectral_s42, while latent-view regularization gives smaller consistent gains. These are follow-up candidates rather than replicated conclusions. Reset with warmup improves spectra/photometry with a small coefficient penalty. Removing dropout, increasing the spectral weight to 0.3, adding multiscale spectral supervision, and scoring every decoder coefficient do not improve the joint coefficient/spectral result under the tested settings.",
    ]
    report += [
        "",
        "These are short continuations sharing one pretrained parent. Reconstruction and physical-consistency gains do not establish improved downstream representation; use the separate held-out probe analyses. Raw downloaded files remain unchanged. Counts/order groups/all mask regimes and deliberate experiment policies are preserved in comparison.json and CSV files.",
    ]
    (args.root / "comparison.md").write_text("\n".join(report) + "\n")
    print(
        json.dumps(
            {
                "completed": len(entries),
                "identity_checks": "passed",
                "output": str(args.root),
            }
        )
    )


if __name__ == "__main__":
    main()
