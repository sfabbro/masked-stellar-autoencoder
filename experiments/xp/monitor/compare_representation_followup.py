"""Audit paired frozen-encoder/adaptation probes and inference refinement.

Requires actual paired prediction arrays before reporting probe gains. --partial
permits finished split groups while other splits/jobs remain pending.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

PARENT = "4ee3c98c6db2e75d50aa54611a2beedd5ecbbb70c5af94e4c718d23eae208482"
BENCHMARK = "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
SCALER = "5fd976729dfe5ccd7b4ec2ce080e679c49f1ca5244f779cad45e148fa4836037"
CONTROL = "probe_unchanged_control"
ANCESTOR = "probe_parent47"
TARGETS = ("teff", "logg", "fe_h")
UNITS = {"teff": "K", "logg": "dex", "fe_h": "dex"}
PROBE_ARGUMENTS = (
    "mode",
    "config",
    "benchmark",
    "labels_fits",
    "fits_xp_units",
    "targets",
    "blocked",
    "split",
    "seed",
    "train_rows",
    "tune_rows",
    "test_rows",
    "head_steps",
    "finetune_steps",
    "batch_size",
)
LATENT_ARGUMENTS = (
    "mode",
    "config",
    "benchmark",
    "seed",
    "test_rows",
    "prior_rows",
    "batch_size",
    "refinement_mask",
    "refinement_steps",
    "refinement_lr",
    "prior_weight",
)


def read(path):
    return json.loads(path.read_text())


def equal(value, expected, description):
    if value != expected:
        raise ValueError(f"{description}: {value!r} != {expected!r}")


def percentage(value, reference):
    return 100.0 * (value / reference - 1.0)


def score_pairs(prediction, target, baseline):
    if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
        raise ValueError("Nonfinite paired prediction")
    error = np.abs(prediction.astype(np.float64) - target)
    other = np.abs(baseline.astype(np.float64) - target)
    return {
        "count": len(target),
        "mae": float(error.mean()),
        "baseline_mae": float(other.mean()),
        "paired_mae_difference": float((error - other).mean()),
        "fraction_smaller_absolute_error": float((error < other).mean()),
        "mae_change_percent": percentage(error.mean(), other.mean()),
    }


def verify_manifest(manifest, settings):
    equal(manifest["status"], "complete", settings["run_id"] + " status")
    equal(
        manifest["checkpoint_sha256"],
        settings["checkpoint_sha256"],
        "Declared final checkpoint checksum",
    )
    equal(
        manifest["arguments"]["checkpoint_sha256"],
        settings["checkpoint_sha256"],
        "Runtime checksum assertion",
    )
    equal(
        manifest["arguments"]["checkpoint"], settings["checkpoint"], "Checkpoint path"
    )
    expected_parent = None if settings["checkpoint_sha256"] == PARENT else PARENT
    equal(
        manifest["parent_checkpoint_sha256"], expected_parent, "Epoch47 parent identity"
    )
    equal(
        manifest["benchmark_source_id_sha256"], BENCHMARK, "Benchmark sample identity"
    )
    equal(manifest["scaler_sha256"], SCALER, "Epoch27 reference scaler")
    equal(manifest["arguments"]["seed"], 42, "Probe/mask seed")


def load_group(root, settings, split):
    folder = root / settings["run_id"] / split
    required = [folder / "manifest.json", folder / "results.json"]
    if not all(path.exists() for path in required):
        return None
    manifest = read(required[0])
    if manifest.get("status") != "complete":
        return None
    verify_manifest(manifest, settings)
    results = read(required[1])
    filename = (
        "probe_predictions.npz"
        if settings["mode"] == "probes"
        else "latent_refinement.npz"
    )
    path = folder / filename
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    return {
        "manifest": manifest,
        "results": results,
        "arrays": arrays,
        "folder": folder,
    }


def policies(group, names):
    return {name: group["manifest"]["arguments"][name] for name in names}


def verify_probe(group, reference):
    equal(
        policies(group, PROBE_ARGUMENTS),
        policies(reference, PROBE_ARGUMENTS),
        "Shared supervised policy",
    )
    equal(
        group["manifest"]["script_sha256"],
        reference["manifest"]["script_sha256"],
        "Probe implementation",
    )
    result, base = group["results"], reference["results"]
    for field in (
        "split_counts",
        "split_source_id_sha256",
        "duplicate_source_ids_removed",
    ):
        equal(result[field], base[field], "Label dataset " + field)
    equal(
        result["fits_units_audit"],
        base["fits_units_audit"],
        "Matched-source units audit",
    )
    audit = result["fits_units_audit"]
    if audit["matched_stars"] < 128 or audit["chosen_units"] != "g_normalized":
        raise ValueError("FITS matched-source normalization audit failed")
    if not audit["candidates"]["g_normalized"]["passed"]:
        raise ValueError("FITS units check is not passed")
    equal(tuple(result["targets"]), TARGETS, "Stellar probe labels")
    for target in TARGETS:
        report, base_report = result["targets"][target], base["targets"][target]
        truth = group["arrays"][f"{target}_truth"]
        ids = group["arrays"][f"{target}_source_ids"]
        np.testing.assert_array_equal(ids, reference["arrays"][f"{target}_source_ids"])
        np.testing.assert_array_equal(truth, reference["arrays"][f"{target}_truth"])
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate labelled test source IDs")
        if not np.isfinite(truth).all():
            raise ValueError("Nonfinite labels in paired prediction payload")
        equal(report["ridge"]["count"], len(truth), "Physical label test counts")
        equal(
            set(report["budgets"]),
            set(base_report["budgets"]),
            "Effective finite-label budgets",
        )
        if not {"1000", "10000"}.issubset(report["budgets"]):
            raise ValueError("Missing required few-label budgets")
        largest = max(map(int, report["budgets"]))
        if not 10000 <= largest <= 100000:
            raise ValueError("Unexpected capped training-label budget")
        for field in ("train_target_center", "train_target_scale"):
            equal(
                report["ridge_policy"][field],
                base_report["ridge_policy"][field],
                "Train-only target scaler",
            )
        for budget, measures in report["budgets"].items():
            compare = base_report["budgets"][budget]
            for field in ("head_steps", "head_lr", "encoder_frozen", "finetune_policy"):
                equal(
                    measures[field],
                    compare[field],
                    "Equal supervised adaptation budget",
                )
            for field in ("train_target_center", "train_target_scale"):
                equal(
                    measures["ridge_policy"][field],
                    compare["ridge_policy"][field],
                    "Few-label target scaler",
                )
            for name in ("ridge", "head", "finetuned_encoder_head"):
                equal(
                    measures[name]["count"],
                    len(truth),
                    "Equal labelled evaluation counts",
                )
        keys = [f"{target}_ridge"]
        keys += [
            f"{target}_{method}_{budget}"
            for method in ("head", "finetuned")
            for budget in report["budgets"]
        ]
        for key in keys:
            prediction = group["arrays"][key]
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise ValueError("Malformed/nonfinite prediction " + key)
            method = (
                "ridge"
                if key.endswith("_ridge")
                else "head"
                if "_head_" in key
                else "finetuned_encoder_head"
            )
            declared = (
                report["ridge"]
                if method == "ridge"
                else report["budgets"][key.rsplit("_", 1)[1]][method]
            )
            mae = np.abs(prediction.astype(np.float64) - truth).mean()
            np.testing.assert_allclose(mae, declared["mae"], rtol=1e-12, atol=1e-12)


def probe_comparison(group, references):
    output = {
        "checkpoint_sha256": group["manifest"]["checkpoint_sha256"],
        "targets": {},
    }
    for target in TARGETS:
        report = group["results"]["targets"][target]
        truth = group["arrays"][f"{target}_truth"]
        target_result = {
            "unit": UNITS[target],
            "test_source_id_sha256": hashlib.sha256(
                group["arrays"][f"{target}_source_ids"].tobytes()
            ).hexdigest(),
            "test_target_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
            "ridge": report["ridge"],
            "budgets": report["budgets"],
            "comparisons": {},
        }
        for baseline_name, base in references.items():
            comparisons = {}
            keys = [f"{target}_ridge"] + [
                f"{target}_{method}_{budget}"
                for method in ("head", "finetuned")
                for budget in report["budgets"]
            ]
            for key in keys:
                name = key.removeprefix(target + "_")
                comparisons[name] = score_pairs(
                    group["arrays"][key], truth, base["arrays"][key]
                )
            # Few-label ridge predictions were not archived in the runner.
            # These entries are aggregated and clearly not called paired arrays.
            for budget in report["budgets"]:
                current = report["budgets"][budget]["ridge"]["mae"]
                baseline = base["results"]["targets"][target]["budgets"][budget][
                    "ridge"
                ]["mae"]
                comparisons[f"ridge_{budget}_aggregate_only"] = {
                    "mae": current,
                    "baseline_mae": baseline,
                    "mae_change_percent": percentage(current, baseline),
                    "pairing": "source-ID and target arrays verified; budget-ridge prediction arrays not archived",
                }
            target_result["comparisons"][baseline_name] = comparisons
        output["targets"][target] = target_result
    return output


def verify_latent(group, reference):
    equal(
        policies(group, LATENT_ARGUMENTS),
        policies(reference, LATENT_ARGUMENTS),
        "Inference refinement policy/masks",
    )
    equal(
        group["manifest"]["script_sha256"],
        reference["manifest"]["script_sha256"],
        "Inference implementation",
    )
    for field in (
        "rows",
        "train_prior_source_id_sha256",
        "validation_source_id_sha256",
        "policy",
    ):
        equal(
            group["results"][field],
            reference["results"][field],
            "Inference paired rows " + field,
        )
    for name in ("source_ids", "hidden"):
        np.testing.assert_array_equal(group["arrays"][name], reference["arrays"][name])
    ids = group["arrays"]["source_ids"]
    equal(len(ids), 20000, "Bounded validation-prefix inference rows")
    equal(len(np.unique(ids)), len(ids), "Unique inference validation IDs")
    equal(
        hashlib.sha256(ids.tobytes()).hexdigest(),
        group["results"]["validation_source_id_sha256"],
        "Inference ID payload checksum",
    )
    before, after = group["arrays"]["before"], group["arrays"]["after"]
    if before.shape != after.shape or before.shape[0] != len(ids):
        raise ValueError("Inference prediction shape mismatch")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Nonfinite final inference prediction")
    for name in ("all_hidden", "xp_hidden"):
        equal(
            group["results"]["scores"][name]["before"]["count"],
            reference["results"]["scores"][name]["before"]["count"],
            "Inference heldout count",
        )
        equal(
            group["results"]["scores"][name]["before"]["count"],
            group["results"]["scores"][name]["after"]["count"],
            "Before/after heldout pairing",
        )


def report_markdown(report):
    lines = [
        "# Paired representation probes and latent refinement",
        "",
        f"Status: **{report['status']}**.",
        "",
        "These are paired downstream evaluation pilots, not independent pretraining-seed confirmations. "
        "Labels are held out from supervised fitting; source stars are not certified unseen during unsupervised pretraining. "
        "Physical Teff/logg/metallicity MAE is reported in K/dex. Negative percentages mean lower error.",
        "",
    ]
    if report["pending"]:
        lines += ["Pending groups: " + ", ".join(report["pending"]) + ".", ""]
    for split, data in report["probes"].items():
        lines += [
            f"## {split} holdout",
            "",
            "All source IDs, labels, label budgets, fitted target scales, checkpoint ancestry, benchmark/scaler and 128 matched coefficient-unit checks agree. Comparisons use final fixed-budget supervised adaptation.",
            "",
        ]
        for run_id, result in data["experiments"].items():
            if run_id in (CONTROL, ANCESTOR):
                continue
            parts = []
            for target in TARGETS:
                comparisons = result["targets"][target]["comparisons"]
                current = result["targets"][target]["ridge"]["mae"]
                control = comparisons[CONTROL]["ridge"]["mae_change_percent"]
                parent = comparisons[ANCESTOR]["ridge"]["mae_change_percent"]
                parts.append(
                    f"{target} {current:.4g}{UNITS[target]} ({control:+.2f}% versus control; {parent:+.2f}% versus parent)"
                )
            lines.append(
                f"- **{run_id.removeprefix('probe_')}**, frozen ridge: "
                + "; ".join(parts)
                + "."
            )
            heads = []
            for target in TARGETS:
                c = result["targets"][target]["comparisons"][CONTROL]
                largest = max(map(int, result["targets"][target]["budgets"]))
                heads.append(
                    f"{target} (n={largest}) frozen head {c[f'head_{largest}']['mae_change_percent']:+.2f}%; encoder+head {c[f'finetuned_{largest}']['mae_change_percent']:+.2f}%"
                )
            lines.append(
                "  At the largest available label budget (capped at 100k), versus control: "
                + "; ".join(heads)
                + "."
            )
        lines.append("")
    if report["latent_refinement"]:
        lines += [
            "## Frozen-decoder per-star inference refinement",
            "",
            "Fixed 30 final steps, full XP hidden, 20k validation-prefix stars; visible-feature fit only. This is inference refinement, not improved encoder training.",
            "",
        ]
        for run_id, result in report["latent_refinement"].items():
            xp, all_features = (
                result["scores"]["xp_hidden"],
                result["scores"]["all_hidden"],
            )
            lines.append(
                f"- **{run_id.removeprefix('latent_refine_')}**: hidden XP MAE {xp['before']['mae']:.6f} → {xp['after']['mae']:.6f} ({xp['mae_change_percent']:+.3f}%); all-hidden MAE {all_features['mae_change_percent']:+.3f}%."
            )
    lines += [
        "",
        "The archived JSON includes all three label budgets and paired prediction comparisons. "
        "No best-validation checkpoint, test-selected adaptation step or training-seed significance claim is used.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/xp/results-followup-e47")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xp-followup-analysis.yaml")
    )
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    settings = [
        item
        for item in yaml.safe_load(args.config.read_text())["analyses"]
        if "depends_on" not in item
    ]
    probes = [item for item in settings if item["mode"] == "probes"]
    latent = [item for item in settings if item["mode"] == "latent_refinement"]
    report = {
        "status": "complete",
        "verification": {},
        "pending": [],
        "probes": {},
        "latent_refinement": {},
    }
    for split in ("identity", "sky"):
        groups = {item["run_id"]: load_group(args.root, item, split) for item in probes}
        missing = [f"{name}/{split}" for name, value in groups.items() if value is None]
        if missing:
            report["pending"].extend(missing)
            continue
        control, parent = groups[CONTROL], groups[ANCESTOR]
        verify_probe(parent, control)
        result = {
            "verification": {
                "paired_arrays": True,
                "split_source_id_sha256": control["results"]["split_source_id_sha256"],
                "fits_units_audit": control["results"]["fits_units_audit"],
            },
            "experiments": {},
        }
        for run_id, group in groups.items():
            verify_probe(group, control)
            result["experiments"][run_id] = probe_comparison(
                group, {CONTROL: control, ANCESTOR: parent}
            )
        report["probes"][split] = result
    groups = {
        item["run_id"]: load_group(args.root, item, "inference") for item in latent
    }
    missing = [f"{name}/inference" for name, value in groups.items() if value is None]
    if missing:
        report["pending"].extend(missing)
    else:
        baseline = groups["latent_refine_parent47"]
        for run_id, group in groups.items():
            verify_latent(group, baseline)
            result = {
                "checkpoint_sha256": group["manifest"]["checkpoint_sha256"],
                "scores": group["results"]["scores"],
            }
            for name in result["scores"]:
                metrics = result["scores"][name]
                metrics["mae_change_percent"] = percentage(
                    metrics["after"]["mae"], metrics["before"]["mae"]
                )
            report["latent_refinement"][run_id] = result
    if report["pending"]:
        if not args.partial:
            raise ValueError(
                "Incomplete paired analysis artifacts: " + ", ".join(report["pending"])
            )
        report["status"] = "preliminary_partial"
    if not report["probes"] and not report["latent_refinement"]:
        raise ValueError(
            "No fully paired completed group available; no comparison written"
        )
    report["verification"] = {
        "parent_sha256": PARENT,
        "benchmark_source_id_sha256": BENCHMARK,
        "scaler_sha256": SCALER,
        "uses_paired_archived_arrays": True,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root / "analysis-comparison.json"
    temporary = path.with_suffix(".json.building")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    (args.root / "analysis-comparison.md").write_text(report_markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "probes": list(report["probes"]),
                "latent_refinement": len(report["latent_refinement"]),
                "pending": report["pending"],
            }
        )
    )


if __name__ == "__main__":
    main()
