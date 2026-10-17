"""Audit final dependent-checkpoint probes and descriptive three-seed results.

No scientific significance test is inferred from three continuation seeds. Planned
jobs without a CLI checksum assertion require an independently verified saved
training checkpoint checksum matching the actual probe manifest checksum.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

_SPEC = importlib.util.spec_from_file_location(
    "dependent_representation_helpers",
    Path(__file__).with_name("compare_representation_followup.py"),
)
paired = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(paired)

SEEDS = (42, 43, 44)
FAMILIES = ("control", "spectral", "photometry_spectral")


def resolve_training_proof(root, settings):
    dependency = settings.get("depends_on")
    if not dependency or settings["mode"] != "probes":
        raise ValueError(
            "Unpinned dependent probe requires an explicit planned training dependency"
        )
    if settings["run_id"] != "probe_new_" + dependency:
        raise ValueError("Dependent probe name differs from declared training run")
    suffix = f"/{dependency}/{dependency}/checkpoint.pth"
    if not settings["checkpoint"].endswith(suffix):
        raise ValueError(
            "Dependent probe checkpoint path differs from planned training output"
        )
    proof_path = root / dependency / dependency / "checkpoint_verification.json"
    manifest_path = root / dependency / "manifest.json"
    metrics_path = root / dependency / dependency / "metrics.jsonl"
    if not all(path.exists() for path in (proof_path, manifest_path, metrics_path)):
        return None
    manifest, proof = paired.read(manifest_path), paired.read(proof_path)
    if manifest.get("status") != "complete":
        return None
    paired.equal(manifest["suite_id"], dependency, "Training run identity")
    paired.equal(
        manifest["checkpoint_sha256"], paired.PARENT, "Epoch47 training parent"
    )
    paired.equal(proof["parent_sha256"], paired.PARENT, "Reload proof parent")
    paired.equal(
        proof["reload_predictions_equal"], True, "Saved final checkpoint reload probe"
    )
    if not isinstance(proof.get("sha256"), str) or len(proof["sha256"]) != 64:
        raise ValueError("Malformed independently saved final checkpoint hash")
    paired.equal(
        manifest["presentations_per_experiment"],
        20000000,
        "Planned continuation budget",
    )
    paired.equal(manifest["sample"]["train_rows"], 9000000, "Shared training sample")
    paired.equal(
        manifest["sample"]["validation_rows"], 1000000, "Shared validation sample"
    )
    paired.equal(
        manifest["benchmark_source_id_sha256"],
        paired.BENCHMARK,
        "Training benchmark identity",
    )
    events = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    if not events or events[-1].get("phase") != "complete":
        return None
    final = events[-1]
    paired.equal(
        final["rows_seen_total"], 20000000, "Actual continuation presentations"
    )
    paired.equal(final["optimizer_steps"], 612, "Actual continuation optimizer steps")
    suffix_seed = int(dependency.rsplit("_s", 1)[1])
    paired.equal(final["seed"], suffix_seed, "Continuation training seed")
    return {
        "checkpoint_sha256": proof["sha256"],
        "reload_predictions_equal": True,
        "parent_checkpoint_sha256": paired.PARENT,
        "training_seed": suffix_seed,
        "training_presentations": final["rows_seen_total"],
        "optimizer_steps": final["optimizer_steps"],
        "checkpoint_proof_file": str(proof_path),
    }


def verify_dependent_manifest(manifest, settings, proof):
    if not settings.get("depends_on"):
        raise ValueError("Unpinned checksum exemption requires a declared dependency")
    paired.equal(
        proof["reload_predictions_equal"], True, "Independent final reload proof passed"
    )
    paired.equal(
        proof["parent_checkpoint_sha256"],
        paired.PARENT,
        "Independent final proof parent",
    )
    paired.equal(manifest["status"], "complete", "Dependent probe status")
    paired.equal(
        manifest["checkpoint_sha256"],
        proof["checkpoint_sha256"],
        "Runtime actual checksum versus independent final training proof",
    )
    paired.equal(
        manifest["arguments"]["checkpoint"],
        settings["checkpoint"],
        "Planned dependent checkpoint source",
    )
    planned = settings.get("checkpoint_sha256")
    asserted = manifest["arguments"].get("checkpoint_sha256")
    if asserted is None:
        if planned is not None:
            raise ValueError("Declared CLI checksum assertion was not executed")
        cli_asserted = False
    else:
        paired.equal(
            asserted,
            proof["checkpoint_sha256"],
            "Runtime declared CLI checksum assertion",
        )
        if planned is not None:
            paired.equal(planned, asserted, "Planned checksum and runtime assertion")
        cli_asserted = True
    paired.equal(
        manifest["parent_checkpoint_sha256"],
        paired.PARENT,
        "Dependent representation parent",
    )
    paired.equal(
        manifest["benchmark_source_id_sha256"],
        paired.BENCHMARK,
        "Probe benchmark identity",
    )
    paired.equal(
        manifest["scaler_sha256"], paired.SCALER, "Probe common reference scaler"
    )
    paired.equal(manifest["arguments"]["seed"], 42, "Fixed downstream adaptation seed")
    return {
        "actual_checksum_matches_independent_final_proof": True,
        "runtime_cli_checksum_assertion_executed": cli_asserted,
        "planned_dependency": settings["depends_on"],
    }


def load_dependent(root, settings, split):
    folder = root / settings["run_id"] / split
    paths = [
        folder / name
        for name in ("manifest.json", "results.json", "probe_predictions.npz")
    ]
    if not all(path.exists() for path in paths):
        return None
    manifest = paired.read(paths[0])
    if manifest.get("status") != "complete":
        return None
    proof = resolve_training_proof(root, settings)
    if proof is None:
        return None
    checksum_verification = verify_dependent_manifest(manifest, settings, proof)
    with np.load(paths[2], allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    return {
        "manifest": manifest,
        "results": paired.read(paths[1]),
        "arrays": arrays,
        "folder": folder,
        "training_proof": proof,
        "checksum_verification": checksum_verification,
    }


def baselines_for(dependency):
    stem, suffix = dependency.rsplit("_s", 1)
    seed = int(suffix)
    if stem in FAMILIES:
        baseline = f"probe_new_control_s{seed}"
    else:
        if seed != 42:
            raise ValueError(
                "Single-seed knob comparison requires the declared spectral_s42 reference"
            )
        baseline = "probe_new_spectral_s42"
    result = {"matched_reference": baseline}
    if stem == "photometry_spectral":
        result["matched_spectral"] = f"probe_new_spectral_s{seed}"
    return result


def metric_values(group, target):
    report = group["results"]["targets"][target]
    metrics = {"ridge": report["ridge"]["mae"]}
    for budget, values in report["budgets"].items():
        for method in ("ridge", "head", "finetuned_encoder_head"):
            metrics[f"{method}_{budget}"] = values[method]["mae"]
    return metrics


def descriptive(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def summarize_families(groups):
    output = {}
    for family in FAMILIES:
        available = [
            seed
            for seed in SEEDS
            if f"probe_new_{family}_s{seed}" in groups
            and f"probe_new_control_s{seed}" in groups
        ]
        if not available:
            continue
        summary = {
            "seeds": available,
            "three_seeds_complete": available == list(SEEDS),
            "targets": {},
        }
        for target in paired.TARGETS:
            by_seed = [
                metric_values(groups[f"probe_new_{family}_s{seed}"], target)
                for seed in available
            ]
            controls = [
                metric_values(groups[f"probe_new_control_s{seed}"], target)
                for seed in available
            ]
            paired.equal(
                [set(item) for item in by_seed],
                [set(item) for item in controls],
                "Seed-matched metric budgets",
            )
            target_summary = {"unit": paired.UNITS[target], "metrics": {}}
            for metric in by_seed[0]:
                actual = [item[metric] for item in by_seed]
                baseline = [item[metric] for item in controls]
                delta = [
                    paired.percentage(value, reference)
                    for value, reference in zip(actual, baseline, strict=True)
                ]
                target_summary["metrics"][metric] = {
                    "physical_mae": descriptive(actual),
                    "matched_control_mae": descriptive(baseline),
                    "matched_control_change_percent": descriptive(delta),
                }
            summary["targets"][target] = target_summary
        output[family] = summary
    return output


def markdown(report):
    lines = [
        "# New continuation representation comparison",
        "",
        f"Status: **{report['status']}**.",
        "",
        "Every reported representation uses an independently reload-verified final checkpoint with20M continuation presentations and612 optimizer steps. "
        "Checksum assertions omitted from predeclared dependent jobs are not claimed to have executed: the actual runtime checksum is independently matched to the completed training proof. "
        "Downstream IDs, labels, normalization audit, scaler, head budgets and paired prediction arrays are checked identical. "
        "Negative changes mean lower physical label MAE. Three-seed summaries are descriptive continuation-seed variability, not significance tests or independent fresh-pretraining replications.",
        "",
    ]
    if report["pending"]:
        lines += ["Pending groups: " + ", ".join(report["pending"]) + ".", ""]
    for split, data in report["splits"].items():
        lines += [f"## {split} holdout", ""]
        for family, summary in data["three_seed_families"].items():
            parts = []
            for target in paired.TARGETS:
                statistics = summary["targets"][target]["metrics"]["ridge"][
                    "matched_control_change_percent"
                ]
                std = (
                    "not available"
                    if statistics["sample_std"] is None
                    else f"{statistics['sample_std']:.2f}%"
                )
                parts.append(f"{target}: {statistics['mean']:+.2f}% (sample SD {std})")
            lines.append(
                f"- **{family}**, frozen ridge, n={len(summary['seeds'])} continuation seeds: "
                + "; ".join(parts)
                + "."
            )
        lines.append("")
        for run_id, result in data["experiments"].items():
            dependency = result["checksum_verification"]["planned_dependency"]
            if dependency.rsplit("_s", 1)[0] in FAMILIES:
                continue
            parts = []
            for target in paired.TARGETS:
                report_target = result["targets"][target]
                largest = max(map(int, report_target["budgets"]))
                comparison = report_target["comparisons"]["matched_reference"]
                parts.append(
                    f"{target}: ridge {comparison['ridge']['mae_change_percent']:+.2f}%, frozen head {comparison[f'head_{largest}']['mae_change_percent']:+.2f}%, encoder+head {comparison[f'finetuned_{largest}']['mae_change_percent']:+.2f}%"
                )
            lines.append(
                f"- **{dependency}** versus spectral_s42, largest available label budget: "
                + "; ".join(parts)
                + "."
            )
        lines.append("")
    lines += [
        "The JSON preserves all1k/10k/up-to100k label budgets, physical-unit errors, actual training checkpoint hashes and exact test-ID/target hashes. "
        "The200-step encoder+head results are fixed-budget adaptation pilots, not fully converged downstream performance. "
        "Labelled test stars are held out from supervised fitting but not certified unseen during unsupervised pretraining.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/xp/results-followup-e47")
    )
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/xp-followup-analysis.yaml")
    )
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--include-previous", action="store_true")
    args = parser.parse_args()
    entries = [
        item
        for item in yaml.safe_load(args.spec.read_text())["analyses"]
        if item.get("depends_on") and item.get("mode") == "probes"
    ]
    if len(entries) != 20 or len({item["run_id"] for item in entries}) != 20:
        raise ValueError(
            "Expected the20 explicitly planned distinct dependent representation probes"
        )
    report = {
        "status": "complete",
        "parent_checkpoint_sha256": paired.PARENT,
        "benchmark_source_id_sha256": paired.BENCHMARK,
        "scaler_sha256": paired.SCALER,
        "statistics_policy": "descriptive sample SD across continuation seeds42/43/44; no significance test",
        "pending": [],
        "splits": {},
    }
    previous_settings = [
        item
        for item in yaml.safe_load(args.spec.read_text())["analyses"]
        if item.get("run_id") in (paired.CONTROL, paired.ANCESTOR)
    ]
    for split in ("identity", "sky"):
        loaded = {
            item["run_id"]: load_dependent(args.root, item, split) for item in entries
        }
        report["pending"].extend(
            name + "/" + split for name, value in loaded.items() if value is None
        )
        available = {name: value for name, value in loaded.items() if value is not None}
        if "probe_new_control_s42" not in available:
            continue
        policy_reference = available["probe_new_control_s42"]
        for group in available.values():
            paired.verify_probe(group, policy_reference)
        previous = {}
        if args.include_previous:
            for settings in previous_settings:
                group = paired.load_group(args.root, settings, split)
                if group is None:
                    report["pending"].append(settings["run_id"] + "/" + split)
                else:
                    paired.verify_probe(group, policy_reference)
                    previous[settings["run_id"]] = group
        comparisons = {}
        comparable = {}
        for settings in entries:
            run_id = settings["run_id"]
            if run_id not in available:
                continue
            names = baselines_for(settings["depends_on"])
            missing = [name for name in names.values() if name not in available]
            if missing:
                report["pending"].append(
                    run_id + "/" + split + " (waiting for baseline)"
                )
                continue
            references = {role: available[name] for role, name in names.items()}
            references.update(previous)
            result = paired.probe_comparison(available[run_id], references)
            result["references"] = names
            result["training_proof"] = available[run_id]["training_proof"]
            result["checksum_verification"] = available[run_id]["checksum_verification"]
            comparisons[run_id] = result
            comparable[run_id] = available[run_id]
        if comparisons:
            report["splits"][split] = {
                "paired_arrays_verified": True,
                "experiments": comparisons,
                "three_seed_families": summarize_families(comparable),
            }
    if report["pending"]:
        if not args.partial:
            raise ValueError(
                "Incomplete dependent representation artifacts: "
                + ", ".join(report["pending"])
            )
        report["status"] = "preliminary_partial"
    if not report["splits"]:
        raise ValueError("No completed paired dependent-probe baseline available")
    output = args.root / "new-representation-comparison.json"
    temporary = output.with_suffix(".json.building")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    (args.root / "new-representation-comparison.md").write_text(markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "split_experiment_counts": {
                    name: len(value["experiments"])
                    for name, value in report["splits"].items()
                },
                "pending_count": len(report["pending"]),
            }
        )
    )


if __name__ == "__main__":
    main()
