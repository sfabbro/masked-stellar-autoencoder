"""Compare a later standard checkpoint with saved experiment representations.

This is a practical checkpoint comparison with UNEQUAL encoder pretraining
budgets; it is not a controlled loss-objective ablation. Original reports remain
untouched. Both label holdouts require exact paired prediction payloads.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import yaml

_SPEC = importlib.util.spec_from_file_location(
    "representation_followup_helpers",
    Path(__file__).with_name("compare_representation_followup.py"),
)
paired = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(paired)


def latest_settings(path):
    document = yaml.safe_load(path.read_text())
    entries = document.get("analyses", [document])
    entries = [item for item in entries if item.get("mode") == "probes"]
    if len(entries) != 1:
        raise ValueError(
            "Latest standard specification requires exactly one probes entry"
        )
    result = entries[0]
    result = dict(result)
    result["checkpoint_epoch"] = result.get(
        "checkpoint_epoch", document.get("checkpoint_epoch")
    )
    for key in ("run_id", "checkpoint", "checkpoint_sha256", "checkpoint_epoch"):
        if not result.get(key):
            raise ValueError(f"Latest standard specification lacks {key}")
    epoch = result["checkpoint_epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("Latest standard checkpoint_epoch must be a positive integer")
    filename_epoch = re.search(r"checkpoint_(\d+)\.pth$", result["checkpoint"])
    if filename_epoch is None or int(filename_epoch[1]) != epoch:
        raise ValueError("Declared checkpoint_epoch differs from checkpoint filename")
    return result


def verify_latest_manifest(manifest, settings):
    """Standard checkpoint has no experiment parent; never assume epoch47 ancestry."""
    paired.equal(manifest["status"], "complete", "Latest standard probe status")
    paired.equal(
        manifest["checkpoint_sha256"],
        settings["checkpoint_sha256"],
        "Latest standard pinned checksum",
    )
    paired.equal(
        manifest["arguments"]["checkpoint_sha256"],
        settings["checkpoint_sha256"],
        "Latest standard runtime checksum assertion",
    )
    paired.equal(
        manifest["arguments"]["checkpoint"],
        settings["checkpoint"],
        "Latest standard source path",
    )
    paired.equal(
        manifest["parent_checkpoint_sha256"],
        None,
        "Standard checkpoint has no experiment-only parent",
    )
    paired.equal(
        manifest["benchmark_source_id_sha256"],
        paired.BENCHMARK,
        "Shared benchmark source IDs",
    )
    paired.equal(
        manifest["scaler_sha256"], paired.SCALER, "Shared epoch27 feature scaler"
    )
    paired.equal(manifest["arguments"]["seed"], 42, "Identical supervised probe seed")
    paired.equal(
        manifest["arguments"]["mode"], "probes", "Read-only saved-checkpoint probes"
    )


def load_latest(root, split, settings):
    folder = root / split
    paths = [
        folder / name
        for name in ("manifest.json", "results.json", "probe_predictions.npz")
    ]
    if not all(path.exists() for path in paths):
        return None
    manifest = paired.read(paths[0])
    if manifest.get("status") != "complete":
        return None
    verify_latest_manifest(manifest, settings)
    with np.load(paths[2], allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    return {
        "manifest": manifest,
        "results": paired.read(paths[1]),
        "arrays": arrays,
        "folder": folder,
    }


def markdown(report):
    lines = [
        "# Latest standard checkpoint versus reconstruction experiments",
        "",
        f"Status: **{report['status']}**.",
        "",
        "**Unequal pretraining budgets:** the later standard checkpoint continued full-catalogue training; "
        "the experiment checkpoints started at epoch47 and received20M presentations. This compares currently "
        "available models on identical downstream data, not the causal effect of their objectives. "
        "Supervised fit/tuning/test IDs, physical labels, adaptation budgets, feature scaler, normalization "
        "and probe implementation are checked equal. Negative percentages mean lower error than the latest standard.",
        "",
        f"Latest standard epoch: **{report['latest_checkpoint_epoch']}**. Checksum: `"
        + report["latest_checkpoint_sha256"]
        + "`.",
        "",
    ]
    if report["pending"]:
        lines += ["Pending: " + ", ".join(report["pending"]) + ".", ""]
    for split, data in report["splits"].items():
        lines += [f"## {split} holdout", ""]
        absolute = []
        for target in paired.TARGETS:
            latest = data["latest"]["targets"][target]
            largest = max(map(int, latest["budgets"]))
            absolute.append(
                f"{target}: ridge {latest['ridge']['mae']:.5g}{paired.UNITS[target]}, frozen head {latest['budgets'][str(largest)]['head']['mae']:.5g}{paired.UNITS[target]}, encoder+head {latest['budgets'][str(largest)]['finetuned_encoder_head']['mae']:.5g}{paired.UNITS[target]}"
            )
        lines += [
            "Latest standard, largest available label budget: "
            + "; ".join(absolute)
            + ".",
            "",
        ]
        for run_id, result in data["experiments"].items():
            measures = []
            for target in paired.TARGETS:
                report_target = result["targets"][target]
                largest = max(map(int, report_target["budgets"]))
                changes = report_target["comparisons"]["latest_standard"]
                measures.append(
                    f"{target} ridge {changes['ridge']['mae_change_percent']:+.2f}%, head {changes[f'head_{largest}']['mae_change_percent']:+.2f}%, encoder+head {changes[f'finetuned_{largest}']['mae_change_percent']:+.2f}%"
                )
            lines.append(
                f"- **{run_id.removeprefix('probe_')}**: " + "; ".join(measures) + "."
            )
        lines.append("")
    lines += [
        "Short200-step fine-tuning is a fixed-budget adaptation pilot, not a fully converged downstream model. "
        "No checkpoint, step or adaptation hyperparameter is selected using final test errors. "
        "The JSON retains all label budgets and exact paired-error comparisons.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, default=57)
    parser.add_argument(
        "--latest-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--latest-spec",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments/xp/results-followup-e47"),
    )
    parser.add_argument(
        "--experiments-spec",
        type=Path,
        default=Path("configs/xp-followup-analysis.yaml"),
    )
    parser.add_argument("--partial", action="store_true")
    parser.add_argument(
        "--include-new",
        action="store_true",
        help="Also require all20 planned dependent follow-up checkpoints",
    )
    args = parser.parse_args()
    if args.latest_spec is None:
        args.latest_spec = Path(
            f"configs/xp-latest-standard-e{args.epoch}-analysis.yaml"
        )
    settings = latest_settings(args.latest_spec)
    paired.equal(
        settings["checkpoint_epoch"], args.epoch, "Requested latest checkpoint epoch"
    )
    if args.latest_root is None:
        args.latest_root = (
            Path(f"experiments/xp/results-latest-standard-e{args.epoch}")
            / settings["run_id"]
        )
    all_entries = yaml.safe_load(args.experiments_spec.read_text())["analyses"]
    entries = [
        item
        for item in all_entries
        if item.get("mode") == "probes" and "depends_on" not in item
    ]
    new_entries = []
    new_helpers = None
    if args.include_new:
        new_entries = [
            item
            for item in all_entries
            if item.get("mode") == "probes" and item.get("depends_on")
        ]
        if len(new_entries) != 20:
            raise ValueError(
                "Expected all20 planned dependent checkpoints for --include-new"
            )
        specification = importlib.util.spec_from_file_location(
            "latest_dependent_helpers",
            Path(__file__).with_name("compare_new_representations.py"),
        )
        new_helpers = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(new_helpers)
    report = {
        "status": "complete",
        "role": "unequal_encoder_pretraining_budget_checkpoint_comparison",
        "latest_run_id": settings["run_id"],
        "latest_checkpoint": settings["checkpoint"],
        "latest_checkpoint_epoch": settings["checkpoint_epoch"],
        "latest_checkpoint_sha256": settings["checkpoint_sha256"],
        "latest_parent_checkpoint_sha256": None,
        "experiment_start_parent_sha256": paired.PARENT,
        "benchmark_source_id_sha256": paired.BENCHMARK,
        "scaler_sha256": paired.SCALER,
        "encoder_pretraining_budgets_equal": False,
        "includes_new_dependent_experiments": args.include_new,
        "pending": [],
        "splits": {},
    }
    for split in ("identity", "sky"):
        latest = load_latest(args.latest_root, split, settings)
        if latest is None:
            report["pending"].append(settings["run_id"] + "/" + split)
            continue
        groups = {
            item["run_id"]: paired.load_group(args.experiments_root, item, split)
            for item in entries
        }
        pending = [
            name + "/" + split for name, value in groups.items() if value is None
        ]
        if pending:
            report["pending"].extend(pending)
            continue
        control = groups[paired.CONTROL]
        paired.verify_probe(latest, control)
        data = {
            "verification": {
                "same_test_ids_and_targets": True,
                "same_fit_tune_ids_and_target_scalers": True,
                "same_label_adaptation_budgets": True,
                "fits_units_audit": latest["results"]["fits_units_audit"],
                "split_source_id_sha256": latest["results"]["split_source_id_sha256"],
                "probe_script_sha256": latest["manifest"]["script_sha256"],
            },
            "latest": paired.probe_comparison(
                latest,
                {paired.CONTROL: control, paired.ANCESTOR: groups[paired.ANCESTOR]},
            ),
            "experiments": {},
        }
        for run_id, group in groups.items():
            paired.verify_probe(group, latest)
            data["experiments"][run_id] = paired.probe_comparison(
                group, {"latest_standard": latest}
            )
        for settings_new in new_entries:
            group = new_helpers.load_dependent(
                args.experiments_root, settings_new, split
            )
            if group is None:
                report["pending"].append(settings_new["run_id"] + "/" + split)
                continue
            paired.verify_probe(group, latest)
            result = paired.probe_comparison(group, {"latest_standard": latest})
            result["training_proof"] = group["training_proof"]
            result["checksum_verification"] = group["checksum_verification"]
            data["experiments"][settings_new["run_id"]] = result
        report["splits"][split] = data
    if report["pending"]:
        if not args.partial:
            raise ValueError(
                "Incomplete paired latest-checkpoint artifacts: "
                + ", ".join(report["pending"])
            )
        report["status"] = "preliminary_partial"
    if not report["splits"]:
        raise ValueError("No fully paired latest-standard holdout available")
    output = args.latest_root.parent
    stem = "latest-vs-all-experiments" if args.include_new else "latest-vs-experiments"
    path = output / f"{stem}.json"
    temporary = path.with_suffix(".json.building")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    (output / f"{stem}.md").write_text(markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "splits": list(report["splits"]),
                "pending": report["pending"],
            }
        )
    )


if __name__ == "__main__":
    main()
