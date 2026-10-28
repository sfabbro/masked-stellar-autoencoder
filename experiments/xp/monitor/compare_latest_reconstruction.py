"""Compare a pinned later standard checkpoint with archived reconstruction experiments."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "latest_standard_comparison_helpers",
    Path(__file__).with_name("compare_latest_standard.py"),
)
latest_helpers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(latest_helpers)


def resolve_latest(args):
    spec = args.spec or Path(f"configs/xp-latest-standard-e{args.epoch}-analysis.yaml")
    settings = latest_helpers.latest_settings(spec)
    if settings["checkpoint_epoch"] != args.epoch:
        raise ValueError(
            "Requested reconstruction epoch differs from pinned specification"
        )
    root = args.root or Path(f"experiments/xp/results-latest-standard-e{args.epoch}")
    run_id = args.run_id or f"evaluate_latest_standard{args.epoch}"
    return root, run_id, settings


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, default=57)
    parser.add_argument("--spec", "--latest-spec", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root, run_id, settings = resolve_latest(args)
    folder = root / run_id
    manifest = read(folder / "manifest.json")
    assert manifest["status"] == "complete" and manifest["evaluation_only"]
    expected = settings["checkpoint_sha256"]
    assert manifest["checkpoint_sha256"] == expected
    assert manifest["checkpoint_epoch"] == settings["checkpoint_epoch"]
    assert manifest["checkpoint"] == settings["checkpoint"]
    assert manifest["sample"]["validation_rows"] == 1_000_000
    latest = folder / run_id
    proof = read(latest / "evaluation_verification.json")
    assert proof["source_unchanged"] and proof["reload_predictions_equal"]
    assert proof["evaluation_only"] and proof["checkpoint_sha256"] == expected
    assert proof["optimizer_steps"] == proof["training_presentations"] == 0
    qa = read(latest / "residual_latest.json")
    with np.load(latest / "residual_latest.npz", allow_pickle=False) as file:
        arrays = {k: np.array(file[k]) for k in file.files}
    scores = {}
    baselines = {}
    for suite, names in [
        (
            "photometry",
            [
                "unchanged_control",
                "spectral_auxiliary_mae",
                "observed_photometry_mae",
                "photometry_spectral_mae",
                "anchored_decoder_consistency_mae",
            ],
        ),
        (
            "next",
            ["corrected_signal_weighted_mae", "adam_reset_mae", "adam_reset_lr2_mae"],
        ),
        (
            "followup",
            [
                x["run_id"]
                for x in read(Path("experiments/xp/launch-followup-e47-20261014.json"))[
                    "jobs"
                ]
            ],
        ),
    ]:
        for name in names:
            base = Path(f"experiments/xp/results-{suite}-e47/{name}")
            parent_manifest = read(base / "manifest.json")
            assert parent_manifest["validation_id"] == manifest["validation_id"]
            assert (
                parent_manifest["benchmark_source_id_sha256"]
                == manifest["benchmark_source_id_sha256"]
            )
            assert (
                parent_manifest["spectral_projection"]
                == manifest["spectral_projection"]
            )
            if suite in {"photometry", "followup"}:
                for key in (
                    "photometry_calibration_sha256",
                    "photometry_projection_sha256",
                ):
                    assert parent_manifest[key] == manifest[key]
            reference = read(base / name / "residual_latest.json")
            with np.load(base / name / "residual_init.npz", allow_pickle=False) as file:
                for key in arrays:
                    if (
                        key == "targets"
                        or key == "validation_row_index"
                        or key.endswith("_mask")
                    ):
                        np.testing.assert_array_equal(arrays[key], file[key])
            by_regime = {}
            for regime, value in qa["regimes"].items():
                original = reference["regimes"][regime]
                assert value["overall"]["count"] == original["overall"]["count"]
                row = {}
                for key, new, old in [
                    ("all_mae", value["overall"]["mae"], original["overall"]["mae"]),
                    (
                        "xp_mae",
                        value["blocks"]["xp"]["mae"],
                        original["blocks"]["xp"]["mae"],
                    ),
                    (
                        "spectral_mean",
                        value.get("spectral", {}).get(
                            "integrated_absolute_fraction_mean"
                        ),
                        original.get("spectral", {}).get(
                            "integrated_absolute_fraction_mean"
                        ),
                    ),
                ]:
                    row[key] = {
                        "latest": new,
                        "experiment": old,
                        "latest_change_percent": 100 * (new / old - 1)
                        if new is not None and old
                        else None,
                    }
                by_regime[regime] = row
            baselines[name] = parent_manifest["checkpoint_sha256"]
            scores[name] = by_regime
    result = {
        "status": "complete",
        "latest_checkpoint_epoch": settings["checkpoint_epoch"],
        "latest_checkpoint_sha256": expected,
        "equal_pretraining_budget": False,
        "policy": "Practical comparison of later standard training versus short epoch47 continuations; not a causal loss ablation",
        "validation_id": manifest["validation_id"],
        "baseline_parent_hashes": baselines,
        "comparison_count": len(scores),
        "comparisons": scores,
    }
    (root / "latest-reconstruction-comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        "# Latest standard reconstruction comparison",
        "",
        result["policy"],
        "",
        f"All evaluations use the same 1M validation stars, scaler, diagnostic rows and masks. Latest checkpoint epoch{settings['checkpoint_epoch']} received zero updates in this evaluation. Its encoder pretraining budget is larger, so this is not a matched-budget loss ablation.",
        "",
    ]
    for name, regimes in scores.items():
        common = regimes["common"]
        lines.append(
            f"- Versus {name}: latest XP MAE {common['xp_mae']['latest_change_percent']:+.3f}%; all-feature MAE {common['all_mae']['latest_change_percent']:+.3f}%; spectral mean {common['spectral_mean']['latest_change_percent']:+.3f}%."
        )
    (root / "latest-reconstruction-comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "comparisons": len(scores)}))


if __name__ == "__main__":
    main()
