"""Dependent checkpoints require an independent final proof, not an absent assertion."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "new_representation_comparison",
    Path(__file__).parents[1] / "experiments/xp/monitor/compare_new_representations.py",
)
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)


def dependent_manifest():
    settings = {
        "depends_on": "spectral_s42",
        "run_id": "probe_new_spectral_s42",
        "checkpoint": "/arc/spectral_s42/spectral_s42/checkpoint.pth",
    }
    proof = {
        "checkpoint_sha256": "a" * 64,
        "parent_checkpoint_sha256": new.paired.PARENT,
        "reload_predictions_equal": True,
    }
    manifest = {
        "status": "complete",
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "parent_checkpoint_sha256": new.paired.PARENT,
        "benchmark_source_id_sha256": new.paired.BENCHMARK,
        "scaler_sha256": new.paired.SCALER,
        "arguments": {
            "checkpoint": settings["checkpoint"],
            "checkpoint_sha256": None,
            "seed": 42,
        },
    }
    return manifest, settings, proof


def test_planned_dependency_accepts_independent_actual_hash_without_claiming_cli_assertion():
    manifest, settings, proof = dependent_manifest()
    result = new.verify_dependent_manifest(manifest, settings, proof)
    assert result["actual_checksum_matches_independent_final_proof"]
    assert not result["runtime_cli_checksum_assertion_executed"]
    changed = copy.deepcopy(manifest)
    changed["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="independent final training proof"):
        new.verify_dependent_manifest(changed, settings, proof)


def test_no_dependency_cannot_use_missing_cli_checksum_exemption():
    manifest, settings, proof = dependent_manifest()
    settings.pop("depends_on")
    with pytest.raises(ValueError, match="declared dependency"):
        new.verify_dependent_manifest(manifest, settings, proof)


def test_expected_cli_checksum_cannot_be_silently_omitted():
    manifest, settings, proof = dependent_manifest()
    settings["checkpoint_sha256"] = proof["checkpoint_sha256"]
    with pytest.raises(ValueError, match="was not executed"):
        new.verify_dependent_manifest(manifest, settings, proof)


def test_family_summaries_use_seed_matched_ratios_and_sample_standard_deviation():
    groups = {}
    for seed, baseline in zip(new.SEEDS, (10.0, 20.0, 40.0), strict=True):
        for family, value in (("control", baseline), ("spectral", baseline * 0.9)):
            targets = {
                target: {"ridge": {"mae": value}, "budgets": {}}
                for target in new.paired.TARGETS
            }
            groups[f"probe_new_{family}_s{seed}"] = {"results": {"targets": targets}}
    summary = new.summarize_families(groups)
    statistic = summary["spectral"]["targets"]["teff"]["metrics"]["ridge"][
        "matched_control_change_percent"
    ]
    assert statistic["mean"] == pytest.approx(-10)
    assert statistic["sample_std"] == pytest.approx(0)
    assert summary["spectral"]["three_seeds_complete"]
    assert new.descriptive([1, 2, 3])["sample_std"] == pytest.approx(1)


def test_single_seed_knobs_reference_spectral_and_family_uses_matched_control():
    assert new.baselines_for("spectral_dropout0_s42") == {
        "matched_reference": "probe_new_spectral_s42"
    }
    assert new.baselines_for("spectral_s44") == {
        "matched_reference": "probe_new_control_s44"
    }


def test_training_proof_requires_completed_exact_budget_and_reload(tmp_path):
    import json

    run = "spectral_s42"
    folder = tmp_path / run
    nested = folder / run
    nested.mkdir(parents=True)
    settings = {
        "depends_on": run,
        "run_id": "probe_new_" + run,
        "mode": "probes",
        "checkpoint": f"/arc/{run}/{run}/checkpoint.pth",
    }
    manifest = {
        "status": "complete",
        "suite_id": run,
        "checkpoint_sha256": new.paired.PARENT,
        "presentations_per_experiment": 20000000,
        "sample": {"train_rows": 9000000, "validation_rows": 1000000},
        "benchmark_source_id_sha256": new.paired.BENCHMARK,
    }
    proof = {
        "parent_sha256": new.paired.PARENT,
        "reload_predictions_equal": True,
        "sha256": "c" * 64,
    }
    event = {
        "phase": "complete",
        "rows_seen_total": 20000000,
        "optimizer_steps": 612,
        "seed": 42,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (nested / "checkpoint_verification.json").write_text(json.dumps(proof))
    metrics = nested / "metrics.jsonl"
    metrics.write_text(json.dumps(event) + "\n")
    resolved = new.resolve_training_proof(tmp_path, settings)
    assert resolved["checkpoint_sha256"] == proof["sha256"]
    event["optimizer_steps"] = 611
    metrics.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="Actual continuation optimizer steps"):
        new.resolve_training_proof(tmp_path, settings)
