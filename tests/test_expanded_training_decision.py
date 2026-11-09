import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "expanded_training",
    Path(__file__).resolve().parents[1] / "scripts/run_expanded_six_training.py",
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def setup_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    cfg = dict(
        expanded_source_run="source",
        calibration_run="calibration",
        mapping_candidates=["map_a", "map_b"],
    )
    runs = {
        "source": dict(
            status="expanded_six_source_complete",
            training_source_ready=True,
            prepared_sha256="a" * 64,
        ),
        "calibration": dict(status="complete", results={}),
    }
    for family, score in [("energy", 0.2), ("hybrid", 0.18)]:
        for seed in [1, 2]:
            rid = f"nsfconfirm_{family}_s{seed}_20261106_v1"
            runs[rid] = dict(
                status="remediation_pilot_complete",
                tune_score=score,
                development_score=1000 if family == "hybrid" else 0,
            )
            runs["calibration"]["results"][rid] = dict(status="complete")
    for name, score, mae in [("map_a", 0.17, 0.07), ("map_b", 0.1, 0.15)]:
        runs[name] = dict(
            status="complete",
            tune_score=score,
            selected_checkpoint_sha256="b" * 64,
            arm=dict(map_kind="affine"),
            evaluation=dict(
                tune=dict(
                    sources=[
                        dict(
                            source="APOKASC3_loggSeis",
                            targets=dict(
                                logg=dict(bias=0.01, mae=mae, coverage90=0.88)
                            ),
                        )
                    ]
                )
            ),
        )
    cfg["decision_runs"] = list(runs)

    def write():
        for name, m in runs.items():
            d = tmp_path / name
            d.mkdir(exist_ok=True)
            (d / "manifest.json").write_text(json.dumps(m))

    write()
    return cfg, runs, write


def test_tune_selects_objective_and_gravity_gate_rejects_bad_map_without_using_dev(
    tmp_path, monkeypatch
):
    cfg, runs, write = setup_runs(tmp_path, monkeypatch)
    saved = {}
    decision, source = training.wait_decision(
        cfg, tmp_path, lambda p, v: saved.update({p.name: v})
    )
    assert decision["objective"] == "hybrid" and decision["mapping_run"] == "map_a"
    assert "architecture-decision.json" in saved


def test_failed_required_pair_blocks_large_training(tmp_path, monkeypatch):
    cfg, runs, write = setup_runs(tmp_path, monkeypatch)
    runs["nsfconfirm_hybrid_s2_20261106_v1"]["status"] = "failed"
    write()
    with pytest.raises(RuntimeError, match="paired"):
        training.wait_decision(cfg, tmp_path, lambda p, v: None)


def test_reuse_requires_waiting_no_training_and_no_checkpoint(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    path = output / "manifest.json"
    path.write_text(
        json.dumps(
            dict(
                status="waiting_for_decision",
                training_performed=False,
                completed_updates=0,
            )
        )
    )
    training.prepare_output(output, reuse_waiting=True)
    assert (output / "pre-admission-attempt-1.json").exists()
    training.prepare_output(output, reuse_waiting=True)
    assert (output / "pre-admission-attempt-2.json").exists()
    other = tmp_path / "trained"
    other.mkdir()
    (other / "manifest.json").write_text(
        json.dumps(
            dict(status="training", training_performed=True, completed_updates=1)
        )
    )
    with pytest.raises(ValueError, match="optimizing"):
        training.prepare_output(other, reuse_waiting=True)


def test_explicit_retry_only_allows_unadmitted_zero_update_failure(tmp_path):
    output = tmp_path / "failed"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps(dict(status="failed", training_performed=False, completed_updates=0))
    )
    with pytest.raises(ValueError):
        training.prepare_output(output, reuse_waiting=True)
    training.prepare_output(output, retry_failed=True)
    (output / "architecture-decision.json").write_text("{}")
    with pytest.raises(ValueError):
        training.prepare_output(output, retry_failed=True)
