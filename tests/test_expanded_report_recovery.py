"""Synthetic saved-array tests only: no neural models or real checkpoints."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import expanded_saved_array_reporting as lightweight
import recover_expanded_evaluation_report as recovery


def fixture_policy(tmp_path):
    arm = dict(
        run_id="report",
        producer_run_id="producer",
        head_kind="quantile3",
        checkpoint_sha256="c" * 64,
    )
    cfg = dict(
        output_root=str(tmp_path),
        expected_rows=2,
        producer_source_sha="a" * 40,
        prepared_sha256="b" * 64,
        library_revision="d" * 40,
        optimizer_steps=0,
        model_forward_rows=0,
        arms=[arm],
    )
    manifest = dict(
        status="failed",
        error_type="ValueError",
        error="output array is read-only",
        processed=2,
        total=2,
        source_sha=cfg["producer_source_sha"],
        run_id="producer",
        zero_training=True,
        optimizer_constructed=False,
        reconstruction_presentations=0,
        checkpoint_sha256=arm["checkpoint_sha256"],
    )
    binding = dict(
        source_sha=cfg["producer_source_sha"],
        prepared_sha256=cfg["prepared_sha256"],
        checkpoint_sha256=arm["checkpoint_sha256"],
        target_names=["teff", "logg", "fe_h", "alpha", "age", "distance"],
        library=dict(vcs_info=dict(commit_id=cfg["library_revision"])),
    )
    return cfg, arm, manifest, binding


def synthetic_arrays(folder):
    folder.mkdir()
    observed = np.zeros((2, 6), bool)
    observed[:, 0] = True
    diagnostic = observed.copy()
    diagnostic[:, 3] = True
    arrays = dict(
        source_id=np.array([10, 20], dtype="i8"),
        point_native=np.ones((2, 6)),
        quantiles_native=np.ones((2, 3, 6)),
        target_native=np.ones((2, 6)),
        observed=observed,
        diagnostic_eligible=diagnostic,
        evaluation_mode=np.array(["natural"] * 2),
        xp_status=np.array(["complete_xp"] * 2),
        label_source=np.full((2, 6), "fixture"),
        ancestor_feh_supervised_fit_exposed=np.zeros(2, bool),
    )
    for key, values in arrays.items():
        np.save(folder / (key + ".npy"), values)
    return arrays


def test_external_age_report_accepts_readonly_mask_without_mutation(tmp_path):
    path = tmp_path / "mask.npy"
    np.save(path, np.array([True, True]))
    mask = np.load(path, mmap_mode="r")
    predictions = dict(
        external_age_observed=mask,
        external_age_value=np.array([3.0, np.nan]),
        point_native=np.full((2, 6), 5.0),
    )
    output = tmp_path / "report"
    lightweight.external_report(predictions, output)
    np.testing.assert_array_equal(mask, [True, True])
    result = json.loads((output / "summary.json").read_text())
    assert result["count"] == 1 and result["mae_gyr"] == 2
    assert (output / "age-reference.png").exists()
    assert (output / "age-reference.pdf").exists()


def test_readiness_rejects_running_partial_and_changed_identity(tmp_path):
    cfg, arm, manifest, binding = fixture_policy(tmp_path)
    recovery.validate_ready(manifest, binding, cfg, arm)
    for change in [
        dict(status="running"),
        dict(processed=1),
        dict(error="other"),
        dict(zero_training=False),
    ]:
        with pytest.raises(ValueError):
            recovery.validate_ready(dict(manifest, **change), binding, cfg, arm)
    changed = copy.deepcopy(binding)
    changed["prepared_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="binding"):
        recovery.validate_ready(manifest, changed, cfg, arm)


def test_memmap_layouts_and_disjoint_diagnostic_mask(tmp_path):
    original = synthetic_arrays(tmp_path / "arrays")
    arrays, schemas = recovery.load_arrays(tmp_path / "arrays", 2)
    assert isinstance(arrays["point_native"], np.memmap)
    assert not arrays["point_native"].flags.writeable
    diagnostic = recovery.diagnostic_only_mask(arrays)
    assert not (diagnostic & arrays["observed"]).any()
    assert diagnostic[:, 3].all()
    np.testing.assert_array_equal(
        arrays["diagnostic_eligible"], original["diagnostic_eligible"]
    )
    assert schemas["point_native.npy"]["content_sha256_verified"] is False
    with pytest.raises(ValueError):
        recovery.load_arrays(tmp_path / "arrays", 3)


def test_cpu_guard_before_any_file_or_output_access(tmp_path, monkeypatch):
    monkeypatch.delenv("MSA_EXPERIMENT_CPU_ONLY", raising=False)
    args = SimpleNamespace(
        config=tmp_path / "missing", run_id="none", output=tmp_path / "out"
    )
    with pytest.raises(RuntimeError, match="CANFAR"):
        recovery.run(args)
    assert not args.output.exists()


def test_recovery_reuses_arrays_preserves_original_failure_and_rejects_repeat(
    tmp_path, monkeypatch
):
    import five_wave_report

    cfg, arm, manifest, binding = fixture_policy(tmp_path)
    producer = tmp_path / "producer"
    producer.mkdir()
    manifest_path = producer / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (producer / "bindings.json").write_text(json.dumps(binding))
    synthetic_arrays(producer / "prediction-arrays")
    config = tmp_path / "policy.yaml"
    config.write_text(yaml.safe_dump(cfg))
    args = SimpleNamespace(
        config=config, run_id="report", output=tmp_path / "new-report"
    )
    seen = []

    def fake_report(predictions, output, **kwargs):
        output.mkdir()
        seen.append(np.array(predictions["observed"]))
        (output / "report.json").write_text("{}")

    monkeypatch.setenv("MSA_EXPERIMENT_CPU_ONLY", "1")
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(recovery, "external_report", lambda *a, **k: None)
    monkeypatch.setattr(five_wave_report, "make_report", fake_report)
    before = manifest_path.read_bytes()
    recovery.run(args)
    assert manifest_path.read_bytes() == before
    result = json.loads((args.output / "manifest.json").read_text())
    assert result["status"] == "saved_array_reporting_complete"
    assert (
        result["native_model_execution"] is False
        and result["resampling_performed"] is False
    )
    assert len(seen) == 2 and not (seen[0] & seen[1]).any()
    with pytest.raises(FileExistsError):
        recovery.run(args)


def test_recovery_modules_never_import_neural_libraries():
    import ast

    for module in [lightweight, recovery]:
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not name.name.startswith(("torch", "zuko")) for name in node.names
                )
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("torch", "zuko", "run_expanded_checkpoint_evaluation")
                )


def test_corner_primary_references_use_exact_ids_without_modifying_saved_draws():
    predictions = dict(
        source_id=np.array([10, 20], dtype="i8"),
        target_native=np.ones((2, 6)),
        observed=np.zeros((2, 6), bool),
    )
    predictions["observed"][:, 0] = True
    old_mask = np.ones((2, 6), bool)
    old_mask.setflags(write=False)
    draws = np.ones((2, 256, 6))
    corners = dict(
        source_id=np.array([20, 10], dtype="i8"),
        samples_native=draws,
        observed=old_mask,
        error_observed=old_mask,
        target_native=np.ones((2, 6)),
    )
    result, proof = recovery.primary_corner_references(corners, predictions)
    assert result["samples_native"] is draws
    assert old_mask.all() and corners["observed"].all()
    assert result["observed"][:, 0].all() and not result["observed"][:, 1:].any()
    assert not result["error_observed"][:, 1:].any()
    assert proof["excluded_diagnostic_reference_count"] == [0, 2, 2, 2, 2, 2]
    assert proof["native_draws_changed"] is False
    with pytest.raises(ValueError, match="absent"):
        recovery.primary_corner_references(
            dict(corners, source_id=np.array([20, 30])), predictions
        )


@pytest.mark.parametrize(
    "key,bad",
    [
        ("intrinsic_log_prob_native", np.inf),
        ("observed_log_prob_native", np.nan),
        ("intrinsic_pit", -0.01),
        ("observed_pit", 1.01),
    ],
)
def test_valid_reference_density_or_cdf_failure_cannot_be_silently_dropped(
    tmp_path, key, bad
):
    folder = tmp_path / "arrays"
    synthetic_arrays(folder)
    values = np.zeros((2, 6))
    values[0, 3] = bad  # diagnostic-only coordinate must also be checked
    np.save(folder / (key + ".npy"), values)
    with pytest.raises(FloatingPointError, match="saved (density|CDF)"):
        recovery.load_arrays(folder, 2)


def test_absent_reference_nan_and_baseline_no_pdf_are_supported(tmp_path):
    folder = tmp_path / "arrays"
    synthetic_arrays(folder)
    recovery.load_arrays(folder, 2)  # three quantiles do not imply a PDF
    values = np.zeros((2, 6))
    values[:, 4:] = np.nan  # neither primary nor diagnostic references here
    np.save(folder / "intrinsic_log_prob_native.npy", values)
    recovery.load_arrays(folder, 2)
    point = np.ones((2, 6))
    point[0, 4] = np.inf
    np.save(folder / "point_native.npy", point)
    with pytest.raises(FloatingPointError, match="point"):
        recovery.load_arrays(folder, 2)
