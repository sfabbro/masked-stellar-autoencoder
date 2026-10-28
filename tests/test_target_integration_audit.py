"""Readiness audit must distinguish unusable identity/bounds and known exclusions."""

import importlib.util
from pathlib import Path

import h5py
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "target_audit", Path(__file__).parents[1] / "scripts/audit_target_integration.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_bounds_reject_false_validity_and_duplicate_gaia():
    frame = pd.DataFrame(
        {
            "source_id": [1, 1],
            "age16": [1.0, -1.0],
            "age50": [2.0, 2.0],
            "age84": [3.0, 3.0],
            "age_bounds_valid": [True, True],
        }
    )
    report = audit.audit_bounds(frame)
    assert report["declared_valid_but_invalid"] == 1
    assert report["unique_positive_gaia_ids"] is False
    assert report["status"].startswith("blocked")


def test_cohort_reports_new_anchor_overlap_separately_from_old_holdout(tmp_path):
    path = tmp_path / "cohort.h5"
    with h5py.File(path, "w") as f:
        for key, values in {
            "source_id": [10, 20, 30],
            "strict_fit": [True, False, False],
            "tune": [False, True, False],
            "benchmark_partition": [0, 0, 1],
        }.items():
            f.create_dataset("selected/" + key, data=values)
        f.create_dataset("external_reference/source_id", data=[30])
    report = audit.audit_cohort(path, [10])
    assert report["existing_reference_fit_tune_leaks"] == 0
    assert report["apokasc_strict_fit_overlap"] == 1
    assert report["new_teacher_holdout_redesign_required"] is True


def test_missing_artifacts_fail_closed(tmp_path):
    report = audit.audit(tmp_path)
    assert report["status"].startswith("blocked")
    assert report["age"]["apokasc"]["status"] == "blocked_missing_anchor"
    assert len(report["alpha"]["missing_evidence_gates"]) == 5
    assert (
        "zero_point_recipe_version_units_and_row_binding"
        in report["distance"]["missing_evidence"]
    )
