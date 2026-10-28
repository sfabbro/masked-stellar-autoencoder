"""A verified-subset report must never launder a failed comparator into a pass."""

import importlib.util
import sys
from pathlib import Path

import pytest

MONITOR = Path(__file__).resolve().parents[1] / "experiments/xp/monitor"
sys.path.insert(0, str(MONITOR))
spec = importlib.util.spec_from_file_location(
    "partial_cnf_report", MONITOR / "report_frozen_cnf_partial.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def status(method, passed):
    return dict(
        method=method,
        status="passed" if passed else "failed",
        all100620cpu_predictions=True,
        tolerance_violations=0 if passed else 1,
    )


def test_failed_nll_is_excluded_without_changing_overall_status():
    verification = dict(
        status="failed",
        methods=[status("raw_mse", True), status("nll", False), status("cnf", True)],
    )
    rows = report.verified_subset(verification)
    assert [r["method"] for r in rows] == ["raw_mse", "cnf"]
    assert verification["status"] == "failed"
    assert verification["methods"][1]["status"] == "failed"


def test_failed_included_method_is_rejected():
    verification = dict(
        status="failed",
        methods=[status("raw_mse", True), status("nll", True), status("cnf", False)],
    )
    with pytest.raises(AssertionError, match="cnf"):
        report.verified_subset(verification)
