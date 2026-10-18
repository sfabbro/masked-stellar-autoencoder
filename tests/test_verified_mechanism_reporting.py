"""Support-sensitive source/tail reporting must preserve correlated source layers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

MONITOR = Path(__file__).resolve().parents[1] / "experiments/xp/monitor"
sys.path.insert(0, str(MONITOR))
spec = importlib.util.spec_from_file_location(
    "verified_reports", MONITOR / "report_verified_mechanisms.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_tail_and_source_support_do_not_silently_drop_li():
    measurements = np.array(
        [(-3.1, "li_vmps"), (-2.2, "apogee"), (-1.9, "galah"), (0.0, "galah")],
        dtype=[("value", "f8"), ("survey", "U12")],
    )
    masks = report.conditional_masks(measurements)
    assert masks["below_-2"].sum() == 2
    assert masks["below_-3"].sum() == 1
    assert masks["apogee_galah_below_-2"].sum() == 1
    assert masks["survey:li_vmps:below_-2"].sum() == 1
    rows = report.score_rows("test", measurements["value"], np.zeros(4), masks)
    allrow = next(r for r in rows if r["subset"] == "all")
    assert allrow["count"] == 4
    np.testing.assert_allclose(allrow["mae_dex"], 1.8, rtol=1e-15)


def test_pair_allows_only_documented_input_note_metadata_difference():
    manifest = dict.fromkeys(
        [
            "parent_sha256",
            "cohort_sha256",
            "row_sha256",
            "mask_sha256",
            "supervised_stream_sha256",
        ],
        "same",
    )
    arrays = {"source_id": np.array([11, 13]), "test_input": np.array([[1.0], [2.0]])}
    initial = {"prediction": np.array([[0.0]])}
    reference = manifest, arrays, initial
    report.pair(
        reference,
        manifest,
        {**arrays, "input_note": np.array("SIMEX root metadata")},
        initial,
    )
    bad = {**arrays, "source_id": np.array([11, 14])}
    import pytest

    with pytest.raises(AssertionError):
        report.pair(reference, manifest, bad, initial)


def test_snapshot_dex_conversion_matches_production_float32(tmp_path):
    outputs = np.zeros((2, 1, 2), dtype="f4")
    outputs[:, 0, 0] = [0.1234567, -7.123456]
    path = tmp_path / "snapshot.npz"
    np.savez(path, member_outputs=outputs)
    arrays = {
        "label_scale": np.array(0.318763478934, dtype="f8"),
        "label_center": np.array(-0.251273647, dtype="f8"),
    }
    point = report.snapshot_point(path, arrays)
    expected = outputs[:, :, 0].mean(1) * float(arrays["label_scale"]) + float(
        arrays["label_center"]
    )
    assert point.dtype == np.dtype("f4")
    np.testing.assert_array_equal(point, expected)


def test_affine_snapshot_extrapolation_preserves_label_origin():
    levels = np.array([0, 0.5, 1, 1.5, 2.0])
    constant = np.full((5, 2), [-0.2, 2.0])
    np.testing.assert_array_equal(
        report.independent_extrapolate(levels, constant), [-0.2, 2.0]
    )
