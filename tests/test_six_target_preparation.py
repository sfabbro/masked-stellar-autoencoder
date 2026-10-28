"""Synthetic arrays only: missing labels, role leakage and source estimands."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_six_target_wave import assemble_sparse


def source_fixture():
    ids = np.array([11, 12, 13, 14])
    native = dict(
        source_id=ids,
        value=np.tile([5000.0, 2.5, -0.7], (4, 1)),
        error=np.full((4, 3), 0.1),
        observed=np.ones((4, 3), bool),
        quality=np.ones((4, 3), bool),
        survey=np.full((4, 3), b"galah"),
        fit_rows=np.array([0]),
        tune_rows=np.array([1]),
        calibration_rows=np.array([2]),
        test_rows=np.array([3]),
    )
    alpha = dict(
        source_id=np.array([11, 15]),
        role=np.array(["test", "fit"]),
        target=np.array([[0.2], [-0.1]]),
        quoted_error=np.full((2, 1), 0.04),
        raw_feh=np.full(2, -0.6),
        benchmark_partition=np.zeros(2, int),
    )
    medians = np.array([[5.0, 2.0], [7.0, 1.0]])
    posterior = dict(
        source_id=np.array([12, 16]),
        role=np.array(["cal", "fit"]),
        target=np.log(medians),
        posterior_quantiles=medians[:, :, None] * np.array([0.8, 1.0, 1.2]),
        benchmark_partition=np.zeros(2, int),
    )
    return native, alpha, posterior


def test_sparse_union_preserves_native_units_and_missing_measurement_errors():
    data, conflicts = assemble_sparse(*source_fixture())
    assert conflicts == 2
    rows = {int(s): i for i, s in enumerate(data["source_id"])}
    assert data["role"][rows[11]] == "test"  # Never keep native FIT over alpha TEST.
    assert data["role"][rows[12]] == "cal"  # Never keep native TUNE over CAL.
    assert not data["quality"][
        rows[11], :3
    ].any()  # Original native FIT is not a new TEST label.
    assert data["quality"][rows[11], 3]
    assert data["label_source_role"][rows[12], 0] == "tune"
    assert not data["quality"][rows[12], :3].any()
    np.testing.assert_allclose(data["value"][rows[12], 4:], [5.0, 2.0])
    assert np.isnan(data["error"][rows[12], 4:]).all()
    assert not data["error_observed"][rows[12], 4:].any()
    assert np.isnan(data["value"][rows[15], :3]).all()
    assert data["value"][rows[15], 3] == -0.1  # Signed alpha stays dex.
    assert data["label_source"][rows[11], 0] == "galah"
    assert not data["observed"].all(1).any()


def test_reference_union_excluded_from_all_roles():
    data, _ = assemble_sparse(*source_fixture(), excluded_ids=[11, 16])
    assert not np.isin(data["source_id"], [11, 16]).any()


def test_unknown_native_error_remains_unknown_not_zero():
    native, alpha, posterior = source_fixture()
    native["error"][0, 0] = np.nan
    data, _ = assemble_sparse(native, alpha, posterior)
    row = np.flatnonzero(data["source_id"] == 11)[0]
    assert data["observed"][row, 0]
    assert not data["error_observed"][row, 0]
    assert np.isnan(data["error"][row, 0])


@pytest.mark.parametrize("case", ["ids", "roles", "posterior", "benchmark", "alpha"])
def test_invalid_source_evidence_fails_before_assembly(case):
    native, alpha, posterior = copy.deepcopy(source_fixture())
    if case == "ids":
        alpha["source_id"][1] = alpha["source_id"][0]
    elif case == "roles":
        native["test_rows"] = native["fit_rows"]
    elif case == "posterior":
        posterior["target"][0, 0] += 0.1
    elif case == "benchmark":
        posterior["benchmark_partition"][1] = 1
    else:
        alpha["raw_feh"][1] = -2.1
    with pytest.raises(ValueError):
        assemble_sparse(native, alpha, posterior)


def test_actual_native_one_survey_tag_per_star_is_broadcast_to_targets():
    native, alpha, posterior = source_fixture()
    native["survey"] = np.array([b"galah", b"apogee", b"galah", b"apogee"])
    data, _ = assemble_sparse(native, alpha, posterior)
    row = np.flatnonzero(data["source_id"] == 14)[0]
    assert data["label_source"][row, :3].tolist() == ["apogee"] * 3


def test_calibration_priority_preserves_calibration_without_using_test_labels():
    native, alpha, posterior = source_fixture()
    posterior["source_id"][0] = 14  # Native TEST, but posterior CAL.
    old, _ = assemble_sparse(native, alpha, posterior)
    new, _ = assemble_sparse(native, alpha, posterior, calibration_priority=True)
    old_row = np.flatnonzero(old["source_id"] == 14)[0]
    row = np.flatnonzero(new["source_id"] == 14)[0]
    assert old["role"][old_row] == "test" and not old["quality"][old_row, 4:].any()
    assert new["role"][row] == "cal" and new["quality"][row, 4:].all()
    assert not new["quality"][
        row, :3
    ].any()  # Native TEST labels are not reused as CAL.
    assert not np.isin(
        new["source_id"][new["role"] == "fit"],
        new["source_id"][np.isin(new["role"], ["tune", "cal", "test"])],
    ).any()
