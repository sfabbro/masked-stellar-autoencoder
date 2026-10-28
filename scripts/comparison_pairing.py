"""Fail-closed contracts for released-catalogue comparisons."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def bound_predictions(
    cohort: Mapping, predictions: Mapping, role: str, *, variance: bool = False
):
    """Return native means and optional standard deviations after ordered-ID checks."""
    ids = np.asarray(cohort["source_id"])
    rows = np.asarray(cohort[f"{role}_rows"])
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.unique(ids).size != ids.size:
        raise ValueError("cohort source IDs must be unique integers")
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or np.unique(rows).size != rows.size
        or np.any(rows < 0)
        or np.any(rows >= len(ids))
    ):
        raise ValueError(f"invalid {role} rows")
    if "calibration_rows" in cohort and "test_rows" in cohort:
        if np.intersect1d(cohort["calibration_rows"], cohort["test_rows"]).size:
            raise ValueError("calibration and test rows overlap")
    key = f"{role}_source_id"
    if key not in predictions:
        raise ValueError(f"missing {key}: ordered prediction identity is required")
    actual_ids = np.asarray(predictions[key])
    if actual_ids.dtype.kind not in "iu" or not np.array_equal(actual_ids, ids[rows]):
        raise ValueError(f"{role} prediction source IDs differ from cohort order")
    center = np.asarray(cohort["label_center"], dtype="f8")
    scale = np.asarray(cohort["label_scale"], dtype="f8")
    if (
        center.shape != (3,)
        or scale.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0)
    ):
        raise ValueError("expected finite three-label center and positive scale")
    truth = np.asarray(cohort["value"])[rows]
    mean = np.asarray(predictions[f"{role}_mean"], dtype="f8")
    if (
        truth.shape != (len(rows), 3)
        or mean.shape != truth.shape
        or not np.all(np.isfinite(mean))
    ):
        raise ValueError(f"invalid {role} prediction/truth shape or mean")
    native = mean * scale + center
    if not variance:
        return native
    var = np.asarray(predictions[f"{role}_variance"], dtype="f8")
    if var.shape != mean.shape or not np.all(np.isfinite(var)) or np.any(var <= 0):
        raise ValueError(f"invalid {role} variance; positive finite values required")
    return native, np.sqrt(var) * scale
