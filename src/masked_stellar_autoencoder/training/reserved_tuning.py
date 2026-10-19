"""Label-blind native role-1 tuning sample; never a fresh final holdout."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch

from .spectrophotometric_inputs import spectrophotometric_inputs


def reserved_native_rows(
    source_id: np.ndarray,
    feature_row: np.ndarray,
    available: np.ndarray,
    partition: np.ndarray,
    *,
    identity_split: Callable,
    size: int = 4096,
    seed: int = 20261019,
) -> np.ndarray:
    """Select raw-native row indices using identity/availability only.

    The canonical identity_split is injected by the caller, preventing a second
    drifting implementation of the programme's source-ID role definition.
    Sampling sorted source IDs makes selection invariant to native row order.
    No measurement values, errors or survey labels enter this function.
    """
    ids = np.asarray(source_id)
    rows = np.asarray(feature_row)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Feature source IDs must be unique integers")
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or np.any(rows < 0)
        or np.any(rows >= len(ids))
    ):
        raise ValueError("Invalid native feature associations")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Native source associations must be unique")
    if np.shape(available) != ids.shape or np.shape(partition) != ids.shape:
        raise ValueError("Feature availability/partition shape differs")
    roles = np.asarray(identity_split(ids))
    if roles.shape != ids.shape or not np.isin(roles, [0, 1, 2]).all():
        raise ValueError("Canonical identity split must assign roles 0, 1, 2")
    if not isinstance(size, int) or size < 1:
        raise ValueError("Sample size must be a positive integer")
    eligible = np.flatnonzero(
        np.asarray(available)[rows]
        & (np.asarray(partition)[rows] == 0)
        & (roles[rows] == 1)
    )
    eligible = eligible[np.argsort(ids[rows[eligible]], kind="stable")]
    if len(eligible) < size:
        raise ValueError(
            f"Only {len(eligible)} available partition-0 role-1 native stars; require {size}"
        )
    chosen = np.random.default_rng(seed).choice(eligible, size=size, replace=False)
    return chosen[np.argsort(ids[rows[chosen]], kind="stable")]


def load_reserved_tuning(
    path: str | Path,
    *,
    identity_split: Callable,
    size: int = 4096,
    seed: int = 20261019,
) -> dict:
    """Read existing audited cohort, keeping parent input and fit target scales.

    This returns raw dex truth only. Head target center/scale MUST come from the
    native role-0 fit cohort; this helper does not estimate target scales.
    Existing native_inputs includes EBV by default; the same policy is used.
    Historical role-1 use is not audited and pretraining overlap is not excluded.
    """
    with h5py.File(path) as source:
        ids = source["features/source_id"][:]
        rows = source["layer_index/native/feature_row"][:]
        available = source["features/feature_available"][:]
        partition = source["features/benchmark_partition"][:]
        roles = np.asarray(identity_split(ids))
        np.testing.assert_array_equal(source["features/label_split"][:], roles)
        chosen = reserved_native_rows(
            ids,
            rows,
            available,
            partition,
            identity_split=identity_split,
            size=size,
            seed=seed,
        )
        selected_rows = rows[chosen]
        # Fancy HDF5 indexing requires sorted row indices, restore source-ID order.
        row_order = np.argsort(selected_rows)
        inverse = np.argsort(row_order)
        raw = source["features/values"][selected_rows[row_order]][inverse]
        native_order = np.argsort(chosen)
        measurements = source["raw_layers/native/measurements"][chosen[native_order]][
            np.argsort(native_order)
        ]
        center = source["reference_scaler/center"][:]
        scale = source["reference_scaler/scale"][:]
        columns = json.loads(source.attrs["feature_columns_json"])
        candidate_count = int(
            np.sum(available[rows] & (partition[rows] == 0) & (roles[rows] == 1))
        )
    if (
        center.shape != (raw.shape[1],)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid fixed parent feature scaler")
    selected_ids = ids[selected_rows]
    np.testing.assert_array_equal(measurements["source_id"], selected_ids)
    safe, visible = spectrophotometric_inputs(
        torch.from_numpy(((raw - center) / scale).astype("f4")),
        torch.from_numpy(np.isfinite(raw)),
        columns,
    )
    return dict(
        source_id=selected_ids,
        native_row=chosen,
        feature_row=selected_rows,
        measurements=measurements,
        raw_values=raw,
        safe=safe.numpy(),
        visible=visible.numpy(),
        feature_columns=columns,
        center=center,
        scale=scale,
        audit=dict(
            role=1,
            benchmark_partition=0,
            size=size,
            candidate_count=candidate_count,
            sample_seed=seed,
            selection_label_blind=True,
            label_scaler_refitted=False,
            source_id_sha256=hashlib.sha256(
                selected_ids.astype("<i8").tobytes()
            ).hexdigest(),
            historical_role1_use_audited=False,
            unsupervised_overlap_excluded=False,
        ),
    )


def evaluate_reserved_predictions(
    sample: dict, prediction: np.ndarray, source_id: np.ndarray
) -> dict:
    """Evaluate point predictions in dex, requiring exact tuning-star association."""
    np.testing.assert_array_equal(np.asarray(source_id), sample["source_id"])
    truth = np.asarray(sample["measurements"]["value"], dtype="f8")
    point = np.asarray(prediction, dtype="f8")
    if (
        point.shape != truth.shape
        or not np.isfinite(point).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Require finite dex point predictions with exact tuning shape")
    residual = point - truth
    return dict(
        n=len(truth),
        mae_dex=float(np.abs(residual).mean()),
        bias_dex=float(residual.mean()),
        rmse_dex=float(np.sqrt(np.mean(residual**2))),
        purpose="role1_tuning_not_final_identity_test",
        **sample["audit"],
    )
