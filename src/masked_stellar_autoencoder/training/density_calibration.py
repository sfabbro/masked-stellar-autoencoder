"""Post-fit joint density level-set calibration with FIT-only magnitude groups.

This module calibrates a frozen score; it neither fits a density nor selects a
model. Membership is ``-log p(y_standardized | x) <= threshold`` in three
dimensions, never three rectangular intervals. Exchangeability assumptions are
required for conformal coverage; empirical group coverage is not a claim about
arbitrary conditional coverage or physical truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch


def _vector(values, name: str, *, finite: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty vector")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def numpy_threshold(scores, alpha: float) -> float:
    """Independent exact finite-sample order statistic, including infinity."""
    scores = _vector(scores, "scores")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else math.inf


def fit_magnitude_tertiles(fit_g) -> np.ndarray:
    """Boundaries use only FIT apparent G, linear sample quantiles."""
    fit_g = _vector(fit_g, "fit_g")
    boundaries = np.quantile(fit_g, [1 / 3, 2 / 3], method="linear")
    if not boundaries[0] < boundaries[1]:
        raise ValueError("FIT apparent G tertiles are degenerate")
    return boundaries


def magnitude_groups(apparent_g, boundaries) -> np.ndarray:
    """Boundary equality belongs to the higher bin: [-inf,b0), [b0,b1), ..."""
    boundaries = np.asarray(boundaries, dtype=np.float64)
    if boundaries.shape != (2,) or not np.isfinite(boundaries).all():
        raise ValueError("two finite magnitude boundaries are required")
    if not boundaries[0] < boundaries[1]:
        raise ValueError("magnitude boundaries must increase")
    return np.searchsorted(boundaries, _vector(apparent_g, "apparent_g"), side="right")


def evaluate_density_scores(
    log_prob: Callable,
    context: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Evaluate any frozen likelihood, including positive-context contrastive flow.

    ``log_prob(context, targets)`` must return one JOINT log-density per row;
    targets must already use the fitted three-label standardization. Evaluator
    mode and model freezing are the caller's responsibility.
    """
    if targets.ndim != 2 or targets.shape[1] != 3 or len(context) != len(targets):
        raise ValueError("row-aligned three-dimensional standardized targets required")
    if batch_size < 1 or not len(targets):
        raise ValueError("positive batch size and nonempty targets required")
    chunks = []
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            stop = min(start + batch_size, len(targets))
            score = -log_prob(context[start:stop], targets[start:stop])
            if score.shape not in ((stop - start,), (stop - start, 1)):
                raise ValueError("log_prob must return one joint density per row")
            chunks.append(score.detach().cpu().to(torch.float64).numpy().reshape(-1))
    return _vector(np.concatenate(chunks), "density scores")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def calibrate_density_scores(
    *,
    calibration_scores,
    test_scores,
    fit_g,
    calibration_g,
    test_g,
    fit_ids,
    calibration_ids,
    test_ids,
    provenance: Mapping[str, str],
    api: str = "density",
    alphas: tuple[float, ...] = (0.1, 0.05),
) -> tuple[dict, dict[str, np.ndarray]]:
    """Check torchregress membership against independent NumPy order statistics.

    Scores must be frozen-model negative JOINT log-densities in standardized
    label space. ``api='contrastive'`` exercises ContrastiveFlowConformal on
    the SAME precomputed positive-density scores, not a learned classifier
    ratio or a second fitted model. Only uniform, unweighted calibration is
    supported. Role IDs must be unique and mutually disjoint.
    """
    if api not in ("density", "contrastive"):
        raise ValueError("api must be density or contrastive")
    if (
        not alphas
        or len(set(alphas)) != len(alphas)
        or any(not 0 < a < 1 for a in alphas)
    ):
        raise ValueError("alphas must be distinct miscoverage levels in (0, 1)")
    required = ("checkpoint_sha256", "scaler_sha256", "source_sha256")
    for key in required:
        value = provenance.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be an actual lowercase SHA256 receipt")
    arrays = {
        "calibration_scores": _vector(calibration_scores, "calibration_scores"),
        "test_scores": _vector(test_scores, "test_scores"),
        "fit_g": _vector(fit_g, "fit_g"),
        "calibration_g": _vector(calibration_g, "calibration_g"),
        "test_g": _vector(test_g, "test_g"),
    }
    for role in ("fit", "calibration", "test"):
        ids = np.asarray(locals()[f"{role}_ids"])
        if ids.ndim != 1 or ids.dtype.kind not in "iuUS":
            raise ValueError("role IDs must be integer or string vectors")
        ids = ids.astype(str)
        if len(ids) != len(arrays[f"{role}_g"]) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"{role} IDs must be unique and row aligned")
        arrays[f"{role}_ids"] = ids
    for left, right in (
        ("fit", "calibration"),
        ("fit", "test"),
        ("calibration", "test"),
    ):
        if np.intersect1d(arrays[f"{left}_ids"], arrays[f"{right}_ids"]).size:
            raise ValueError("FIT, calibration and test roles must be disjoint")
    for role in ("calibration", "test"):
        if len(arrays[f"{role}_scores"]) != len(arrays[f"{role}_g"]):
            raise ValueError(f"{role} scores and magnitudes must be row aligned")
        arrays[f"{role}_weights"] = np.ones(len(arrays[f"{role}_g"]), dtype=np.float64)
    boundaries = fit_magnitude_tertiles(arrays["fit_g"])
    arrays["g_boundaries"] = boundaries
    for role in ("fit", "calibration", "test"):
        arrays[f"{role}_groups"] = magnitude_groups(arrays[f"{role}_g"], boundaries)
    counts = {
        role: np.bincount(arrays[f"{role}_groups"], minlength=3).tolist()
        for role in ("fit", "calibration", "test")
    }
    if min(counts["calibration"]) == 0:
        raise ValueError("each FIT-defined magnitude bin needs calibration rows")
    from torchregress.losses.conformal import (
        ConformalizedDensity,
        ContrastiveFlowConformal,
    )

    scores = torch.from_numpy(
        np.concatenate([arrays["calibration_scores"], arrays["test_scores"]])
    )
    n_cal = len(arrays["calibration_scores"])
    cal_context = torch.arange(n_cal)[:, None]
    test_context = torch.arange(n_cal, len(scores))[:, None]
    # CTI uses targets only for row count; actual standardized targets were
    # consumed by the score producer. These placeholders never enter a model.
    cal_target = torch.zeros((n_cal, 3), dtype=torch.float64)
    test_target = torch.zeros((len(test_context), 3), dtype=torch.float64)

    class FrozenScores:
        def log_prob(self, context, target):
            return -scores[context[:, 0].long()]

    adapter = FrozenScores()
    calibrations = []
    for alpha in alphas:
        for grouped in (False, True):
            model = (
                ConformalizedDensity(alpha=alpha)
                if api == "density"
                else ContrastiveFlowConformal(adapter, alpha=alpha)
            )
            kwargs = (
                {"groups": torch.from_numpy(arrays["calibration_groups"])}
                if grouped
                else {}
            )
            if api == "density":
                model.calibrate(
                    lambda c, y: -adapter.log_prob(c, y),
                    cal_context,
                    cal_target,
                    **kwargs,
                )
            else:
                model.calibrate(cal_context, cal_target, **kwargs)
            thresholds = np.array(
                [
                    numpy_threshold(
                        arrays["calibration_scores"][arrays["calibration_groups"] == g],
                        alpha,
                    )
                    for g in range(3)
                ]
                if grouped
                else [numpy_threshold(arrays["calibration_scores"], alpha)]
            )
            library = model.cti.q_hat
            actual = np.array(
                [float(library[g]) for g in range(3)] if grouped else [float(library)]
            )
            np.testing.assert_array_equal(actual, thresholds)
            test_kwargs = (
                {"groups": torch.from_numpy(arrays["test_groups"])} if grouped else {}
            )
            if api == "density":
                membership = model.contains(
                    lambda c, y: -adapter.log_prob(c, y),
                    test_context,
                    test_target,
                    **test_kwargs,
                )
            else:
                membership = model.contains(test_context, test_target, **test_kwargs)
            per_row = (
                thresholds[arrays["test_groups"]]
                if grouped
                else np.full(len(test_context), thresholds[0])
            )
            expected = arrays["test_scores"] <= per_row
            np.testing.assert_array_equal(membership.numpy(), expected)
            key = f"{'mondrian' if grouped else 'global'}_alpha_{alpha:g}"
            arrays[f"{key}_thresholds"] = thresholds
            arrays[f"{key}_membership"] = expected
            arrays[f"{key}_per_test_threshold"] = per_row
            calibrations.append(
                {
                    "key": key,
                    "alpha": alpha,
                    "grouped": grouped,
                    "thresholds": [
                        float(q) if np.isfinite(q) else None for q in thresholds
                    ],
                    "unbounded": np.isinf(thresholds).tolist(),
                    "test_coverage": float(expected.mean()),
                    "group_test_coverage": [
                        float(expected[arrays["test_groups"] == g].mean())
                        if counts["test"][g]
                        else None
                        for g in range(3)
                    ],
                }
            )
    summary = {
        "status": "density_level_set_arithmetic_passed",
        "api": api,
        "score": "negative_joint_log_density_standardized_three_labels",
        "group_definition": "FIT apparent G linear tertiles; boundary equality in higher bin",
        "counts": counts,
        "provenance": dict(provenance),
        "calibrations": calibrations,
        "array_sha256": {key: _digest(value) for key, value in arrays.items()},
        "weighting": "unweighted; stored weights all one",
        "limits": "Frozen-score arithmetic only, not independent model replay. Joint level-set membership, not rectangular intervals. Exchangeability required; no arbitrary conditional or physical-truth coverage claim.",
    }
    return summary, arrays


def save_density_calibration(
    path: Path, summary: dict, arrays: Mapping[str, np.ndarray]
) -> None:
    """Preserve auditable scores, role IDs, group rules, memberships and receipts."""
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "density-calibration-arrays.npz", **arrays)
    (path / "density-calibration-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
