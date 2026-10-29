"""Differentiable energy scores for partly observed multivariate predictions.

This module scores samples; it is not a marginal log likelihood. In particular,
it does not certify the earlier failed importance-sampling/NLL gates, identify
unobserved correlations, or choose a physical abundance scale.

Inputs are in a common observed-coordinate layout. A caller with multiple
surveys must construct that layout from the SAME full latent draw for all
records, implement suitable source observation maps, and explicitly handle
shared-spectrum dependencies. This module does not provide those mappings.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_inputs(
    samples: Tensor, target: Tensor, observed: Tensor, scale, weights, chunk_size
):
    if not all(isinstance(value, Tensor) for value in (samples, target, observed)):
        raise ValueError("Samples, targets and masks must be tensors")
    if (
        samples.ndim != 3
        or samples.shape[0] < 2
        or samples.shape[1] < 1
        or samples.shape[2] < 1
        or samples.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Samples must be finite float32/64 [S,B,D] with S >= 2")
    _, batch, dims = samples.shape
    if (
        target.shape != (batch, dims)
        or target.dtype != samples.dtype
        or target.device != samples.device
        or observed.shape != (batch, dims)
        or observed.dtype != torch.bool
        or observed.device != samples.device
    ):
        raise ValueError(
            "Target and boolean mask must align in shape, dtype and device"
        )
    if (
        not torch.isfinite(samples).all()
        or not observed.any(dim=1).all()
        or not torch.isfinite(target[observed]).all()
    ):
        raise ValueError("Every star needs finite observed labels and finite samples")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("Chunk size must be a positive integer")
    scale = torch.as_tensor(scale, dtype=samples.dtype, device=samples.device)
    if (
        scale.shape != (dims,)
        or scale.requires_grad
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Fixed finite positive per-coordinate scales are required")
    if weights is not None:
        if (
            not isinstance(weights, Tensor)
            or not weights.is_floating_point()
            or weights.shape != (batch,)
            or weights.device != samples.device
            or weights.requires_grad
            or not torch.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("Star weights must be fixed finite positive [B] tensors")
        weights = weights.to(dtype=samples.dtype)
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(
                "Star weights must remain finite and positive in sample dtype"
            )
    return scale, weights


def masked_energy_score(
    samples: Tensor,
    target: Tensor,
    observed: Tensor,
    *,
    scale,
    weights: Tensor | None = None,
    reduction: str = "mean",
    chunk_size: int = 128,
) -> Tensor:
    """Score finite [S,B,D] vector samples on each star's observed coordinates.

    Scales are fixed coordinate units chosen without test leakage. The first
    expectation is estimated with S samples; the second uses the unbiased
    off-diagonal U-statistic over S*(S-1) ordered pairs. No nonnegativity
    clamp or admission gate is applied. Missing target coordinates may be NaN.

    If weights are supplied, none returns per-star weighted scores, sum returns
    their sum, and mean divides that sum by the total weight; this self-normalized
    weighted mean is not automatically an unbiased estimate of a population mean.
    The caller must validate its sampling/weight contract. Chunking limits
    pairwise storage to O(chunk_size*S*S).
    """
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("Reduction must be none, sum or mean")
    scale, weights = _validate_inputs(
        samples, target, observed, scale, weights, chunk_size
    )
    count, batch, _ = samples.shape
    patterns, inverse = torch.unique(observed, dim=0, return_inverse=True)
    per_star = samples.new_zeros(batch)
    for pattern_index, pattern in enumerate(patterns):
        rows = torch.nonzero(inverse == pattern_index, as_tuple=False).flatten()
        columns = torch.nonzero(pattern, as_tuple=False).flatten()
        for offset in range(0, len(rows), chunk_size):
            selected_rows = rows[offset : offset + chunk_size]
            prediction = samples[:, selected_rows][:, :, columns] / scale[columns]
            truth = target[selected_rows][:, columns] / scale[columns]
            # vector_norm has a finite zero subgradient at exact equality.
            first = torch.linalg.vector_norm(
                prediction - truth.unsqueeze(0), ord=2, dim=-1
            ).mean(dim=0)
            by_star = prediction.permute(1, 0, 2)
            distances = torch.cdist(
                by_star,
                by_star,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            second = distances.sum(dim=(1, 2)) / (2 * count * (count - 1))
            per_star = per_star.index_copy(0, selected_rows, first - second)
    effective = per_star if weights is None else per_star * weights
    if reduction == "none":
        return effective
    if reduction == "sum":
        return effective.sum()
    if weights is None:
        return effective.mean()
    return effective.sum() / weights.sum()


def masked_noise_aware_energy_score(
    samples: Tensor,
    target: Tensor,
    observed: Tensor,
    error: Tensor,
    error_known: Tensor,
    *,
    scale,
    generator: torch.Generator,
    assume_independent_errors: bool,
    unknown_error_floor=None,
    weights: Tensor | None = None,
    reduction: str = "mean",
    chunk_size: int = 128,
) -> Tensor:
    """Score vector draws convolved with an explicit diagonal Gaussian error model.

    Known observed standard deviations must be strictly positive and finite.
    Every observed unknown error requires an explicit strictly positive floor.
    A supplied floor can be scalar, [D], or [B,D], and must be fixed and
    nonnegative everywhere. Unknown errors are never turned into zero noise.

    The caller explicitly accepts independent measurement errors across the
    provided coordinates. Independent Gaussian noise is sampled for each
    vector draw, star and coordinate using the provided generator. Persist its
    state for resume and advance it between updates. Catalog posterior widths
    are not automatically measurement errors. Shared spectral/systematic
    covariance requires a different observation/noise model.
    """
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("Reduction must be none, sum or mean")
    scale, weights = _validate_inputs(
        samples, target, observed, scale, weights, chunk_size
    )
    if assume_independent_errors is not True:
        raise ValueError("Explicit independent-error assumption is required")
    if not isinstance(generator, torch.Generator):
        raise ValueError("An explicit torch.Generator is required")
    generator_device = torch.device(generator.device)
    if generator_device.type != samples.device.type or (
        generator_device.index is not None
        and generator_device.index != samples.device.index
    ):
        raise ValueError("Generator and sample devices differ")
    if (
        not isinstance(error, Tensor)
        or error.shape != target.shape
        or error.dtype != samples.dtype
        or error.device != samples.device
        or error.requires_grad
        or not isinstance(error_known, Tensor)
        or error_known.shape != observed.shape
        or error_known.dtype != torch.bool
        or error_known.device != observed.device
    ):
        raise ValueError("Fixed errors and known-error masks must align with targets")
    if (error_known & ~observed).any():
        raise ValueError("Known-error flags require observed labels")
    if not torch.isfinite(error[error_known]).all() or (error[error_known] <= 0).any():
        raise ValueError("Known observed errors must be finite and strictly positive")
    unknown = observed & ~error_known
    if unknown_error_floor is None:
        if unknown.any():
            raise ValueError(
                "Unknown observed errors require an explicit positive floor"
            )
        floor = torch.zeros_like(target)
    else:
        floor = torch.as_tensor(unknown_error_floor, device=samples.device)
        if floor.dtype == torch.bool or floor.is_complex():
            raise ValueError("Unknown-error floor must be real numeric, not boolean")
        floor = floor.to(dtype=samples.dtype)
        if floor.shape not in ((), (target.shape[1],), target.shape):
            raise ValueError("Unknown-error floor must be scalar, [D] or [B,D]")
        try:
            floor = torch.broadcast_to(floor, target.shape)
        except RuntimeError as error_exc:
            raise ValueError(
                "Unknown-error floor must broadcast to [B,D]"
            ) from error_exc
        if (
            floor.requires_grad
            or not torch.isfinite(floor).all()
            or (floor < 0).any()
            or (floor[unknown] <= 0).any()
        ):
            raise ValueError(
                "Unknown observed errors need fixed finite positive floors"
            )
    sigma = torch.where(observed, torch.where(error_known, error, floor), 0.0)
    noise = torch.randn(
        samples.shape, dtype=samples.dtype, device=samples.device, generator=generator
    )
    return masked_energy_score(
        samples + noise * sigma.unsqueeze(0),
        target,
        observed,
        scale=scale,
        weights=weights,
        reduction=reduction,
        chunk_size=chunk_size,
    )
