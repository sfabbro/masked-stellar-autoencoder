"""Deterministic supervised XP augmentation and point-ensemble aggregation.

Inputs already use the parent scaler and the -9999 missing-value convention.
Noise uses the *per-star*, G-normalized, parent-scaled reported XP sigma. It is
extra conditional diagonal measurement noise, not a physical covariance model.
No reconstruction targets, labels, photometry or withheld inputs are modified.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

MISSING = -9999.0


def canonical_xp_columns(feature_names: Sequence[str]) -> tuple[int, ...]:
    """Explicit BP1..55 then RP1..55 ordering for both values and sigma."""
    names = [str(name).casefold() for name in feature_names]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate feature names make XP alignment ambiguous")
    expected = [f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)]
    if any(name not in names for name in expected):
        raise ValueError("All110 named XP coefficients are required")
    return tuple(names.index(name) for name in expected)


def supervised_augmentation(
    values: Tensor,
    visible: Tensor,
    xp_columns: Sequence[int],
    *,
    generator: torch.Generator,
    sigma_xp: Tensor | None = None,
    noise_scale: float = 0.0,
    mask_probability: float = 0.0,
    mask_mode: str = "coefficients",
) -> tuple[Tensor, Tensor, dict[str, int | float | str]]:
    """Augment a clone using independent RNG; primary test inputs stay untouched.

    ``xp_columns`` and ``sigma_xp`` must share canonical BP1..55,RP1..55 order.
    Coefficient masks are independent Bernoulli draws. Band masking selects at
    most ONE of BP/RP per selected row, preserving the other observed band.
    Both noise and mask draws are consumed for every policy, including control,
    so paired policy variants retain common augmentation random numbers.
    """
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("values must be a floating [row,feature] matrix")
    if visible.shape != values.shape or visible.dtype != torch.bool:
        raise ValueError("visible must be a same-shape boolean matrix")
    if visible.device != values.device:
        raise ValueError("values and visible must share device")
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and nonnegative")
    if not math.isfinite(mask_probability) or not 0 <= mask_probability <= 1:
        raise ValueError("mask_probability must be finite in[0,1]")
    if mask_mode not in {"coefficients", "band"}:
        raise ValueError("mask_mode must be coefficients or band")
    columns = torch.as_tensor(xp_columns, dtype=torch.long, device=values.device)
    if (
        columns.ndim != 1
        or len(columns) != 110
        or len(torch.unique(columns)) != 110
        or torch.any(columns < 0)
        or torch.any(columns >= values.shape[1])
    ):
        raise ValueError("Require110 unique in-range XP columns")
    if not torch.all(torch.isfinite(values[visible]) & (values[visible] != MISSING)):
        raise ValueError("Visible values must be finite and not the missing sentinel")
    if not torch.all(values[~visible] == MISSING):
        raise ValueError(
            "Invisible inputs must already use the parent missing sentinel"
        )

    seen = visible[:, columns]
    selected = values[:, columns]
    if noise_scale:
        if sigma_xp is None or sigma_xp.shape != seen.shape:
            raise ValueError("Per-star standardized XP sigma must be [row,110]")
        if sigma_xp.device != values.device or not sigma_xp.is_floating_point():
            raise ValueError("sigma_xp must be floating on the input device")
        if not torch.all(torch.isfinite(sigma_xp[seen]) & (sigma_xp[seen] > 0)):
            raise ValueError("Observed XP sigma must be finite and positive")
        sigma = torch.where(seen, sigma_xp, 0).to(values.dtype)
    else:
        sigma = torch.zeros_like(selected)

    noise = torch.randn(
        selected.shape, dtype=values.dtype, device=values.device, generator=generator
    )
    coefficient_draw = torch.rand(
        selected.shape, dtype=values.dtype, device=values.device, generator=generator
    )
    row_draw = torch.rand(
        (len(values),), dtype=values.dtype, device=values.device, generator=generator
    )
    band_draw = torch.randint(
        2, (len(values),), device=values.device, generator=generator
    )
    if mask_mode == "coefficients":
        hidden = coefficient_draw < mask_probability
    else:
        band_membership = (
            torch.arange(110, device=values.device)[None, :] // 55 == band_draw[:, None]
        )
        hidden = (row_draw < mask_probability)[:, None] & band_membership
    hidden &= seen
    perturbed = selected + noise_scale * sigma * noise if noise_scale else selected
    augmented = torch.where(seen, perturbed, MISSING)
    augmented = torch.where(hidden, MISSING, augmented)
    result = values.clone()
    result[:, columns] = augmented
    remaining = visible.clone()
    remaining[:, columns] = seen & ~hidden
    if not torch.all(torch.isfinite(result)):
        raise FloatingPointError("Augmentation produced nonfinite values")
    return (
        result,
        remaining,
        dict(
            observed_xp=int(seen.sum()),
            hidden_xp=int(hidden.sum()),
            noise_scale=float(noise_scale),
            mask_probability=float(mask_probability),
            mask_mode=mask_mode,
        ),
    )


def aggregate_point_ensemble(
    member_prediction: Tensor, member_source_id: Tensor
) -> tuple[Tensor, Tensor]:
    """Return float64 point mean and population member spread, never calibrated UQ.

    All members must contain identical source IDs in identical order. The
    ensemble costs the sum of its independent member training budgets.
    """
    if (
        member_prediction.ndim != 2
        or len(member_prediction) < 2
        or member_prediction.shape[1] == 0
        or not member_prediction.is_floating_point()
        or not torch.isfinite(member_prediction).all()
    ):
        raise ValueError("Require finite floating [member>=2,star] predictions")
    if member_source_id.shape != member_prediction.shape:
        raise ValueError("Each prediction requires its member/star source ID")
    if member_source_id.is_floating_point() or member_source_id.dtype == torch.bool:
        raise ValueError("Source IDs must retain integer precision")
    if not torch.all(member_source_id == member_source_id[0]):
        raise ValueError("Ensemble member source IDs/order differ")
    if len(torch.unique(member_source_id[0])) != member_prediction.shape[1]:
        raise ValueError("Duplicate stars are not independent evaluation rows")
    predictions = member_prediction.double()
    return predictions.mean(0), predictions.std(0, correction=0)
