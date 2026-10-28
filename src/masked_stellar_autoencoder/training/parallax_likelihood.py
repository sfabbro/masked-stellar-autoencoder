"""Proposed frequentist distance head: likelihood of signed parallaxes.

This module is not wired into an accepted experiment or the running trainer.
Errors are formal measurement errors in mas; Gaussian independent errors are
an assumption, not a certificate of calibrated total uncertainty. No distance
prior, observed-parallax inversion, positivity cut, or error floor is applied.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def distance_modulus_to_parallax_mas(distance_modulus: Tensor) -> Tensor:
    """Convert mu=5*log10(d/pc)-5 to positive predicted parallax in mas.

    mu=10 corresponds to d=1 kpc and q=1 mas. Measured parallaxes must
    never be passed through this transformation. No numerical clipping occurs.
    """
    if not distance_modulus.is_floating_point():
        raise TypeError("distance modulus must be floating point")
    return torch.exp((2.0 - distance_modulus / 5.0) * math.log(10.0))


def signed_parallax_nll(
    distance_modulus: Tensor,
    measured_parallax_mas: Tensor,
    formal_error_mas: Tensor,
    *,
    zero_point_mas: float = 0.0,
    eligible: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Return mean Gaussian measurement NLL and the usable-label count.

    The observation mean is q(mu)+zero_point_mas. Eligibility is combined
    with finite measurement and finite positive formal error; measured sign
    never determines eligibility. Missing labels are selected out before any
    arithmetic. An empty batch returns differentiable zero with count zero.
    Only a fixed, explicitly supplied zero point is supported in this primitive.
    """
    if not distance_modulus.is_floating_point():
        raise TypeError("distance modulus must be floating point")
    if not (
        distance_modulus.shape == measured_parallax_mas.shape == formal_error_mas.shape
    ):
        raise ValueError("prediction, parallax and error shapes must match exactly")
    if not (
        distance_modulus.device
        == measured_parallax_mas.device
        == formal_error_mas.device
    ):
        raise ValueError("prediction, parallax and error must use the same device")
    if not math.isfinite(zero_point_mas):
        raise ValueError("zero point must be finite and expressed in mas")
    valid = (
        torch.isfinite(measured_parallax_mas)
        & torch.isfinite(formal_error_mas)
        & (formal_error_mas > 0)
    )
    if eligible is not None:
        if eligible.shape != valid.shape or eligible.dtype != torch.bool:
            raise ValueError(
                "eligibility must be a boolean mask of the exact label shape"
            )
        if eligible.device != valid.device:
            raise ValueError("eligibility and labels must use the same device")
        valid = valid & eligible
    count = int(valid.sum().item())
    if count == 0:
        return distance_modulus[valid].sum(), 0
    dtype = torch.promote_types(
        distance_modulus.dtype,
        torch.promote_types(measured_parallax_mas.dtype, formal_error_mas.dtype),
    )
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    mu = distance_modulus[valid].to(dtype)
    predicted = distance_modulus_to_parallax_mas(mu)
    if not torch.all(torch.isfinite(mu) & torch.isfinite(predicted) & (predicted > 0)):
        raise ValueError(
            "usable labels require finite, strictly positive model parallaxes"
        )
    measured = measured_parallax_mas[valid].to(dtype)
    sigma = formal_error_mas[valid].to(dtype)
    residual = (measured - zero_point_mas - predicted) / sigma
    loss = 0.5 * residual.square() + sigma.log() + 0.5 * math.log(2.0 * math.pi)
    if not torch.isfinite(loss).all():
        raise ValueError("parallax likelihood overflow; no silent numerical clipping")
    return loss.mean(), count
