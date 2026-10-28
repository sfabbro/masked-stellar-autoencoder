"""Separate Gaussian distance-modulus likelihood for raw signed parallaxes.

This optional branch requires the newer torchregress forward-quadrature API.
It is not integrated into the main atmosphere trainer or accepted job configs.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torchregress.losses.forward_quadrature import (
    GaussianLatentForwardObservationNLLLoss,
)

from .parallax_likelihood import distance_modulus_to_parallax_mas


class SignedParallaxQuadratureNLLLoss(nn.Module):
    """Marginalize a Gaussian in native distance modulus against signed mas data.

    ``distance_modulus_mean/std`` describe mu = 5 log10(d/pc) - 5, in
    magnitudes (not standardized coordinates). The observation convention is
    ``measured_parallax_mas = 10**(2 - 0.2*mu) + zero_point_mas + noise``.
    Thus mu=10 gives physical parallax 1 mas. A zero point of -0.02 mas
    shifts the observed mean to 0.98 mas. The measured parallax is never
    inverted, clipped, or filtered by sign.

    A user-specified nonnegative systematic floor adds independent Gaussian
    variance to *known positive* formal errors. It is an assumption, not a
    calibrated uncertainty estimate, and never substitutes for missing errors.
    Missing/nonpositive errors and nonfinite measurements are excluded. Other
    usable latent/map failures raise explicitly rather than being dropped.

    Caller must explicitly declare whether the same measured parallax entered
    the model inputs. That double use is rejected: this branch assumes an
    independent forward observation given the photometric/spectral features.
    The flag is a provenance declaration, not automatic feature inspection.
    Other label/feature conditioning dependencies still require source review.
    Gauss–Hermite integration is approximate; verify convergence across orders
    when the measurement likelihood is narrow relative to latent spread.
    """

    def __init__(
        self,
        quadrature_points: int = 64,
        reduction: Literal["none", "sum", "mean"] = "mean",
        *,
        zero_point_mas: float = 0.0,
        systematic_floor_mas: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(zero_point_mas):
            raise ValueError("zero_point_mas must be a finite fixed offset in mas")
        if not math.isfinite(systematic_floor_mas) or systematic_floor_mas < 0:
            raise ValueError("systematic_floor_mas must be finite and nonnegative")
        self.zero_point_mas = float(zero_point_mas)
        self.systematic_floor_mas = float(systematic_floor_mas)
        self.likelihood = GaussianLatentForwardObservationNLLLoss(
            quadrature_points=quadrature_points, reduction=reduction
        )

    def forward(
        self,
        distance_modulus_mean: Tensor,
        distance_modulus_std: Tensor,
        measured_parallax_mas: Tensor,
        formal_error_mas: Tensor,
        *,
        parallax_used_as_model_input: bool,
        eligible: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, int]:
        """Return weighted NLL and the number of usable positive-weight labels."""
        if not isinstance(parallax_used_as_model_input, bool):
            raise TypeError("Explicit boolean parallax_used_as_model_input is required")
        if parallax_used_as_model_input:
            raise ValueError(
                "Cannot use the same parallax as both model input and independent target"
            )
        tensors = (
            distance_modulus_mean,
            distance_modulus_std,
            measured_parallax_mas,
            formal_error_mas,
        )
        shape = distance_modulus_mean.shape
        if distance_modulus_mean.ndim != 1 or any(t.shape != shape for t in tensors):
            raise ValueError(
                "Distance modulus, parallax and formal errors must match [B]"
            )
        device = distance_modulus_mean.device
        if any(not t.is_floating_point() or t.device != device for t in tensors):
            raise ValueError(
                "All numerical inputs must be floating point on the same device"
            )
        valid = (
            torch.isfinite(measured_parallax_mas)
            & torch.isfinite(formal_error_mas)
            & (formal_error_mas > 0)
        )
        if eligible is not None:
            if (
                eligible.shape != shape
                or eligible.dtype != torch.bool
                or eligible.device != device
            ):
                raise ValueError("eligible must be boolean [B] on the input device")
            valid = valid & eligible
        dtype = torch.float32
        for tensor in tensors:
            dtype = torch.promote_types(dtype, tensor.dtype)
        with torch.autocast(device_type=device.type, enabled=False):
            # Invalid errors remain masked, even when a nonzero floor was supplied.
            safe_error = torch.where(valid, formal_error_mas.to(dtype), 1.0)
            error = torch.hypot(
                safe_error, safe_error.new_tensor(self.systematic_floor_mas)
            )

            def observation_map(mu: Tensor) -> Tensor:
                physical = distance_modulus_to_parallax_mas(mu)
                if not torch.isfinite(physical).all() or (physical <= 0).any():
                    raise ValueError(
                        "Distance-modulus quadrature overflow/underflow; no silent clipping"
                    )
                return physical + self.zero_point_mas

            loss = self.likelihood(
                distance_modulus_mean,
                distance_modulus_std,
                measured_parallax_mas,
                error,
                observation_map,
                mask=valid,
                weights=weights,
            )
        usable = valid if weights is None else valid & (weights > 0)
        return loss, int(usable.sum().item())
