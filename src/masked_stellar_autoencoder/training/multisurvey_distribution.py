"""Distribution diagnostics in native survey observation space.

The canonical covariance describes the feature-conditioned latent Gaussian.
Coverage and PIT below instead include the declared survey map and measurement
noise. Neither is a certificate of calibration against physical stellar truth.
"""

from __future__ import annotations

import torch


def observation_marginals(mean, lower, packed):
    """Return mapped means, variances and PITs for the packed observed labels."""
    covariance = lower @ lower.transpose(-1, -2)
    jacobian = torch.where(packed.mask[..., None], packed.jacobian, 0.0)
    projected = jacobian @ covariance @ jacobian.transpose(-1, -2)
    variance = (projected + packed.measurement_covariance).diagonal(dim1=-2, dim2=-1)
    active = packed.mask
    if not torch.isfinite(variance[active]).all() or (variance[active] <= 0).any():
        raise FloatingPointError("Observed predictive variances must be positive")
    safe_variance = torch.where(active, variance, torch.ones_like(variance))
    residual = torch.where(active, packed.values - packed.mapped_mean, 0.0)
    z = residual / safe_variance.sqrt()
    pit = 0.5 * (1 + torch.erf(z / 2**0.5))
    return packed.mapped_mean, variance, torch.where(active, pit, torch.nan)


def joint_gaussian_draws(mean, lower, count, *, generator=None):
    """Generate coupled [B,K,D] draws, retaining cross-target covariance."""
    if count < 1 or mean.ndim != 2 or lower.shape != (*mean.shape, mean.shape[-1]):
        raise ValueError("Positive count and aligned [B,D], [B,D,D] required")
    noise = torch.randn(
        (len(mean), count, mean.shape[-1]),
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    return mean[:, None] + noise @ lower.transpose(-1, -2)
