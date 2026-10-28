"""Reliable first joint stellar head and native-observation likelihoods.

The three Gaussian coordinates are train-standardized *native* Teff, logg and
FeH. Temperature is linear Kelvin, not log temperature: a quoted Kelvin error
must not be silently treated as Gaussian in log temperature. The Gaussian
conditional covariance is added to the quoted measurement covariance before
scoring. Its interpretation as latent scatter depends on the measurement model;
unknown survey systematics/covariances are not fixed by this loss.

Distance is a separate log(kpc) point output fitted to signed raw parallax.
This module does NOT supply a joint stellar/distance PDF, a distance interval,
Bayesian inference or a calibration certificate. Alpha and age are absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossSum:
    """Add numerators/counts across microbatches before normalizing.

    Gaussian normalization uses observed scalar-label count, not local row
    count. Backpropagate numerator / *whole optimizer-batch* observation count
    on each microbatch. A wholly missing batch has differentiable zero.
    """

    numerator: Tensor
    observation_count: int
    row_count: int

    @property
    def mean(self) -> Tensor:
        return self.numerator / max(1, self.observation_count)


class ResidualJointHead(nn.Module):
    """Joint native-parameter Gaussian plus a log-distance point estimate."""

    def __init__(
        self,
        input_dim: int = 256,
        width: int = 128,
        target_dim: int = 3,
        initial_log_distance: float = 0.0,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("width", width),
            ("target_dim", target_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if target_dim > 8:
            raise ValueError("at most 8 targets supported by explicit mask grouping")
        if not math.isfinite(initial_log_distance):
            raise ValueError("initial log distance must be finite")
        self.target_dim = target_dim
        self.input = nn.Linear(input_dim, width)
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, max(1, width // 4)),
            nn.GELU(),
            nn.Linear(max(1, width // 4), width),
        )
        n_tril = target_dim * (target_dim + 1) // 2
        self.output = nn.Linear(width, target_dim + n_tril + 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[-1] = initial_log_distance
        index = torch.tril_indices(target_dim, target_dim)
        self.register_buffer("tril_row", index[0], persistent=False)
        self.register_buffer("tril_col", index[1], persistent=False)

    def forward(self, latent: Tensor) -> dict[str, Tensor]:
        if latent.ndim != 2 or not latent.is_floating_point():
            raise ValueError("latent must be a floating [row, feature] tensor")
        hidden = torch.nn.functional.gelu(self.input(latent))
        raw = self.output(hidden + self.block(hidden))
        if not torch.isfinite(raw).all():
            raise ValueError("joint head produced nonfinite outputs")
        d = self.target_dim
        packed = raw[:, d:-1]
        diag = self.tril_row == self.tril_col
        # Exponentiate only the used diagonal. torch.where(..., exp(raw), raw)
        # also evaluates unused off-diagonal exponentials and can create NaN
        # gradients from 0*inf even when the selected forward value is finite.
        diagonal = packed[:, diag].exp()
        packed = packed.clone()
        packed[:, diag] = diagonal
        if not torch.isfinite(packed).all() or not (packed[:, diag] > 0).all():
            raise ValueError(
                "invalid joint covariance; no silent clipping or error floor"
            )
        tril = packed.new_zeros((len(latent), d, d))
        tril[:, self.tril_row, self.tril_col] = packed
        return {"mean": raw[:, :d], "scale_tril": tril, "log_distance": raw[:, -1]}


def _float_tensors_same_device(*values: Tensor) -> None:
    if not all(v.is_floating_point() for v in values):
        raise TypeError("predictions, measurements and errors must be floating point")
    if not all(v.device == values[0].device for v in values):
        raise ValueError("all tensors must be on the same device")


def _eligible(valid: Tensor, eligible: Tensor | None) -> Tensor:
    if eligible is None:
        return valid
    if eligible.shape != valid.shape or eligible.dtype != torch.bool:
        raise ValueError(
            "eligibility must be a boolean tensor of the exact target shape"
        )
    if eligible.device != valid.device:
        raise ValueError("eligibility and observations must use the same device")
    return valid & eligible


def masked_gaussian_measurement_nll(
    mean: Tensor,
    scale_tril: Tensor,
    target: Tensor,
    formal_error: Tensor,
    *,
    eligible: Tensor | None = None,
    measurement_covariance: Tensor | None = None,
) -> LossSum:
    """Exact Gaussian observed-subvector likelihood, with missing data omitted.

    All coordinates/errors/covariances must already use one consistent fit-only
    standardized native coordinate system. Default measurement covariance is
    diagonal formal_error**2, an explicit independence assumption. If supplied,
    every USED submatrix must be symmetric positive semidefinite and its diagonal
    must agree with formal_error**2. No missing-error imputation or extra floor.
    """
    _float_tensors_same_device(mean, scale_tril, target, formal_error)
    if mean.ndim != 2 or target.shape != mean.shape or formal_error.shape != mean.shape:
        raise ValueError("mean, target and formal error must share [row, target] shape")
    n, d = mean.shape
    if not 1 <= d <= 8 or scale_tril.shape != (n, d, d):
        raise ValueError(
            "scale_tril must be [row,target,target], with 1 through 8 targets"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(scale_tril).all():
        raise ValueError("all model parameters must be finite")
    if not (scale_tril.diagonal(dim1=-2, dim2=-1) > 0).all():
        raise ValueError("model scale diagonal must be strictly positive")
    if torch.count_nonzero(torch.triu(scale_tril, diagonal=1)):
        raise ValueError("model scale must be lower triangular")
    valid = _eligible(
        torch.isfinite(target) & torch.isfinite(formal_error) & (formal_error > 0),
        eligible,
    )
    count = int(valid.sum().item())
    rows = int(valid.any(1).sum().item())
    if measurement_covariance is not None:
        _float_tensors_same_device(mean, measurement_covariance)
        if measurement_covariance.shape != (n, d, d):
            raise ValueError(
                "measurement covariance must have exact [row,target,target] shape"
            )
    # A differentiable zero from known finite model values, not NaN labels.
    total = mean.reshape(-1)[:0].sum() + scale_tril.reshape(-1)[:0].sum()
    if count == 0:
        return LossSum(total, 0, 0)
    # Linalg in at least float32 even when the runner uses mixed precision.
    dtype = torch.promote_types(mean.dtype, target.dtype)
    dtype = torch.promote_types(dtype, formal_error.dtype)
    dtype = torch.promote_types(dtype, scale_tril.dtype)
    if measurement_covariance is not None:
        dtype = torch.promote_types(dtype, measurement_covariance.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    with torch.autocast(device_type=mean.device.type, enabled=False):
        model_scale = scale_tril.to(dtype)
        covariance = model_scale @ model_scale.transpose(-1, -2)
        code = (valid.to(torch.int64) * (2 ** torch.arange(d, device=mean.device))).sum(
            1
        )
        for pattern in range(1, 2**d):
            keep = code == pattern
            if not keep.any():
                continue
            dims = [i for i in range(d) if pattern & (1 << i)]
            observed = target[keep][:, dims].to(dtype)
            errors = formal_error[keep][:, dims].to(dtype)
            sub = covariance[keep][:, dims][:, :, dims]
            noise = torch.diag_embed(errors.square())
            if measurement_covariance is not None:
                noise = measurement_covariance[keep][:, dims][:, :, dims].to(dtype)
                if not torch.isfinite(noise).all():
                    raise ValueError("used measurement covariance must be finite")
                tolerance = 32 * torch.finfo(dtype).eps
                if not torch.allclose(
                    noise, noise.transpose(-1, -2), rtol=tolerance, atol=0
                ):
                    raise ValueError("measurement covariance must be symmetric")
                if not torch.allclose(
                    noise.diagonal(dim1=-2, dim2=-1),
                    errors.square(),
                    rtol=tolerance,
                    atol=0,
                ):
                    raise ValueError(
                        "measurement covariance diagonal must match formal errors"
                    )
                # A genuine covariance can be singular, so validate PSD without jitter.
                if (
                    torch.linalg.eigvalsh(noise)
                    < -tolerance * errors.square().amax(1)[:, None]
                ).any():
                    raise ValueError(
                        "measurement covariance must be positive semidefinite"
                    )
            combined = sub + noise
            chol, info = torch.linalg.cholesky_ex(combined)
            if (info != 0).any():
                raise ValueError(
                    "combined covariance is not positive definite; no silent jitter"
                )
            residual = observed - mean[keep][:, dims].to(dtype)
            whitened = torch.linalg.solve_triangular(
                chol, residual[..., None], upper=False
            )
            nll = 0.5 * (
                whitened.square().sum((1, 2)) + len(dims) * math.log(2 * math.pi)
            )
            nll = nll + chol.diagonal(dim1=-2, dim2=-1).log().sum(1)
            if not torch.isfinite(nll).all():
                raise ValueError("nonfinite joint measurement likelihood")
            total = total + nll.sum()
    return LossSum(total, count, rows)


def signed_log_distance_nll(
    log_distance: Tensor,
    measured_parallax_mas: Tensor,
    formal_error_mas: Tensor,
    *,
    zero_point_mas: Tensor | float = 0.0,
    eligible: Tensor | None = None,
) -> LossSum:
    """Point log(kpc) prediction scored against raw signed mas observations.

    The observation mean is exp(-log_distance)+zero_point_mas. A fixed scalar
    or an audited per-star zero-point tensor is accepted. Measured parallax is
    neither inverted nor sign-cut. This supplies no distance PDF or interval.
    """
    _float_tensors_same_device(log_distance, measured_parallax_mas, formal_error_mas)
    if log_distance.ndim != 1 or not (
        log_distance.shape == measured_parallax_mas.shape == formal_error_mas.shape
    ):
        raise ValueError("log distance, parallax and error must share [row] shape")
    valid = _eligible(
        torch.isfinite(measured_parallax_mas)
        & torch.isfinite(formal_error_mas)
        & (formal_error_mas > 0),
        eligible,
    )
    if isinstance(zero_point_mas, Tensor):
        _float_tensors_same_device(log_distance, zero_point_mas)
        if zero_point_mas.shape != log_distance.shape:
            raise ValueError("per-star zero point must have exact [row] shape")
        if not torch.isfinite(zero_point_mas[valid]).all():
            raise ValueError("used zero points must be finite")
        zero_point = zero_point_mas[valid]
    else:
        if not math.isfinite(zero_point_mas):
            raise ValueError("zero point must be finite")
        zero_point = zero_point_mas
    count = int(valid.sum().item())
    if count == 0:
        return LossSum(log_distance[valid].sum(), 0, 0)
    dtype = torch.promote_types(
        torch.promote_types(log_distance.dtype, measured_parallax_mas.dtype),
        formal_error_mas.dtype,
    )
    if isinstance(zero_point, Tensor):
        dtype = torch.promote_types(dtype, zero_point.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    with torch.autocast(device_type=log_distance.device.type, enabled=False):
        predicted = torch.exp(-log_distance[valid].to(dtype))
        if not (torch.isfinite(predicted) & (predicted > 0)).all():
            raise ValueError(
                "model parallax must be finite and positive; no silent clipping"
            )
        sigma = formal_error_mas[valid].to(dtype)
        residual = (
            measured_parallax_mas[valid].to(dtype) - zero_point - predicted
        ) / sigma
        nll = 0.5 * residual.square() + sigma.log() + 0.5 * math.log(2 * math.pi)
        if not torch.isfinite(nll).all():
            raise ValueError("nonfinite signed-parallax likelihood")
    return LossSum(nll.sum(), count, count)
