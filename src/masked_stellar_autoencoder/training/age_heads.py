"""Age head in ``tau = log(age / Gyr)``.

``tau`` is the **natural** logarithm of age in Gyr. This matches TRAINING_PLAN
Sec. 6 and ``05-heads-losses-metrics.md`` Sec. 5; the ``log10`` label in
``07-inputs-architecture.md`` Sec. 4 is superseded here and must not be mixed
with these outputs (a log10/log ratio is a constant shift in ``tau`` and would
silently rescale every width).

Three formulations, in the order the plan builds them:

1. **Multi-quantile (primary).** ``AgeQuantileHead`` predicts
   ``(0.16, 0.5, 0.84)`` and optionally ``(0.05, 0.95)`` of ``tau``. Trained by
   the pinball loss plus a crossover penalty. Pinball is proper for the
   quantiles and natively asymmetric, which is what the StarHorse
   ``16/50/84`` summaries are.
2. **Split-normal (control, documented approximation).**
   ``p(tau) = sqrt(2/pi)/(s_l + s_r) exp(-(tau - m)^2 / 2 s_{l,r}^2)`` with the
   density maximized at ``m`` (the *mode*, not the median for ``s_l != s_r``).
   Its 16/84 percentiles are only approximately ``m -/+ s`` (exactly
   ``0.1587`` for equal widths): that error is reported, never claimed away.
3. **Heteroscedastic Gaussian (control).** ``mu``, ``log sigma`` on ``tau``.

Evidence status: the labels are StarHorse (or similar) *posterior percentiles*.
They are noisy, asymmetric, model-conditioned summaries, **not truth**. Fitting
network quantiles to label quantiles puts errors-in-variables on the spread
itself, so predicted widths come out narrower than the true predictive spread;
the estimand is ``p(age_observed | x)``. APOKASC3 is held out entirely and is a
calibration anchor, not a truth set. No function here claims a true age.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

#: Declared evidence status for every age product of this module.
AGE_EVIDENCE_STATUS = "observed_label (StarHorse-conditioned); not physical truth"

#: Primary quantile grid; the 5/95 pair is opt-in and never silently added.
PRIMARY_AGE_QUANTILES = (0.16, 0.50, 0.84)
WIDE_AGE_QUANTILES = (0.05, 0.16, 0.50, 0.84, 0.95)


@dataclass(frozen=True)
class AgeLoss:
    """Numerator/count pair, field-compatible with ``joint_prediction.LossSum``.

    ``observation_count`` is the number of usable rows times the number of
    quantiles, so ``mean`` is the per-row-per-quantile pinball value (declare
    ``loss_denominator="per_row_per_quantile"`` in the certificate).
    ``crossing_fraction`` is the diagnostic fraction of rows whose quantiles
    cross, and is NaN-free: it is zero for an empty batch.
    """

    numerator: Tensor
    observation_count: int
    row_count: int
    crossing_fraction: float = 0.0

    @property
    def mean(self) -> Tensor:
        return self.numerator / max(1, self.observation_count)


# --------------------------------------------------------------------------- #
# Positive-age / log-age transforms
# --------------------------------------------------------------------------- #
def log_age_gyr(age_gyr: Tensor) -> Tensor:
    """Return ``tau = log(age / Gyr)`` for a strictly positive age.

    Non-positive, non-finite or non-floating ages raise: there is no clipping,
    no absolute value and no ``log1p`` rescue, because every one of those would
    silently change the estimand.
    """
    if not age_gyr.is_floating_point():
        raise TypeError("age must be floating point")
    if not (torch.isfinite(age_gyr) & (age_gyr > 0)).all():
        raise ValueError("age must be finite and strictly positive; no clipping")
    return age_gyr.log()


def age_gyr_from_log_age(tau: Tensor) -> Tensor:
    """Return ``age = exp(tau)`` Gyr: positive by construction, never clipped."""
    if not tau.is_floating_point():
        raise TypeError("log age must be floating point")
    age = tau.exp()
    if not torch.isfinite(age).all():
        raise ValueError("age overflowed; no silent clipping of log age")
    if not (age > 0).all():
        raise ValueError("exponentiated age must be strictly positive")
    return age


# --------------------------------------------------------------------------- #
# Quantile helpers
# --------------------------------------------------------------------------- #
def _check_quantiles(quantiles) -> Tensor:
    values = torch.as_tensor(tuple(quantiles), dtype=torch.float64)
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("quantiles must be a non-empty 1-D sequence")
    if not torch.all((values > 0) & (values < 1)):
        raise ValueError("quantiles must lie strictly between 0 and 1")
    if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
        raise ValueError("quantiles must be strictly increasing and unique")
    return values


def _eligible(valid: Tensor, eligible: Tensor | None) -> Tensor:
    if eligible is None:
        return valid
    if eligible.shape != valid.shape or eligible.dtype != torch.bool:
        raise ValueError("eligibility must be a boolean mask of the exact label shape")
    if eligible.device != valid.device:
        raise ValueError("eligibility and labels must use the same device")
    return valid & eligible


def pinball_loss(
    predictions: Tensor,
    target: Tensor,
    quantiles: Tensor | tuple[float, ...],
    *,
    eligible: Tensor | None = None,
) -> AgeLoss:
    """Mean pinball (quantile) loss over observed ``(row, quantile)`` pairs.

    ``predictions`` is ``[row, quantile]`` in ``tau`` units, ``target`` is
    ``[row]`` in ``tau`` units. The pinball loss is proper for the target
    quantiles and needs no distributional assumption; it is evaluated in
    ``tau``, so the printed value is *not* in Gyr.
    """
    grid = _check_quantiles(quantiles).to(
        device=predictions.device, dtype=predictions.dtype
    )
    if predictions.ndim != 2 or predictions.shape[1] != grid.numel():
        raise ValueError("predictions must be [row, len(quantiles)]")
    if target.ndim != 1 or target.shape[0] != predictions.shape[0]:
        raise ValueError("target must be [row] and match the prediction rows")
    if not (predictions.is_floating_point() and target.is_floating_point()):
        raise TypeError("predictions and target must be floating point")
    if predictions.device != target.device or grid.device != predictions.device:
        raise ValueError("predictions, target and quantiles must share a device")
    if not torch.isfinite(predictions).all():
        raise ValueError("quantile predictions must be finite")
    valid = _eligible(torch.isfinite(target), eligible)
    count_rows = int(valid.sum().item())
    total = predictions.reshape(-1)[:0].sum()
    if count_rows == 0:
        return AgeLoss(total, 0, 0, 0.0)
    error = target[valid][:, None] - predictions[valid]
    loss = torch.maximum(grid * error, (grid - 1.0) * error)
    if not torch.isfinite(loss).all():
        raise ValueError("nonfinite pinball loss")
    crossed = (predictions[valid][:, 1:] < predictions[valid][:, :-1]).any(1)
    return AgeLoss(
        loss.sum(),
        count_rows * grid.numel(),
        count_rows,
        float(crossed.double().mean().item()),
    )


def crossover_penalty(
    predictions: Tensor,
    quantiles: Tensor | tuple[float, ...],
    *,
    eligible: Tensor | None = None,
) -> Tensor:
    """Mean adjacent crossing penalty ``mean_i sum_j relu(q_ij - q_i,j+1)``.

    Returns a differentiable zero for an empty batch. Crossings are penalized,
    never silently sorted away; use ``AgeQuantileHead(monotone=True)`` when a
    non-crossing output is required for interval reporting.
    """
    grid = _check_quantiles(quantiles).to(
        device=predictions.device, dtype=predictions.dtype
    )
    if predictions.ndim != 2 or predictions.shape[1] != grid.numel():
        raise ValueError("predictions must be [row, len(quantiles)]")
    if not predictions.is_floating_point():
        raise TypeError("predictions must be floating point")
    if not torch.isfinite(predictions).all():
        raise ValueError("quantile predictions must be finite")
    if grid.numel() < 2:
        return predictions.reshape(-1)[:0].sum()
    valid = _eligible(
        torch.ones(predictions.shape[0], dtype=torch.bool, device=predictions.device),
        eligible,
    )
    if not bool(valid.any()):
        return predictions.reshape(-1)[:0].sum()
    gaps = predictions[valid][:, 1:] - predictions[valid][:, :-1]
    return torch.relu(-gaps).sum(1).mean()


def age_quantile_loss(
    predictions: Tensor,
    target: Tensor,
    quantiles: Tensor | tuple[float, ...],
    *,
    eligible: Tensor | None = None,
    crossover_weight: float = 1.0,
) -> AgeLoss:
    """Primary age objective: pinball plus ``crossover_weight`` x crossing.

    ``crossover_weight`` must be finite and non-negative; a negative weight
    would reward crossings and is rejected.
    """
    if not math.isfinite(crossover_weight) or crossover_weight < 0:
        raise ValueError("crossover_weight must be finite and non-negative")
    pinball = pinball_loss(predictions, target, quantiles, eligible=eligible)
    penalty = crossover_penalty(predictions, quantiles, eligible=eligible)
    return AgeLoss(
        pinball.numerator + crossover_weight * penalty,
        pinball.observation_count,
        pinball.row_count,
        pinball.crossing_fraction,
    )


def interval_from_quantiles(
    predictions: Tensor,
    quantiles: Tensor | tuple[float, ...],
    *,
    lower: float = 0.16,
    upper: float = 0.84,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(median, lower, upper, asymmetry)`` from quantile predictions.

    ``asymmetry = (upper - median) - (median - lower)`` in ``tau``: positive
    means a right-skewed (older) tail. Inputs are never reordered silently; a
    crossing pair is returned as-is so the caller can report it.
    """
    grid = _check_quantiles(quantiles)
    if predictions.ndim != 2 or predictions.shape[1] != grid.numel():
        raise ValueError("predictions must be [row, len(quantiles)]")
    names = grid.tolist()
    for name, value in (("lower", lower), ("upper", upper)):
        if value not in names:
            raise ValueError(f"{name} quantile {value} is not part of the head grid")
    if not lower < upper:
        raise ValueError("lower must be below upper")
    median_index = names.index(0.5) if 0.5 in names else None
    if median_index is None:
        raise ValueError("a 0.5 quantile is required to report asymmetry")
    lower_q = predictions[:, names.index(lower)]
    upper_q = predictions[:, names.index(upper)]
    median = predictions[:, median_index]
    return median, lower_q, upper_q, (upper_q - median) - (median - lower_q)


# --------------------------------------------------------------------------- #
# Split-normal density (control)
# --------------------------------------------------------------------------- #
def split_normal_log_prob(
    value: Tensor,
    peak: Tensor,
    sigma_lower: Tensor,
    sigma_upper: Tensor,
) -> Tensor:
    """Exact split-normal log density in ``tau`` (peak = mode, not median).

    ``log p = log sqrt(2/pi) - log(s_l + s_r) - (value - m)^2 / (2 s^2)`` with
    ``s = s_l`` below the mode and ``s = s_r`` above it. Widths must be finite
    and strictly positive: no floor is substituted.
    """
    for name, width in (("sigma_lower", sigma_lower), ("sigma_upper", sigma_upper)):
        if width.shape != peak.shape or value.shape != peak.shape:
            raise ValueError("value, peak and widths must share one shape")
        if not width.is_floating_point() or not peak.is_floating_point():
            raise TypeError("split-normal parameters must be floating point")
        if not (torch.isfinite(width).all() and (width > 0).all()):
            raise ValueError(f"{name} must be finite and strictly positive")
    if not (torch.isfinite(peak).all() and torch.isfinite(value).all()):
        raise ValueError("split-normal value and peak must be finite")
    width = torch.where(value < peak, sigma_lower, sigma_upper)
    standardized = (value - peak) / width
    return (
        math.log(_SQRT_2_OVER_PI)
        - (sigma_lower + sigma_upper).log()
        - 0.5 * standardized.square()
    )


def split_normal_cdf(
    value: Tensor, peak: Tensor, sigma_lower: Tensor, sigma_upper: Tensor
) -> Tensor:
    """Exact split-normal CDF in ``tau`` (continuous at the mode)."""
    for name, width in (("sigma_lower", sigma_lower), ("sigma_upper", sigma_upper)):
        if not (torch.isfinite(width).all() and (width > 0).all()):
            raise ValueError(f"{name} must be finite and strictly positive")
    total = sigma_lower + sigma_upper
    below = torch.special.ndtr((value - peak) / sigma_lower)
    above = torch.special.ndtr((value - peak) / sigma_upper)
    mass_below_mode = sigma_lower / total
    return torch.where(
        value < peak,
        2.0 * mass_below_mode * below,
        mass_below_mode + 2.0 * (sigma_upper / total) * (above - 0.5),
    )


def split_normal_quantiles(
    peak: Tensor,
    sigma_lower: Tensor,
    sigma_upper: Tensor,
    probabilities: Tensor | tuple[float, ...],
) -> Tensor:
    """Exact quantiles of the split normal, shape ``[..., len(probabilities)]``.

    The 0.5 quantile is **not** ``peak`` unless ``sigma_lower == sigma_upper``;
    that is the documented approximation error of reading ``tau_16/tau_84`` as
    ``tau_50 -/+ sigma``.
    """
    probs = _check_quantiles(probabilities).to(device=peak.device, dtype=peak.dtype)
    if not (peak.ndim == sigma_lower.ndim == sigma_upper.ndim == 1):
        raise ValueError("peak and widths must be 1-D [row] tensors")
    if peak.shape != sigma_lower.shape or peak.shape != sigma_upper.shape:
        raise ValueError("peak and widths must share [row] shape")
    total = sigma_lower + sigma_upper
    mass_below_mode = sigma_lower / total
    lower_tail = probs[None, :] < mass_below_mode[:, None]
    lower_z = torch.special.ndtri(
        probs[None, :] * total[:, None] / (2.0 * sigma_lower[:, None])
    )
    inside = (probs[None, :] * total[:, None] - sigma_lower[:, None]) / sigma_upper[
        :, None
    ]
    upper_z = torch.special.ndtri(0.5 * (1.0 + inside))
    return peak[:, None] + torch.where(
        lower_tail,
        sigma_lower[:, None] * lower_z,
        sigma_upper[:, None] * upper_z,
    )


def label_split_normal_widths(
    label_tau_16: Tensor, label_tau_50: Tensor, label_tau_84: Tensor
) -> tuple[Tensor, Tensor]:
    """Map StarHorse ``16/50/84`` percentiles to split-normal widths.

    ``sigma_lower = tau_50 - tau_16`` and ``sigma_upper = tau_84 - tau_50``.
    This is a *diagnostic* conversion (the split normal's own width
    parameterization) so predicted and label widths can be compared in one
    unit. Crossed or non-finite percentiles raise instead of being reordered.
    """
    for name, value in (
        ("label_tau_16", label_tau_16),
        ("label_tau_50", label_tau_50),
        ("label_tau_84", label_tau_84),
    ):
        if value.shape != label_tau_50.shape or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor of one shape")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if not ((label_tau_16 <= label_tau_50) & (label_tau_50 <= label_tau_84)).all():
        raise ValueError("label percentiles are crossed; refusing to reorder them")
    return label_tau_50 - label_tau_16, label_tau_84 - label_tau_50


def split_normal_nll(
    peak: Tensor,
    log_sigma_lower: Tensor,
    log_sigma_upper: Tensor,
    label_tau: Tensor,
    *,
    eligible: Tensor | None = None,
) -> AgeLoss:
    """NLL of the observed ``tau`` median under the predicted split normal.

    The label is the StarHorse **median** ``tau_50``; the 16/84 percentiles are
    consumed by the primary quantile head and by
    :func:`label_split_normal_widths`, never a second time here. Widths are
    ``exp(log_sigma)`` -- no softplus and no variance floor.
    """
    for name, value in (
        ("peak", peak),
        ("log_sigma_lower", log_sigma_lower),
        ("log_sigma_upper", log_sigma_upper),
    ):
        if value.shape != peak.shape or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor matching peak")
        if not torch.isfinite(value).all():
            raise ValueError(f"all model parameters must be finite ({name})")
    if label_tau.shape != peak.shape:
        raise ValueError("label_tau must match the prediction rows")
    sigma_lower = log_sigma_lower.exp()
    sigma_upper = log_sigma_upper.exp()
    if not (torch.isfinite(sigma_lower).all() and torch.isfinite(sigma_upper).all()):
        raise ValueError("split-normal widths overflowed; no clipping")
    valid = _eligible(torch.isfinite(label_tau), eligible)
    count = int(valid.sum().item())
    if count == 0:
        return AgeLoss(peak.reshape(-1)[:0].sum(), 0, 0, 0.0)
    nll = -split_normal_log_prob(
        label_tau[valid],
        peak[valid],
        sigma_lower[valid],
        sigma_upper[valid],
    )
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite split-normal likelihood")
    return AgeLoss(nll.sum(), count, count, 0.0)


def gaussian_nll(
    mean: Tensor,
    log_scale: Tensor,
    label_tau: Tensor,
    *,
    eligible: Tensor | None = None,
) -> AgeLoss:
    """Heteroscedastic Gaussian NLL on ``tau`` (closed form, control model)."""
    if mean.shape != log_scale.shape or label_tau.shape != mean.shape:
        raise ValueError("mean, log_scale and label_tau must share [row] shape")
    if not (mean.is_floating_point() and log_scale.is_floating_point()):
        raise TypeError("Gaussian parameters must be floating point")
    if not (torch.isfinite(mean).all() and torch.isfinite(log_scale).all()):
        raise ValueError("all model parameters must be finite")
    scale = log_scale.exp()
    if not (torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("invalid Gaussian scale; no clipping or variance floor")
    valid = _eligible(torch.isfinite(label_tau), eligible)
    count = int(valid.sum().item())
    if count == 0:
        return AgeLoss(mean.reshape(-1)[:0].sum(), 0, 0, 0.0)
    residual = (label_tau[valid] - mean[valid]) / scale[valid]
    nll = 0.5 * residual.square() + scale[valid].log() + 0.5 * _LOG_2PI
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite Gaussian age likelihood")
    return AgeLoss(nll.sum(), count, count, 0.0)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class _AgeTrunk(nn.Module):
    """Small residual trunk shared by the age heads."""

    def __init__(self, input_dim: int, width: int, outputs: int):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("width", width),
            ("outputs", outputs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.input = nn.Linear(input_dim, width)
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, max(1, width // 4)),
            nn.GELU(),
            nn.Linear(max(1, width // 4), width),
        )
        self.output = nn.Linear(width, outputs)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError("features must be a floating [row, feature] tensor")
        if features.shape[1] != self.input.in_features:
            raise ValueError("feature width does not match the head input dimension")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        hidden = torch.nn.functional.gelu(self.input(features))
        raw = self.output(hidden + self.block(hidden))
        if not torch.isfinite(raw).all():
            raise ValueError("age head produced nonfinite parameters")
        return raw


class AgeQuantileHead(nn.Module):
    """Multi-quantile head on ``tau = log(age/Gyr)`` (primary age model).

    ``monotone=False`` (default) keeps raw outputs and relies on the crossover
    penalty; ``monotone=True`` applies a differentiable monotone rearrangement
    (a sort along the quantile axis) for interval reporting only.
    """

    def __init__(
        self,
        input_dim: int = 256,
        width: int = 128,
        *,
        quantiles: tuple[float, ...] = PRIMARY_AGE_QUANTILES,
        monotone: bool = False,
    ):
        super().__init__()
        grid = _check_quantiles(quantiles)
        self.quantiles = tuple(float(q) for q in grid.tolist())
        self.monotone = bool(monotone)
        self.input_dim = input_dim
        self.trunk = _AgeTrunk(input_dim, width, len(self.quantiles))
        self.register_buffer("quantile_grid", grid.float(), persistent=True)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.trunk(features)
        return torch.sort(raw, dim=1).values if self.monotone else raw

    def loss(
        self,
        features: Tensor,
        label_tau: Tensor,
        **kwargs,
    ) -> AgeLoss:
        """Convenience wrapper: head outputs -> :func:`age_quantile_loss`."""
        return age_quantile_loss(self(features), label_tau, self.quantiles, **kwargs)


class AgeSplitNormalHead(nn.Module):
    """Split-normal control head: ``(peak, log sigma_lower, log sigma_upper)``."""

    def __init__(self, input_dim: int = 256, width: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.trunk = _AgeTrunk(input_dim, width, 3)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        raw = self.trunk(features)
        peak = raw[:, 0]
        log_sigma_lower = raw[:, 1]
        log_sigma_upper = raw[:, 2]
        sigma_lower = log_sigma_lower.exp()
        sigma_upper = log_sigma_upper.exp()
        if not (
            torch.isfinite(sigma_lower).all()
            and torch.isfinite(sigma_upper).all()
            and (sigma_lower > 0).all()
            and (sigma_upper > 0).all()
        ):
            raise ValueError("invalid split-normal widths; no clipping or floor")
        return {
            "peak": peak,
            "log_sigma_lower": log_sigma_lower,
            "log_sigma_upper": log_sigma_upper,
            "sigma_lower": sigma_lower,
            "sigma_upper": sigma_upper,
        }

    def loss(self, features: Tensor, label_tau: Tensor, **kwargs) -> AgeLoss:
        """Convenience wrapper: head outputs -> :func:`split_normal_nll`."""
        outputs = self(features)
        return split_normal_nll(
            outputs["peak"],
            outputs["log_sigma_lower"],
            outputs["log_sigma_upper"],
            label_tau,
            **kwargs,
        )


class AgeGaussianHead(nn.Module):
    """Heteroscedastic Gaussian control head on ``tau``."""

    def __init__(
        self, input_dim: int = 256, width: int = 128, *, initial_log_scale: float = 0.0
    ):
        super().__init__()
        if not math.isfinite(initial_log_scale):
            raise ValueError("initial log scale must be finite")
        self.input_dim = input_dim
        self.trunk = _AgeTrunk(input_dim, width, 2)
        with torch.no_grad():
            self.trunk.output.bias[1] = initial_log_scale

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        raw = self.trunk(features)
        log_scale = raw[:, 1]
        scale = log_scale.exp()
        if not (torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("invalid age scale; no clipping or variance floor")
        return {"mean": raw[:, 0], "log_scale": log_scale, "scale": scale}

    def loss(self, features: Tensor, label_tau: Tensor, **kwargs) -> AgeLoss:
        """Convenience wrapper: head outputs -> :func:`gaussian_nll`."""
        outputs = self(features)
        return gaussian_nll(outputs["mean"], outputs["log_scale"], label_tau, **kwargs)
