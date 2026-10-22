"""Source-tagged alpha interface -- **no training, no numbers**.

APOGEE ``alpha/M``, GALAH ``alpha/Fe`` and Gaia-RVS ``alphafe`` are *different
observables*. This module only fixes the contract the labels will land in:

* a shared latent alpha output ``mu_star`` (one number per star) with a
  per-source affine link ``alpha_o = a_o + b_o mu_star + eps_o``,
  ``eps_o ~ N(0, kappa_o^2)``;
* a per-source ``(mu_o, log sigma_o)`` output layer (one head per source tag);
* a **gauge** so the affine family is identified: either an anchor source with
  ``(a, b) = (0, 1)``, or ``sum_o n_o a_o = 0`` together with a weighted-mean
  slope of one (fixing only the offsets would leave the scale free);
* a per-source ``kappa_o`` nuisance hook, fitted from measured overlaps, used to
  inflate the quoted label error -- never invented, never floored.

The head is **disabled by default** and raises unless ``enabled=True``, and the
loss drops rows without a native label before any arithmetic: missing labels are
never imputed and a usable row without a source tag raises instead of silently
inheriting source 0.

``alpha_claim_allowed`` encodes the five gates of ``05`` Sec. 6. Until all five
pass there is no alpha number and F05 stays "NOT RUN".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

_LOG_2PI = math.log(2.0 * math.pi)

#: Declared status of every alpha product of this module.
ALPHA_HEAD_STATUS = "interface_only"

#: The five gates that must all pass before any alpha number may be quoted.
ALPHA_GATES: tuple[str, ...] = (
    "native_alpha_column_with_units_and_flags",
    "measured_overlap_offsets_and_scatter_with_n",
    "source_tagged_beats_single_alpha_on_heldout_overlap",
    "calibration_split_coverage",
    "joint_feh_alpha_level_set_coverage",
)

GAUGE_MODES = ("anchor", "sum_zero")


@dataclass(frozen=True)
class AlphaGauge:
    """Declared gauge for the per-source affine link.

    ``anchor``
        ``reference_source`` carries ``(a, b) = (0, 1)``; every other source is
        expressed relative to it.
    ``sum_zero``
        ``sum_o n_o a_o = 0`` and ``sum_o n_o b_o = sum_o n_o`` (weighted-mean
        slope one), which identifies both the shift and the scale.
    """

    mode: str = "anchor"
    reference_source: str | None = None
    counts: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.mode not in GAUGE_MODES:
            raise ValueError(f"gauge mode must be one of {GAUGE_MODES}")
        if self.mode == "anchor" and self.reference_source is None:
            raise ValueError("anchor gauge requires an explicit reference source")
        if self.mode == "sum_zero":
            if self.counts is None:
                raise ValueError("sum_zero gauge requires per-source label counts")
            if any(not math.isfinite(c) or c < 0 for c in self.counts):
                raise ValueError("gauge counts must be finite and non-negative")
            if sum(self.counts) <= 0:
                raise ValueError("sum_zero gauge requires at least one label")


def _weighted_mean(values: Tensor, counts: Tensor) -> Tensor:
    total = counts.sum()
    if not float(total) > 0:
        raise ValueError("gauge weights must sum to a positive value")
    return (values * counts).sum() / total


def apply_gauge(
    offsets: Tensor,
    slopes: Tensor,
    gauge: AlphaGauge,
    *,
    source_names: tuple[str, ...],
) -> tuple[Tensor, Tensor]:
    """Return gauge-fixed ``(a_o, b_o)`` for the declared mode.

    ``offsets``/``slopes`` are the raw ``[n_sources]`` parameters, in the order
    of ``source_names``. The anchor source is returned as exactly ``(0, 1)`` and
    the ``sum_zero`` gauge returns offsets whose weighted mean is zero and slopes
    whose weighted mean is one (both to floating-point round-off).
    """
    if offsets.ndim != 1 or slopes.shape != offsets.shape:
        raise ValueError("offsets and slopes must share [n_sources] shape")
    if offsets.numel() != len(source_names):
        raise ValueError("one offset/slope per source tag is required")
    if gauge.mode == "anchor":
        if gauge.reference_source not in source_names:
            raise ValueError(f"unknown reference source {gauge.reference_source!r}")
        index = source_names.index(gauge.reference_source)
        return offsets - offsets[index], slopes - slopes[index] + 1.0
    counts = torch.as_tensor(gauge.counts, dtype=offsets.dtype, device=offsets.device)
    if counts.numel() != len(source_names):
        raise ValueError("gauge counts must cover every source tag")
    return offsets - _weighted_mean(offsets, counts), slopes / _weighted_mean(
        slopes, counts
    )


def effective_alpha_sigma(
    intrinsic_sigma: Tensor,
    quoted_sigma: Tensor,
    kappa: Tensor,
) -> Tensor:
    """Return ``sqrt(sigma_intrinsic^2 + (kappa quoted_sigma)^2)``.

    ``kappa`` is the fitted per-source excess factor. No floor is added and
    non-positive inputs raise: an "invented" or silently floored label error is
    exactly what the plan forbids.
    """
    for name, value in (
        ("intrinsic_sigma", intrinsic_sigma),
        ("quoted_sigma", quoted_sigma),
        ("kappa", kappa),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if not (intrinsic_sigma > 0).all() or not (quoted_sigma > 0).all():
        raise ValueError("sigmas must be strictly positive; no floor is substituted")
    if not (kappa > 0).all():
        raise ValueError("kappa must be strictly positive")
    return torch.sqrt(intrinsic_sigma.square() + (kappa * quoted_sigma).square())


def source_tagged_alpha_nll(
    mean: Tensor,
    log_sigma: Tensor,
    label_alpha: Tensor,
    source_index: Tensor,
    *,
    eligible: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Gaussian NLL of labelled alpha **per source tag**; returns (sum, count).

    Rows whose label is missing (non-finite), whose ``source_index`` is negative
    or whose ``eligible`` bit is false are dropped *before* any arithmetic; a
    usable row with a negative source index raises. An all-missing batch returns
    a differentiable zero with count zero, which is the only admissible value
    when no alpha labels exist yet.
    """
    if mean.shape != log_sigma.shape or label_alpha.shape != mean.shape:
        raise ValueError("mean, log_sigma and label_alpha must share [row] shape")
    if mean.ndim != 1:
        raise ValueError("alpha tensors must be [row]")
    if source_index.shape != mean.shape or source_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("source_index must be an integer [row] tensor")
    if mean.device != label_alpha.device or source_index.device != mean.device:
        raise ValueError("alpha tensors must share a device")
    if not (mean.is_floating_point() and log_sigma.is_floating_point()):
        raise TypeError("alpha parameters must be floating point")
    if not (torch.isfinite(mean).all() and torch.isfinite(log_sigma).all()):
        raise ValueError("all model parameters must be finite")
    valid = torch.isfinite(label_alpha)
    if eligible is not None:
        if eligible.shape != valid.shape or eligible.dtype != torch.bool:
            raise ValueError("eligible must be a boolean mask of the exact label shape")
        if eligible.device != valid.device:
            raise ValueError("eligible and labels must share a device")
        valid = valid & eligible
    if bool((valid & (source_index < 0)).any()):
        raise ValueError(
            "every labelled alpha row needs a valid source tag; there is no "
            "default source"
        )
    count = int(valid.sum().item())
    if count == 0:
        return mean.reshape(-1)[:0].sum(), 0
    scale = log_sigma[valid].exp()
    if not (torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("invalid alpha scale; no clipping or variance floor")
    residual = (label_alpha[valid] - mean[valid]) / scale
    nll = 0.5 * residual.square() + scale.log() + 0.5 * _LOG_2PI
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite alpha likelihood")
    return nll.sum(), count


class SourceTaggedAlphaHead(nn.Module):
    """Per-source ``(mu, log sigma)`` output layer with an affine gauge.

    ``forward`` needs an integer ``source_index`` per row and gathers that
    source's two outputs, so a single shared trunk cannot leak one survey's
    label scale into another's. ``log_kappa`` is a per-source parameter exposed
    as :meth:`kappa` for the label-noise term.

    The module is disabled unless ``enabled=True``; there are no alpha labels in
    the current cohort, so constructing it enabled is an explicit opt-in.
    """

    def __init__(
        self,
        input_dim: int = 256,
        width: int = 128,
        *,
        sources: tuple[str, ...] = ("APOGEE", "GALAH", "DESI"),
        gauge: AlphaGauge | None = None,
        enabled: bool = False,
        initial_log_scale: float = -1.0,
    ):
        super().__init__()
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim < 1
        ):
            raise ValueError("input_dim must be a positive integer")
        if len(sources) < 1 or len(set(sources)) != len(sources):
            raise ValueError("sources must be a non-empty tuple of unique tags")
        if not math.isfinite(initial_log_scale):
            raise ValueError("initial log scale must be finite")
        self.sources = tuple(sources)
        self.enabled = bool(enabled)
        self.gauge = (
            gauge
            if gauge is not None
            else AlphaGauge(mode="anchor", reference_source=self.sources[0])
        )
        if len(self.gauge.counts or ()) not in (0, len(self.sources)):
            raise ValueError("gauge counts must cover every source tag exactly")
        if (
            self.gauge.mode == "anchor"
            and self.gauge.reference_source not in self.sources
        ):
            raise ValueError("gauge reference source is not in the source list")
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, 2 * len(self.sources)),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].bias[len(self.sources) :] = initial_log_scale
        self.offsets = nn.Parameter(torch.zeros(len(self.sources)))
        self.slopes = nn.Parameter(torch.ones(len(self.sources)))
        self.log_kappa = nn.Parameter(torch.zeros(len(self.sources)))

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "SourceTaggedAlphaHead is disabled by default: there is no audited "
                "alpha label source. Construct with enabled=True only after the five "
                f"ALPHA_GATES pass ({ALPHA_GATES[0]} first)"
            )

    def kappa(self) -> Tensor:
        """Per-source excess-noise factors ``exp(log_kappa)`` (strictly positive)."""
        kappa = self.log_kappa.exp()
        if not (torch.isfinite(kappa).all() and (kappa > 0).all()):
            raise ValueError("invalid kappa; no clipping or floor")
        return kappa

    def gauge_parameters(self) -> tuple[Tensor, Tensor]:
        """Gauge-fixed ``(a_o, b_o)`` in source order."""
        return apply_gauge(
            self.offsets, self.slopes, self.gauge, source_names=self.sources
        )

    def forward(self, features: Tensor, source_index: Tensor) -> dict[str, Tensor]:
        """Return ``mu``/``log_sigma``/``sigma`` for the rows' source tags."""
        self._require_enabled()
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must be [row, {self.input_dim}]")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        if source_index.shape != (features.shape[0],):
            raise ValueError("source_index must be [row]")
        if source_index.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("source_index must be an integer tensor")
        if source_index.device != features.device:
            raise ValueError("source_index and features must share a device")
        if bool((source_index < 0).any()) or bool(
            (source_index >= len(self.sources)).any()
        ):
            raise ValueError("source_index is out of range; no default source")
        raw = self.net(features)
        n = len(self.sources)
        mu_latent = raw[:, 0]
        log_sigma = torch.gather(raw[:, n:], 1, source_index[:, None]).squeeze(1)
        offsets, slopes = self.gauge_parameters()
        mean = offsets[source_index] + slopes[source_index] * mu_latent
        sigma = log_sigma.exp()
        if not (torch.isfinite(mean).all() and torch.isfinite(sigma).all()):
            raise ValueError("nonfinite alpha parameters")
        if not (sigma > 0).all():
            raise ValueError("invalid alpha scale; no clipping or floor")
        return {"mean": mean, "log_sigma": log_sigma, "sigma": sigma}


def alpha_claim_allowed(gates: Mapping[str, bool]) -> bool:
    """Return True only when all five :data:`ALPHA_GATES` are explicitly passed.

    Unknown keys raise (a typo must not silently count as a passed gate) and a
    missing gate counts as failed, so the default answer is "no alpha number".
    """
    unknown = set(gates) - set(ALPHA_GATES)
    if unknown:
        raise ValueError(f"unknown alpha gate names: {sorted(unknown)}")
    return all(bool(gates.get(name, False)) for name in ALPHA_GATES)
