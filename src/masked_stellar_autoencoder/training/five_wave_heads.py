"""Matched three-quantile and normalized-density heads for the five-job wave.

The ordered quantile head predicts exactly three conditional quantiles. It is
not promoted to a PDF, full CDF, PIT or six-dimensional sampling distribution.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .final_six_recipes import SparseSixHead

QUANTILES = (0.05, 0.5, 0.95)


class QuantileThreeHead(nn.Module):
    """Residual MLP with six ordered lower/median/upper output triplets."""

    kind = "quantile3"
    has_density = False

    def __init__(self, latent_dim=256, width=128):
        super().__init__()
        self.input = nn.Linear(latent_dim, width)
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 4),
            nn.GELU(),
            nn.Linear(width // 4, width),
        )
        self.output = nn.Linear(width, 18)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("levels", torch.tensor(QUANTILES))

    def forward(self, latent):
        hidden = F.gelu(self.input(latent))
        raw = self.output(hidden + self.block(hidden))
        left, median, right = raw.chunk(3, -1)
        return torch.stack(
            (median - F.softplus(left), median, median + F.softplus(right)), 1
        )

    def criterion(self, latent, target, sigma, mask, source_id):
        del sigma, source_id
        if target.shape != (len(latent), 6) or mask.shape != target.shape:
            raise ValueError("Six-target quantile layout differs")
        if not torch.isfinite(target).all():
            raise ValueError(
                "Missing labels must be masked, with safe finite arithmetic values"
            )
        difference = target[:, None] - self(latent)
        q = self.levels.to(difference)[None, :, None]
        values = torch.maximum(q * difference, (q - 1) * difference)
        return values.masked_fill(~mask[:, None], 0).sum() / (
            mask.sum() * len(QUANTILES)
        ).clamp_min(1)


def build_head(arm, latent_dim=256):
    if arm["kind"] == "quantile3":
        return QuantileThreeHead(latent_dim)
    return SparseSixHead(
        arm["kind"],
        latent_dim,
        noise_aware=True,
        mixed_weight=arm.get("mixed_weight", 0),
    )


def standardized_quantiles(head, latent, levels=QUANTILES):
    """Actual intrinsic conditional quantiles, never a moment-Gaussian proxy."""
    if head.kind == "quantile3":
        if tuple(levels) != QUANTILES:
            raise ValueError("Three-quantile baseline has no arbitrary inverse CDF")
        return head(latent)
    probability = latent.new_tensor(levels)
    z = math.sqrt(2.0) * torch.erfinv(2 * probability - 1)
    if head.kind == "gaussian":
        mean, lv = head.head(latent).chunk(2, -1)
        return mean[:, None] + (lv / 2).exp()[:, None] * z[None, :, None]
    if head.kind != "factorized_nsf":
        raise ValueError("This matched wave supports Gaussian or scalar NSF densities")
    result = []
    for scalar in head.heads:
        base = z[:, None, None].expand(-1, len(latent), 1)
        values = scalar.distribution(latent).transform.inv(base)
        result.append(values.permute(1, 0, 2))
    return torch.cat(result, -1)


def native_quantiles(head, latent, center, scale, levels=QUANTILES):
    values = center + scale * standardized_quantiles(head, latent, levels)
    # Positive monotone exponentiation preserves age/distance quantiles exactly.
    return torch.cat((values[..., :4], values[..., 4:].exp()), -1)


def modality_dropout(values, visible, xp, probability, generator):
    """Whole-XP supervised-input dropout only; no invented natural availability."""
    if values.shape != visible.shape or not 0 <= probability <= 1 or not xp:
        raise ValueError("Valid input/mask/dropout/XPlayout required")
    before = visible[:, xp].any(-1)
    dropped = (
        torch.rand(len(values), device=values.device, generator=generator) < probability
    ) & before
    result, after_visible = values.clone(), visible.clone()
    rows = dropped.nonzero().squeeze(1)
    if len(rows):
        columns = torch.tensor(xp, device=values.device)
        result[rows[:, None], columns] = -9999
        after_visible[rows[:, None], columns] = False
    return (
        result,
        after_visible,
        dict(
            input_xp_before=before,
            input_xp_after=after_visible[:, xp].any(-1),
            xp_dropped=dropped,
        ),
    )


def native_fit_point_scale(value, mask, fit):
    """FIT-only native target standard deviations make the median MAEs comparable."""
    scales = []
    for column in range(6):
        values = np.asarray(value)[
            np.asarray(fit)[np.asarray(mask)[fit, column]], column
        ]
        if len(values) < 16 or not np.isfinite(values).all() or values.std() <= 0:
            raise ValueError("Insufficient nondegenerate FIT native point scale")
        scales.append(values.std())
    return np.array(scales)


def masked_native_median_score(prediction, target, observed, scale):
    """Macro per-label native median MAE, standardized by fixed FIT scales."""
    per_label = []
    for column in range(6):
        use = np.asarray(observed)[:, column]
        if use.sum() < 16:
            raise ValueError("Every TUNE target needs at least16 observed references")
        residual = np.abs(
            np.asarray(prediction)[use, column] - np.asarray(target)[use, column]
        )
        if not np.isfinite(residual).all():
            raise FloatingPointError("Nonfinite TUNE median error")
        per_label.append(float(residual.mean() / scale[column]))
    return float(np.mean(per_label)), per_label
