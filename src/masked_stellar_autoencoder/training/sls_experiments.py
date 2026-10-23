"""Joint SLS regions and frontier-center diagnostics; never normalized PDFs."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torchregress.losses import SLSLoss


class SLSHead(nn.Module):
    """Adapter for the common native training loop with an explicit schedule."""

    kind = "sls"
    has_density = False
    density_kind = "none"
    loss_denominator = "joint_region_objective"
    noise_aware = False
    flow_objective = "none"

    def __init__(
        self,
        *,
        target_dim=3,
        latent_dim=256,
        mode="full",
        noise_aware=False,
        measurement_contrastive=False,
        **_,
    ):
        super().__init__()
        if noise_aware or measurement_contrastive:
            raise ValueError(
                "SLS has no measurement-error convolution or contrastive objective"
            )
        self.target_dim = target_dim
        self.sls = SLSLoss(
            d=target_dim,
            context_dim=latent_dim,
            mode=mode,
            rank=2,
            hidden_dim=64,
            n_transforms=4,
            tau=0.9,
            warmup_steps=50,
        )
        self.register_buffer("criterion_calls", torch.zeros((), dtype=torch.int64))

    def criterion(self, latent, target, sigma=None, **_):
        # The schedule counts supervised minibatch calls, including head warmup.
        # Pass explicit step and persist it; the library's Python fallback is
        # deliberately not relied on for checkpoint reloads.
        if self.training:
            self.criterion_calls.add_(1)
        return self.sls(latent, target, step=int(self.criterion_calls.item()))

    def moments(self, latent):
        # This is a geometric frontier center, NOT E[Y|X]. No variance exists.
        mu, _ = self.sls.frontier._get_params(latent)
        center = self.sls.frontier.flow.inverse(mu, latent)
        return center, torch.full_like(center, float("nan"))

    def region_values(self, latent, target):
        frontier, log_precision_root = self.sls.evaluate_frontier(target, latent)
        quantile = self.sls.quantile_net(latent)[..., 1].clamp_min(1e-8)
        return frontier, quantile, log_precision_root


def finite_sample_threshold(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    scores = np.asarray(scores, dtype="f8")
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite nonempty calibration vector required")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return np.inf if rank > len(scores) else np.sort(scores)[rank - 1]


def region_log_volume(quantile, log_precision_root, threshold, scale):
    """Exact K=1 native volume: unit ball * radius^d / sqrt(det precision).

    Translation coupling has determinant one; the frontier matrix is a
    precision factor (not a covariance factor). Units are K*dex*dex.
    """
    d = len(scale)
    return (
        d / 2 * np.log(np.pi)
        - math.lgamma(d / 2 + 1)
        + d / 2 * np.log(np.asarray(quantile) * threshold)
        - np.asarray(log_precision_root)
        + np.log(scale).sum()
    )
