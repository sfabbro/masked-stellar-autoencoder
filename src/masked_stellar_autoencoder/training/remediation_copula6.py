"""Six-label coupled spline-flow control with analytic missing-label marginals.

This is a Gaussian-copula spline flow, not the unrestricted autoregressive NSF
used in the earlier pilot. Measurement-noise integration remains finite-QMC;
its initial and selected checkpoints must pass separate numerical gates.
"""

from __future__ import annotations

import torch
from torch import nn

from .remediation_joint6 import normal_qmc_nodes


class CopulaSplineSixHead(nn.Module):
    """Six-output density, with genuinely coupled whole-vector samples."""

    def __init__(self, latent_dim: int, width: int, particles: int, seed: int):
        super().__init__()
        from torchregress.models import GaussianCopulaSplineHead

        if isinstance(particles, bool) or particles < 8 or particles % 2:
            raise ValueError("An even particle count of at least eight is required")
        self.flow = GaussianCopulaSplineHead(latent_dim, 6, width=width)
        self.particles, self.seed = particles, seed
        self.last_loss_components = {}

    def log_prob(self, context, target, sigma, mask, *, particles=None, seed=None):
        count = self.particles if particles is None else particles
        nodes = normal_qmc_nodes(
            count, seed=self.seed if seed is None else seed, like=context
        )
        return self.flow.observed_log_prob(context, target, sigma, mask, nodes[:, :6])

    def criterion(self, context, target, sigma, mask, source_id=None):
        if source_id is not None and len(source_id) != len(context):
            raise ValueError("Source IDs must align with contexts")
        observed = mask.sum()
        if not observed:
            return context.sum() * 0
        lp, _ = self.log_prob(context, target, sigma, mask)
        loss = -lp.sum() / observed
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite copula spline observed-label NLL")
        self.last_loss_components = {"observed_nll": loss.detach()}
        return loss

    def sample(self, context, count, generator=None):
        return self.flow.sample(context, count, generator=generator)

    @torch.no_grad()
    def diagnostics(self, context, target, sigma, mask, *, particles=None):
        count = self.particles if particles is None else particles
        first, relative_ess = self.log_prob(
            context, target, sigma, mask, particles=count
        )
        replica, _ = self.log_prob(
            context, target, sigma, mask, particles=count, seed=self.seed + 1
        )
        refined, _ = self.log_prob(
            context, target, sigma, mask, particles=2 * count, seed=self.seed + 2
        )
        return {
            "method": "analytic_missing_marginal_gaussian_copula_spline_plus_noise_qmc",
            "particles": count,
            "exact": False,
            "relative_ess": relative_ess,
            "ess": relative_ess * count,
            "log_prob": first,
            "independent_replica_log_prob_gap": (first - replica).abs(),
            "refinement_log_prob_gap": (first - refined).abs(),
            "scope": "Missing-label marginal analytic; finite noise quadrature remains approximate",
        }
