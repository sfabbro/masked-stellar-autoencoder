"""Observed-label conditional flows with identity initialization and fixed moments.

A scalar affine MAF remains a conditional Gaussian; it is a control for the
nonlinear NSF density, not a more flexible scalar distribution. These heads
model reported standardized FeH, without claiming latent-truth deconvolution.
"""

import math

import numpy as np
import torch
from torch import nn


class ConditionalFlowHead(nn.Module):
    def __init__(
        self,
        *,
        kind="nsf",
        bins=8,
        context=256,
        transforms=3,
        moment_order=64,
        bound=5.0,
    ):
        super().__init__()
        from torchregress.losses.nflows import BoundedNSF
        from zuko.flows import MAF

        if kind not in {"nsf", "maf"}:
            raise ValueError("Conditional flow must be NSF or affine MAF")
        if moment_order < 8 or moment_order % 2:
            raise ValueError("Moment order must be even and at least8")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("Flow bound must be finite and positive")
        self.kind, self.bins, self.moment_order = kind, bins, moment_order
        self.bound = float(bound)
        options = dict(
            features=1, context=context, transforms=transforms, hidden_features=(64, 64)
        )
        self.flow = (
            BoundedNSF(bins=bins, bound=bound, **options)
            if kind == "nsf"
            else MAF(**options)
        )
        # Both zuko RQS and affine transforms are identity for zero raw params:
        # uniform RQS widths/heights, exp(0)=1 derivatives; shift=logscale=0.
        for transform in self.flow.transform.transforms:
            final = [
                module
                for module in transform.hyper.modules()
                if hasattr(module, "weight") and hasattr(module, "bias")
            ][-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def distribution(self, latent):
        if latent.ndim != 2 or not torch.isfinite(latent).all():
            raise ValueError("Finite [row,context] latents are required")
        return self.flow(latent)

    def criterion(self, latent, target):
        if target.shape != (len(latent),) or not torch.isfinite(target).all():
            raise ValueError("Finite scalar standardized labels are required")
        return -self.distribution(latent).log_prob(target[:, None]).mean()

    def moments(self, latent, *, order=None):
        order = self.moment_order if order is None else order
        if order < 8 or order % 2:
            raise ValueError("Moment order must be even and at least8")
        distribution = self.distribution(latent)
        if self.kind == "maf":
            zero = latent.new_zeros(len(latent), 1)
            mean = distribution.transform.inv(zero).squeeze(-1)
            scale = distribution.transform.inv(zero + 1).squeeze(-1) - mean
            return mean, scale.square()
        # Composite eight-point Legendre quadrature in base-normal z. Every
        # spline is identity outside +/-bound, including their composition;
        # hence the omitted normal-tail first/second moments are analytic.
        # order means number of equal panels, not Hermite nodes. Dense panels
        # resolve spline bends much better than a single global Hermite rule.
        node, weight = np.polynomial.legendre.leggauss(8)
        edges = np.linspace(-self.bound, self.bound, order + 1)
        middle = (edges[:-1] + edges[1:]) / 2
        half = self.bound / order
        z = latent.new_tensor((middle[:, None] + half * node).reshape(-1))
        weights = latent.new_tensor(np.tile(weight * half, order))
        weights = weights * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
        half_count = len(z) // 2
        positive = z[half_count:, None, None].expand(-1, len(latent), 1)
        yp = distribution.transform.inv(positive).squeeze(-1)
        yn = distribution.transform.inv(-positive).squeeze(-1)
        w = weights[half_count:, None]
        mean = ((yp + yn) * w).sum(0)
        tail_second = 2 * (
            self.bound * math.exp(-0.5 * self.bound**2) / math.sqrt(2 * math.pi)
            + 0.5 * math.erfc(self.bound / math.sqrt(2))
        )
        second = ((yp.square() + yn.square()) * w).sum(0) + tail_second
        variance = second - mean.square()
        if not torch.all(torch.isfinite(variance) & (variance > 0)):
            raise FloatingPointError("Conditional flow moments invalid")
        return mean, variance

    def forward(self, latent):
        mean, variance = self.moments(latent)
        return torch.stack((mean, variance.log()), -1)[:, None, :]

    def quantiles(self, latent, probabilities):
        p = torch.as_tensor(probabilities, dtype=latent.dtype, device=latent.device)
        if p.ndim != 1 or not torch.all(torch.isfinite(p) & (p > 0) & (p < 1)):
            raise ValueError("Quantile probabilities must lie strictly in(0,1)")
        z = (math.sqrt(2) * torch.erfinv(2 * p - 1))[:, None, None].expand(
            -1, len(latent), 1
        )
        values = self.distribution(latent).transform.inv(z).squeeze(-1)
        return values.movedim(0, 1)

    def cdf(self, latent, target):
        if target.shape != (len(latent),):
            raise ValueError("Scalar targets must align with latent rows")
        z = self.distribution(latent).transform(target[:, None]).squeeze(-1)
        return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    @torch.no_grad()
    def diagnostics(self, latent, target):
        distribution = self.distribution(latent)
        refined_mean, refined_variance = self.moments(
            latent, order=2 * self.moment_order
        )
        return dict(
            observed_label_log_density=distribution.log_prob(target[:, None]),
            cdf=self.cdf(latent, target),
            quantiles=self.quantiles(latent, [0.025, 0.16, 0.5, 0.84, 0.975]),
            moment_refinement_mean=refined_mean,
            moment_refinement_variance=refined_variance,
        )
