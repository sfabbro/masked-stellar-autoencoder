"""Joint flexible densities under one shared multi-survey observation model.

Reference Gaussian conditioning is an integration proposal, never a prediction:
catalogue samples come from the feature-conditioned head alone. All heads use
the same fixed affine survey map inherited from the reference fit. Finite-QMC
importance integrals require initial, training and selected-state checks.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def ensemble_crps(samples, observed):
    """Unbiased univariate ensemble CRPS, preserving every native target unit."""
    k = len(samples)
    if k < 2 or samples.shape[1:] != observed.shape:
        raise ValueError("At least two aligned predictive samples required")
    ordered = samples.sort(dim=0).values
    rank = torch.arange(1, k + 1, device=samples.device, dtype=samples.dtype)
    rank = rank.reshape(k, *((1,) * (samples.ndim - 1)))
    pair = ((2 * rank - k - 1) * ordered).sum(0) / (k * (k - 1))
    return (samples - observed).abs().mean(0) - pair


def whiten_observations(reference_lower, packed):
    """Whiten measured residuals/noise and physical latent coordinates.

    The reference latent is z=mu_ref+L_ref*u with u~N(0,I). Returns the exact
    whitened affine observation likelihood, including its native-unit Jacobian.
    Unobserved slots are independent unit-noise padding with zero residual/design.
    """
    mask = packed.mask
    active_pair = mask[:, :, None] & mask[:, None, :]
    k = mask.shape[1]
    noise = torch.where(active_pair, packed.measurement_covariance, 0.0)
    noise = noise + torch.diag_embed((~mask).to(noise.dtype))
    residual = torch.where(mask, packed.values - packed.mapped_mean, 0.0)
    design = torch.where(mask[..., None], packed.jacobian, 0.0) @ reference_lower
    chol = torch.linalg.cholesky(noise)
    design = torch.linalg.solve_triangular(chol, design, upper=False)
    residual = torch.linalg.solve_triangular(
        chol, residual[..., None], upper=False
    ).squeeze(-1)
    if (
        design.shape[1] != k
        or not torch.isfinite(design).all()
        or not torch.isfinite(residual).all()
    ):
        raise FloatingPointError("Nonfinite whitened source observations")
    return dict(
        design=design,
        residual=residual,
        noise_logdet=chol.diagonal(dim1=-2, dim2=-1).log().sum(-1),
        count=mask.sum(-1),
    )


def gaussian_observation_log_prob(mean, lower, observation):
    """Exact noisy multi-survey density for a Gaussian in reference coordinates."""
    h, value = observation["design"], observation["residual"]
    projected = h @ lower
    covariance = projected @ projected.transpose(-1, -2)
    covariance = covariance + torch.eye(h.shape[1], device=h.device, dtype=h.dtype)
    chol = torch.linalg.cholesky(covariance)
    residual = value - (h @ mean[..., None]).squeeze(-1)
    standardized = torch.linalg.solve_triangular(chol, residual[..., None], upper=False)
    return (
        -0.5
        * (
            observation["count"].to(mean) * math.log(2 * math.pi)
            + standardized.square().sum((-2, -1))
        )
        - chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        - observation["noise_logdet"]
    )


def reference_posterior(observation):
    """Gaussian proposal p0(u | observed source measurements), in float64."""
    h = observation["design"]
    eye = torch.eye(h.shape[-1], device=h.device, dtype=h.dtype).expand(len(h), -1, -1)
    precision = eye + h.transpose(-1, -2) @ h
    precision_lower = torch.linalg.cholesky(precision)
    covariance = torch.cholesky_inverse(precision_lower)
    mean = (
        covariance @ h.transpose(-1, -2) @ observation["residual"][..., None]
    ).squeeze(-1)
    lower = torch.linalg.cholesky(covariance)
    log_prob = gaussian_observation_log_prob(torch.zeros_like(mean), eye, observation)
    return mean, lower, log_prob


def normal_nodes(count, batch, dims, *, seed, like):
    """Independent scrambled Sobol coordinates per star; [K,B,D]."""
    if count < 8 or count & (count - 1):
        raise ValueError("QMC count must be a power of two of at least eight")
    uniform = torch.quasirandom.SobolEngine(
        batch * dims, scramble=True, seed=seed
    ).draw(count, dtype=torch.float64)
    eps = torch.finfo(uniform.dtype).eps
    return (
        torch.special.ndtri(uniform.clamp(eps, 1 - eps))
        .reshape(count, batch, dims)
        .to(like)
    )


def importance_log_prob(
    head, context, posterior_mean, posterior_lower, reference_lp, count, nodes
):
    """Shared-latent observation integral using a label-conditioned proposal.

    Identity: p_q(y)=p_0(y) E_{p_0(u|y)}[q(u|x)/phi(u)]. The prediction q depends
    only on context x. Every source record for a star enters the same integral.
    """
    samples = posterior_mean[None] + torch.einsum(
        "bij,kbj->kbi", posterior_lower, nodes
    )
    log_reference = -0.5 * (
        samples.square().sum(-1) + samples.shape[-1] * math.log(2 * math.pi)
    )
    weights = head.log_prob(context, samples) - log_reference
    if not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite joint importance density ratio")
    log_sum = torch.logsumexp(weights, dim=0)
    lp = reference_lp + log_sum - math.log(len(nodes))
    ess = (2 * log_sum - torch.logsumexp(2 * weights, dim=0)).exp() / len(nodes)
    active = count > 0
    return torch.where(active, lp, lp * 0), torch.where(
        active, ess, torch.ones_like(ess)
    ).detach()


class GaussianResidualHead(nn.Module):
    def __init__(self, context_dim, dims=3, width=128):
        super().__init__()
        from .multisurvey_model import CanonicalGaussianHead

        self.gaussian = CanonicalGaussianHead(
            context_dim, torch.zeros(dims), torch.ones(dims), width=width
        ).double()
        out = self.gaussian.net[-1]
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        with torch.no_grad():
            diagonal = self.gaussian.row == self.gaussian.column
            out.bias[dims:][diagonal] = math.log(
                math.expm1(1 - self.gaussian.min_normalized_scale)
            )

    def forward(self, context):
        return self.gaussian(context)

    def log_prob(self, context, value):
        mean, lower = self(context)
        return torch.distributions.MultivariateNormal(mean, scale_tril=lower).log_prob(
            value
        )

    def sample(self, context, count):
        mean, lower = self(context)
        eps = torch.randn(count, *mean.shape, device=mean.device, dtype=mean.dtype)
        return mean[None] + torch.einsum("bij,kbj->kbi", lower, eps)


class CopulaResidualHead(nn.Module):
    def __init__(self, context_dim, dims=3, width=128, *, bound=5.0, bins=16):
        super().__init__()
        from torchregress.models import GaussianCopulaSplineHead

        self.flow = GaussianCopulaSplineHead(
            context_dim, dims, width=width, bound=bound, bins=bins
        ).double()
        with torch.no_grad():
            diagonal = self.flow.row == self.flow.col
            n = len(self.flow.row)
            self.flow._out.bias[dims : dims + n][diagonal] = math.log(
                math.expm1(1 - self.flow.min_scale)
            )

    def log_prob(self, context, value):
        mask = torch.ones(
            (len(context), self.flow.target_dim),
            device=context.device,
            dtype=torch.bool,
        )
        return self.flow.intrinsic_log_prob(context, value, mask)

    def sample(self, context, count):
        return self.flow.sample(context, count)


class NSFResidualHead(nn.Module):
    """Unrestricted autoregressive spline flow, not a Gaussian copula."""

    def __init__(self, context_dim, dims=3, width=128, *, bound=5.0, bins=16):
        super().__init__()
        from torchregress.losses.nflows import create_flow_model

        self.flow = create_flow_model(
            n_features=dims,
            context_dim=context_dim,
            flow_type="nsf",
            n_transforms=3,
            hidden_features=(width, width),
            bins=bins,
            bound=bound,
        )
        # Uniform RQS parameters give identity transforms; permutations preserve
        # the initial standard-normal density. Check this numerically before fit.
        for module in self.flow.modules():
            if hasattr(module, "hyper"):
                nn.init.zeros_(module.hyper[-1].weight)
                nn.init.zeros_(module.hyper[-1].bias)

    def log_prob(self, context, value):
        from torchregress.utils.pytorch_compat import _collect_transform_cycles

        with _collect_transform_cycles():
            return self.flow(context).log_prob(value)

    def sample(self, context, count):
        from torchregress.utils.pytorch_compat import _collect_transform_cycles

        with _collect_transform_cycles():
            return self.flow(context).sample((count,))


class MixtureResidualHead(nn.Module):
    """Conditional multimodal mixture with full-covariance components."""

    def __init__(self, context_dim, dims=3, width=128, components=4):
        super().__init__()
        self.dims, self.components = dims, components
        self.register_buffer("row", torch.tril_indices(dims, dims)[0])
        self.register_buffer("column", torch.tril_indices(dims, dims)[1])
        n = len(self.row)
        self.net = nn.Sequential(
            nn.Linear(context_dim, width),
            nn.GELU(),
            nn.Linear(width, components * (1 + dims + n)),
        ).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            raw = self.net[-1].bias.reshape(components, -1)
            raw[:, 1 : 1 + dims] = 0.03 * torch.randn(components, dims)
            raw[:, 1 + dims :][:, self.row == self.column] = math.log(
                math.expm1(1 - 1e-4)
            )

    def forward(self, context):
        raw = self.net(context).reshape(len(context), self.components, -1)
        mean, packed = raw[..., 1 : 1 + self.dims], raw[..., 1 + self.dims :]
        packed = torch.where(self.row == self.column, F.softplus(packed) + 1e-4, packed)
        lower = raw.new_zeros((*raw.shape[:2], self.dims, self.dims))
        lower[..., self.row, self.column] = packed
        return raw[..., 0].log_softmax(-1), mean, lower

    def log_prob(self, context, value):
        weights, mean, lower = self(context)
        lp = torch.distributions.MultivariateNormal(mean, scale_tril=lower).log_prob(
            value[..., None, :]
        )
        return torch.logsumexp(lp + weights, -1)

    def observation_log_prob(self, context, observation):
        weights, means, lowers = self(context)
        scores = torch.stack(
            [
                gaussian_observation_log_prob(means[:, c], lowers[:, c], observation)
                for c in range(self.components)
            ],
            -1,
        )
        return torch.logsumexp(weights + scores, -1)

    def sample(self, context, count):
        weights, means, lowers = self(context)
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=weights),
            torch.distributions.MultivariateNormal(means, scale_tril=lowers),
        )
        return distribution.sample((count,))
