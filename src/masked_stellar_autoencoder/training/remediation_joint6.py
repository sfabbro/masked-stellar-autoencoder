"""Dependent six-target densities and explicit sparse-observation estimators.

Gaussian/mixture observed-subvector likelihoods are analytic. The genuinely
six-dimensional NSF has a fixed-QMC importance estimate for arbitrary missing
coordinates and diagonal quoted measurement noise. Its log estimate is biased at
finite particle count; ESS/replica/refinement diagnostics are admission gates,
not a certificate of an exact marginal likelihood. StarHorse age/distance
medians remain catalogue-teacher targets, not physical-truth measurements.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")
DIMENSIONS = 6
_LOG_2PI = math.log(2 * math.pi)


def normal_qmc_nodes(count: int, *, seed: int, like: Tensor) -> Tensor:
    """Twelve independent Sobol coordinates: six noise plus six missing values."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 8 or count % 2:
        raise ValueError("An even QMC particle count of at least eight is required")
    uniform = torch.quasirandom.SobolEngine(12, scramble=True, seed=seed).draw(
        count // 2, dtype=torch.float64
    )
    if not ((uniform > 0) & (uniform < 1)).all():
        raise FloatingPointError("QMC endpoint cannot map to finite normal")
    nodes = torch.special.ndtri(uniform)
    return torch.cat((nodes, -nodes), 0).to(device=like.device, dtype=like.dtype)


def _check_arrays(target: Tensor, sigma: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if target.ndim != 2 or target.shape[1] != DIMENSIONS:
        raise ValueError("Targets must be [row,6]")
    if (
        sigma.shape != target.shape
        or mask.shape != target.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("Six-target errors and boolean observed masks must align")
    if not torch.isfinite(target[mask]).all():
        raise ValueError("Observed targets must be finite")
    if not torch.isfinite(sigma[mask]).all() or (sigma[mask] < 0).any():
        raise ValueError("Observed measurement errors must be finite and nonnegative")
    # Safe placeholders never enter a density as an observed value: missing
    # coordinates are analytically marginalized or explicitly integrated below.
    return target.masked_fill(~mask, 0), sigma.masked_fill(~mask, 0)


def masked_gaussian_log_prob(
    mean: Tensor, scale_tril: Tensor, target: Tensor, sigma: Tensor, mask: Tensor
) -> Tensor:
    """Exact N(mean, LL^T + diag(sigma²)) law on observed subcoordinates."""
    target, sigma = _check_arrays(target, sigma, mask)
    if mean.shape != target.shape or scale_tril.shape != (*target.shape, DIMENSIONS):
        raise ValueError("Gaussian parameters must have [row,6]/[row,6,6] shapes")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale_tril).all():
        raise FloatingPointError("Nonfinite Gaussian parameters")
    if (torch.diagonal(scale_tril, dim1=-2, dim2=-1) <= 0).any():
        raise ValueError("Gaussian scale requires positive diagonal")
    if torch.count_nonzero(torch.triu(scale_tril, diagonal=1)):
        raise ValueError("Gaussian scale must be lower triangular")
    covariance = scale_tril @ scale_tril.transpose(-1, -2)
    code = (mask.long() * (2 ** torch.arange(6, device=mask.device))).sum(-1)
    result = mean.sum(-1) * 0
    for pattern in torch.unique(code).tolist():
        if not pattern:
            continue
        rows = (code == pattern).nonzero().flatten()
        columns = [j for j in range(6) if pattern & (1 << j)]
        selected = covariance[rows][:, columns][:, :, columns]
        selected = selected + torch.diag_embed(sigma[rows][:, columns].square())
        law = torch.distributions.MultivariateNormal(
            mean[rows][:, columns], covariance_matrix=selected
        )
        result = result.index_copy(0, rows, law.log_prob(target[rows][:, columns]))
    return result


def importance_observed_log_prob(
    density: Callable[[Tensor, Tensor], Tensor],
    proposal_mean: Tensor,
    proposal_scale_tril: Tensor,
    target: Tensor,
    sigma: Tensor,
    mask: Tensor,
    nodes: Tensor,
) -> tuple[Tensor, dict[str, Tensor | int | str]]:
    """Integrate missing labels and observed errors with an explicit q correction.

    ``density(row_indices, values[K,B,6])`` returns the true joint log density.
    For each noise draw, q(missing | perturbed observed) is the conditional of
    the proposal Gaussian. The proposal parameters are detached, so training
    cannot improve this objective simply by moving its integration proposal.
    """
    target, sigma = _check_arrays(target, sigma, mask)
    if proposal_mean.shape != target.shape or proposal_scale_tril.shape != (
        *target.shape,
        6,
    ):
        raise ValueError("Proposal Gaussian shapes differ from six-target data")
    if nodes.ndim != 2 or nodes.shape[1] != 12 or len(nodes) < 2:
        raise ValueError("Integration nodes must be finite [particle,12]")
    if not torch.isfinite(nodes).all():
        raise FloatingPointError("Nonfinite integration nodes")
    mean, lower = proposal_mean.detach(), proposal_scale_tril.detach()
    if not torch.isfinite(mean).all() or not torch.isfinite(lower).all():
        raise FloatingPointError("Nonfinite proposal parameters")
    if (torch.diagonal(lower, dim1=-2, dim2=-1) <= 0).any():
        raise ValueError("Proposal scale requires positive diagonal")
    if torch.count_nonzero(torch.triu(lower, diagonal=1)):
        raise ValueError("Proposal scale must be lower triangular")
    covariance = lower @ lower.transpose(-1, -2)
    code = (mask.long() * (2 ** torch.arange(6, device=mask.device))).sum(-1)
    result = target.sum(-1) * 0
    ess = target.new_full((len(target),), len(nodes))
    half_gap = target.new_zeros(len(target))
    for pattern in torch.unique(code).tolist():
        if not pattern:
            continue
        rows = (code == pattern).nonzero().flatten()
        observed = [j for j in range(6) if pattern & (1 << j)]
        missing = [j for j in range(6) if not pattern & (1 << j)]
        # Full, noiseless rows are evaluated exactly, without MC work or q.
        if not missing and not sigma[rows].any():
            lp = density(rows, target[rows][None]).squeeze(0)
            result = result.index_copy(0, rows, lp)
            continue
        values = target[rows][None].expand(len(nodes), -1, -1).clone()
        noisy = target[rows][:, observed][None] + (
            nodes[:, None, observed] * sigma[rows][:, observed][None]
        )
        values[:, :, observed] = noisy
        log_q = target.new_zeros((len(nodes), len(rows)))
        if missing:
            cov = covariance[rows]
            oo = cov[:, observed][:, :, observed]
            om = cov[:, observed][:, :, missing]
            mm = cov[:, missing][:, :, missing]
            gain = torch.cholesky_solve(om, torch.linalg.cholesky(oo)).transpose(-1, -2)
            conditional_covariance = mm - gain @ om
            conditional_lower = torch.linalg.cholesky(conditional_covariance)
            conditional_mean = mean[rows][:, missing][None] + torch.einsum(
                "bmo,kbo->kbm", gain, noisy - mean[rows][:, observed][None]
            )
            eps = nodes[:, None, [6 + j for j in missing]].expand(-1, len(rows), -1)
            values[:, :, missing] = conditional_mean + torch.einsum(
                "bij,kbj->kbi", conditional_lower, eps
            )
            log_q = (
                -0.5 * (len(missing) * _LOG_2PI + eps.square().sum(-1))
                - (
                    torch.diagonal(conditional_lower, dim1=-2, dim2=-1)
                    .log()
                    .sum(-1)[None]
                )
            )
        log_weight = density(rows, values) - log_q
        if log_weight.shape != (len(nodes), len(rows)):
            raise ValueError("Joint density must return [particle,row]")
        if not torch.isfinite(log_weight).all():
            raise FloatingPointError("Nonfinite importance log weights")
        log_sum = torch.logsumexp(log_weight, 0)
        lp = log_sum - math.log(len(nodes))
        row_ess = (2 * log_sum - torch.logsumexp(2 * log_weight, 0)).exp()
        half = len(nodes) // 2
        gap = (
            torch.logsumexp(log_weight[:half], 0)
            - math.log(half)
            - torch.logsumexp(log_weight[half:], 0)
            + math.log(len(nodes) - half)
        ).abs()
        result = result.index_copy(0, rows, lp)
        ess = ess.index_copy(0, rows, row_ess)
        half_gap = half_gap.index_copy(0, rows, gap)
    if not torch.isfinite(result).all() or not torch.isfinite(ess).all():
        raise FloatingPointError("Nonfinite integrated likelihood or ESS")
    return result, dict(
        method="fixed_qmc_importance_with_conditional_gaussian_proposal",
        particles=len(nodes),
        ess=ess.detach(),
        relative_ess=(ess / len(nodes)).detach(),
        antithetic_half_log_prob_gap=half_gap.detach(),
    )


class _GaussianParameters(nn.Module):
    def __init__(self, latent_dim: int, width: int, components: int = 1):
        super().__init__()
        self.components = components
        self.net = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.Linear(width, components * (1 + 6 + 21)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if components > 1:
            # Break component symmetry while keeping the initial covariance I.
            with torch.no_grad():
                bias = self.net[-1].bias.view(components, 28)
                bias[:, 1:7] = torch.linspace(-0.2, 0.2, components)[:, None]
        index = torch.tril_indices(6, 6)
        self.register_buffer("row", index[0], persistent=False)
        self.register_buffer("column", index[1], persistent=False)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.net(context).view(len(context), self.components, 28)
        weights = raw[:, :, 0].log_softmax(-1)
        mean = raw[:, :, 1:7]
        packed = raw[:, :, 7:].clone()
        diagonal = self.row == self.column
        packed[:, :, diagonal] = packed[:, :, diagonal].exp()
        lower = packed.new_zeros((len(context), self.components, 6, 6))
        lower[:, :, self.row, self.column] = packed
        if not torch.isfinite(lower).all() or not torch.isfinite(mean).all():
            raise FloatingPointError("Nonfinite Gaussian parameters; no clipping")
        return weights, mean, lower


class JointSixHead(nn.Module):
    """Full covariance Gaussian/mixture or a genuinely coupled six-feature NSF.

    ``criterion`` divides the joint observed-row log densities by the number of
    observed coordinates. The NSF adds a declared auxiliary Gaussian-proposal
    objective. It does not imply six independent scalar likelihoods.
    """

    def __init__(
        self,
        kind="gaussian",
        *,
        latent_dim=256,
        width=128,
        components=4,
        particles=32,
        seed=20261026,
        proposal_weight=0.1,
    ):
        super().__init__()
        if kind not in {"gaussian", "mdn", "nsf"}:
            raise ValueError("Joint six-target kind must be gaussian, mdn or nsf")
        if (
            not isinstance(components, int)
            or isinstance(components, bool)
            or components < 1
        ):
            raise ValueError("Mixture components must be a positive integer")
        if proposal_weight < 0 or not math.isfinite(proposal_weight):
            raise ValueError("Proposal objective weight must be finite and nonnegative")
        self.kind, self.latent_dim = kind, latent_dim
        self.particles, self.seed = particles, seed
        self.proposal_weight = proposal_weight
        if (
            not isinstance(particles, int)
            or isinstance(particles, bool)
            or particles < 8
            or particles % 2
        ):
            raise ValueError("Even particle count of at least eight required")
        self.gaussian = _GaussianParameters(
            latent_dim, width, components if kind == "mdn" else 1
        )
        self.flow = None
        if kind == "nsf":
            from .multilabel_experiments import MultiLabelHead

            self.flow = MultiLabelHead(
                "flow",
                target_dim=6,
                latent_dim=latent_dim,
                width=width,
                flow_type="nsf",
            )
        self.last_diagnostics = {}
        self.last_loss_components = {}

    def _check_context(self, context):
        if context.ndim != 2 or context.shape[1] != self.latent_dim:
            raise ValueError("Context must be [row,latent_dim]")
        if not torch.isfinite(context).all():
            raise ValueError("Context must be finite")

    def proposal_parameters(self, context):
        self._check_context(context)
        _, mean, lower = self.gaussian(context)
        return mean[:, 0], lower[:, 0]

    def log_prob(
        self,
        context,
        target,
        sigma,
        mask,
        *,
        observed=True,
        particles=None,
        seed=None,
        return_diagnostics=False,
    ):
        self._check_context(context)
        target, sigma = _check_arrays(target, sigma, mask)
        if len(context) != len(target):
            raise ValueError("Context and target rows differ")
        error = sigma if observed else torch.zeros_like(sigma)
        weights, means, lowers = self.gaussian(context)
        if self.kind in {"gaussian", "mdn"}:
            components = [
                masked_gaussian_log_prob(means[:, k], lowers[:, k], target, error, mask)
                for k in range(means.shape[1])
            ]
            lp = torch.logsumexp(weights + torch.stack(components, -1), -1)
            diag = dict(method="analytic_observed_subvector", exact=True)
        else:
            count = self.particles if particles is None else particles
            nodes = normal_qmc_nodes(
                count, seed=self.seed if seed is None else seed, like=context
            )

            def density(rows, values):
                return self.flow.distribution(context[rows]).log_prob(values)

            lp, diag = importance_observed_log_prob(
                density, means[:, 0], lowers[:, 0], target, error, mask, nodes
            )
            diag["exact"] = False
        if not torch.isfinite(lp).all():
            raise FloatingPointError("Nonfinite joint log density")
        self.last_diagnostics = diag
        return (lp, diag) if return_diagnostics else lp

    def criterion(self, context, target, sigma, mask, source_id=None):
        if source_id is not None and len(source_id) != len(context):
            raise ValueError("Source IDs must align with contexts")
        count = mask.sum()
        if not count:
            return context.sum() * 0
        lp = self.log_prob(context, target, sigma, mask)
        objective = -lp.sum() / count
        proposal = context.sum() * 0
        if self.kind == "nsf":
            mean, lower = self.proposal_parameters(context)
            proposal = (
                -masked_gaussian_log_prob(mean, lower, target, sigma, mask).sum()
                / count
            )
            objective = objective + self.proposal_weight * proposal
        self.last_loss_components = dict(
            observed_nll=(-lp.sum() / count).detach(),
            proposal_nll=proposal.detach(),
            proposal_weight=self.proposal_weight if self.kind == "nsf" else 0.0,
        )
        return objective

    def sample(self, context, count, generator=None):
        """Whole [sample,row,6] vectors; no independent-label reassembly."""
        self._check_context(context)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Positive sample count required")
        eps = torch.randn(
            (count, len(context), 6),
            dtype=context.dtype,
            device=context.device,
            generator=generator,
        )
        if self.kind == "nsf":
            return self.flow.distribution(context).transform.inv(eps)
        weights, means, lower = self.gaussian(context)
        component = torch.multinomial(
            weights.exp(), count, replacement=True, generator=generator
        ).T
        rows = torch.arange(len(context), device=context.device)[None]
        selected_mean, selected_lower = means[rows, component], lower[rows, component]
        return selected_mean + torch.einsum("kbij,kbj->kbi", selected_lower, eps)

    @torch.no_grad()
    def diagnostics(self, context, target, sigma, mask, *, particles=None):
        count = self.particles if particles is None else particles
        first, diag = self.log_prob(
            context,
            target,
            sigma,
            mask,
            particles=count,
            seed=self.seed,
            return_diagnostics=True,
        )
        if self.kind != "nsf":
            return dict(diag, log_prob=first)
        replica = self.log_prob(
            context, target, sigma, mask, particles=count, seed=self.seed + 1
        )
        refined = self.log_prob(
            context, target, sigma, mask, particles=2 * count, seed=self.seed + 2
        )
        return dict(
            diag,
            log_prob=first,
            independent_replica_log_prob_gap=(first - replica).abs(),
            refinement_log_prob_gap=(first - refined).abs(),
            scope="Finite-QMC diagnostic only; no automatic marginal certification",
        )

    def native_log_prob(self, context, target, sigma, mask, center, scale, **kwargs):
        """Native-coordinate Jacobian; age/distance teacher widths are forbidden R."""
        if (
            center.shape != (6,)
            or scale.shape != (6,)
            or not torch.isfinite(center).all()
        ):
            raise ValueError("Six finite normalization coordinates required")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Positive finite FIT-only scales required")
        value, errors = _check_arrays(target, sigma, mask)
        if (mask[:, 4:] & (value[:, 4:] <= 0)).any():
            raise ValueError("Observed native age/distance must be positive")
        if (errors[:, 4:] != 0).any():
            raise ValueError("Posterior age/distance widths are not measurement errors")
        coordinate = value.clone()
        coordinate[:, 4:] = torch.where(mask[:, 4:], value[:, 4:], 1).log()
        standardized = (coordinate - center) / scale
        jacobian = scale.log().expand_as(value).clone()
        jacobian[:, 4:] += torch.where(mask[:, 4:], value[:, 4:], 1).log()
        lp = self.log_prob(context, standardized, errors / scale, mask, **kwargs)
        if isinstance(lp, tuple):
            return lp[0] - jacobian.masked_fill(~mask, 0).sum(-1), lp[1]
        return lp - jacobian.masked_fill(~mask, 0).sum(-1)

    def native_samples(self, context, count, center, scale, generator=None):
        if (
            center.shape != (6,)
            or scale.shape != (6,)
            or not torch.isfinite(center).all()
        ):
            raise ValueError("Six finite normalization coordinates required")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Positive finite scales required")
        value = self.sample(context, count, generator) * scale + center
        value = torch.cat((value[..., :4], value[..., 4:].exp()), -1)
        if not torch.isfinite(value).all() or (value[..., 4:] <= 0).any():
            raise FloatingPointError("Nonfinite native samples; no clipping")
        return value
