"""Controlled native Teff/logg/FeH heads, calibration and label-free refinement.

All joint-density methods use the same complete-case native rows. Quoted errors
are diagonal survey measurement errors, not a physical-truth certificate.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

TARGET_NAMES = ("teff", "logg", "fe_h")


def complete_roles(observed, quality, roles):
    observed, quality = np.asarray(observed), np.asarray(quality)
    if observed.shape != quality.shape or observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("Three aligned native targets required")
    complete = (observed & quality).all(1)
    result = {
        role: np.flatnonzero(np.asarray(roles[role], dtype=bool) & complete)
        for role in ("fit", "tune", "calibration", "test")
    }
    for i, left in enumerate(result):
        for right in list(result)[i + 1 :]:
            if np.intersect1d(result[left], result[right]).size:
                raise ValueError("Native source roles overlap")
    return result


def label_scaling(value, fit):
    values = np.asarray(value)[fit]
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite complete fit labels required")
    center, scale = values.mean(0), values.std(0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Degenerate fit-only target scale")
    return center, scale


def qmc_normal_nodes(count=256, dimensions=3, seed=2026):
    if count < 8 or count % 2:
        raise ValueError("Even antithetic QMC count of at least eight required")
    uniform = (
        torch.quasirandom.SobolEngine(dimensions, scramble=True, seed=seed)
        .draw(count // 2)
        .double()
    )
    if not ((uniform > 0) & (uniform < 1)).all():
        raise FloatingPointError("Sobol endpoint cannot be mapped to finite normal")
    normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    return torch.cat([normal, -normal]).float()


def supervised_fit_rows(rng, fit, batch_size=256):
    """Draw a bounded FIT minibatch without repeated source positions."""
    if batch_size < 1 or len(fit) < 2:
        raise ValueError("At least two FIT objects and a positive batch are required")
    return rng.choice(fit, size=min(batch_size, len(fit)), replace=False)


HEAD_KINDS = (
    "huber",
    "gaussian",
    "flow",
    "mvgaussian",
    "lowrank",
    "mdn",
    "quantile",
    "expectile",
)
DENSITY_KINDS = ("gaussian", "flow", "mvgaussian", "lowrank", "mdn")
NON_DENSITY_KINDS = ("huber", "quantile", "expectile")


def _observed_mask(shape, mask, device):
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.shape != shape:
        raise ValueError(f"observed mask must have shape {tuple(shape)}")
    return mask


def _gaussian_diagonal_log_prob(residual, variance):
    if not torch.isfinite(variance).all() or not (variance > 0).all():
        raise FloatingPointError("Non-positive or nonfinite predictive variance")
    return -0.5 * (
        residual.square() / variance + variance.log() + math.log(2 * math.pi)
    )


def masked_gaussian_row_log_prob(mean, scale_tril, target, sigma=None, *, mask=None):
    """Per-row exact Gaussian log density of the observed target subvector.

    Iterates the ``2**d - 1`` observed patterns (same arithmetic as
    ``joint_prediction.masked_gaussian_measurement_nll``), so a row with missing
    coordinates gets the marginal density of the coordinates it does have. Rows
    with no observed coordinate return a log density of ``0``; callers must count
    them separately. A non-positive combined covariance raises: no jitter.
    """
    if mean.ndim != 2:
        raise ValueError("mean must be a [row, target] tensor")
    rows, dims = mean.shape
    if target.shape != (rows, dims) or scale_tril.shape != (rows, dims, dims):
        raise ValueError("target/scale_tril must share the [row, target] layout")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale_tril).all():
        raise ValueError("model parameters must be finite")
    if not (scale_tril.diagonal(dim1=-2, dim2=-1) > 0).all():
        raise ValueError("model scale diagonal must be strictly positive")
    if torch.count_nonzero(torch.triu(scale_tril, diagonal=1)):
        raise ValueError("model scale must be lower triangular")
    valid = _observed_mask((rows, dims), mask, mean.device) & torch.isfinite(target)
    if sigma is not None:
        if sigma.shape != target.shape:
            raise ValueError("measurement errors must share the target layout")
        valid = valid & torch.isfinite(sigma) & (sigma > 0)
    out = mean.new_zeros(rows)
    code = (valid.to(torch.int64) * (2 ** torch.arange(dims, device=mean.device))).sum(1)
    for pattern in range(1, 2**dims):
        keep = code == pattern
        if not keep.any():
            continue
        selected = [i for i in range(dims) if pattern & (1 << i)]
        # Keep ALL factor columns: a lower-triangular submatrix is a Cholesky
        # factor only for a *contiguous* leading subvector. The marginal factor
        # is the selected ROWS of the full factor, so Sigma_obs = L_obs L_obs^T.
        factor = scale_tril[keep][:, selected, :]
        covariance = factor @ factor.transpose(-1, -2)
        if sigma is not None:
            covariance = covariance + torch.diag_embed(
                sigma[keep][:, selected].square()
            )
        chol, info = torch.linalg.cholesky_ex(covariance)
        if (info != 0).any():
            raise ValueError(
                "observed-subvector covariance is not positive definite; no jitter"
            )
        residual = target[keep][:, selected] - mean[keep][:, selected]
        whitened = torch.linalg.solve_triangular(chol, residual[..., None], upper=False)
        density = -0.5 * (
            whitened.square().sum((1, 2)) + len(selected) * math.log(2 * math.pi)
        ) - chol.diagonal(dim1=-2, dim2=-1).log().sum(1)
        if not torch.isfinite(density).all():
            raise FloatingPointError("Nonfinite masked Gaussian density")
        index = torch.nonzero(keep).squeeze(1)
        out = out.index_copy(0, index, density)
    return out


def masked_mixture_row_log_prob(
    log_weights, means, log_stds, target, sigma=None, *, mask=None
):
    """Per-row diagonal-mixture density of the observed target subvector.

    Exact marginal: every component is a diagonal Gaussian, so dropping a
    coordinate means dropping its factor from each component and keeping the
    mixture weights unchanged. No renormalization, no imputation.
    """
    if means.ndim != 3 or log_weights.ndim != 2:
        raise ValueError("log_weights [row, component] and means [row, K, d] required")
    rows, components, dims = means.shape
    if log_stds.shape != means.shape or log_weights.shape != (rows, components):
        raise ValueError("mixture parameter shapes disagree")
    if target.shape != (rows, dims):
        raise ValueError("target must be [row, target]")
    if not torch.isfinite(means).all() or not torch.isfinite(log_stds).all():
        raise ValueError("mixture parameters must be finite")
    variance_all = (2 * log_stds).exp()
    if not (variance_all > 0).all() or not torch.isfinite(variance_all).all():
        raise FloatingPointError("invalid mixture component variance")
    valid = _observed_mask((rows, dims), mask, target.device) & torch.isfinite(target)
    if sigma is not None:
        if sigma.shape != target.shape:
            raise ValueError("measurement errors must share the target layout")
        valid = valid & torch.isfinite(sigma) & (sigma > 0)
    out = target.new_zeros(rows)
    code = (valid.to(torch.int64) * (2 ** torch.arange(dims, device=target.device))).sum(1)
    for pattern in range(1, 2**dims):
        keep = code == pattern
        if not keep.any():
            continue
        selected = [i for i in range(dims) if pattern & (1 << i)]
        variance = variance_all[keep][:, :, selected]
        if sigma is not None:
            variance = variance + sigma[keep][:, None, selected].square()
        residual = target[keep][:, None, selected] - means[keep][:, :, selected]
        component = _gaussian_diagonal_log_prob(residual, variance).sum(-1)
        density = torch.logsumexp(component + log_weights[keep], -1)
        index = torch.nonzero(keep).squeeze(1)
        out = out.index_copy(0, index, density)
    return out


def _standard_normal_quantiles(levels):
    values = torch.tensor(tuple(float(level) for level in levels), dtype=torch.float64)
    return torch.special.ndtri(values)


def _standard_normal_expectiles(levels):
    """Constants ``c_tau`` with ``E_tau[Z] = c_tau`` for ``Z ~ N(0, 1)``.

    Bisection on ``tau*A(c) = (1-tau)*B(c)`` with ``A = phi(c) - c(1-Phi(c))`` and
    ``B = c*Phi(c) + phi(c)``; ``A`` is the upper partial first moment and ``B`` the
    lower one. Exact to float64 before the buffer is stored in the head dtype.
    """
    tau = torch.tensor(tuple(float(level) for level in levels), dtype=torch.float64)
    root2 = math.sqrt(2.0)
    inverse_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    def misfit(c):
        normal = 0.5 * (1.0 + torch.erf(c / root2))
        density = torch.exp(-0.5 * c * c) * inverse_sqrt_2pi
        upper = density - c * (1.0 - normal)
        lower = c * normal + density
        return tau * upper - (1.0 - tau) * lower

    low = torch.full_like(tau, -8.0)
    high = torch.full_like(tau, 8.0)
    for _ in range(200):
        middle = 0.5 * (low + high)
        above = misfit(middle) > 0
        low = torch.where(above, middle, low)
        high = torch.where(above, high, middle)
    return 0.5 * (low + high)


def _level_sequence(levels, name, minimum=2):
    values = tuple(float(level) for level in levels)
    if len(values) < minimum:
        raise ValueError(f"at least {minimum} {name} levels required")
    if not all(0.0 < value < 1.0 for value in values):
        raise ValueError(f"{name} levels must lie strictly inside (0, 1)")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} levels must be strictly ascending")
    return values


class MultiLabelHead(nn.Module):
    """Native multi-target head: Huber, Gaussian, flow, MDN, quantile, expectile.

    Every kind shares one encoder latent interface and one output block. The
    declared ``target_dim`` sets the output layout and replaces the old hardcoded
    three-target normalization. Density kinds return per-``density_kind``
    log densities; point/interval kinds (Huber, quantile, expectile) return no
    density, and the scoring path must record that explicitly instead of storing
    a silently-aliased intrinsic value.
    """

    def __init__(
        self,
        kind="gaussian",
        *,
        target_dim=3,
        latent_dim=256,
        width=128,
        flow_type="nsf",
        noise_aware=False,
        mc_samples=32,
        flow_bound=8.0,
        flow_objective="nll",
        contrastive_weight=0.1,
        quantiles=(0.16, 0.5, 0.84),
        expectiles=(0.1, 0.5, 0.9),
        quantile_crossover=True,
        quantile_crossover_penalty=10.0,
        mdn_components=4,
        lowrank_rank=2,
    ):
        super().__init__()
        if kind not in HEAD_KINDS:
            raise ValueError(f"Unsupported multilabel head {kind!r}; expected {HEAD_KINDS}")
        if (
            isinstance(target_dim, bool)
            or not isinstance(target_dim, int)
            or not 1 <= target_dim <= 8
        ):
            raise ValueError("target_dim must be an integer in 1..8")
        if isinstance(mdn_components, bool) or not isinstance(mdn_components, int):
            raise ValueError("mdn_components must be a positive integer")
        if mdn_components < 1:
            raise ValueError("mdn_components must be a positive integer")
        if isinstance(lowrank_rank, bool) or not isinstance(lowrank_rank, int):
            raise ValueError("lowrank_rank must be a positive integer")
        if kind == "lowrank" and not 1 <= lowrank_rank < target_dim:
            raise ValueError("lowrank_rank must satisfy 1 <= rank < target_dim")
        if noise_aware and kind in NON_DENSITY_KINDS:
            raise ValueError(f"{kind} is not a density measurement likelihood")
        if flow_objective not in {"nll", "contrastive", "nll_contrastive"}:
            raise ValueError("Unsupported flow objective")
        if flow_objective != "nll" and (kind != "flow" or noise_aware):
            raise ValueError("Contrastive contexts require a clean joint flow")
        if contrastive_weight < 0:
            raise ValueError("Contrastive weight must be nonnegative")
        if quantile_crossover_penalty < 0:
            raise ValueError("Quantile crossover penalty must be nonnegative")
        self.kind = kind
        self.target_dim = target_dim
        self.noise_aware = bool(noise_aware)
        self.flow_type = flow_type
        self.flow_objective = flow_objective
        self.contrastive_weight = float(contrastive_weight)
        self.quantile_crossover_penalty = float(quantile_crossover_penalty)
        self.mdn_components = int(mdn_components)
        self.lowrank_rank = int(lowrank_rank)
        self.quantiles = (
            _level_sequence(quantiles, "quantile") if kind == "quantile" else ()
        )
        self.expectiles = (
            _level_sequence(expectiles, "expectile") if kind == "expectile" else ()
        )
        self.input = nn.Linear(latent_dim, width)
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 4),
            nn.GELU(),
            nn.Linear(width // 4, width),
        )
        self.output = nn.Linear(width, self._output_size())
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.loss_fn = None
        if kind == "gaussian":
            from torchregress.losses.gaussian import GaussianNLLLoss

            self.loss_fn = GaussianNLLLoss(
                log_variance=False,
                min_variance=torch.finfo(torch.float32).tiny,
                eps=0.0,
            )
        elif kind == "flow":
            from torchregress.losses.nflows import (
                NormalizingFlowLoss,
                create_flow_model,
            )

            kwargs = dict(
                n_features=target_dim,
                context_dim=16,
                flow_type=flow_type,
                n_transforms=3,
                hidden_features=[64, 64],
            )
            if flow_type == "nsf":
                kwargs.update(bound=flow_bound, bins=16)
            if flow_type == "bpf":
                kwargs.update(degree=8)
            flow = create_flow_model(**kwargs)
            if flow_objective == "nll":
                base = NormalizingFlowLoss(flow)
            else:
                from torchregress.losses.nflows import ContrastiveFlowLoss

                base = ContrastiveFlowLoss(flow, temperature=1.0, margin=0.0)

            if noise_aware:
                from torchregress.losses.noise_aware_density import (
                    NoiseAwareDensityLoss,
                )

                self.loss_fn = NoiseAwareDensityLoss(
                    base, mode="mc", n_samples=mc_samples, reduction="none"
                )
            else:
                self.loss_fn = base
        elif kind == "quantile":
            from torchregress.losses.quantile import (
                MultiQuantileLoss,
                QuantileCrossoverLoss,
            )

            if quantile_crossover:
                self.loss_fn = QuantileCrossoverLoss(
                    list(self.quantiles), crossover_penalty=quantile_crossover_penalty
                )
            else:
                self.loss_fn = MultiQuantileLoss(list(self.quantiles))
        elif kind == "expectile":
            from torchregress.losses.expectile import MultiExpectileLoss

            self.loss_fn = MultiExpectileLoss(list(self.expectiles))
        self.register_buffer(
            "point_nodes", qmc_normal_nodes(dimensions=target_dim)
        )
        self.register_buffer(
            "measurement_nodes", qmc_normal_nodes(mc_samples, dimensions=target_dim)
        )
        if kind == "quantile":
            self.register_buffer(
                "level_constants", _standard_normal_quantiles(self.quantiles).float()
            )
        elif kind == "expectile":
            self.register_buffer(
                "level_constants", _standard_normal_expectiles(self.expectiles).float()
            )
        index = torch.tril_indices(target_dim, target_dim)
        self.register_buffer("tril_row", index[0], persistent=False)
        self.register_buffer("tril_col", index[1], persistent=False)

    def _output_size(self):
        dims = self.target_dim
        if self.kind == "huber":
            return dims
        if self.kind == "gaussian":
            return 2 * dims
        if self.kind == "flow":
            return 16
        if self.kind == "mvgaussian":
            return dims + dims * (dims + 1) // 2
        if self.kind == "lowrank":
            return dims + dims * self.lowrank_rank + dims
        if self.kind == "mdn":
            return self.mdn_components + 2 * self.mdn_components * dims
        if self.kind == "quantile":
            return dims * len(self.quantiles)
        if self.kind == "expectile":
            return dims * len(self.expectiles)
        raise ValueError(f"Unsupported multilabel head {self.kind!r}")

    @property
    def has_density(self):
        """Whether this head defines a normalized predictive density."""
        return self.kind in DENSITY_KINDS

    @property
    def density_kind(self):
        """Which estimand the density accessors refer to.

        ``intrinsic`` is the latent-label predictive density; ``observed_label``
        is the correctly convolved density of the reported (noisy) labels. A
        head that is not noise-aware can only report the intrinsic one.
        """
        if not self.has_density:
            return None
        return "observed_label" if self.noise_aware else "intrinsic"

    @property
    def loss_denominator(self):
        return "per_coordinate"

    def forward(self, latent):
        hidden = torch.nn.functional.gelu(self.input(latent))
        return self.output(hidden + self.block(hidden))

    def distribution(self, latent):
        if self.kind != "flow":
            raise ValueError("Joint flow required")
        loss = self.loss_fn.base_loss if self.noise_aware else self.loss_fn
        return loss.flow(self(latent))

    def _gaussian_parameters(self, raw):
        mean = raw[:, : self.target_dim]
        dims = self.target_dim
        if self.kind == "mvgaussian":
            packed = raw[:, dims:].clone()
            diagonal = self.tril_row == self.tril_col
            packed[:, diagonal] = packed[:, diagonal].exp()
            scale_tril = packed.new_zeros((len(raw), dims, dims))
            scale_tril[:, self.tril_row, self.tril_col] = packed
            if not torch.isfinite(scale_tril).all() or not (
                scale_tril.diagonal(dim1=-2, dim2=-1) > 0
            ).all():
                raise FloatingPointError("Invalid full covariance; no silent clipping")
            return mean, scale_tril
        factor = raw[:, dims : dims + dims * self.lowrank_rank].view(
            len(raw), dims, self.lowrank_rank
        )
        variance = raw[:, dims + dims * self.lowrank_rank :].exp()
        if not torch.isfinite(variance).all() or not (variance > 0).all():
            raise FloatingPointError("Invalid low-rank Gaussian variance")
        covariance = factor @ factor.transpose(-1, -2) + torch.diag_embed(variance)
        scale_tril, info = torch.linalg.cholesky_ex(covariance)
        if (info != 0).any():
            raise ValueError("Low-rank covariance is not positive definite; no jitter")
        return mean, scale_tril

    def _mixture_parameters(self, raw):
        dims, components = self.target_dim, self.mdn_components
        start = components
        means = raw[:, start : start + components * dims].view(
            len(raw), components, dims
        )
        log_stds = raw[:, start + components * dims :].view(
            len(raw), components, dims
        )
        if not torch.isfinite(means).all() or not torch.isfinite(log_stds).all():
            raise FloatingPointError("Nonfinite mixture parameters")
        return torch.log_softmax(raw[:, :components], -1), means, log_stds

    def _level_values(self, raw):
        levels = len(self.quantiles) if self.kind == "quantile" else len(self.expectiles)
        return raw.view(len(raw), levels, self.target_dim)

    def criterion(self, latent, target, sigma, *, source_ids=None, mask=None):
        if target.shape != (len(latent), self.target_dim) or sigma.shape != target.shape:
            raise ValueError("Complete standardized target rows required")
        if (
            not torch.isfinite(target).all()
            or not torch.isfinite(sigma).all()
            or not (sigma > 0).all()
        ):
            raise ValueError("Invalid native target/error")
        raw = self(latent)
        dims = self.target_dim
        if self.kind == "huber":
            return torch.nn.functional.huber_loss(raw, target)
        if self.kind == "gaussian":
            mean, logvar = raw.chunk(2, -1)
            variance = logvar.exp()
            if self.noise_aware:
                variance = variance + sigma.square()
            if not torch.isfinite(variance).all() or not (variance > 0).all():
                raise FloatingPointError("Nonfinite Gaussian variance")
            return self.loss_fn((mean, variance), target)
        if self.kind in {"mvgaussian", "lowrank"}:
            mean, scale_tril = self._gaussian_parameters(raw)
            if self.noise_aware:
                # Audited exact observed-subvector likelihood (diagonal R).
                from masked_stellar_autoencoder.training.joint_prediction import (
                    masked_gaussian_measurement_nll,
                )

                total = masked_gaussian_measurement_nll(
                    mean, scale_tril, target, sigma, eligible=mask
                )
                if total.observation_count == 0:
                    return mean.reshape(-1)[:0].sum()
                return total.numerator / total.observation_count
            density = masked_gaussian_row_log_prob(
                mean, scale_tril, target, mask=mask
            )
            return -density.mean() / dims
        if self.kind == "mdn":
            log_weights, means, log_stds = self._mixture_parameters(raw)
            density = masked_mixture_row_log_prob(
                log_weights,
                means,
                log_stds,
                target,
                sigma if self.noise_aware else None,
                mask=mask,
            )
            return -density.mean() / dims
        if self.kind in {"quantile", "expectile"}:
            return self.loss_fn(self._level_values(raw), target, mask=mask)
        if self.noise_aware:
            return self.loss_fn(raw, target, sigma=sigma).mean() / dims
        if self.flow_objective != "nll":
            if source_ids is None or len(raw) < 2:
                raise ValueError("Contrastive batches require unique source IDs")
            if len(torch.unique(source_ids)) != len(raw):
                raise ValueError("Contrastive negatives cannot contain duplicate stars")
            # One fixed cyclic derangement; both context branches retain gradients.
            negative = torch.roll(raw, shifts=1, dims=0).unsqueeze(1)
            contrastive = self.loss_fn(raw, target, negative_context=negative)
            if self.flow_objective == "contrastive":
                return contrastive / dims
            nll = -self.loss_fn.log_prob(raw, target).mean()
            return (nll + self.contrastive_weight * contrastive) / dims
        return self.loss_fn(raw, target) / dims

    def _location_scale_from_levels(self, values, constants):
        """Least-squares ``level = mean + constant * sigma`` over the level axis."""
        location = constants.to(values.dtype).to(values.device)
        centered = location - location.mean()
        weight = centered.square().sum()
        if not torch.isfinite(weight) or weight <= 0:
            raise ValueError("Level constants must not be all equal")
        deviation = values - values.mean(1, keepdim=True)
        spread = (deviation * centered[None, :, None]).sum(1) / weight
        # Diagnostic point moments only: a degenerate head yields zero spread,
        # which the scoring path reports instead of inventing a variance floor.
        variance = spread.clamp_min(0).square()
        return values.mean(1), variance

    def moments(self, latent):
        if self.kind == "flow":
            dist = self.distribution(latent)
            # Preserve all fixed nodes and their reduction order while bounding
            # the spline inverse's temporary sample-by-context allocation.
            samples = [
                dist.transform.inv(nodes[:, None, :].expand(-1, len(latent), -1))
                for nodes in self.point_nodes.split(16)
            ]
            sample = torch.cat(samples, dim=0)
            mean = sample.double().mean(0).to(latent.dtype)
            covariance = (sample - mean).square().double().mean(0).to(latent.dtype)
            return mean, covariance
        raw = self(latent)
        if self.kind == "huber":
            return raw, torch.full_like(raw, float("nan"))
        if self.kind == "gaussian":
            mean, logvar = raw.chunk(2, -1)
            return mean, logvar.exp()
        if self.kind in {"mvgaussian", "lowrank"}:
            mean, scale_tril = self._gaussian_parameters(raw)
            covariance = (scale_tril @ scale_tril.transpose(-1, -2)).diagonal(
                dim1=-2, dim2=-1
            )
            return mean, covariance
        if self.kind == "mdn":
            log_weights, means, log_stds = self._mixture_parameters(raw)
            weights = log_weights.exp()
            mean = (weights[:, :, None] * means).sum(1)
            second = (weights[:, :, None] * (means.square() + (2 * log_stds).exp())).sum(1)
            return mean, second - mean.square()
        return self._location_scale_from_levels(
            self._level_values(raw), self.level_constants
        )

    def log_prob(self, latent, target, sigma=None, *, mask=None):
        if self.kind == "flow":
            distribution = self.distribution(latent)
            if sigma is None:
                return distribution.log_prob(target)
            perturbed = (
                target.unsqueeze(0)
                + sigma.unsqueeze(0) * self.measurement_nodes[:, None, :]
            )
            return torch.logsumexp(distribution.log_prob(perturbed), 0) - math.log(
                len(self.measurement_nodes)
            )
        if not self.has_density:
            return torch.full((len(latent),), float("nan"), device=latent.device)
        raw = self(latent)
        if self.kind == "gaussian":
            mean, logvar = raw.chunk(2, -1)
            variance = logvar.exp()
            if sigma is not None:
                variance = variance + sigma.square()
            return _gaussian_diagonal_log_prob(target - mean, variance).sum(-1)
        if self.kind in {"mvgaussian", "lowrank"}:
            mean, scale_tril = self._gaussian_parameters(raw)
            return masked_gaussian_row_log_prob(mean, scale_tril, target, sigma, mask=mask)
        log_weights, means, log_stds = self._mixture_parameters(raw)
        return masked_mixture_row_log_prob(
            log_weights, means, log_stds, target, sigma, mask=mask
        )
def conformal_intervals(
    calibration_prediction, calibration_target, test_prediction, alpha=0.1
):
    """Marginal library CP and conservative simultaneous Bonferroni boxes."""
    from torchregress.losses.conformal import MultiTargetConformal

    marginal = MultiTargetConformal(alpha=alpha)
    marginal.calibrate(calibration_prediction, calibration_target)
    simultaneous = MultiTargetConformal(alpha=alpha / 3)
    simultaneous.calibrate(calibration_prediction, calibration_target)
    lower, upper = marginal.predict_interval(test_prediction)
    joint_lower, joint_upper = simultaneous.predict_interval(test_prediction)
    return dict(
        marginal_lower=lower,
        marginal_upper=upper,
        simultaneous_lower=joint_lower,
        simultaneous_upper=joint_upper,
        marginal_threshold=marginal.q_hat,
        simultaneous_threshold=simultaneous.q_hat,
    )


def hash_selected(ids, count=512, seed=1042):
    keys = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{int(s)}:{seed}".encode()).digest()[:8], "little"
            )
            for s in ids
        ],
        dtype="u8",
    )
    return np.argsort(keys, kind="stable")[: min(len(ids), count)]


@contextmanager
def refinement_scope(model, head):
    """Freeze predictor/decoder; restore complete per-object LN parameter state."""
    parameters = list(model.parameters()) + list(head.parameters())
    flags = [p.requires_grad for p in parameters]
    modes = [
        (module, module.training) for root in (model, head) for module in root.modules()
    ]
    selected = [
        p
        for module in model.encoder.modules()
        if isinstance(module, nn.LayerNorm)
        for p in module.parameters(recurse=False)
    ]
    if not selected:
        raise ValueError("No encoder LayerNorm affine parameters")
    saved = [p.detach().clone() for p in selected]
    gradients = [
        None if p.grad is None else p.grad.detach().clone() for p in parameters
    ]
    try:
        for p in parameters:
            p.requires_grad_(False)
        for p in selected:
            p.requires_grad_(True)
        model.eval()
        head.eval()
        yield selected, saved
    finally:
        with torch.no_grad():
            for p, v in zip(selected, saved, strict=True):
                p.copy_(v)
        for p, flag, grad in zip(parameters, flags, gradients, strict=True):
            p.requires_grad_(flag)
            p.grad = grad
        for module, mode in modes:
            module.training = mode


def refine_one(
    model, head, x, visible, xp_columns, source_id, *, steps=3, lr=1e-4, seed=1042
):
    """MSA reconstruction TTT; labels/errors never enter this local objective."""
    if x.shape[0] != 1 or visible.shape != x.shape:
        raise ValueError("One object with its observed input mask required")
    with refinement_scope(model, head) as (parameters, original):
        valid = [int(c) for c in xp_columns if bool(visible[0, c])]
        if len(valid) < 2:
            raise ValueError("At least two observed XP coefficients required")
        key = hashlib.sha256(f"{int(source_id)}:{seed}:holdout".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        chosen = [c for c in valid if rng.random() < 0.2]
        if not chosen:
            chosen = [valid[int(rng.integers(len(valid)))]]
        if len(chosen) == len(valid):
            chosen = chosen[:-1]
        hidden = x.clone()
        hidden[:, chosen] = -9999.0
        with torch.no_grad():
            baseline = head.moments(model.encoder(x))[0]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            latent = model.encoder(hidden)
            reconstruction = model.reconstruction_layer(model.decoder(latent))
            if getattr(model, "heteroscedastic", False):
                reconstruction = reconstruction.chunk(2, -1)[0]
            loss = (reconstruction[:, chosen] - x[:, chosen]).abs().mean()
            drift = sum(
                (p - v).square().mean()
                for p, v in zip(parameters, original, strict=True)
            ) / len(parameters)
            mean = head.moments(model.encoder(x))[0]
            loss = loss + 0.1 * drift + 0.1 * (mean - baseline).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local refinement")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = head.moments(model.encoder(x))[0].detach()
        return prediction, dict(
            steps=steps, heldout_count=len(chosen), final_objective=float(loss.detach())
        )


def native_xp_error_subset(errors, scale, xp, g_magnitude, used_rows):
    """Normalize only the model's actual rows, never unrelated missing-XP stars."""
    from masked_stellar_autoencoder.training.input_measurement import (
        standardized_xp_errors,
    )

    used = np.unique(np.asarray(used_rows, dtype="i8"))
    if not len(used) or used[0] < 0 or used[-1] >= len(errors):
        raise ValueError("Actual XP source rows required")
    sigma = standardized_xp_errors(
        np.asarray(errors)[used],
        scale,
        xp,
        g_magnitude=np.asarray(g_magnitude)[used],
        error_convention="raw_xp",
    )
    return used, sigma
