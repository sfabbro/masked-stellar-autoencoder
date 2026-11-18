"""Anchored survey maps and the joint per-star Student-t likelihood (v6).

The model predicts, per star, a multivariate Student-t over the common-scale
labels (Teff, log g, [Fe/H], then own labels such as E(B-V) and parallax):
location mu and scale matrix S = L L^T (``StudentTHead``). Survey s reports
the mapped parameters

    y_s = x + delta_s(x) + noise,  noise ~ sigma_s (learned) (+) quoted error.

The common scale is fixed by construction, not by weights:

* anchor surveys (seismic log g, IRFM Teff) use the identity map;
* [Fe/H] "core" references (high-resolution) get offsets only, summing to
  zero over the core (weighted by stars), so the core fixes the [Fe/H] slope
  and zero point;
* every other survey's map capacity follows its data: offset only, offset +
  linear (with cross terms), or offset + linear + a shared learned basis
  (``tier``), so a small survey cannot absorb a distortion of the scale.

The likelihood of all of a star's observations (its records and its own
labels) is one multivariate Student-t, with the maps linearised at mu:
z = A x + noise, mean = stacked mapped means, scale = A S A^T + D. Records
are conditionally independent given the star, so overlaps between surveys
identify each survey's noise sigma_s; quoted errors add in quadrature.
Unobserved components are padded so they contribute nothing.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

IDENTITY, OFFSET, LINEAR, FULL = 0, 1, 2, 3


def assign_tiers(counts, anchor, core, *, linear_min=5000, full_min=50000):
    """(surveys, params) tier from record counts; anchors identity, core offset."""
    counts = np.asarray(counts)
    tier = np.where(
        counts >= full_min, FULL, np.where(counts >= linear_min, LINEAR, OFFSET)
    )
    tier = np.where(np.asarray(core), OFFSET, tier)
    tier = np.where(np.asarray(anchor), IDENTITY, tier)
    return np.where(counts > 0, tier, IDENTITY).astype("i8")


class AnchoredSurveyMaps(nn.Module):
    """delta_s(x) = b_s + L_s x + C_s phi(x), masked by each survey's tier."""

    def __init__(
        self,
        tier,
        core,
        core_weight,
        *,
        basis: int = 8,
        hidden: int = 64,
        min_slope: float = 0.25,
        sigma_init: float = 0.1,
    ):
        super().__init__()
        tier = torch.as_tensor(np.asarray(tier), dtype=torch.long)
        m, p = tier.shape
        self.n_params = p
        self.register_buffer("tier", tier)
        self.register_buffer(
            "core", torch.as_tensor(np.asarray(core), dtype=torch.bool)
        )
        w = torch.as_tensor(np.asarray(core_weight), dtype=torch.float32) * self.core
        self.register_buffer("core_weight", w / w.sum(0).clamp_min(1e-12))
        self.register_buffer("use_bias", (tier >= OFFSET).float())
        self.register_buffer(
            "use_linear", (tier >= LINEAR).float()[:, :, None].expand(m, p, p).clone()
        )
        self.register_buffer(
            "use_mix", (tier >= FULL).float()[:, :, None].expand(m, p, basis).clone()
        )
        self.min_slope = float(min_slope)
        self.bias = nn.Parameter(torch.zeros(m, p))
        self.linear = nn.Parameter(torch.zeros(m, p, p))
        self.mix = nn.Parameter(torch.zeros(m, p, basis))
        self.log_sigma = nn.Parameter(torch.full((m, p), math.log(sigma_init)))
        self.phi = nn.Sequential(
            nn.Linear(p, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, basis),
            nn.Tanh(),
        )

    def coefficients(self):
        bias = self.bias * self.use_bias
        # Core references: offsets sum to zero (star-weighted) per parameter.
        core_mean = (self.core_weight * bias).sum(0, keepdim=True)
        bias = bias - core_mean * self.core
        return bias, self.linear * self.use_linear, self.mix * self.use_mix

    def forward(self, x: torch.Tensor, survey: torch.Tensor):
        """Mapped values (R, p) and Jacobians d y / d x (R, p, p) at ``x``."""
        bias, linear, mix = self.coefficients()
        b, lin, c = bias[survey], linear[survey], mix[survey]
        phi = self.phi(x)
        dphi = torch.func.vmap(torch.func.jacrev(self.phi))(x)  # (R, K, p)
        y = x + b + torch.einsum("rjk,rk->rj", lin, x)
        y = y + torch.einsum("rjk,rk->rj", c, phi)
        jac = torch.eye(self.n_params, device=x.device) + lin
        jac = jac + torch.einsum("rjk,rkl->rjl", c, dphi)
        return y, jac

    def sigma(self, survey: torch.Tensor) -> torch.Tensor:
        return self.log_sigma[survey].exp().clamp_min(1e-3)

    def regularization(self) -> torch.Tensor:
        """L2 prior (zero = identity) on the coefficients each tier uses."""
        bias, linear, mix = self.coefficients()
        n = self.use_bias.sum() + self.use_linear.sum() + self.use_mix.sum()
        total = bias.square().sum() + linear.square().sum() + mix.square().sum()
        return total / n.clamp_min(1.0)

    def slope_penalties(self, jac: torch.Tensor):
        """(log-slope prior, monotone shortfall) from record Jacobians."""
        diag = torch.diagonal(jac, dim1=1, dim2=2)
        log_slope = diag.clamp_min(1e-3).log().square().mean()
        short = (self.min_slope - diag).clamp_min(0.0).square().mean()
        return log_slope, short


class StudentTHead(nn.Module):
    """Multivariate Student-t over the labels: location and Cholesky scale.

    ``forward`` returns (N, P, 3) marginal 16/50/84 % quantiles, so code that
    expects the quantile head (evaluation, prediction) works unchanged;
    ``params`` returns (mu, L) for the likelihood.

    With ``context_dim`` > 0, per-star metadata (``ctx`` = (values, visible),
    see ``context_inputs``) conditions the latent additively through a
    zero-initialised ``ContextLatent``, so the head is unchanged at step 0.
    """

    def __init__(
        self, latent_size: int, n_labels: int, activation, nu: float = 5.0,
        context_dim: int = 0,
    ):  # fmt: skip
        super().__init__()
        self.n_labels, self.nu = n_labels, float(nu)
        self.context = None
        if context_dim:
            from .context_latent import ContextLatent

            self.context = ContextLatent(latent_size, context_dim)
        self.shared = nn.Sequential(
            nn.Linear(latent_size, 2048), activation,
            nn.Linear(2048, 2048), activation,
            nn.Linear(2048, 1024), activation,
            nn.Linear(1024, 512), activation,
            nn.Linear(512, 256), activation,
        )  # fmt: skip
        self.loc = nn.Linear(256, n_labels)
        self.log_scale = nn.Linear(256, n_labels)
        self.n_off = n_labels * (n_labels - 1) // 2
        self.off = nn.Linear(256, max(self.n_off, 1))
        rows, cols = torch.tril_indices(n_labels, n_labels, offset=-1)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        from scipy.stats import t as student_t

        self.q84 = float(student_t.ppf(0.84, self.nu))

    def params(self, z: torch.Tensor, ctx=None):
        if self.context is not None:
            if ctx is None:
                raise ValueError("This head needs its metadata context")
            z = self.context(z, *ctx)
        h = self.shared(z)
        mu = self.loc(h)
        scale = nn.functional.softplus(self.log_scale(h)) + 1e-3
        L = torch.diag_embed(scale)
        if self.n_off:
            L = L.index_put(
                (torch.arange(len(z), device=z.device)[:, None], self.rows, self.cols),
                self.off(h)[:, : self.n_off],
            )
        return mu, L

    def forward(self, z: torch.Tensor, ctx=None) -> torch.Tensor:
        mu, L = self.params(z, ctx)
        sd = L.square().sum(-1).sqrt()
        return torch.stack([mu - self.q84 * sd, mu, mu + self.q84 * sd], dim=2)


def student_t_nll(resid, A, L, noise_var, valid, nu):
    """Per-star negative log density of z ~ t_nu(mean, A S A^T + D).

    resid (B, D) = z - mean, A (B, D, P), L (B, P, P) with S = L L^T,
    noise_var (B, D), valid (B, D). Invalid components are replaced by
    independent unit variables with zero residual, which leaves the density
    of the valid ones unchanged.
    """
    vf = valid.to(resid.dtype)
    AL = (A * vf[:, :, None]) @ L
    cov = AL @ AL.transpose(1, 2)
    # Padding: unit variance, zero residual. Jitter only on observed components.
    cov = cov + torch.diag_embed(
        torch.where(valid, noise_var + 1e-6, torch.ones_like(noise_var))
    )
    chol = torch.linalg.cholesky(cov)
    r = (resid * vf).unsqueeze(-1)
    sol = torch.linalg.solve_triangular(chol, r, upper=False).squeeze(-1)
    q = sol.square().sum(-1)
    logdet = 2 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(-1)
    d = vf.sum(-1)
    logp = (
        torch.lgamma((nu + d) / 2) - math.lgamma(nu / 2)
        - d / 2 * math.log(nu * math.pi) - logdet / 2
        - (nu + d) / 2 * torch.log1p(q / nu)
    )  # fmt: skip
    return -logp, d


def censored_upper_nll(upper, mean, var):
    """-log P(y <= upper) for y ~ N(mean, var): a record at a survey's floor.

    A survey that cannot report [Fe/H] below its grid edge says only that the
    star is at or below ``upper`` (floor + margin). The Gaussian CDF stands in
    for the Student-t one (torch has no t CDF); the term only bites when the
    prediction sits above the limit, where the two agree to first order.
    """
    return -torch.special.log_ndtr((upper - mean) / var.clamp_min(1e-12).sqrt())


def condition_on_observation(mu, L, j, z, var, nu):
    """Condition the head's joint Student-t on a noisy observation of label j.

    The head gives x ~ t_nu(mu, S = L L^T) per star. An independent
    measurement z = x_j + e, e ~ N(0, var) (approximated as part of the same
    t), updates every label: the conditional of a multivariate t is a t with
    nu + 1 degrees of freedom, location mu + S[:, j] (z - mu_j) / s_jj and
    scale (S - S[:, j] S[j, :] / s_jj) (nu + q) / (nu + 1), with
    s_jj = S_jj + var and q = (z - mu_j)^2 / s_jj. Rows where z or var is not
    finite are returned unchanged. Returns (location, marginal scale) per row;
    the conditional t has nu + 1 degrees of freedom (unconditioned rows: nu).

    Used at inference with Gaia parallax: the head is trained with parallax as
    an own label, so its covariance links parallax to log g and Teff without
    any stellar model, and conditioning uses the measurement without letting
    the network copy it.
    """
    S = L @ L.transpose(1, 2)
    ok = torch.isfinite(z) & torch.isfinite(var) & (var >= 0)
    zj = torch.where(ok, z, mu[:, j])
    vj = torch.where(ok, var, torch.zeros_like(var))
    s_col = S[:, :, j]
    s_jj = S[:, j, j] + vj
    r = zj - mu[:, j]
    loc = mu + s_col * (r / s_jj)[:, None]
    q = r * r / s_jj
    cond = S - s_col[:, :, None] * s_col[:, None, :] / s_jj[:, None, None]
    factor = (nu + q) / (nu + 1)
    sd_c = (torch.diagonal(cond, dim1=1, dim2=2).clamp_min(0) * factor[:, None]).sqrt()
    sd_0 = torch.diagonal(S, dim1=1, dim2=2).sqrt()
    loc = torch.where(ok[:, None], loc, mu)
    sd = torch.where(ok[:, None], sd_c, sd_0)
    return loc, sd
