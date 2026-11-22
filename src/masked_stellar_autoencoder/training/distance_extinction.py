"""Joint distance-extinction heads trained on Gaia parallaxes (S3).

Design note ``docs/design-20261122-data-driven-inference.md`` (sections 1-3, 6):
distance and E(B-V) as first-class, data-driven outputs. Distance is geometry,
not a label: the training signal is every star's own Gaia parallax through its
measurement likelihood, never an inverted parallax or a model-based distance
catalogue, and no parallax / error cut (low-S/N and negative parallaxes stay).

Coordinates: ``s = ln(d / kpc)`` (model parallax ``h = exp(-s)`` mas, distance
modulus ``mu = 10 + 5 s / ln 10``) and ``E = E(B-V)`` [mag]. A head gives a
joint predictive density ``p(s, E | z)`` on a frozen latent ``z`` (computed with
the parallax input hidden), trained by maximising, per star, the exact marginal
likelihood of the star's measurements::

    L_i = int int p(s, E | z_i) N(plx_i - zp_i | e^-s, sig_eff_i^2)
                                [N(E_obs_i | E, e_i^2)] ds dE

* parallax: ``plx - zp`` with the Lindegren+2021 zero point,
  ``sig_eff^2 = (kappa_g sig)^2 + sig_zp^2``; ``kappa_g`` is a fitted inflation
  of the quoted error per G bin (``KAPPA_G_EDGES``), ``sig_zp`` a fixed
  zero-point uncertainty (default 0);
* E(B-V) label (errors in variables) where one exists; without one E is
  marginalised exactly (the factor is 1).

Distance and extinction are degenerate in photometry, so the head is joint and
represents their correlation. Two families (``KINDS``), each run as a
K-member ensemble (equal-weight mixture of members):

* ``gmix``: an MLP trunk -> a 2-D full-covariance Gaussian mixture in
  ``(s, e)``, ``e = (E - e_loc) / e_scale`` (torchregress ``MixtureDensityLoss``,
  ``df = inf``, scale-Cholesky parameterisation: ``StudentTMixture.scale_tril``
  of the precision form goes through ``qr(mode="r")``, which has no gradient,
  so ``condition`` cannot be trained through it). The E label integral is
  closed form: the label's marginal ``component_observation_log_prob`` and the
  label-conditioned mixture ``StudentTMixture.condition``; the parallax
  integral is 1-D in ``s``.
* ``flow``: an MLP trunk -> context ``c``; a conditional rational-quadratic
  spline flow in the autoregressive order (s, E):
  ``p(s, E | z) = p(s | c) p(e | s, c) / e_scale`` with two torchregress
  ``create_flow_model(1, ., "nsf")`` flows (``flow_s`` on ``c``, ``flow_e`` on
  ``[c, s]``): a 2-D spline flow with a fixed variable order, whose ``s``
  marginal is exact (no E integral when the label is missing) and whose
  conditional ``p(e | s)`` has spline parameters computed once per ``s`` node
  (zuko's 1-D element-wise transform), so the E label integral is a cheap
  inner quadrature.

Quadrature (why it is exact up to a controlled discretisation): the integrand
in ``s`` is the product of the predictive (its mass inside a window ``W_p``
that maps the base-space interval ``[-U_TAIL, U_TAIL]``, i.e. all but ~4e-8 of
the probability) and the parallax likelihood (its mass inside ``W_m``: ``h``
within ``K_MEAS`` errors of ``plx - zp``). The integral is the trapezoid rule
on ``n`` nodes over ``W_m`` when it is the narrower window (it then contains
the product's mass wherever it sits, so precise parallaxes are resolved,
including those in the predictive's far tail), else over ``W_p n W_m`` (a
broad likelihood: the product lies inside ``W_p``; without overlap, the gap
plus the narrower window; ``combine_windows``). gmix:
nodes uniform in ``s``; the integrand is analytic and decays at both window
ends, where the trapezoid rule converges geometrically. flow: the spline
density is only continuous at its knots (O(h^2) convergence in ``s``), so the
nodes are uniform in the flow's base coordinate ``u = T(s)`` over the same
window (``p(s) ds = phi(u) du``, a continuously differentiable integrand); the
inner E integral likewise uses base-coordinate nodes of ``p(e | s)`` over
``[E_obs +- K_MEAS e]`` cut by ``|u| <= U_TAIL``. ``quadrature_check``
measures the discretisation error against 4x the nodes on real rows.

torchregress gap: it has no exact 1-D/2-D measurement convolution of a
nonlinear observation (``h = e^-s``) for a flow predictive (its
``forward_quadrature`` is Gauss-Hermite for a Gaussian latent; the MC
``NoiseAwareDensityLoss`` is a biased lower bound for narrow likelihoods); the
window-adaptive trapezoid here should move upstream.

PIT of an observation under the convolved predictive uses integration by parts
so that the exact predictive CDF and the likelihood kernel are both resolved:
``P(plx_rep <= plx) = int (1 - F_s(s)) N(plx | e^-s, sig) e^-s ds`` on the
union of ``W_p`` and ``W_m`` (plus closed-form tails), and
``P(E_rep <= E_obs) = int F_E(E) N(E | E_obs, e^2) dE``.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from .distributional_heads import mlp

KINDS = ("gmix", "flow")
#: Global box of s = ln(d / kpc): 0.9 pc .. 1.1 Mpc (model parallax 1100 .. 9e-4 mas).
S_BOUNDS = (-7.0, 7.0)
H_MIN = math.exp(-S_BOUNDS[1])
#: Base-space half-width of the predictive windows (two-sided tail 3.8e-8).
U_TAIL = 5.5
#: Measurement windows: +- this many (effective) errors.
K_MEAS = 7.0
#: Smallest window width (s or e units) so the trapezoid weights stay positive.
MIN_WIDTH = 1e-4
#: Mixture components below this log weight do not set windows.
LOGW_MIN = -12.0
#: E(B-V) label errors: floor and cap [mag] (the labels' quoted errors).
E_ERROR_FLOOR, E_ERROR_CAP = 0.01, 0.5
#: G bins of the fitted parallax-error inflation kappa_g (bin b: edges[b-1] < G <= edges[b]).
KAPPA_G_EDGES = (13.0, 15.0, 17.0, 18.0, 19.0)
KAPPA_MIN = 0.3
_LOG_2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


def distance_modulus(s):
    """mu = 5 log10(d / 10 pc) for s = ln(d / kpc) (numpy or torch)."""
    return 10.0 + 5.0 * s / math.log(10.0)


def kappa_bin(G):
    """Index of the kappa G bin (NaN G -> the faintest bin)."""
    g = np.asarray(G, dtype=np.float64)
    idx = np.searchsorted(np.asarray(KAPPA_G_EDGES), g, side="left")
    return np.where(np.isfinite(g), idx, len(KAPPA_G_EDGES)).astype(np.int64)


def _normal_logpdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd).square() - sd.log() - 0.5 * _LOG_2PI


def _ndtr(x):
    return 0.5 * torch.erfc(-x / _SQRT2)


def parallax_log_likelihood(s, plx, sig):
    """log N(plx | exp(-s), sig^2) at nodes ``s [N, B]`` for ``plx, sig [B]``."""
    return _normal_logpdf(plx, torch.exp(-s), sig)


def uniform_nodes(lo, hi, n):
    """Trapezoid nodes ``[n, B]`` and log weights on ``[lo, hi]`` (per row)."""
    t = torch.linspace(0.0, 1.0, int(n), dtype=lo.dtype, device=lo.device)[:, None]
    width = hi - lo
    x = lo + width * t
    w = torch.ones(int(n), 1, dtype=lo.dtype, device=lo.device)
    w[0] = w[-1] = 0.5
    return x, torch.log(w * (width / (int(n) - 1)))


def sorted_trapezoid_log_weights(x):
    """Log trapezoid weights of nodes ``x [N, B]`` sorted along dim 0."""
    dx = (x[1:] - x[:-1]).clamp_min(0.0)
    w = torch.zeros_like(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return torch.log(w.clamp_min(1e-30))


def combine_windows(lo_p, hi_p, lo_m, hi_m, informative, bounds=None):
    """Integration window of predictive x measurement (each a window of its mass).

    A measurement window narrower than the predictive one contains the
    product's mass (the likelihood is negligible outside it, and the product
    is no wider than it), wherever it sits, including the predictive's far
    tail: it is used as is. Otherwise (a broad measurement) the product lies
    inside the predictive window: the intersection, or, without overlap, the
    gap between the windows plus the narrower one. Rows without an informative
    measurement: the predictive window. Returns (lo, hi, tension) with
    ``tension`` marking rows whose windows do not overlap.
    """
    if bounds is not None:
        lo_p, hi_p = lo_p.clamp(*bounds), hi_p.clamp(*bounds)
        lo_m, hi_m = lo_m.clamp(*bounds), hi_m.clamp(*bounds)
    lo_i, hi_i = torch.maximum(lo_p, lo_m), torch.minimum(hi_p, hi_m)
    overlap = hi_i > lo_i
    narrow_p = (hi_p - lo_p) <= (hi_m - lo_m)
    f_lo = torch.minimum(torch.minimum(hi_p, hi_m), torch.where(narrow_p, lo_p, lo_m))
    f_hi = torch.maximum(torch.maximum(lo_p, lo_m), torch.where(narrow_p, hi_p, hi_m))
    narrow_m = (hi_m - lo_m) < (hi_p - lo_p)
    lo = torch.where(overlap, lo_i, f_lo)
    hi = torch.where(overlap, hi_i, f_hi)
    lo = torch.where(informative, torch.where(narrow_m, lo_m, lo), lo_p)
    hi = torch.where(informative, torch.where(narrow_m, hi_m, hi), hi_p)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo).clamp_min(MIN_WIDTH)
    return mid - half, mid + half, informative & ~overlap


def parallax_window(plx, sig, has):
    """s window of the parallax likelihood (h within K_MEAS sig of plx).

    ``informative`` is False where the parallax is missing or where every
    positive model parallax is more than K_MEAS errors above the measurement
    (the likelihood is then monotone over the box).
    """
    up = plx + K_MEAS * sig
    dn = plx - K_MEAS * sig
    informative = has & (up > H_MIN)
    lo = -torch.log(up.clamp_min(H_MIN))
    hi = torch.where(
        dn > H_MIN, -torch.log(dn.clamp_min(H_MIN)), torch.full_like(dn, S_BOUNDS[1])
    )
    return lo, hi, informative


def _marginal_mixture(mix, j):
    """1-D marginal mixture of coordinate ``j`` (torchregress StudentTMixture)."""
    from torchregress.losses import StudentTMixture

    return StudentTMixture(
        mix.log_weights, mix.loc[..., j : j + 1],
        scale_diag=mix.component_scale[..., j : j + 1], df=mix.df,
    )  # fmt: skip


def _mixture_window(loc, scale, logw):
    """Window ``[min_k mu - U sd, max_k mu + U sd]`` over components with weight."""
    big = torch.finfo(loc.dtype).max
    use = logw > LOGW_MIN
    lo = torch.where(use, loc - U_TAIL * scale, torch.full_like(loc, big)).amin(-1)
    hi = torch.where(use, loc + U_TAIL * scale, torch.full_like(loc, -big)).amax(-1)
    return lo, hi


class Member(nn.Module):
    """One ensemble member: an MLP trunk on the standardized latent and a joint
    density of (s, E) (``kind``: ``gmix`` or ``flow``), plus the fitted
    parallax-error inflation ``kappa_g`` per G bin."""

    def __init__(self, kind, in_dim, *, hidden=(256, 256), context=64, transforms=3,
                 flow_hidden=(128, 128), bins=16, s_bound=7.5, e_bound=8.0,
                 components=4, e_loc=0.0, e_scale=1.0, fit_kappa=True):  # fmt: skip
        super().__init__()
        from torchregress.losses import MixtureDensityLoss, NormalizingFlowLoss
        from torchregress.losses.nflows import create_flow_model

        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if not e_scale > 0:
            raise ValueError("e_scale must be positive")
        self.kind = kind
        self.config = dict(kind=kind, in_dim=int(in_dim), hidden=list(hidden),
                           context=int(context), transforms=int(transforms),
                           flow_hidden=list(flow_hidden), bins=int(bins),
                           s_bound=float(s_bound), e_bound=float(e_bound),
                           components=int(components), e_loc=float(e_loc),
                           e_scale=float(e_scale), fit_kappa=bool(fit_kappa))  # fmt: skip
        self.e_loc, self.e_scale = float(e_loc), float(e_scale)
        if kind == "flow":
            self.trunk = mlp(in_dim, hidden, context)
            kw = dict(
                n_transforms=transforms, hidden_features=list(flow_hidden), bins=bins
            )
            self.flow_s = NormalizingFlowLoss(
                create_flow_model(1, context, "nsf", bound=s_bound, **kw),
                reduction="none",
            )
            self.flow_e = NormalizingFlowLoss(
                create_flow_model(1, context + 1, "nsf", bound=e_bound, **kw),
                reduction="none",
            )
        else:
            self.density = MixtureDensityLoss(
                int(components), 2, "full", full_parameterization="covariance",
                df=math.inf, learn_df=False, reduction="none",
            )  # fmt: skip
            self.trunk = mlp(in_dim, hidden, self.density.expected_output_size)
        raw0 = math.log(math.expm1(1.0 - KAPPA_MIN))  # kappa = 1 at start
        raw = torch.full((len(KAPPA_G_EDGES) + 1,), raw0)
        if fit_kappa:
            self.kappa_raw = nn.Parameter(raw)
        else:
            self.register_buffer("kappa_raw", raw)

    # ── parameters ──────────────────────────────────────────────────────
    def forward(self, x):
        """Context (flow) or mixture parameters (gmix) of standardized latents."""
        return self.trunk(x)

    def kappa(self):
        return KAPPA_MIN + nn.functional.softplus(self.kappa_raw)

    def sigma_eff(self, sig, gbin, zp_sigma=0.0):
        k = self.kappa()[gbin]
        return torch.sqrt((k * sig).square() + float(zp_sigma) ** 2)

    def _e(self, E):
        return (E - self.e_loc) / self.e_scale

    def _E(self, e):
        return e * self.e_scale + self.e_loc

    def mixture(self, p):
        return self.density.distribution(p)

    def _cond_e(self, p, s):
        """flow: the distribution of e | (s, c) for every node, batch ``[N * B]``."""
        n, b = s.shape
        ctx = torch.cat([p.unsqueeze(0).expand(n, b, p.shape[-1]), s.unsqueeze(-1)], -1)
        return self.flow_e.distribution(ctx.reshape(n * b, -1))

    # ── densities (physical units: s, E) ────────────────────────────────
    def log_prob_s(self, p, s):
        """Exact marginal log density of s at nodes ``s [N, B]``."""
        if self.kind == "flow":
            return self.flow_s.distribution(p).log_prob(s.unsqueeze(-1))
        return _marginal_mixture(self.mixture(p), 0).log_prob(s.unsqueeze(-1))

    def cdf_s(self, p, s):
        """Exact marginal CDF of s at nodes ``s [N, B]``."""
        if self.kind == "flow":
            u = self.flow_s.distribution(p).transform(s.unsqueeze(-1))[..., 0]
            return _ndtr(u)
        return _marginal_mixture(self.mixture(p), 0).marginal_cdf(s.unsqueeze(-1))[
            ..., 0
        ]

    def log_prob_joint(self, p, s, E):
        """log p(s_i, E_j) ``[Ns, Ne, B]`` at ``s [Ns, B]`` and ``E [Ne, B]``."""
        ns, b = s.shape
        ne = E.shape[0]
        e = self._e(E)
        if self.kind == "flow":
            lps = self.log_prob_s(p, s)
            d = self._cond_e(p, s)
            ee = e.unsqueeze(1).expand(ne, ns, b).reshape(ne, ns * b, 1)
            lpe = d.log_prob(ee).reshape(ne, ns, b).transpose(0, 1)
            return lps.unsqueeze(1) + lpe - math.log(self.e_scale)
        pts = torch.stack([s.unsqueeze(1).expand(ns, ne, b),
                           e.unsqueeze(0).expand(ns, ne, b)], -1)  # fmt: skip
        return self.mixture(p).log_prob(pts) - math.log(self.e_scale)

    def label_log_prob_s(self, p, s, e_obs, e_sig, has_e, n_e=24):
        """``log p(s) + log int p(E | s) N(E_obs | E, e_sig^2) dE`` at ``s [N, B]``.

        Rows without a label (``has_e`` False) give ``log p(s)`` (E
        marginalised exactly). Physical units (densities per kpc-log and per
        mag).
        """
        log_scale = math.log(self.e_scale)
        if self.kind == "gmix":
            mix = self.mixture(p)
            e = torch.where(has_e, self._e(e_obs), torch.zeros_like(e_obs))
            var = torch.where(
                has_e, (e_sig / self.e_scale).square(), torch.ones_like(e_sig)
            )
            design = torch.tensor([[0.0, 1.0]], dtype=e.dtype, device=e.device)
            comp = mix.component_observation_log_prob(
                e[:, None],
                design.reshape(1, 1, 1, 2),
                var[:, None],
                valid=has_e[:, None],
            )
            log_pe = torch.logsumexp(mix.log_weights + comp, -1)
            nan = torch.full_like(e, math.nan)
            cond = mix.condition(
                1, torch.where(has_e, e, nan), torch.where(has_e, var, nan)
            )
            lp = _marginal_mixture(cond, 0).log_prob(s.unsqueeze(-1))
            return lp + torch.where(has_e, log_pe - log_scale, torch.zeros_like(log_pe))
        lps = self.log_prob_s(p, s)
        idx = torch.nonzero(has_e).reshape(-1)
        if not len(idx):
            return lps
        ss = s[:, idx]
        ns, b = ss.shape
        d = self._cond_e(p[idx], ss)
        eo = self._e(e_obs[idx]).repeat(ns)
        sg = (e_sig[idx] / self.e_scale).repeat(ns)
        # int p(e | s) N(e_obs | e, sg^2) de = int phi(u) N(e_obs | e(u), sg^2) du on
        # the label window mapped to the base coordinate u (cut by +-U_TAIL):
        # smooth in u, whereas the spline density has kinks at its knots.
        u, logw = self._base_nodes(d, eo - K_MEAS * sg, eo + K_MEAS * sg, n_e)
        nodes = d.transform.inv(u.unsqueeze(-1))[..., 0]
        lphi = -0.5 * u.square() - 0.5 * _LOG_2PI
        term = torch.logsumexp(logw + lphi + _normal_logpdf(eo, nodes, sg), 0)
        out = lps.clone()
        out[:, idx] = lps[:, idx] + term.reshape(ns, b) - log_scale
        return out

    @staticmethod
    def _base_nodes(dist, lo, hi, n):
        """flow: trapezoid nodes in the base coordinate u over the window
        ``[lo, hi]`` (target units) combined with the predictive ``|u| <=
        U_TAIL`` by ``combine_windows``."""
        with torch.no_grad():
            ue = dist.transform(torch.stack([lo, hi]).unsqueeze(-1))[..., 0]
            full = torch.full_like(lo, U_TAIL)
            u_lo, u_hi, _ = combine_windows(-full, full, ue[0], ue[1],
                                            torch.ones_like(lo, dtype=torch.bool))  # fmt: skip
        return uniform_nodes(u_lo, u_hi, n)

    def quadrature_nodes(self, p, lo, hi, n):
        """Nodes ``s [n, B]`` and log weights such that ``sum exp(logw) f(s)`` is
        ``int_lo^hi f(s) ds`` for integrands ``f = p(s) g(s)``.

        gmix: uniform in s (the Gaussian mixture is analytic: the trapezoid rule
        converges geometrically). flow: uniform in the base coordinate
        ``u = T(s)`` over ``[T(lo), T(hi)]`` with weights ``w_u / T'(s)``, so
        that ``p(s) ds = phi(u) du``: the spline density is only continuous at
        its knots (O(h^2) in s), the integrand in u is continuously
        differentiable.
        """
        if self.kind == "gmix":
            return uniform_nodes(lo, hi, n)
        d = self.flow_s.distribution(p)
        u, logw = self._base_nodes(d, lo, hi, n)
        s = d.transform.inv(u.unsqueeze(-1))[..., 0]
        lphi = -0.5 * u.square() - 0.5 * _LOG_2PI
        return s, logw + lphi - d.log_prob(s.unsqueeze(-1))

    # ── windows (detached) ──────────────────────────────────────────────
    @staticmethod
    def _inverse_window(dist, n, like):
        """flow: x at base quantiles -+U_TAIL (the predictive window), ``[n]`` each."""
        u = torch.tensor([-U_TAIL, U_TAIL], dtype=like.dtype, device=like.device)
        x = dist.transform.inv(u.reshape(2, 1, 1).expand(2, n, 1))[..., 0]
        return x[0], x[1]

    @torch.no_grad()
    def s_window(self, p, e_obs=None, e_sig=None, has_e=None):
        """Predictive window of s per row (``U_TAIL`` base-space half-width).

        gmix with labels: the window of the label-conditioned mixture.
        """
        if self.kind == "flow":
            d = self.flow_s.distribution(p)
            lo, hi = self._inverse_window(d, p.shape[0], p)
        else:
            mix = self.mixture(p)
            if has_e is not None and bool(has_e.any()):
                e = torch.where(has_e, self._e(e_obs), torch.full_like(e_obs, math.nan))
                var = torch.where(has_e, (e_sig / self.e_scale).square(),
                                  torch.full_like(e_sig, math.nan))  # fmt: skip
                mix = mix.condition(1, e, var)
            lo, hi = _mixture_window(mix.loc[..., 0], mix.component_scale[..., 0],
                                     mix.log_weights)  # fmt: skip
        return lo.clamp(*S_BOUNDS), hi.clamp(*S_BOUNDS)

    @torch.no_grad()
    def e_window_given_s(self, p, s):
        """Predictive window of E | s (physical) at nodes ``s [N, B]`` -> ``[N, B]``."""
        n, b = s.shape
        if self.kind == "flow":
            d = self._cond_e(p, s)
            lo, hi = self._inverse_window(d, n * b, s)
            return self._E(lo.reshape(n, b)), self._E(hi.reshape(n, b))
        mix = self.mixture(p)
        L = mix.scale_tril
        cov = L @ L.mT  # [B, K, 2, 2]
        a, c, v = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
        slope = c / a
        m = mix.loc[..., 1].unsqueeze(0) + slope.unsqueeze(0) * (
            s.unsqueeze(-1) - mix.loc[..., 0].unsqueeze(0)
        )
        sd = torch.sqrt((v - c * slope).clamp_min(1e-12)).unsqueeze(0).expand_as(m)
        lo, hi = _mixture_window(m, sd, mix.log_weights.unsqueeze(0).expand_as(m))
        return self._E(lo), self._E(hi)


def observation_tensors(obs, rows, device):
    """Per-row observation tensors (float32) of table ``rows``.

    ``obs``: arrays per table row: ``parallax_corrected`` (plx - zp, mas),
    ``parallax_error``, ``kappa_bin``, ``ebv``, ``e_ebv`` (NaN where missing).
    """
    f32 = dict(dtype=torch.float32, device=device)
    plx = torch.as_tensor(np.asarray(obs["parallax_corrected"])[rows], **f32)
    sig = torch.as_tensor(np.asarray(obs["parallax_error"])[rows], **f32)
    e = torch.as_tensor(np.asarray(obs["ebv"])[rows], **f32)
    es = torch.as_tensor(np.asarray(obs["e_ebv"])[rows], **f32)
    has_plx = torch.isfinite(plx) & torch.isfinite(sig) & (sig > 0)
    has_e = torch.isfinite(e) & torch.isfinite(es) & (es > 0)
    return dict(
        plx=torch.where(has_plx, plx, torch.zeros_like(plx)),
        sig=torch.where(has_plx, sig, torch.ones_like(sig)),
        gbin=torch.as_tensor(np.asarray(obs["kappa_bin"])[rows], dtype=torch.long,
                             device=device),
        e_obs=torch.where(has_e, e, torch.zeros_like(e)),
        e_sig=torch.where(has_e, es, torch.ones_like(es)),
        has_plx=has_plx, has_e=has_e,
    )  # fmt: skip


def subset(obs_t, idx):
    return {k: v[idx] for k, v in obs_t.items()}


def log_marginal_likelihood(member, p, ob, *, n_s=48, n_e=24, zp_sigma=0.0,
                            use_plx=True, use_e=True, return_fallback=False):  # fmt: skip
    """Per-row log p(observations | z) [B] of one member (physical units).

    ``use_plx`` / ``use_e`` drop the parallax / E-label factor (e.g. the
    parallax-only score ``log p(plx | z)``). Rows with neither observation
    give log 1 = 0 (callers mask them out of a loss).
    """
    has_plx = ob["has_plx"] & bool(use_plx)
    has_e = ob["has_e"] & bool(use_e)
    sig_eff = member.sigma_eff(ob["sig"], ob["gbin"], zp_sigma)
    with torch.no_grad():
        lo_p, hi_p = member.s_window(p, ob["e_obs"], ob["e_sig"], has_e)
        lo_m, hi_m, inf = parallax_window(ob["plx"], sig_eff.detach(), has_plx)
        lo, hi, fb = combine_windows(lo_p, hi_p, lo_m, hi_m, inf, S_BOUNDS)
    s, logw = member.quadrature_nodes(p, lo, hi, n_s)
    ll = torch.where(has_plx, parallax_log_likelihood(s, ob["plx"], sig_eff),
                     torch.zeros_like(s))  # fmt: skip
    lab = member.label_log_prob_s(p, s, ob["e_obs"], ob["e_sig"], has_e, n_e)
    out = torch.logsumexp(logw + ll + lab, 0)
    return (out, fb) if return_fallback else out


@torch.no_grad()
def parallax_pit(member, p, ob, *, n_s=96, zp_sigma=0.0):
    """P(plx_rep <= plx_obs | z) per row under the member's convolved predictive.

    ``int (1 - F_s(s)) N(plx | e^-s, sig) e^-s ds`` on the union of the
    predictive and measurement windows, plus closed-form tails beyond them
    (``1 - F`` held at its edge value). NaN without a parallax.
    """
    sig = member.sigma_eff(ob["sig"], ob["gbin"], zp_sigma)
    plx = ob["plx"]
    lo_p, hi_p = member.s_window(p)
    lo_m, hi_m, inf = parallax_window(plx, sig, ob["has_plx"])
    lo_m = torch.where(inf, lo_m, lo_p).clamp(*S_BOUNDS)
    hi_m = torch.where(inf, hi_m, hi_p).clamp(*S_BOUNDS)
    s1, _ = uniform_nodes(lo_p, torch.maximum(hi_p, lo_p + MIN_WIDTH), n_s)
    s2, _ = uniform_nodes(lo_m, torch.maximum(hi_m, lo_m + MIN_WIDTH), n_s)
    s = torch.sort(torch.cat([s1, s2], 0), 0).values
    logw = sorted_trapezoid_log_weights(s)
    surv = (1.0 - member.cdf_s(p, s)).clamp(0.0, 1.0)
    kern = parallax_log_likelihood(s, plx, sig) - s
    body = torch.exp(logw + kern) * surv
    pit = body.sum(0)
    h_lo, h_hi = torch.exp(-s[-1]), torch.exp(-s[0])
    pit = pit + surv[-1] * (_ndtr((h_lo - plx) / sig) - _ndtr(-plx / sig)).clamp_min(0)
    pit = pit + surv[0] * (1.0 - _ndtr((h_hi - plx) / sig)).clamp_min(0)
    return torch.where(
        ob["has_plx"], pit.clamp(0.0, 1.0), torch.full_like(pit, math.nan)
    )


# ── training ────────────────────────────────────────────────────────────────


def train_member(member, std, z_train, obs_train, z_stop, obs_stop, *, n_s=48, n_e=24,
                 zp_sigma=0.0, batch=2048, max_epochs=20, patience=3, lr=1e-3,
                 weight_decay=1e-4, seed=0, eval_batch=8192, log=print):  # fmt: skip
    """Fit one member by maximum marginal likelihood; early stopping on the stop set.

    ``z_*``: raw latents (torch, e.g. float16, on the training device);
    ``obs_*``: ``observation_tensors`` dicts on the same device. The loss is
    the mean per-star -log p(observations | z) over rows with a parallax or
    an E label. Returns the history; ``member`` holds the best state.
    """
    dev = z_train.device
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    kap = [p for n, p in member.named_parameters() if n == "kappa_raw"]
    rest = [p for n, p in member.named_parameters() if n != "kappa_raw"]
    groups = [dict(params=rest, lr=lr, weight_decay=weight_decay)]
    if kap:
        groups.append(dict(params=kap, lr=lr, weight_decay=0.0))
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    use_tr = torch.nonzero(obs_train["has_plx"] | obs_train["has_e"]).reshape(-1).cpu()
    use_st = torch.nonzero(obs_stop["has_plx"] | obs_stop["has_e"]).reshape(-1).to(dev)

    def rows_nll(z, ob):
        p = member(std.latent(z))
        return -log_marginal_likelihood(
            member, p, ob, n_s=n_s, n_e=n_e, zp_sigma=zp_sigma
        )

    def stop_loss():
        member.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(use_st), eval_batch):
                idx = use_st[i : i + eval_batch]
                nll = rows_nll(z_stop[idx], subset(obs_stop, idx))
                ok = torch.isfinite(nll)
                tot += float(nll[ok].sum())
                cnt += int(ok.sum())
        member.train()
        return tot / max(cnt, 1)

    best, best_state, bad = math.inf, None, 0
    hist = dict(train=[], valid=[], lr=[], skipped=[], seconds=[], kappa=[])
    t_start = time.time()
    member.train()
    for epoch in range(max_epochs):
        t0 = time.time()
        perm = use_tr[torch.randperm(len(use_tr), generator=gen)].to(dev)
        tot, cnt, skipped = 0.0, 0, 0
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            nll = rows_nll(z_train[idx], subset(obs_train, idx))
            loss = nll.mean()
            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(member.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        v = stop_loss()
        sched.step(v)
        hist["train"].append(tot / max(cnt, 1))
        hist["valid"].append(v)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["skipped"].append(skipped)
        hist["seconds"].append(time.time() - t0)
        hist["kappa"].append(member.kappa().detach().cpu().tolist())
        log(f"  epoch {epoch}: train {hist['train'][-1]:.4f} stop {v:.4f} "
            f"({time.time() - t0:.0f} s{', skipped ' + str(skipped) if skipped else ''})")  # fmt: skip
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = copy.deepcopy(member.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        member.load_state_dict(best_state)
    member.eval()
    hist.update(best_valid=best, epochs=len(hist["valid"]), n_train=int(len(use_tr)),
                n_valid=int(len(use_st)), batch_size=int(batch), seed=int(seed),
                n_s=int(n_s), n_e=int(n_e), kappa_final=member.kappa().detach().cpu().tolist(),
                train_seconds=time.time() - t_start)  # fmt: skip
    return hist


def save_member(module, path, std, **extra):
    torch.save(dict(config=module.config, state_dict=module.state_dict(),
                    standardizer=std.state(), **extra), path)  # fmt: skip


def load_member(path, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(blob["config"])
    member = Member(cfg.pop("kind"), cfg.pop("in_dim"), **cfg)
    member.load_state_dict(blob["state_dict"])
    member.to(device).eval().requires_grad_(False)
    return member, blob


# ── grid summaries ──────────────────────────────────────────────────────────


def _grid_cdf(x, lp):
    """Normalized cumulative trapezoid CDF ``[N, B]`` and density weights of
    unnormalized log densities ``lp [N, B]`` at sorted nodes ``x [N, B]``."""
    f = torch.exp(lp - lp.amax(0, keepdim=True))
    f = torch.nan_to_num(f, nan=0.0)
    seg = 0.5 * (f[1:] + f[:-1]) * (x[1:] - x[:-1])
    cdf = torch.cat([torch.zeros_like(x[:1]), torch.cumsum(seg, 0)], 0)
    tot = cdf[-1:].clamp_min(1e-300)
    return cdf / tot, f / tot


def grid_quantiles(x, cdf, f, levels):
    """Quantiles ``[B, Q]`` from a grid CDF and its normalized densities ``f``
    (second order: the density is linear within a segment, so the CDF is
    quadratic there, consistent with the trapezoid CDF at the nodes)."""
    c, xs, fs = cdf.T.contiguous(), x.T.contiguous(), f.T.contiguous()
    q = torch.as_tensor(levels, dtype=c.dtype, device=c.device).expand(c.shape[0], -1)
    i = torch.searchsorted(c, q.contiguous()).clamp(1, c.shape[1] - 1)
    c0, x0, x1 = c.gather(1, i - 1), xs.gather(1, i - 1), xs.gather(1, i)
    f0, f1 = fs.gather(1, i - 1), fs.gather(1, i)
    h = x1 - x0
    a, b = 0.5 * h * (f1 - f0), h * f0
    dq = (q - c0).clamp_min(0.0)
    disc = (b.square() + 4.0 * a * dq).clamp_min(0.0)
    t = (2.0 * dq / (b + disc.sqrt()).clamp_min(1e-30)).clamp(0.0, 1.0)
    return x0 + t * h


def interp_cdf(x, cdf, xq):
    """Linear interpolation of a grid CDF at ``xq [M, B]`` (0 below, 1 above)."""
    xs, c = x.T.contiguous(), cdf.T.contiguous()
    q = xq.T.contiguous()
    i = torch.searchsorted(xs, q).clamp(1, xs.shape[1] - 1)
    x0, x1 = xs.gather(1, i - 1), xs.gather(1, i)
    c0, c1 = c.gather(1, i - 1), c.gather(1, i)
    t = ((q - x0) / (x1 - x0).clamp_min(1e-30)).clamp(0.0, 1.0)
    out = c0 + t * (c1 - c0)
    out = torch.where(q < xs[:, :1], torch.zeros_like(out), out)
    out = torch.where(q > xs[:, -1:], torch.ones_like(out), out)
    return out.T


def _moments(x, wf):
    """Mean and sd of a grid density (trapezoid weights ``wf`` sum to ~1)."""
    m = (wf * x).sum(0)
    v = (wf * (x - m).square()).sum(0)
    return m, v.clamp_min(0).sqrt()


def _trap_mass(x, f):
    """Trapezoid masses per node of densities ``f`` at sorted nodes ``x``."""
    w = torch.zeros_like(x)
    dx = x[1:] - x[:-1]
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w * f


SUMMARY_Q = (0.16, 0.5, 0.84)


class EnsembleDE:
    """K members of one kind (equal-weight mixture) on the frozen latent."""

    def __init__(self, members, std, *, zp_sigma=0.0):
        kinds = {m.kind for m in members}
        if len(kinds) != 1:
            raise ValueError(f"members of one kind expected, got {kinds}")
        self.members, self.std = list(members), std
        self.kind = kinds.pop()
        self.zp_sigma = float(zp_sigma)

    def kappas(self):
        return np.array([m.kappa().detach().cpu().numpy() for m in self.members])

    def _ens(self, per_member):
        return torch.logsumexp(torch.stack(per_member), 0) - math.log(len(per_member))

    def _e_window(self, ps, s, mass):
        """Union over members of the E | s windows at nodes with mass."""
        big = torch.finfo(s.dtype).max
        keep = mass > 1e-7 * mass.amax(0, keepdim=True)
        lo = torch.full_like(s[0], big)
        hi = torch.full_like(s[0], -big)
        for m, p in zip(self.members, ps, strict=True):
            a, b = m.e_window_given_s(p, s)
            lo = torch.minimum(
                lo, torch.where(keep, a, torch.full_like(a, big)).amin(0)
            )
            hi = torch.maximum(
                hi, torch.where(keep, b, torch.full_like(b, -big)).amax(0)
            )
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo).clamp_min(MIN_WIDTH)
        return mid - half, mid + half

    def _joint(self, ps, s, E, extra=None):
        """Ensemble log p(s, E) ``[Ns, Ne, B]`` (+ per-member ``extra[k] [Ns, B]``)."""
        per = []
        for k, (m, p) in enumerate(zip(self.members, ps, strict=True)):
            lj = m.log_prob_joint(p, s, E)
            if extra is not None:
                lj = lj + extra[k].unsqueeze(1)
            per.append(lj)
        return self._ens(per)

    def _s_grid(self, ps, lo, hi, n, extra=None):
        """Ensemble log p(s) (+ per-member ``extra[k](s)``) on ``n`` uniform nodes."""
        s, _ = uniform_nodes(lo, torch.maximum(hi, lo + MIN_WIDTH), n)
        per = []
        for k, (m, p) in enumerate(zip(self.members, ps, strict=True)):
            lp = m.log_prob_s(p, s)
            per.append(lp if extra is None else lp + extra[k](s))
        return s, self._ens(per)

    def _e_marginal(self, ps, s, mass_s, n_e, extra=None):
        """E marginal of the ensemble joint on s nodes x E nodes: (E nodes, CDF,
        normalized densities, node masses [Ne, B], joint masses [Ns, Ne, B])."""
        e_lo, e_hi = self._e_window(ps, s, mass_s)
        E, _ = uniform_nodes(e_lo, e_hi, n_e)
        ex = None if extra is None else [g(s) for g in extra]
        lj = self._joint(ps, s, E, extra=ex)
        ws = _trap_mass(s, torch.ones_like(s)).unsqueeze(1)
        we = _trap_mass(E, torch.ones_like(E))
        pj = torch.exp(lj - lj.amax((0, 1), keepdim=True)) * ws * we.unsqueeze(0)
        pj = pj / pj.sum((0, 1), keepdim=True).clamp_min(1e-300)
        pe = pj.sum(0)
        cdf, f = _grid_cdf(E, torch.log((pe / we.clamp_min(1e-30)).clamp_min(1e-300)))
        return E, cdf, f, pe, pj

    @torch.no_grad()
    def predict(self, z, ob, *, n_fine=256, n_joint=48, n_e=48, n_s_lik=40, n_e_lik=24,
                device="cpu"):  # fmt: skip
        """Per-row summaries, scores and PITs for raw latents ``z`` [B] (dict of
        numpy arrays; NaN where an observation is missing).

        s marginals (predictive and parallax-combined) on ``n_fine`` nodes; the
        E marginals and the (s, E) correlation on ``n_joint`` x ``n_e`` grids;
        scores with the training quadrature (``n_s_lik``, ``n_e_lik``).
        """
        x = self.std.latent(z, device)
        ps = [m(x) for m in self.members]
        has_plx, has_e = ob["has_plx"], ob["has_e"]
        B = x.shape[0]
        nan = torch.full((B,), math.nan, device=x.device)
        res = {}
        # Predictive s marginal on the union of the members' windows.
        wins = [m.s_window(p) for m, p in zip(self.members, ps, strict=True)]
        lo = torch.stack([w[0] for w in wins]).amin(0)
        hi = torch.stack([w[1] for w in wins]).amax(0)
        s, lps = self._s_grid(ps, lo, hi, n_fine)
        cdf_s, f_s = _grid_cdf(s, lps)
        res["s_q"] = grid_quantiles(s, cdf_s, f_s, SUMMARY_Q)
        mass_s = _trap_mass(s, f_s)
        res["s_mean"], res["s_sd"] = _moments(s, mass_s)
        h = torch.exp(-s)
        res["plx_mean"] = (mass_s * h).sum(0)
        res["plx_sd"] = (
            (mass_s * (h - res["plx_mean"]).square()).sum(0).clamp_min(0).sqrt()
        )
        # Joint grid: E marginal and the (s, E) correlation.
        sj, lpj = self._s_grid(ps, lo, hi, n_joint)
        _, fj = _grid_cdf(sj, lpj)
        E, cdf_e, f_e, pe, pj = self._e_marginal(ps, sj, _trap_mass(sj, fj), n_e)
        res["e_q"] = grid_quantiles(E, cdf_e, f_e, SUMMARY_Q)
        e_mean = (pe * E).sum(0)
        e_sd = (pe * (E - e_mean).square()).sum(0).clamp_min(0).sqrt()
        ds = sj.unsqueeze(1) - (pj * sj.unsqueeze(1)).sum((0, 1))
        sd_sj = (pj * ds.square()).sum((0, 1)).clamp_min(1e-30).sqrt()
        cov = (pj * ds * (E.unsqueeze(0) - e_mean)).sum((0, 1))
        res["rho"] = cov / (sd_sj * e_sd.clamp_min(1e-30))
        res["e_mean"], res["e_sd"] = e_mean, e_sd
        # Scores: log p(observations), parallax alone, E label alone; PITs.
        lik = dict(n_s=n_s_lik, n_e=n_e_lik, zp_sigma=self.zp_sigma)
        lo_obs, lo_plx, lo_e, pits, fbs = [], [], [], [], []
        for m, p in zip(self.members, ps, strict=True):
            a, fb = log_marginal_likelihood(m, p, ob, return_fallback=True, **lik)
            lo_obs.append(a)
            fbs.append(fb)
            lo_plx.append(log_marginal_likelihood(m, p, ob, use_e=False, **lik))
            lo_e.append(log_marginal_likelihood(m, p, ob, use_plx=False, **lik))
            pits.append(parallax_pit(m, p, ob, n_s=n_fine // 2, zp_sigma=self.zp_sigma))
        any_obs = has_plx | has_e
        res["lp_obs"] = torch.where(any_obs, self._ens(lo_obs), nan)
        res["lp_plx"] = torch.where(has_plx, self._ens(lo_plx), nan)
        res["lp_e"] = torch.where(has_e, self._ens(lo_e), nan)
        res["pit_plx"] = torch.stack(pits).mean(0)
        res["window_fallback"] = torch.stack(fbs).any(0) & has_plx
        # PIT of the E label: int F_E(E) N(E | E_obs, e^2) dE (+ upper tail).
        eo, es = ob["e_obs"], ob["e_sig"]
        lab, _ = uniform_nodes(eo - K_MEAS * es, eo + K_MEAS * es, n_e)
        ee = torch.sort(torch.cat([E, lab], 0), 0).values
        fe = interp_cdf(E, cdf_e, ee)
        w = sorted_trapezoid_log_weights(ee)
        pit_e = (torch.exp(w + _normal_logpdf(ee, eo, es)) * fe).sum(0)
        pit_e = pit_e + (1.0 - _ndtr((ee[-1] - eo) / es))
        res["pit_e"] = torch.where(has_e, pit_e.clamp(0, 1), nan)
        # Spectrophotometric + Gaia parallax: predictive x measurement.
        sigs = [m.sigma_eff(ob["sig"], ob["gbin"], self.zp_sigma) for m in self.members]
        lo_m, hi_m, inf = parallax_window(ob["plx"], torch.stack(sigs).amax(0), has_plx)
        c_lo, c_hi, _ = combine_windows(lo, hi, lo_m, hi_m, inf, S_BOUNDS)
        lik_s = [
            lambda s_, sg=sg: parallax_log_likelihood(s_, ob["plx"], sg) for sg in sigs
        ]
        sc, lc = self._s_grid(ps, c_lo, c_hi, n_fine, extra=lik_s)
        cdf_c, f_c = _grid_cdf(sc, lc)
        cq = grid_quantiles(sc, cdf_c, f_c, SUMMARY_Q)
        scj, lcj = self._s_grid(ps, c_lo, c_hi, n_joint, extra=lik_s)
        _, fcj = _grid_cdf(scj, lcj)
        Ec, cdf_ec, f_ec, _, _ = self._e_marginal(ps, scj, _trap_mass(scj, fcj), n_e,
                                                  extra=lik_s)  # fmt: skip
        ceq = grid_quantiles(Ec, cdf_ec, f_ec, SUMMARY_Q)
        keep = has_plx.unsqueeze(1)
        res["comb_s_q"] = torch.where(keep, cq, torch.full_like(cq, math.nan))
        res["comb_e_q"] = torch.where(keep, ceq, torch.full_like(ceq, math.nan))
        return {k: v.float().cpu().numpy() for k, v in res.items()}

    @torch.no_grad()
    def grids(self, z, ob, *, n_s=96, n_e=64, device="cpu"):
        """Ensemble predictive and parallax-combined joint grids (examples)."""
        x = self.std.latent(z, device)
        ps = [m(x) for m in self.members]
        wins = [m.s_window(p) for m, p in zip(self.members, ps, strict=True)]
        lo = torch.stack([w[0] for w in wins]).amin(0)
        hi = torch.stack([w[1] for w in wins]).amax(0)
        s, _ = uniform_nodes(lo, torch.maximum(hi, lo + MIN_WIDTH), n_s)
        lps = self._ens(
            [m.log_prob_s(p, s) for m, p in zip(self.members, ps, strict=True)]
        )
        _, f_s = _grid_cdf(s, lps)
        e_lo, e_hi = self._e_window(ps, s, _trap_mass(s, f_s))
        E, _ = uniform_nodes(e_lo, e_hi, n_e)
        lj = self._joint(ps, s, E)
        sigs = [m.sigma_eff(ob["sig"], ob["gbin"], self.zp_sigma) for m in self.members]
        lls = [torch.where(ob["has_plx"], parallax_log_likelihood(s, ob["plx"], sg),
                           torch.zeros_like(s)) for sg in sigs]  # fmt: skip
        ljc = self._joint(ps, s, E, extra=lls)
        return {k: v.float().cpu().numpy() for k, v in
                dict(s=s, E=E, log_p=lj, log_p_comb=ljc).items()}  # fmt: skip


@torch.no_grad()
def quadrature_check(ens, z, ob, *, n_s=48, n_e=24, factor=4, device="cpu"):
    """|log L(n nodes) - log L(factor x n nodes)| per row and member (max over
    members) for the training objective: the discretisation error on real rows."""
    x = ens.std.latent(z, device)
    out = torch.zeros(x.shape[0], device=x.device)
    for m in ens.members:
        p = m(x)
        a = log_marginal_likelihood(m, p, ob, n_s=n_s, n_e=n_e, zp_sigma=ens.zp_sigma)
        b = log_marginal_likelihood(m, p, ob, n_s=factor * n_s, n_e=factor * n_e,
                                    zp_sigma=ens.zp_sigma)  # fmt: skip
        out = torch.maximum(out, (a - b).abs())
    use = ob["has_plx"] | ob["has_e"]
    return out[use].cpu().numpy()
