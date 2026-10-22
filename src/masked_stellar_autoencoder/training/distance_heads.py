"""Log-distance head and the convolved raw-parallax likelihood.

Latent coordinate ``s = log(d / kpc)``. The model parallax is therefore

    h(s) = exp(-s)   [mas]

so ``s = 0`` is exactly 1 mas at exactly 1 kpc and ``s = log(10)`` is 0.1 mas at
10 kpc. There is no 1000x factor anywhere: 1 mas corresponds to 1 kpc, not to
1 pc. The measured parallax is *never* inverted, sign-cut or clipped; it enters
only through the Gaussian residual and through the change of variables below.

The training objective is the marginal likelihood of the *raw signed* parallax,

    L_A = -sum_i log int p_theta(s | x_i) N(plx_i | z_i + exp(-s), sigma_eff,i) ds

with ``sigma_eff^2 = kappa^2 sigma_quoted^2 + sigma_zp^2`` (no variance floor,
no clipping). Two fixed Gauss-Hermite rules implement the integral, both with
``logsumexp`` over nodes so that a star whose likelihood is only constrained in
the far tail still gives a finite log-density:

``gh_prior``
    nodes in the prior coordinate, ``s_k = mu + sigma_s u_k``. This is the
    quadrature prescribed by the plan. It is accurate when the likelihood is
    broad compared with the prior, i.e. when ``sigma_eff / (sigma_s h)`` is not
    small.

``gh_residual``
    nodes in the standardized parallax residual, ``h_k = (plx - z) - sigma_eff
    r_k``, ``s_k = -log h_k``, with the exact change-of-variables Jacobian
    ``|ds/dh| = 1/h``. This rule is *exact* in the narrow-error limit
    ``sigma_eff -> 0`` (the limit is ``p_theta(s*) / h*``) and is smooth and
    rapidly convergent when ``sigma_eff << h*``. It requires ``h_k > 0`` at
    every node, which fails for non-positive parallaxes and for
    ``sigma_eff > h*/6.63`` (order 16).

``nodes`` selects the node set: ``"gh"`` (the fixed Gauss-Hermite rules above,
default) or ``"qmc"`` (a saved antithetic Sobol set, ``QMC_DEFAULT_COUNT=64``,
``seed=2026``). The QMC weights are ``sqrt(2 pi)/count``, so it is the same
Gaussian expectation and the residual rule is still exact in the narrow limit;
it is a *fallback* -- measured agreement with GH-256 is 4e-2 at
``sigma_eff = h*`` and 1e-3 at ``1.2 h*`` on the reference case -- and it exists
so a non-Gaussian predictive (flow sample set) can use the same code path.

``auto`` (default) uses ``gh_residual`` on the rows where it is representable
(``sigma_eff < h*/6.63``) and ``gh_prior`` above it. Measured against
``scipy.integrate.quad``: the residual rule is exact to ``<1e-9`` over its whole
domain, so the *only* approximation risk is in the prior branch, whose GH-32
error at the join (``sigma_eff = 0.15 h*``) is ``5e-9`` for ``sigma_s = 0.2``,
``5e-5`` for ``0.3``, ``8e-3`` for ``0.5`` and ``0.27`` for ``0.8``. That is why
``DistanceLoss.residual_rule_rows`` and ``DistanceLoss.min_prior_resolution``
(= min ``sigma_eff/(sigma_s h*)`` over prior-rule rows) are reported: a run must
record them and pass the GH16-vs-GH256 convergence gate on the *actual*
``sigma_eff/h`` population instead of assuming GH-32 in ``s`` is always enough
(plain GH-32 in ``s`` is off by 1.7e-3 at ``sigma_eff = 0.1 h*`` and by orders of
magnitude at ``1e-3 h*``).

Two products, named differently forever:

1. ``p_theta(s|x)`` -- a normalized *predictive* density (population
   dependent). Trained by ``convolved_parallax_nll``.
2. ``parallax_profile_confidence_set`` -- the assisted profile-likelihood
   confidence set ``C_alpha = {s: -2[l(s) - l(s_hat)] <= chi2_1(1-alpha)}`` with
   ``l(s) = log N(plx | z + exp(-s), sigma_eff)``. Because ``l`` is exactly a
   downward parabola in ``h`` the set is available in closed form; it may be
   *empty* or *unbounded* and is never cosmetically capped. Its complement
   ``p_theta(s) L(s) / Z`` is a posterior under a learned population prior, not
   a frequentist interval, and must be labelled as such.

``narrow_error_limit_nll`` and ``forward_gaussian_control`` are controls only
(the sigma_eff -> 0 limit and first-order propagation). They are not
likelihoods and must not be reported as the primary distance objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor, nn

#: Model parallax in mas at s = 0. Exactly 1.0 by the definition of the mas.
PARALLAX_MAS_AT_ONE_KPC = 1.0
#: log(d/kpc) at exactly 1 kpc.
LOG_DISTANCE_AT_ONE_KPC = 0.0
#: Allowed values of the quadrature-rule selector.
QUADRATURE_RULES = ("auto", "gh_prior", "gh_residual")
#: Allowed node sets: fixed Gauss-Hermite, or a saved fixed QMC set.
NODE_SETS = ("gh", "qmc")
#: Default size of the saved antithetic QMC set (QMC-64 per the plan).
QMC_DEFAULT_COUNT = 64
QMC_DEFAULT_SEED = 2026
#: The residual-space integrand is smooth in ``r`` over its whole representability
#: domain, so this rule converges by order 8. Capping it at 16 keeps the
#: representability boundary at ``sigma_eff < h*/6.6`` -- exactly the point where
#: the prior-space rule becomes accurate at order 32 -- instead of ``h*/10.1``.
RESIDUAL_RULE_MAX_ORDER = 16

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class DistanceLoss:
    """Numerator/count pair, field-compatible with ``joint_prediction.LossSum``.

    ``observation_count`` is the number of usable rows (one signed parallax per
    row), so ``mean`` is the per-star NLL. ``residual_rule_rows`` is a
    diagnostic that must be reported: it says how many rows used the
    exact-narrow-limit rule instead of the prior-space rule.
    """

    numerator: Tensor
    observation_count: int
    row_count: int
    residual_rule_rows: int = 0
    #: Smallest ``sigma_eff / (sigma_s h*)`` among the prior-rule rows, i.e. the
    #: least-resolved row in the batch (``inf`` when every row used the exact
    #: residual rule). Values below ~0.4 mean the prior-space GH-32 rule is
    #: under-resolved on those rows; record it, do not ignore it.
    min_prior_resolution: float = float("inf")

    @property
    def mean(self) -> Tensor:
        return self.numerator / max(1, self.observation_count)


# --------------------------------------------------------------------------- #
# Units and transforms
# --------------------------------------------------------------------------- #
def parallax_mas_from_log_distance(log_distance: Tensor) -> Tensor:
    """Return the model parallax ``exp(-s)`` in mas (``s = log(d/kpc)``).

    ``s = 0`` gives exactly 1 mas (1 kpc); ``s = log(10)`` gives 0.1 mas
    (10 kpc). Never call this on a *measured* parallax: measured values only
    ever enter the Gaussian residual, never a reciprocal.
    """
    if not log_distance.is_floating_point():
        raise TypeError("log distance must be floating point")
    predicted = torch.exp(-log_distance)
    if not torch.isfinite(predicted).all():
        raise ValueError(
            "model parallax overflowed; no silent clipping of log distance"
        )
    return predicted


def log_distance_from_parallax_mas(parallax_mas: Tensor) -> Tensor:
    """Return ``s = log(d/kpc)`` for a *strictly positive* model parallax.

    This is the change of variables used by the ``sigma_eff -> 0`` control. It
    is not a distance estimator: a measured parallax must not be passed here
    outside that declared control, because ``1/plx`` is undefined, biased and
    can be negative.
    """
    if not parallax_mas.is_floating_point():
        raise TypeError("parallax must be floating point")
    if not (torch.isfinite(parallax_mas) & (parallax_mas > 0)).all():
        raise ValueError(
            "log distance requires finite, strictly positive parallaxes; "
            "no clipping, sign cut or 1/plx fallback"
        )
    return -parallax_mas.log()


def effective_parallax_error_mas(
    sigma_quoted_mas: Tensor,
    *,
    kappa_parallax: float = 1.0,
    sigma_zp_mas: Tensor | float = 0.0,
) -> Tensor:
    """Return ``sqrt(kappa^2 sigma_quoted^2 + sigma_zp^2)`` in mas.

    ``kappa`` is a fitted, per-survey excess-noise factor; ``sigma_zp`` is the
    declared zero-point uncertainty (scalar or a per-star audited recipe). No
    floor is added: a non-positive or non-finite quoted error raises instead of
    being replaced.
    """
    if not sigma_quoted_mas.is_floating_point():
        raise TypeError("quoted parallax error must be floating point")
    if not math.isfinite(kappa_parallax) or kappa_parallax <= 0:
        raise ValueError("kappa_parallax must be finite and strictly positive")
    if not (torch.isfinite(sigma_quoted_mas) & (sigma_quoted_mas > 0)).all():
        raise ValueError(
            "quoted parallax errors must be finite and strictly positive; "
            "no error floor is substituted"
        )
    zp = torch.as_tensor(sigma_zp_mas, dtype=sigma_quoted_mas.dtype)
    if zp.device != sigma_quoted_mas.device:
        raise ValueError("zero-point uncertainty must share the parallax device")
    if not (torch.isfinite(zp) & (zp >= 0)).all():
        raise ValueError("zero-point uncertainty must be finite and non-negative")
    return ((kappa_parallax * sigma_quoted_mas).square() + zp.square()).sqrt()


# --------------------------------------------------------------------------- #
# Fixed Gauss-Hermite rule
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=16)
def _hermite_cpu(order: int) -> tuple[Tensor, Tensor]:
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("quadrature order must be an integer of at least two")
    node, weight = np.polynomial.hermite_e.hermegauss(order)
    if not (np.isfinite(node).all() and np.isfinite(weight).all()):
        raise FloatingPointError("non-finite Gauss-Hermite nodes or weights")
    if abs(weight.sum() - math.sqrt(2.0 * math.pi)) > 1e-12 * weight.sum():
        raise FloatingPointError("Gauss-Hermite weights do not integrate the Gaussian")
    return torch.from_numpy(node), torch.from_numpy(weight)


def gauss_hermite(
    order: int, *, dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return fixed probabilists' nodes/weights for ``int f(u) N(u) du``.

    The rule is ``sum_k w_k f(u_k) = int f(u) exp(-u^2/2) du``, so a Gaussian
    expectation is ``sum_k w_k f(u_k) / sqrt(2 pi)``. Nodes are fixed, saved
    constants; they are never adapted to the data.
    """
    if dtype in (torch.float16, torch.bfloat16):
        raise TypeError("parallax quadrature requires at least float32")
    node, weight = _hermite_cpu(order)
    return node.to(device=device, dtype=dtype), weight.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def _qmc_cpu(count: int, seed: int) -> tuple[Tensor, Tensor]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 8 or count % 2:
        raise ValueError("QMC node count must be an even integer of at least eight")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("QMC seed must be a non-negative integer")
    engine = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed)
    uniform = engine.draw(count // 2).double()
    if not ((uniform > 0.0) & (uniform < 1.0)).all():
        raise FloatingPointError("Sobol endpoints cannot be mapped to finite normals")
    half = (math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)).reshape(-1)
    node = torch.cat([half, -half])
    if node.ndim != 1:
        raise ValueError("QMC nodes must be one-dimensional")
    weight = torch.full((count,), math.sqrt(2.0 * math.pi) / count, dtype=torch.float64)
    return node, weight


def fixed_qmc_nodes(
    count: int = QMC_DEFAULT_COUNT,
    seed: int = QMC_DEFAULT_SEED,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return saved antithetic QMC nodes and equal weights ``sqrt(2 pi)/count``.

    This is the declared fallback integration hook: the weights are chosen so
    that ``sum_k w_k f(z_k) / sqrt(2 pi) = (1/count) sum_k f(z_k)``, i.e. the
    same Gaussian expectation the Gauss-Hermite rule computes. A flow/MDN
    predictive that cannot be integrated component-wise is meant to consume
    this node set; :func:`convolved_parallax_nll_mixture` already accepts it for
    Gaussian components. Nodes are drawn once per ``(count, seed)`` and cached,
    never re-randomized during training; save ``count``/``seed`` in the
    certificate.
    """
    if dtype in (torch.float16, torch.bfloat16):
        raise TypeError("parallax quadrature requires at least float32")
    node, weight = _qmc_cpu(count, seed)
    return node.to(device=device, dtype=dtype), weight.to(device=device, dtype=dtype)


def chi2_1dof_quantile(level: float) -> float:
    """Return the ``level`` quantile of chi-square with one degree of freedom.

    ``chi2_1(1-alpha) = 2 erfinv(1-alpha)^2``, computed here so the
    confidence-set helper needs no SciPy dependency.
    """
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        raise TypeError("confidence level must be a real number")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("confidence level must lie strictly between 0 and 1")
    # chi2_1(1 - alpha) = (Phi^{-1}(1 - alpha/2))^2 = 2 erfinv(1 - alpha)^2.
    return float(
        2.0 * torch.special.erfinv(torch.tensor(float(level), dtype=torch.float64)) ** 2
    )


# --------------------------------------------------------------------------- #
# Shared validation
# --------------------------------------------------------------------------- #
def _compute_dtype(*values: Tensor) -> torch.dtype:
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return dtype


def _zero_point_tensor(
    zero_point_mas: Tensor | float, reference: Tensor, *, valid: Tensor
) -> Tensor:
    """Broadcast a fixed scalar or audited per-star zero point to ``reference``."""
    if isinstance(zero_point_mas, Tensor):
        if not zero_point_mas.is_floating_point():
            raise TypeError("zero point must be floating point")
        if zero_point_mas.device != reference.device:
            raise ValueError("zero point must share the parallax device")
        if zero_point_mas.shape not in (torch.Size([]), reference.shape):
            raise ValueError(
                "per-star zero point must have the exact measurement shape"
            )
        zero = zero_point_mas.to(dtype=reference.dtype).expand_as(reference)
        if not torch.isfinite(zero[valid]).all():
            raise ValueError("zero points used by the likelihood must be finite")
        return zero
    if isinstance(zero_point_mas, bool) or not isinstance(zero_point_mas, (int, float)):
        raise TypeError("zero point must be a tensor or a real scalar")
    if not math.isfinite(float(zero_point_mas)):
        raise ValueError("zero point must be finite and expressed in mas")
    return torch.full_like(reference, float(zero_point_mas))


def _eligible(valid: Tensor, eligible: Tensor | None) -> Tensor:
    if eligible is None:
        return valid
    if eligible.shape != valid.shape or eligible.dtype != torch.bool:
        raise ValueError("eligibility must be a boolean mask of the exact label shape")
    if eligible.device != valid.device:
        raise ValueError("eligibility and measurements must share a device")
    return valid & eligible


def _usable(
    measured_parallax_mas: Tensor, sigma_eff_mas: Tensor, eligible: Tensor | None
) -> Tensor:
    if measured_parallax_mas.ndim != 1:
        raise ValueError("measured parallax must be a [row] tensor")
    if sigma_eff_mas.shape != measured_parallax_mas.shape:
        raise ValueError("measured parallax and sigma_eff must share [row] shape")
    if (
        not measured_parallax_mas.is_floating_point()
        or not sigma_eff_mas.is_floating_point()
    ):
        raise TypeError("measured parallax and sigma_eff must be floating point")
    if sigma_eff_mas.device != measured_parallax_mas.device:
        raise ValueError("measured parallax and sigma_eff must share a device")
    valid = (
        torch.isfinite(measured_parallax_mas)
        & torch.isfinite(sigma_eff_mas)
        & (sigma_eff_mas > 0)
    )
    return _eligible(valid, eligible)


def _validate_predictive_shapes(mean: Tensor, log_scale: Tensor) -> None:
    if mean.ndim != 1 or log_scale.shape != mean.shape:
        raise ValueError("mean and log_scale must share [row] shape")
    if mean.device != log_scale.device:
        raise ValueError("mean and log_scale must share a device")
    if not mean.is_floating_point() or not log_scale.is_floating_point():
        raise TypeError("mean and log_scale must be floating point")
    if not torch.isfinite(mean).all() or not torch.isfinite(log_scale).all():
        raise ValueError("all model parameters must be finite")


def _differentiable_zero(*values: Tensor) -> Tensor:
    zero = values[0].reshape(-1)[:0].sum()
    for value in values[1:]:
        zero = zero + value.reshape(-1)[:0].sum()
    return zero


# --------------------------------------------------------------------------- #
# The convolved likelihood
# --------------------------------------------------------------------------- #
def _log_integral_prior(
    means: Tensor,
    log_scales: Tensor,
    logits: Tensor | None,
    measured: Tensor,
    sigma_eff: Tensor,
    zero_point: Tensor,
    node: Tensor,
    log_weight: Tensor,
) -> Tensor:
    """GH nodes in the prior coordinate: ``s_kj = mu_k + sigma_k u_j``.

    With ``s = mu + sigma_s u`` the prior density cancels the Jacobian and the
    Gaussian weight, so the integral is just the *expectation of the likelihood*
    under the predictive: ``int p(s) N(plx|z+h(s), sigma) ds = E_u[N(...)]``.
    The mixture weights ``pi_k`` are the only prior information left in the sum.
    """
    scale = log_scales.exp()
    s = means[:, :, None] + scale[:, :, None] * node[None, None, :]
    h = torch.exp(-s)
    if not (torch.isfinite(h).all() and (h > 0).all()):
        raise ValueError("model parallax overflowed; no silent clipping")
    residual = (measured[:, None, None] - zero_point[:, None, None] - h) / sigma_eff[
        :, None, None
    ]
    log_likelihood = (
        -0.5 * residual.square() - sigma_eff.log()[:, None, None] - 0.5 * _LOG_2PI
    )
    terms = log_likelihood + log_weight[None, None, :]
    if logits is not None:
        terms = terms + logits[:, :, None]
    return torch.logsumexp(terms.reshape(len(means), -1), dim=1)


def _log_integral_residual(
    means: Tensor,
    log_scales: Tensor,
    logits: Tensor | None,
    h_star: Tensor,
    sigma_eff: Tensor,
    node: Tensor,
    log_weight: Tensor,
) -> Tensor:
    """GH nodes in the standardized residual with the exact Jacobian ``1/h``.

    ``h_j = h* - sigma_eff r_j``, ``s_j = -log h_j`` and
    ``int p(s) N(plx|z+h,sigma) ds = E_r[p(s(r)) / h(r)]``. Requires ``h_j > 0``
    on every node; callers guarantee representability before calling.
    """
    h = h_star[:, None] - sigma_eff[:, None] * node[None, :]
    log_h = h.log()
    s = -log_h
    scale = log_scales.exp()
    log_prior = (
        -0.5 * ((s[:, :, None] - means[:, None, :]) / scale[:, None, :]).square()
    )
    log_prior = log_prior - log_scales[:, None, :] - 0.5 * _LOG_2PI
    terms = log_prior - log_h[:, :, None] + log_weight[None, :, None]
    if logits is not None:
        terms = terms + logits[:, None, :]
    return torch.logsumexp(terms.reshape(len(means), -1), dim=1)


def _convolved_nll(
    means: Tensor,
    log_scales: Tensor,
    logits: Tensor | None,
    measured_parallax_mas: Tensor,
    sigma_eff_mas: Tensor,
    zero_point_mas: Tensor | float,
    eligible: Tensor | None,
    order: int,
    rule: str,
    nodes: str,
    qmc_count: int,
    qmc_seed: int,
) -> DistanceLoss:
    """Shared core for the Gaussian and mixture predictives (``K >= 1``)."""
    if rule not in QUADRATURE_RULES:
        raise ValueError(f"rule must be one of {QUADRATURE_RULES}")
    if nodes not in NODE_SETS:
        raise ValueError(f"nodes must be one of {NODE_SETS}")
    valid = _usable(measured_parallax_mas, sigma_eff_mas, eligible)
    zero_point = _zero_point_tensor(zero_point_mas, measured_parallax_mas, valid=valid)
    count = int(valid.sum().item())
    if count == 0:
        # Differentiable zero from known finite model values, not from NaN labels.
        return DistanceLoss(_differentiable_zero(means, log_scales), 0, 0, 0)
    dtype = _compute_dtype(
        means, log_scales, measured_parallax_mas, sigma_eff_mas, zero_point
    )
    with torch.autocast(device_type=means.device.type, enabled=False):
        m = means[valid].to(dtype)
        ls = log_scales[valid].to(dtype)
        lg = None
        if logits is not None:
            weights = logits[valid].to(dtype)
            # Component weights must be a probability vector; raw logits would
            # add log(sum exp(logit)) to every row's log-integral.
            lg = weights - torch.logsumexp(weights, dim=1, keepdim=True)
        plx = measured_parallax_mas[valid].to(dtype)
        sigma = sigma_eff_mas[valid].to(dtype)
        zp = zero_point[valid].to(dtype)
        if nodes == "gh":
            node, weight = gauss_hermite(order, dtype=dtype, device=means.device)
            residual_node, residual_weight = gauss_hermite(
                min(order, RESIDUAL_RULE_MAX_ORDER), dtype=dtype, device=means.device
            )
        else:
            node, weight = fixed_qmc_nodes(
                qmc_count, qmc_seed, dtype=dtype, device=means.device
            )
            residual_node, residual_weight = node, weight
        log_weight = weight.log() - 0.5 * _LOG_2PI
        log_residual_weight = residual_weight.log() - 0.5 * _LOG_2PI
        h_star = plx - zp
        if rule == "gh_prior":
            use_residual = torch.zeros(count, dtype=torch.bool, device=means.device)
        else:
            use_residual = (
                h_star[:, None] - sigma[:, None] * residual_node[None, :] > 0
            ).all(1)
            if rule == "gh_residual" and not bool(use_residual.all()):
                raise ValueError(
                    "gh_residual requires plx - zero_point > sigma_eff * max(residual "
                    "node) on every row; use rule='auto' or rule='gh_prior'"
                )
        log_integral = torch.empty(count, dtype=dtype, device=means.device)
        if bool(use_residual.any()):
            log_integral = log_integral.index_put(
                (use_residual.nonzero(as_tuple=True)[0],),
                _log_integral_residual(
                    m[use_residual],
                    ls[use_residual],
                    None if lg is None else lg[use_residual],
                    h_star[use_residual],
                    sigma[use_residual],
                    residual_node,
                    log_residual_weight,
                ),
            )
        if bool((~use_residual).any()):
            log_integral = log_integral.index_put(
                ((~use_residual).nonzero(as_tuple=True)[0],),
                _log_integral_prior(
                    m[~use_residual],
                    ls[~use_residual],
                    None if lg is None else lg[~use_residual],
                    plx[~use_residual],
                    sigma[~use_residual],
                    zp[~use_residual],
                    node,
                    log_weight,
                ),
            )
        nll = -log_integral
        if not torch.isfinite(nll).all():
            raise ValueError("nonfinite convolved parallax likelihood")
        min_prior_resolution = float("inf")
        if bool((~use_residual).any()):
            # sigma_eff / (sigma_s h*) with sigma_s = exp(log_scale), h* = exp(-mean)
            resolution = sigma[~use_residual] * torch.exp(
                m[~use_residual][:, 0] - ls[~use_residual][:, 0]
            )
            min_prior_resolution = float(resolution.min().item())
    return DistanceLoss(
        nll.sum(),
        count,
        count,
        int(use_residual.sum().item()),
        min_prior_resolution,
    )


def convolved_parallax_nll(
    mean: Tensor,
    log_scale: Tensor,
    measured_parallax_mas: Tensor,
    sigma_eff_mas: Tensor,
    *,
    zero_point_mas: Tensor | float = 0.0,
    eligible: Tensor | None = None,
    order: int = 32,
    rule: str = "auto",
    nodes: str = "gh",
    qmc_count: int = QMC_DEFAULT_COUNT,
    qmc_seed: int = QMC_DEFAULT_SEED,
) -> DistanceLoss:
    """Marginal NLL of raw signed parallaxes under Gaussian ``p_theta(s|x)``.

    Parameters are native units: ``mean`` is ``s = log(d/kpc)``, ``log_scale``
    is ``log sigma_s`` in the same unit, ``measured_parallax_mas`` is the
    *signed* raw parallax in mas and ``sigma_eff_mas`` the total per-star error
    from :func:`effective_parallax_error_mas`. Negative and zero parallaxes are
    retained; missing labels are selected out before any arithmetic and an
    empty batch returns a differentiable zero with count zero.
    """
    _validate_predictive_shapes(mean, log_scale)
    return _convolved_nll(
        mean[:, None],
        log_scale[:, None],
        None,
        measured_parallax_mas,
        sigma_eff_mas,
        zero_point_mas,
        eligible,
        order,
        rule,
        nodes,
        qmc_count,
        qmc_seed,
    )


def convolved_parallax_nll_mixture(
    logits: Tensor,
    means: Tensor,
    log_scales: Tensor,
    measured_parallax_mas: Tensor,
    sigma_eff_mas: Tensor,
    *,
    zero_point_mas: Tensor | float = 0.0,
    eligible: Tensor | None = None,
    order: int = 32,
    rule: str = "auto",
    nodes: str = "gh",
    qmc_count: int = QMC_DEFAULT_COUNT,
    qmc_seed: int = QMC_DEFAULT_SEED,
) -> DistanceLoss:
    """Same objective for a 1-D MDN/flow placeholder with ``K`` components.

    Each component is Gaussian in ``s``, so the mixture integral is a
    ``logsumexp`` over ``(component, node)`` and stays exact in the
    ``sigma_eff -> 0`` limit. ``logits`` are unnormalized component logits
    ``[row, K]``.
    """
    if (
        logits.ndim != 2
        or means.shape != logits.shape
        or log_scales.shape != logits.shape
    ):
        raise ValueError(
            "logits, means and log_scales must share [row, components] shape"
        )
    if logits.device != means.device or log_scales.device != means.device:
        raise ValueError("mixture parameters must share a device")
    if not (
        logits.is_floating_point()
        and means.is_floating_point()
        and log_scales.is_floating_point()
    ):
        raise TypeError("mixture parameters must be floating point")
    if not (
        torch.isfinite(means).all()
        and torch.isfinite(log_scales).all()
        and torch.isfinite(logits).all()
    ):
        raise ValueError("all model parameters must be finite")
    return _convolved_nll(
        means,
        log_scales,
        logits,
        measured_parallax_mas,
        sigma_eff_mas,
        zero_point_mas,
        eligible,
        order,
        rule,
        nodes,
        qmc_count,
        qmc_seed,
    )


# --------------------------------------------------------------------------- #
# Controls (never the primary objective)
# --------------------------------------------------------------------------- #
def narrow_error_limit_nll(
    mean: Tensor,
    log_scale: Tensor,
    measured_parallax_mas: Tensor,
    *,
    zero_point_mas: Tensor | float = 0.0,
    eligible: Tensor | None = None,
) -> Tensor:
    """Per-row ``sigma_eff -> 0`` limit, i.e. the change-of-variables control.

    ``L -> -log p_theta(s*) + log h*`` with ``s* = -log h*`` and
    ``h* = plx - z``: the exact limit of the convolved likelihood for a
    noiseless parallax. Returns ``+inf`` for rows where the limit does not exist
    (``plx - z <= 0``) instead of clipping: a non-positive parallax has no
    noiseless positive-parallax preimage. Control only -- never a loss.
    """
    _validate_predictive_shapes(mean, log_scale)
    if measured_parallax_mas.ndim != 1 or measured_parallax_mas.shape != mean.shape:
        raise ValueError(
            "measured parallax must share the [row] shape of the prediction"
        )
    if not measured_parallax_mas.is_floating_point():
        raise TypeError("measured parallax must be floating point")
    valid = _eligible(torch.isfinite(measured_parallax_mas), eligible)
    zero_point = _zero_point_tensor(zero_point_mas, measured_parallax_mas, valid=valid)
    dtype = _compute_dtype(mean, log_scale, measured_parallax_mas, zero_point)
    h_star = measured_parallax_mas.to(dtype) - zero_point.to(dtype)
    positive = valid & (h_star > 0)
    safe_h = torch.where(positive, h_star, torch.ones_like(h_star))
    log_h = safe_h.log()
    s_star = -log_h
    log_prior = -0.5 * ((s_star - mean.to(dtype)) / log_scale.to(dtype).exp()).square()
    log_prior = log_prior - log_scale.to(dtype) - 0.5 * _LOG_2PI
    values = torch.where(
        positive, -log_prior + log_h, torch.full_like(h_star, float("inf"))
    )
    return values


def forward_gaussian_control(
    mean: Tensor,
    log_scale: Tensor,
    measured_parallax_mas: Tensor,
    sigma_eff_mas: Tensor,
    *,
    zero_point_mas: Tensor | float = 0.0,
    eligible: Tensor | None = None,
) -> DistanceLoss:
    """First-order (delta-method) propagation control, *not* the likelihood.

    Approximates the predictive parallax density by
    ``N(plx | z + exp(-mu), sigma_s^2 exp(-2 mu) + sigma_eff^2)``. This loses the
    asymmetry of the ``exp(-s)`` map and the prior/likelihood convolution; it is
    kept only so an ablation can quantify what the exact quadrature buys.
    """
    _validate_predictive_shapes(mean, log_scale)
    valid = _usable(measured_parallax_mas, sigma_eff_mas, eligible)
    zero_point = _zero_point_tensor(zero_point_mas, measured_parallax_mas, valid=valid)
    count = int(valid.sum().item())
    if count == 0:
        return DistanceLoss(_differentiable_zero(mean, log_scale), 0, 0, 0)
    dtype = _compute_dtype(
        mean, log_scale, measured_parallax_mas, sigma_eff_mas, zero_point
    )
    predicted = torch.exp(-mean[valid].to(dtype))
    if not torch.isfinite(predicted).all():
        raise ValueError("model parallax overflowed; no silent clipping")
    model_sd = log_scale[valid].to(dtype).exp() * predicted
    sigma = torch.sqrt(model_sd.square() + sigma_eff_mas[valid].to(dtype).square())
    residual = (
        measured_parallax_mas[valid].to(dtype) - zero_point[valid].to(dtype) - predicted
    ) / sigma
    nll = 0.5 * residual.square() + sigma.log() + 0.5 * _LOG_2PI
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite forward-Gaussian control")
    return DistanceLoss(nll.sum(), count, count, 0)


# --------------------------------------------------------------------------- #
# Assisted profile-likelihood confidence set (product B)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProfileConfidenceSet:
    """Closed-form profile-likelihood set for ``plx ~ N(z + exp(-s), sigma_eff)``.

    ``s_lower``/``s_upper`` are ``log(d/kpc)`` bounds and are NaN on rows that
    are empty or ineligible; ``s_upper`` is ``+inf`` where the set is unbounded
    above, ``mode`` is ``+inf`` when the likelihood has no interior maximum
    (non-positive parallax). Nothing is capped or replaced by a finite fallback.
    """

    s_lower: Tensor
    s_upper: Tensor
    mode: Tensor
    empty: Tensor
    upper_unbounded: Tensor
    usable: Tensor
    level: float
    threshold: float

    def fractions(self) -> dict[str, float]:
        """Return ``frac_ci_empty`` / ``frac_ci_upper_unbounded`` plus counts."""
        n = int(self.usable.sum().item())
        n_empty = int((self.empty & self.usable).sum().item())
        n_unbounded = int((self.upper_unbounded & self.usable).sum().item())
        return {
            "n": float(n),
            "n_ci_empty": float(n_empty),
            "n_ci_upper_unbounded": float(n_unbounded),
            "frac_ci_empty": n_empty / n if n else 0.0,
            "frac_ci_upper_unbounded": n_unbounded / n if n else 0.0,
        }


def parallax_profile_confidence_set(
    measured_parallax_mas: Tensor,
    sigma_eff_mas: Tensor,
    *,
    zero_point_mas: Tensor | float = 0.0,
    level: float = 0.90,
    eligible: Tensor | None = None,
) -> ProfileConfidenceSet:
    """Exact assisted confidence set from the raw signed parallax likelihood.

    ``l(h) = log N(plx | z + h, sigma_eff)`` is a downward parabola in
    ``h = exp(-s)``, so
    ``C = {s: (plx - z - exp(-s))^2 / sigma_eff^2 <= chi2_1(level)}`` is the
    interval ``h in [h* - sigma_eff sqrt(q), h* + sigma_eff sqrt(q)]``
    intersected with ``h > 0``. Consequences, reported honestly:

    * ``h* + sigma_eff sqrt(q) <= 0`` -- the set is **empty**;
    * ``h* - sigma_eff sqrt(q) <= 0`` -- ``s`` is **unbounded above**
      (``d -> infinity`` stays inside the set);
    * otherwise ``s_lower = -log(h_hi)`` and ``s_upper = -log(h_lo)``.

    This is a confidence set from the measurement likelihood alone: it uses no
    learned prior and is *not* a normalized ``p_theta``. Rows with non-finite or
    non-positive ``sigma_eff`` are invalid inputs, not silently dropped.
    """
    if (
        not measured_parallax_mas.is_floating_point()
        or not sigma_eff_mas.is_floating_point()
    ):
        raise TypeError("parallax and sigma_eff must be floating point")
    if measured_parallax_mas.ndim != 1:
        raise ValueError("measured parallax must be a [row] tensor")
    if sigma_eff_mas.shape != measured_parallax_mas.shape:
        raise ValueError("parallax and sigma_eff must share [row] shape")
    if sigma_eff_mas.device != measured_parallax_mas.device:
        raise ValueError("parallax and sigma_eff must share a device")
    if not (torch.isfinite(sigma_eff_mas) & (sigma_eff_mas > 0)).all():
        raise ValueError(
            "sigma_eff must be finite and strictly positive on every row "
            "(mask invalid rows instead of substituting a floor)"
        )
    threshold = chi2_1dof_quantile(level)
    zero_point = _zero_point_tensor(
        zero_point_mas,
        measured_parallax_mas,
        valid=torch.ones_like(measured_parallax_mas, dtype=torch.bool),
    )
    dtype = _compute_dtype(measured_parallax_mas, sigma_eff_mas, zero_point)
    plx = measured_parallax_mas.to(dtype)
    usable = _eligible(torch.isfinite(plx), eligible)
    centre = plx - zero_point.to(dtype)
    half_width = sigma_eff_mas.to(dtype) * math.sqrt(threshold)
    h_hi = centre + half_width
    h_lo = centre - half_width
    empty = h_hi <= 0
    upper_unbounded = (~empty) & (h_lo <= 0)
    s_lower = -torch.where(h_hi > 0, h_hi, torch.ones_like(h_hi)).log()
    s_upper = torch.where(
        upper_unbounded,
        torch.full_like(h_lo, float("inf")),
        -torch.where(h_lo > 0, h_lo, torch.ones_like(h_lo)).log(),
    )
    mode = torch.where(
        centre > 0,
        -torch.where(centre > 0, centre, torch.ones_like(centre)).log(),
        torch.full_like(centre, float("inf")),
    )
    keep = usable & ~empty
    s_lower = torch.where(keep, s_lower, torch.full_like(s_lower, float("nan")))
    s_upper = torch.where(keep, s_upper, torch.full_like(s_upper, float("nan")))
    mode = torch.where(usable, mode, torch.full_like(mode, float("nan")))
    return ProfileConfidenceSet(
        s_lower=s_lower,
        s_upper=s_upper,
        mode=mode,
        empty=empty,
        upper_unbounded=upper_unbounded,
        usable=usable,
        level=float(level),
        threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class _DistanceTrunk(nn.Module):
    """Small residual trunk mapping ``[latent || context]`` to head features."""

    def __init__(self, input_dim: int, width: int, outputs: int):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("width", width),
            ("outputs", outputs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.input = nn.Linear(input_dim, width)
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, max(1, width // 4)),
            nn.GELU(),
            nn.Linear(max(1, width // 4), width),
        )
        self.output = nn.Linear(width, outputs)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError("features must be a floating [row, feature] tensor")
        if features.shape[1] != self.input.in_features:
            raise ValueError("feature width does not match the head input dimension")
        hidden = torch.nn.functional.gelu(self.input(features))
        raw = self.output(hidden + self.block(hidden))
        if not torch.isfinite(raw).all():
            raise ValueError("distance head produced nonfinite parameters")
        return raw


class LogDistanceGaussianHead(nn.Module):
    """Predict ``s = log(d/kpc)`` and ``log sigma_s`` (Gaussian ``p_theta``).

    Outputs are native: ``mean`` in log-kpc and ``scale`` in log-kpc. There is
    no softplus and no variance floor; ``scale = exp(log_scale)`` and a
    non-finite parameter raises.
    """

    def __init__(
        self,
        input_dim: int = 384,
        width: int = 128,
        *,
        initial_log_distance: float = 0.0,
        initial_log_scale: float = -1.0,
    ):
        super().__init__()
        if not math.isfinite(initial_log_distance) or not math.isfinite(
            initial_log_scale
        ):
            raise ValueError("initial distance parameters must be finite")
        self.input_dim = input_dim
        self.trunk = _DistanceTrunk(input_dim, width, 2)
        with torch.no_grad():
            self.trunk.output.bias[0] = initial_log_distance
            self.trunk.output.bias[1] = initial_log_scale

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        raw = self.trunk(features)
        mean = raw[:, 0]
        log_scale = raw[:, 1]
        scale = log_scale.exp()
        if not (torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("invalid distance scale; no clipping or error floor")
        return {"mean": mean, "log_scale": log_scale, "scale": scale}

    def loss(
        self,
        features: Tensor,
        measured_parallax_mas: Tensor,
        sigma_eff_mas: Tensor,
        **kwargs,
    ) -> DistanceLoss:
        """Convenience wrapper: head outputs -> :func:`convolved_parallax_nll`."""
        outputs = self(features)
        return convolved_parallax_nll(
            outputs["mean"],
            outputs["log_scale"],
            measured_parallax_mas,
            sigma_eff_mas,
            **kwargs,
        )


class LogDistanceMixtureHead(nn.Module):
    """1-D MDN placeholder: ``K`` Gaussian components in ``s``.

    A scalar mixture is the cheapest stand-in for a flow-shaped predictive: it
    can be bimodal (near/far ambiguity) and its convolved integral is still
    exact per component, so it trains without a Monte-Carlo step. A real flow
    has to supply its own fixed node set (``nodes="qmc"`` in
    :func:`convolved_parallax_nll_mixture` is the declared hook and already
    integrates Gaussian components with saved QMC nodes).
    """

    def __init__(
        self,
        input_dim: int = 384,
        width: int = 128,
        *,
        components: int = 4,
        initial_log_distance: float = 0.0,
        initial_log_scale: float = -1.0,
    ):
        super().__init__()
        if (
            isinstance(components, bool)
            or not isinstance(components, int)
            or components < 1
        ):
            raise ValueError("components must be a positive integer")
        if not math.isfinite(initial_log_distance) or not math.isfinite(
            initial_log_scale
        ):
            raise ValueError("initial distance parameters must be finite")
        self.components = components
        self.input_dim = input_dim
        self.trunk = _DistanceTrunk(input_dim, width, 3 * components)
        with torch.no_grad():
            self.trunk.output.bias[:components] = 0.0
            self.trunk.output.bias[components : 2 * components] = initial_log_distance
            self.trunk.output.bias[2 * components :] = initial_log_scale

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        raw = self.trunk(features)
        k = self.components
        logits = raw[:, :k]
        mean = raw[:, k : 2 * k]
        log_scale = raw[:, 2 * k :]
        scale = log_scale.exp()
        if not (torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("invalid mixture scale; no clipping or error floor")
        return {"logits": logits, "mean": mean, "log_scale": log_scale, "scale": scale}

    def log_prob(self, features: Tensor, log_distance: Tensor) -> Tensor:
        """Mixture log-density ``log p_theta(s|x)`` at ``log_distance`` [row]."""
        outputs = self(features)
        if log_distance.ndim != 1 or log_distance.shape != outputs["mean"][:, 0].shape:
            raise ValueError("log_distance must have [row] shape")
        standardized = (log_distance[:, None] - outputs["mean"]) / outputs["scale"]
        component = -0.5 * standardized.square() - outputs["log_scale"] - 0.5 * _LOG_2PI
        log_weights = outputs["logits"] - torch.logsumexp(
            outputs["logits"], dim=1, keepdim=True
        )
        return torch.logsumexp(log_weights + component, dim=1)

    def loss(
        self,
        features: Tensor,
        measured_parallax_mas: Tensor,
        sigma_eff_mas: Tensor,
        **kwargs,
    ) -> DistanceLoss:
        """Convenience wrapper: head outputs -> mixture convolved likelihood."""
        outputs = self(features)
        return convolved_parallax_nll_mixture(
            outputs["logits"],
            outputs["mean"],
            outputs["log_scale"],
            measured_parallax_mas,
            sigma_eff_mas,
            **kwargs,
        )
