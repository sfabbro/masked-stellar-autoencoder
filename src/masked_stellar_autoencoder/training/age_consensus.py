"""Age consensus C0: one latent log age per star from many label sources, no network.

Design: ``docs/design-20261122-age-consensus.md`` (sections 3 and 7). The latent
is ``t = log10(age / Gyr)``. No label is ground truth: every record (anchor
included) is a noisy measurement of the latent through a fitted map and a
fitted noise model; group ages are fitted latents.

Model. Star ``i`` has a latent ``t_i``. A record of source map ``k`` reports

    y = f_k(t + u_{i,e}, x, z, c) + e,

* ``f_k`` comes from a pluggable *map family* (``MapFamily``): ages use
  ``AgeMaps`` (identity for the gauge anchor, else offset / + slope, class and
  [Fe/H] terms / + polynomial terms, by tier); non-age observables (ln P_rot,
  Li EW, [C/N], ...) use ``MonotoneMaps`` (a monotone spline in ``t`` plus
  covariate terms). A new label type is a new family, or a new source of an
  existing family: the fit below only needs ``f`` and ``df/dt`` per record.
* ``e`` is Student-t (``nu``) with scale^2 = kappa_n^2 sigma^2 + omega_n^2 per
  noise unit ``n`` (source x class); ``sigma`` is the quoted error (0 if none).
* ``u_{i,e} ~ N(0, eta_e^2)`` is an age error shared by all records of star
  ``i`` in effect family ``e`` (sources with shared inputs: isochrone
  pipelines on the same Gaia photometry/parallax and spectra; seismic
  pipelines on the same light curves), so correlated catalogues are not
  counted as independent votes. It cancels in contrasts within a family, so
  ``eta`` is learned only from independent constraints (another family or a
  co-natal partner).
* records at a catalogue cap (floor) are right- (left-) censored;
* co-natal groups (clusters, wide binaries) share a fitted age latent T_g:
  ``t_i ~ N(T_g, s_g^2)`` for a member with responsibility ``rho_i``
  (membership prior x likelihood, against a field density); T_g has a flat
  prior. A literature group age would enter as one more mapped source of T_g.

Estimation (maximum likelihood, CPU, memory bounded):

* maps: the profile (errors-in-variables, "Deming") objective, the latents
  and group ages profiled out in closed form (generalised least squares with
  the shared effects); its expectation is minimised at the true maps whatever
  the slopes when the noise model is right, unlike the flat-prior marginal
  likelihood whose ``-log|a|`` Jacobian pulls slopes to zero. Maps are
  linearised in ``t`` at the current latents (Gauss-Newton outer loop);
  L-BFGS on the map coefficients with the latents re-solved at every
  evaluation (envelope gradient).
* noise (kappa, omega, eta): the restricted likelihood with the latents and
  group ages integrated under flat priors (REML: no Neyman-Scott shrinkage),
  L-BFGS; the Student-t enters through EM weights.
* censoring: EM imputation of the censored value given the latent.

Only stars with at least two constraints inform maps and noise;
``solve_stars`` then gives every star its consensus (flat prior: a measurement
on the anchor's scale, no population prior).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

IDENTITY, OFFSET, LINEAR, FLEX = 0, 1, 2, 3
T0 = 0.6  # pivot log10(age / Gyr), about 4 Gyr
FEH_CLIP = (-2.5, 0.6)
N_FLEX = 4
DT = torch.float64
LOG_OMEGA0 = math.log(0.03)
#: Sanity bounds on the noise (dex): no record is credited with a per-star
#: floor below 0.005 dex or an inflation below 0.2 of its quoted error.
LOG_KAPPA_RANGE = (math.log(0.2), math.log(20.0))
LOG_OMEGA_RANGE = (math.log(0.005), math.log(3.0))


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class Records:
    """Measurements, one row per (star, source) record."""

    star: np.ndarray  # (R,) int64 in [0, n_star)
    map_id: np.ndarray  # (R,) int64 global map index (see MapSet)
    cls: np.ndarray  # (R,) int64 class index
    noise_id: np.ndarray  # (R,) int64 noise unit
    y: np.ndarray  # (R,) reported value (log10 age for ages)
    sig: np.ndarray  # (R,) quoted error; 0 or NaN when none
    feh: np.ndarray  # (R,) [Fe/H]; NaN when unknown
    censor: np.ndarray  # (R,) int8: +1 y is a lower limit (cap), -1 upper limit
    n_star: int
    effect: np.ndarray | None = None  # (R,) int64 shared-effect family, -1 none
    extra: np.ndarray | None = None  # (R, q) further covariates (e.g. colour)

    def __post_init__(self):
        n = len(self.y)
        if self.effect is None:
            self.effect = np.full(n, -1, "i8")
        if self.extra is None:
            self.extra = np.zeros((n, 0))
        for name in ("star", "map_id", "cls", "noise_id", "sig", "feh", "censor", "effect",
                     "extra"):  # fmt: skip
            if len(getattr(self, name)) != n:
                raise ValueError(f"Records.{name} has {len(getattr(self, name))} rows, y has {n}")

    def subset(self, keep: np.ndarray) -> Records:
        return Records(
            self.star[keep], self.map_id[keep], self.cls[keep], self.noise_id[keep],
            self.y[keep], self.sig[keep], self.feh[keep], self.censor[keep], self.n_star,
            self.effect[keep], self.extra[keep],
        )  # fmt: skip


@dataclass
class Groups:
    """Co-natal groups; ``gid`` is -1 for field stars."""

    gid: np.ndarray  # (n_star,) int64
    prior: np.ndarray  # (n_star,) membership prior in [0, 1]
    spread: np.ndarray  # (n_group,) intrinsic age spread in dex

    @staticmethod
    def none(n_star: int) -> Groups:
        return Groups(np.full(n_star, -1, "i8"), np.zeros(n_star), np.zeros(0))


def clip_feh(feh) -> np.ndarray:
    x = np.asarray(feh, float)
    return np.where(np.isfinite(x), np.clip(x, *FEH_CLIP), 0.0)


# --------------------------------------------------------------------------- #
# Map families
# --------------------------------------------------------------------------- #
class MapFamily:
    """A family of measurement maps ``y = f(t, x, z, c; coef)``, one coefficient
    row per map. Subclasses define the coefficient layout, the free mask, the
    prior and ``evaluate`` (value and derivative in ``t``)."""

    names: list[str]

    def n_coef(self) -> int:
        raise NotImplementedError

    def mask(self) -> np.ndarray:  # (n_maps, n_coef) bool
        raise NotImplementedError

    def prior(self) -> tuple[np.ndarray, np.ndarray]:  # mean, sd per coefficient
        raise NotImplementedError

    def init(self) -> np.ndarray:
        return np.tile(self.prior()[0], (len(self.names), 1))

    def evaluate(self, coef, m, t, x, z, cls):
        """(f, df/dt) for records of local map ``m`` at latent ``t``; ``x`` =
        clipped [Fe/H], ``z`` = (R, q) further covariates, ``cls`` = class."""
        raise NotImplementedError

    def penalty(self, dfdt) -> torch.Tensor:
        return dfdt.new_zeros(())

    def column_names(self) -> list[str]:
        return [f"c{j}" for j in range(self.n_coef())]


class AgeMaps(MapFamily):
    """Maps of catalogue ages onto the latent (both log10 Gyr):

    f = t + b + db_c + (A + dA_c) u + beta x + g1 u^2 + g2 u^3 + g3 x u + g4 x^2,
    u = t - T0. Tiers: IDENTITY (gauge anchor), OFFSET (b), LINEAR (+ A, class
    deltas db_c/dA_c where ``class_terms``, beta where ``use_feh``), FLEX (+ g).
    The slope df/dt is kept above ``a_min`` (monotone maps). ``x_range`` and
    ``t_range`` ((M, 2), optional) bound the data each map was fitted on: the
    [Fe/H] terms use x clamped to ``x_range`` and the polynomial terms use t
    clamped to ``t_range`` (no extrapolation; linear continuation in t)."""

    def __init__(self, names, tier, class_terms, use_feh, a_min: float = 0.2,
                 x_range=None, t_range=None):  # fmt: skip
        self.names = list(names)
        self.tier = np.asarray(tier, "i8")
        self.class_terms = np.asarray(class_terms, bool)
        self.use_feh = np.asarray(use_feh, bool)
        self.a_min = a_min
        n = len(self.names)
        xr = np.tile(FEH_CLIP, (n, 1)) if x_range is None else np.asarray(x_range, float)
        tr = np.tile((-3.0, 3.0), (n, 1)) if t_range is None else np.asarray(t_range, float)
        self.x_range = torch.as_tensor(xr, dtype=DT)
        self.t_range = torch.as_tensor(tr, dtype=DT)

    @property
    def n_cls(self) -> int:
        return self.class_terms.shape[1]

    def n_coef(self) -> int:
        return 3 + 2 * self.n_cls + N_FLEX

    def mask(self) -> np.ndarray:
        m, c = len(self.names), self.n_cls
        out = np.zeros((m, self.n_coef()), bool)
        out[:, 0] = self.tier >= OFFSET
        out[:, 1] = self.tier >= LINEAR
        out[:, 2] = (self.tier >= LINEAR) & self.use_feh
        ct = self.class_terms & (self.tier >= LINEAR)[:, None]
        out[:, 3 : 3 + c] = ct
        out[:, 3 + c : 3 + 2 * c] = ct
        flex = self.tier >= FLEX
        out[:, 3 + 2 * c] = flex
        out[:, 4 + 2 * c] = flex
        out[:, 5 + 2 * c] = flex & self.use_feh
        out[:, 6 + 2 * c] = flex & self.use_feh
        return out

    def prior(self):
        c = self.n_cls
        sd = np.array([0.5, 0.5, 0.5] + [0.3] * (2 * c) + [0.3] * N_FLEX)
        return np.zeros_like(sd), sd

    def column_names(self):
        c = self.n_cls
        return (["b", "A", "beta"] + [f"db{j}" for j in range(c)]
                + [f"dA{j}" for j in range(c)] + ["g_uu", "g_uuu", "g_xu", "g_xx"])  # fmt: skip

    def evaluate(self, coef, m, t, x, z, cls):
        n = self.n_cls
        u = t - T0
        x = torch.minimum(torch.maximum(x, self.x_range[m, 0]), self.x_range[m, 1])
        tc = torch.minimum(torch.maximum(t, self.t_range[m, 0]), self.t_range[m, 1])
        inside = ((t >= self.t_range[m, 0]) & (t <= self.t_range[m, 1])).to(t.dtype)
        uc = tc - T0
        b = coef[m, 0] + coef[m, 3 + cls]
        a1 = coef[m, 1] + coef[m, 3 + n + cls]
        g = coef[m, 3 + 2 * n :]
        f = (t + b + a1 * u + coef[m, 2] * x + g[:, 0] * uc * uc + g[:, 1] * uc * uc * uc
             + g[:, 2] * x * uc + g[:, 3] * x * x)  # fmt: skip
        dpoly = 2.0 * g[:, 0] * uc + 3.0 * g[:, 1] * uc * uc + g[:, 2] * x
        dfdt = 1.0 + a1 + inside * dpoly
        return f, dfdt

    def penalty(self, dfdt):
        return 1e3 * ((self.a_min - dfdt).clamp_min(0.0) ** 2).sum()


class MonotoneMaps(MapFamily):
    """Monotone maps of a non-age observable (e.g. ln P_rot, Li EW, [C/N]):

    f = c0 + c1 x + c2 z_0 + direction * sum_k softplus(w_k) h_k(t),
    h_k(t) = s softplus((t - knot_k) / s): smooth ramps, so f is monotone in
    ``t`` by construction (increasing for direction +1). ``use_x`` /
    ``use_z`` switch the [Fe/H] and first extra-covariate terms on per map."""

    def __init__(self, names, direction, knots=(-1.0, -0.4, 0.2, 0.6, 1.0),
                 smooth: float = 0.15, use_x=None, use_z=None, y_scale=None):  # fmt: skip
        self.names = list(names)
        self.direction = torch.as_tensor(np.asarray(direction, float), dtype=DT)
        self.knots = torch.as_tensor(np.asarray(knots, float), dtype=DT)
        self.s = smooth
        n = len(self.names)
        self.use_x = np.ones(n, bool) if use_x is None else np.asarray(use_x, bool)
        self.use_z = np.zeros(n, bool) if use_z is None else np.asarray(use_z, bool)
        self.y_scale = np.ones(n) if y_scale is None else np.asarray(y_scale, float)

    def n_coef(self) -> int:
        return 3 + len(self.knots)

    def mask(self):
        out = np.ones((len(self.names), self.n_coef()), bool)
        out[:, 1] = self.use_x
        out[:, 2] = self.use_z
        return out

    def prior(self):
        k = len(self.knots)
        mean = np.r_[0.0, 0.0, 0.0, np.full(k, -1.0)]
        sd = np.r_[100.0, 10.0, 10.0, np.full(k, 5.0)]
        return mean, sd

    def evaluate(self, coef, m, t, x, z, cls):
        w = torch.nn.functional.softplus(coef[m, 3:])  # (R, K) >= 0
        arg = (t[:, None] - self.knots[None, :]) / self.s
        h = self.s * torch.nn.functional.softplus(arg)
        dh = torch.sigmoid(arg)
        d = self.direction[m]
        zz = z[:, 0] if z.shape[1] else torch.zeros_like(t)
        f = coef[m, 0] + coef[m, 1] * x + coef[m, 2] * zz + d * (w * h).sum(1)
        dfdt = d * (w * dh).sum(1)
        return f, dfdt


class MapSet:
    """All maps of a fit: families concatenated; global map ids are contiguous
    per family. The free coefficients of all families form one vector."""

    def __init__(self, families: list[MapFamily]):
        self.families = list(families)
        self.offsets = np.cumsum([0] + [len(f.names) for f in self.families])
        self.names = [n for f in self.families for n in f.names]
        self.masks = [torch.as_tensor(f.mask()) for f in self.families]
        self.sizes = [int(m.sum()) for m in self.masks]
        pri = [f.prior() for f in self.families]
        self.prior_mean = torch.cat([torch.as_tensor(np.broadcast_to(p[0], m.shape).copy(),
                                                     dtype=DT)[m] for p, m in zip(pri, self.masks)])  # fmt: skip
        self.prior_sd = torch.cat([torch.as_tensor(np.broadcast_to(p[1], m.shape).copy(),
                                                   dtype=DT)[m] for p, m in zip(pri, self.masks)])  # fmt: skip

    def family_of(self, map_id: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.offsets, map_id, side="right") - 1

    def locate(self, name: str) -> tuple[int, int]:
        g = self.names.index(name)
        j = int(self.family_of(np.array([g]))[0])
        return j, g - int(self.offsets[j])

    def theta0(self, coefs=None) -> torch.Tensor:
        coefs = coefs or [f.init() for f in self.families]
        return torch.cat([torch.as_tensor(c, dtype=DT)[m] for c, m in zip(coefs, self.masks)])

    def unpack(self, theta: torch.Tensor) -> list[torch.Tensor]:
        out, k = [], 0
        for f, m, n in zip(self.families, self.masks, self.sizes):
            base = torch.as_tensor(f.init(), dtype=DT) * (~m)
            out.append(base.masked_scatter(m, theta[k : k + n]))
            k += n
        return out

    def theta_slice(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(positions in theta, coefficient columns) of one map's free coefficients."""
        j, local = self.locate(name)
        start = int(sum(self.sizes[:j]))
        mask = self.masks[j].numpy()
        flat = np.flatnonzero(mask.ravel())
        sel = np.flatnonzero(flat // mask.shape[1] == local)
        return start + sel, flat[sel] % mask.shape[1]


# --------------------------------------------------------------------------- #
# Fit result
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    maps: MapSet
    coefs: list  # per family (n_maps, n_coef) arrays
    theta_cov: np.ndarray | None  # covariance of the free coefficients
    log_kappa: np.ndarray  # (U,)
    log_omega: np.ndarray  # (U,)
    log_eta: np.ndarray  # (E,)
    var_cov: np.ndarray | None  # covariance of (log kappa, log omega, log eta)
    t: np.ndarray  # (n_star,) latent of the fit stars
    var: np.ndarray
    rho: np.ndarray  # (n_star,) co-natal responsibility
    weight: np.ndarray  # (R,) Student-t weights of the fit records
    history: list = field(default_factory=list)
    nu: float = 5.0

    def coef_of(self, name: str) -> np.ndarray:
        j, local = self.maps.locate(name)
        return self.coefs[j][local]


# --------------------------------------------------------------------------- #
# Internal tensor state
# --------------------------------------------------------------------------- #
class _State:
    def __init__(self, rec: Records, groups: Groups, maps: MapSet, n_noise: int, n_effect: int):
        self.maps = maps
        self.n_star = rec.n_star
        self.R = len(rec.y)
        self.star = torch.as_tensor(rec.star, dtype=torch.long)
        fam = maps.family_of(rec.map_id)
        self.fam_idx = [torch.as_tensor(np.flatnonzero(fam == j)) for j in range(len(maps.families))]
        self.local = [torch.as_tensor(rec.map_id[fam == j] - maps.offsets[j]) for j in range(len(maps.families))]
        self.cls = torch.as_tensor(rec.cls, dtype=torch.long)
        self.noise_id = torch.as_tensor(rec.noise_id, dtype=torch.long)
        self.y_obs = torch.as_tensor(rec.y, dtype=DT)
        sig = np.where(np.isfinite(rec.sig) & (rec.sig > 0), rec.sig, 0.0)
        self.sig2 = torch.as_tensor(sig, dtype=DT) ** 2
        self.x = torch.as_tensor(clip_feh(rec.feh), dtype=DT)
        self.z = torch.as_tensor(np.nan_to_num(np.asarray(rec.extra, float)), dtype=DT)
        self.censor = torch.as_tensor(rec.censor.astype("i8"))
        self.cens = self.censor != 0
        self.has_sig = np.zeros(n_noise, bool)
        np.logical_or.at(self.has_sig, rec.noise_id, sig > 0)
        self.n_noise, self.n_effect = n_noise, n_effect
        # shared-effect blocks: (star, effect family); family -1 has eta = 0
        eff = np.asarray(rec.effect, "i8")
        key = rec.star.astype("i8") * (n_effect + 1) + (eff + 1)
        uk, blk = np.unique(key, return_inverse=True)
        self.blk = torch.as_tensor(blk)
        self.n_blk = len(uk)
        self.bstar = torch.as_tensor(uk // (n_effect + 1))
        self.bfam = torch.as_tensor(uk % (n_effect + 1) - 1)
        gid = np.asarray(groups.gid, "i8")
        self.member = torch.as_tensor(gid >= 0)
        self.gid = torch.as_tensor(np.where(gid >= 0, gid, 0))
        self.prior = torch.as_tensor(np.asarray(groups.prior, float), dtype=DT)
        self.n_group = int(len(groups.spread))
        self.s2 = torch.as_tensor(np.asarray(groups.spread, float), dtype=DT) ** 2
        self.has_rec = torch.zeros(self.n_star, dtype=torch.bool)
        self.has_rec[self.star] = True

    def scatter(self, values):
        return torch.zeros(self.n_star, dtype=DT).index_add(0, self.star, values)

    def bsum(self, values):
        return torch.zeros(self.n_blk, dtype=DT).index_add(0, self.blk, values)

    def b2star(self, values):
        return torch.zeros(self.n_star, dtype=DT).index_add(0, self.bstar, values)

    def gscatter(self, values, members):
        return torch.zeros(max(self.n_group, 1), dtype=DT).index_add(0, self.gid[members], values[members])

    def evaluate(self, coefs, t_rec):
        """(f, df/dt) for all records at per-record latent ``t_rec``."""
        f = torch.zeros(self.R, dtype=DT)
        a = torch.zeros(self.R, dtype=DT)
        pen = t_rec.new_zeros(())
        for fam, idx, loc, coef in zip(self.maps.families, self.fam_idx, self.local, coefs):
            if not len(idx):
                continue
            fj, aj = fam.evaluate(coef, loc, t_rec[idx], self.x[idx], self.z[idx], self.cls[idx])
            f = f.index_copy(0, idx, fj)
            a = a.index_copy(0, idx, aj)
            pen = pen + fam.penalty(aj)
        return f, a, pen

    def eta2_blk(self, le):
        e2 = torch.exp(2 * le) if self.n_effect else torch.zeros(0, dtype=DT)
        e2 = torch.cat([torch.zeros(1, dtype=DT), e2])
        return e2[self.bfam + 1]


def _noise_var(st: _State, lk, lw):
    return torch.exp(2 * lk)[st.noise_id] * st.sig2 + torch.exp(2 * lw)[st.noise_id]


def _posterior(st: _State, pi, a, yp, eta2b):
    """Records-only flat-prior posterior (m, P) with the shared age errors
    integrated out: per block the covariance is diag(v) + eta^2 a a^T
    (Sherman-Morrison). Returns (m, P, c, s_aa, s_ay, S) with the per-block
    c = eta^2 / (1 + eta^2 s_aa), s_aa = sum pi a^2, s_ay = sum pi a y'."""
    saa = st.bsum(pi * a * a)
    say = st.bsum(pi * a * yp)
    c = eta2b / (1.0 + eta2b * saa)
    P = st.scatter(pi * a * a) - st.b2star(c * saa * saa)
    S = st.scatter(pi * a * yp) - st.b2star(c * saa * say)
    m = torch.where(P > 0, S / P.clamp_min(1e-300), torch.full_like(P, float("nan")))
    return m, P, c, saa, say, S


def _group_solve(st: _State, m, P, rho):
    """Latent posteriors with the co-natal terms, group ages integrated (flat prior).

    Returns (t, var, T_loo, var_T_loo, kappa, T_hat): marginal mean/variance of
    t_i (= the joint profile mode, the posterior being Gaussian), the
    leave-one-out group pseudo-measurement N(T_loo, s^2/rho + var_T_loo) with
    precision kappa, and the group mode T_hat over all members."""
    t, var = m.clone(), 1.0 / P
    T_loo = torch.full_like(m, float("nan"))
    vT_loo = torch.full_like(m, float("inf"))
    kappa = torch.zeros_like(m)
    if st.n_group == 0:
        return t, var, T_loo, vT_loo, kappa, m.clone()
    mem = st.member & st.has_rec & (P > 0)
    s2 = st.s2[st.gid]
    r = rho.clamp(1e-12, 1.0)
    mm = torch.nan_to_num(m)
    om = torch.where(mem, r * P / (r + s2 * P), torch.zeros_like(P))
    W0, W1 = st.gscatter(om, mem)[st.gid], st.gscatter(om * mm, mem)[st.gid]
    rest = W0 - om
    ok = mem & (rest > 1e-12)
    T_loo = torch.where(ok, (W1 - om * mm) / rest.clamp_min(1e-300), T_loo)
    vT_loo = torch.where(ok, 1.0 / rest.clamp_min(1e-300), vT_loo)
    kappa = torch.where(ok, 1.0 / (s2 / r + vT_loo), kappa)
    prec = P + kappa
    t = torch.where(ok, (P * m + kappa * torch.nan_to_num(T_loo)) / prec, t)
    var = torch.where(ok, 1.0 / prec, var)
    T_hat = torch.where(mem & (W0 > 0), W1 / W0.clamp_min(1e-300), m)
    return t, var, T_loo, vT_loo, kappa, T_hat


def _impute(st: _State, mu, v):
    """E[y*] and Var[y*] of censored records given the predicted mean ``mu``."""
    y = st.y_obs.clone()
    cvar = torch.zeros_like(y)
    if not bool(st.cens.any()):
        return y, cvar
    sd = v.sqrt()
    up, lo = st.censor > 0, st.censor < 0
    a = (st.y_obs - mu) / sd
    lphi = -0.5 * a * a - 0.5 * math.log(2 * math.pi)
    lam = torch.exp(lphi - torch.special.log_ndtr(-a))  # true value >= c
    y = torch.where(up, mu + sd * lam, y)
    cvar = torch.where(up, v * (1 + a * lam - lam * lam).clamp_min(1e-6), cvar)
    lam2 = torch.exp(lphi - torch.special.log_ndtr(a))  # true value <= c
    y = torch.where(lo, mu - sd * lam2, y)
    cvar = torch.where(lo, v * (1 - a * lam2 - lam2 * lam2).clamp_min(1e-6), cvar)
    return y, cvar


def _responsibility(st: _State, m, P, rho, field_mu, field_sd):
    if st.n_group == 0:
        return rho
    _, _, T_loo, vT_loo, _, _ = _group_solve(st, m, P, rho)
    V = 1.0 / P.clamp_min(1e-300)
    ok = st.member & st.has_rec & torch.isfinite(T_loo) & (P > 0)
    vc = V + st.s2[st.gid] + vT_loo
    lc = -0.5 * (m - T_loo) ** 2 / vc - 0.5 * torch.log(vc)
    vf = V + field_sd**2
    lf = -0.5 * (m - field_mu) ** 2 / vf - 0.5 * torch.log(vf)
    p = st.prior.clamp(1e-6, 1 - 1e-6)
    new = torch.sigmoid(torch.log(p) - torch.log1p(-p) + lc - lf)
    return torch.where(ok, new, torch.where(st.member, st.prior, torch.zeros_like(rho)))


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def _map_objective(st: _State, theta, w, v, y, t_lin, rho, eta2b):
    """Profile objective (latents and group ages profiled; envelope gradient)."""
    maps = st.maps
    coefs = maps.unpack(theta)
    tl = t_lin[st.star]
    f, a, pen = st.evaluate(coefs, tl)
    yp = y - f + a * tl
    pi = w / v
    with torch.no_grad():
        m, P, c, _, _, _ = _posterior(st, pi, a.detach(), yp.detach(), eta2b)
        ts, _, _, _, _, T_hat = _group_solve(st, m, P, rho)
        gterm = 0.0
        if st.n_group:
            mem = st.member & st.has_rec
            gterm = (rho / st.s2[st.gid] * (ts - T_hat) ** 2)[mem].sum()
    r = yp - a * ts[st.star]
    br = st.bsum(pi * a * r)
    F = (pi * r * r).sum() - (c * br * br).sum() + gterm
    F = F + (((theta - maps.prior_mean) / maps.prior_sd) ** 2).sum() + pen
    return F


def _var_objective(st: _State, p, w, y, a, f, t_lin, rho, has_sig):
    """-2 log restricted likelihood of (log kappa, log omega, log eta)."""
    U, E = st.n_noise, st.n_effect
    lk = torch.where(has_sig, p[:U], torch.zeros_like(p[:U]))
    lw, le = p[U : 2 * U], p[2 * U : 2 * U + E]
    v = _noise_var(st, lk, lw)
    pi = w / v
    eta2b = st.eta2_blk(le)
    yp = y - f + a * t_lin[st.star]
    m, P, c, saa, say, S = _posterior(st, pi, a, yp, eta2b)
    has = st.has_rec & (P > 0)
    Pc = P.clamp_min(1e-300)
    Q = st.scatter(pi * yp * yp) - st.b2star(c * say * say) - torch.where(has, S * S / Pc, 0.0)
    F = torch.log(v).sum() + torch.log1p(eta2b * saa).sum() + Q[has].sum()
    if st.n_group:
        mem = st.member & has
        s2 = st.s2[st.gid]
        r = rho.clamp(1e-6, 1.0)
        F = F + torch.where(has & ~mem, torch.log(Pc), 0.0).sum()
        F = F + torch.where(mem, torch.log1p(Pc * s2 / r), 0.0).sum()
        om = torch.where(mem, r * Pc / (r + s2 * Pc), torch.zeros_like(P))
        mm = torch.nan_to_num(m)
        G0, G1 = st.gscatter(om, mem), st.gscatter(om * mm, mem)
        G2 = st.gscatter(om * mm * mm, mem)
        okg = G0 > 0
        F = F + (G2 - torch.where(okg, G1 * G1 / G0.clamp_min(1e-300), 0.0))[okg].sum()
        F = F + torch.log(G0[okg]).sum()
    else:
        F = F + torch.log(Pc[has]).sum()
    F = F + (lk**2).sum() + (((lw - LOG_OMEGA0) / 1.5) ** 2).sum()
    F = F + (((le - LOG_OMEGA0) / 1.5) ** 2).sum()
    return F


def _lbfgs(fn, x0, iters):
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=iters, history_size=20, tolerance_grad=1e-10,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")  # fmt: skip

    def closure():
        opt.zero_grad()
        val = fn(x)
        val.backward()
        return val

    opt.step(closure)
    return x.detach()


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_consensus(
    rec: Records,
    groups: Groups,
    maps: MapSet,
    n_noise: int,
    n_effect: int = 0,
    *,
    nu: float = 5.0,
    n_outer: int = 40,
    tol: float = 1e-3,
    inner_iter: int = 80,
    coefs0: list | None = None,
    lk0: np.ndarray | None = None,
    lw0: np.ndarray | None = None,
    le0: np.ndarray | None = None,
    t0: np.ndarray | None = None,
    covariance: bool = True,
    fix_noise: np.ndarray | None = None,
    verbose: bool = False,
    log=print,
) -> FitResult:
    """Alternating maximum likelihood (see the module docstring).

    ``fix_noise``: boolean per noise unit; those units keep their initial
    (``lk0``, ``lw0``) values (e.g. a pseudo-record of known variance)."""
    st = _State(rec, groups, maps, n_noise, n_effect)
    R, U, E = st.R, n_noise, n_effect
    theta = maps.theta0(coefs0)
    lk = torch.as_tensor(lk0 if lk0 is not None else np.zeros(U), dtype=DT)
    lw = torch.as_tensor(lw0 if lw0 is not None else np.full(U, LOG_OMEGA0), dtype=DT)
    le = torch.as_tensor(le0 if le0 is not None else np.full(E, LOG_OMEGA0), dtype=DT)
    has_sig = torch.as_tensor(st.has_sig)
    fixed = torch.as_tensor(np.zeros(U, bool) if fix_noise is None else np.asarray(fix_noise, bool))
    w = torch.ones(R, dtype=DT)
    rho = st.prior * st.member
    if t0 is not None:
        t = torch.as_tensor(np.nan_to_num(t0, nan=T0), dtype=DT)
    else:  # quoted-error weighted mean of the raw values of age-family records
        age = torch.zeros(R, dtype=torch.bool)
        for fam, idx in zip(maps.families, st.fam_idx):
            if isinstance(fam, AgeMaps):
                age[idx] = True
        pi0 = age.to(DT) / (st.sig2 + 0.05**2)
        P0 = st.scatter(pi0)
        t = torch.where(P0 > 0, st.scatter(pi0 * st.y_obs) / P0.clamp_min(1e-300),
                        torch.full((st.n_star,), T0, dtype=DT))  # fmt: skip
    var = torch.full_like(t, 0.01)
    y = st.y_obs.clone()
    history = []
    for it in range(n_outer):
        v = _noise_var(st, lk, lw)
        eta2b = st.eta2_blk(le)
        theta_old = theta.clone()
        if theta.numel():
            theta = _lbfgs(lambda th: _map_objective(st, th, w, v, y, t, rho, eta2b) / R,
                           theta, inner_iter)  # fmt: skip
        coefs = maps.unpack(theta)
        with torch.no_grad():
            pi = w / v
            for _ in range(2):  # Gauss-Newton steps on the latents
                f, a, _ = st.evaluate(coefs, t[st.star])
                m, P, _, _, _, _ = _posterior(st, pi, a, y - f + a * t[st.star], eta2b)
                t_new, var, _, _, _, _ = _group_solve(st, m, P, rho)
                t = torch.where(st.has_rec, t_new, t)
            good = st.has_rec & torch.isfinite(m)
            fm = torch.median(m[good])
            fsd = (1.4826 * torch.median((m[good] - fm).abs())).clamp_min(0.1)
            rho = _responsibility(st, m, P, rho, fm, fsd)
            f, a, _ = st.evaluate(coefs, t[st.star])
        # ---- variances: REML ----------------------------------------------
        p0 = torch.cat([lk, lw, le])
        fixp = torch.cat([fixed, fixed, torch.zeros(E, dtype=torch.bool)])
        p = _lbfgs(lambda q: _var_objective(st, torch.where(fixp, p0, q), w, y, a, f, t, rho,
                                            has_sig) / R, p0, 40)  # fmt: skip
        lk_new = torch.where(has_sig, p[:U].clamp(*LOG_KAPPA_RANGE), torch.zeros_like(p[:U]))
        lk = torch.where(fixed, lk, lk_new)
        lw = torch.where(fixed, lw, p[U : 2 * U].clamp(*LOG_OMEGA_RANGE))
        le = p[2 * U :].clamp(*LOG_OMEGA_RANGE)
        # ---- imputation and Student-t weights ------------------------------
        with torch.no_grad():
            v = _noise_var(st, lk, lw)
            eta2b = st.eta2_blk(le)
            pi = w / v
            yp = y - f + a * t[st.star]
            _, _, c, _, _, _ = _posterior(st, pi, a, yp, eta2b)
            u_hat = (c * st.bsum(pi * a * (yp - a * t[st.star])))[st.blk]  # shared age error
            y, _ = _impute(st, f + a * u_hat, v)
            # Student-t weights judge a record against its own star's other
            # records (records-only latent): disagreement between a star and
            # its co-natal group is handled by the membership responsibility,
            # so a group can never turn a star's records into outliers.
            yp = y - f + a * t[st.star]
            m_r, P_r, c, saa, _, _ = _posterior(st, pi, a, yp, eta2b)
            t_r = torch.where(P_r > 0, m_r, t)
            u_r = (c * st.bsum(pi * a * (yp - a * t_r[st.star])))[st.blk]
            resid = yp - a * (t_r[st.star] + u_r)
            # posterior variance of t + u_b (t and u_b are anti-correlated):
            # u_b | t has mean c sum pi a (y' - a t) and variance c, so
            # Var(t + u_b) = Var(t) / (1 + eta^2 s_aa)^2 + c.
            shrink = (1.0 / (1.0 + eta2b * saa))[st.blk]
            var_r = (1.0 / P_r.clamp_min(1e-300))[st.star]
            Er = resid**2 + a * a * (var_r * shrink**2 + c[st.blk])
            w = torch.where(st.cens, torch.ones_like(Er), (nu + 1.0) / (nu + Er / v))
        dtheta = float((theta - theta_old).abs().max()) if theta.numel() else 0.0
        history.append({"iter": it, "dtheta": dtheta, "eta": torch.exp(le).tolist()})
        if verbose:
            log(f"  outer {it:02d} max|dtheta| {dtheta:.2e} eta {np.round(torch.exp(le).numpy(), 4)}")
        if it >= 3 and dtheta < tol:
            break
    coefs = maps.unpack(theta)
    v = _noise_var(st, lk, lw)
    eta2b = st.eta2_blk(le)
    with torch.no_grad():
        f, a, _ = st.evaluate(coefs, t[st.star])
        m, P, _, _, _, _ = _posterior(st, w / v, a, y - f + a * t[st.star], eta2b)
        t, var, _, _, _, _ = _group_solve(st, m, P, rho)
    cov = vcov = None
    if covariance and theta.numel():
        cov = _fd_covariance(lambda th: _map_objective(st, th, w, v, y, t, rho, eta2b), theta)
    if covariance:
        p0 = torch.cat([lk, lw, le])
        H = torch.autograd.functional.hessian(
            lambda q: _var_objective(st, q, w, y, a, f, t, rho, has_sig), p0)
        keep = torch.cat([has_sig & ~fixed, ~fixed, torch.ones(E, dtype=torch.bool)]).numpy()
        vcov = np.full((len(p0), len(p0)), np.nan)
        Hk = H.numpy()[np.ix_(keep, keep)]
        try:
            vcov[np.ix_(keep, keep)] = 2.0 * np.linalg.inv(Hk)
        except np.linalg.LinAlgError:
            vcov[np.ix_(keep, keep)] = 2.0 * np.linalg.pinv(Hk)
    t_out = torch.where(st.has_rec, t, torch.full_like(t, float("nan")))
    return FitResult(
        maps=maps, coefs=[c.numpy() for c in coefs], theta_cov=cov, log_kappa=lk.numpy(),
        log_omega=lw.numpy(), log_eta=le.numpy(), var_cov=vcov, t=t_out.numpy(),
        var=var.numpy(), rho=rho.numpy(), weight=w.numpy(), history=history, nu=nu,
    )  # fmt: skip


def _fd_covariance(objective, theta, h: float = 1e-5):
    """2 H^-1 of the profiled -2 log L (finite differences of the exact
    envelope gradient)."""

    def grad(th):
        th = th.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(objective(th), th)
        return g.detach()

    n = theta.numel()
    H = torch.zeros(n, n, dtype=DT)
    for k in range(n):
        e = torch.zeros(n, dtype=DT)
        e[k] = h
        H[:, k] = (grad(theta + e) - grad(theta - e)) / (2 * h)
    H = 0.5 * (H + H.T)
    try:
        return 2.0 * np.linalg.inv(H.numpy())
    except np.linalg.LinAlgError:
        return 2.0 * np.linalg.pinv(H.numpy())


# --------------------------------------------------------------------------- #
# Consensus for every star, residuals
# --------------------------------------------------------------------------- #
def _setup(rec, groups, fit: FitResult, n_noise, n_effect):
    st = _State(rec, groups, fit.maps, n_noise, n_effect)
    coefs = [torch.as_tensor(c, dtype=DT) for c in fit.coefs]
    v = _noise_var(st, torch.as_tensor(fit.log_kappa, dtype=DT), torch.as_tensor(fit.log_omega, dtype=DT))
    eta2b = st.eta2_blk(torch.as_tensor(fit.log_eta, dtype=DT))
    return st, coefs, v, eta2b


def solve_stars(rec: Records, groups: Groups, fit: FitResult, n_noise: int, n_effect: int = 0,
                rho: np.ndarray | None = None, iters: int = 25) -> dict:  # fmt: skip
    """Flat-prior consensus for every star: records only (``t_rec``) and with
    the co-natal terms (``t``). Censored records enter through the censored
    likelihood (EM for the mean, observed information for the variance). A
    star whose only records are censored gets a limit (``limit`` = +1 lower,
    -1 upper): ``t`` is where its maps reach the reported values, ``sd`` NaN."""
    st, coefs, v, eta2b = _setup(rec, groups, fit, n_noise, n_effect)
    pi = 1.0 / v
    unc = (~st.cens).to(DT)
    n_unc = st.scatter(unc)
    only_cens = st.has_rec & (n_unc == 0)
    with torch.no_grad():
        P0 = st.scatter(pi * unc)
        t = torch.where(P0 > 0, st.scatter(pi * unc * st.y_obs) / P0.clamp_min(1e-300),
                        st.scatter(st.y_obs) / st.scatter(torch.ones_like(v)).clamp_min(1))  # fmt: skip
        for _ in range(iters):
            f, a, _ = st.evaluate(coefs, t[st.star])
            y, _ = _impute(st, f, v)
            yp = y - f + a * t[st.star]
            m, P, _, _, _, _ = _posterior(st, pi, a, yp, eta2b)
            ycl = torch.where(st.cens, st.y_obs - f + a * t[st.star], torch.zeros_like(f))
            tl = st.scatter(ycl * a) / st.scatter(torch.where(st.cens, a * a, 0.0)).clamp_min(1e-300)
            t = torch.where(only_cens, tl, torch.where(st.has_rec & (P > 0), m, t))
        f, a, _ = st.evaluate(coefs, t[st.star])
        sd = v.sqrt()
        z = torch.where(st.censor > 0, (f - st.y_obs) / sd, (st.y_obs - f) / sd)
        mills = torch.exp(-0.5 * z * z - 0.5 * math.log(2 * math.pi) - torch.special.log_ndtr(z))
        fac = torch.where(st.cens, mills * (z + mills), torch.ones_like(z))
        y, _ = _impute(st, f, v)
        _, P, _, _, _, _ = _posterior(st, pi * fac, a, y - f + a * t[st.star], eta2b)
        P = torch.where(only_cens, torch.zeros_like(P), P)
        limit = torch.where(only_cens, st.scatter(st.censor.to(DT)).sign(), torch.zeros_like(P)).long()
        rho_t = torch.as_tensor(rho, dtype=DT) if rho is not None else st.prior * st.member
        good = st.has_rec & (P > 0)
        tg, vg, T_loo, _, kap, _ = _group_solve(st, torch.where(good, t, 0.0), P.clamp_min(1e-300), rho_t)
        nan = torch.full_like(t, float("nan"))
        t_rec = torch.where(st.has_rec, t, nan)
        sd_rec = torch.where(good, P.clamp_min(1e-300).rsqrt(), nan)
        t_con = torch.where(good, tg, t_rec)
        sd_con = torch.where(good, vg.sqrt(), sd_rec)
        use_g = only_cens & torch.isfinite(T_loo) & (kap > 0) & (torch.nan_to_num(T_loo) * limit >= t * limit)
        t_con = torch.where(use_g, T_loo, t_con)
        sd_con = torch.where(use_g, kap.clamp_min(1e-300).rsqrt(), sd_con)
    return {"t_rec": t_rec.numpy(), "sd_rec": sd_rec.numpy(), "t": t_con.numpy(),
            "sd": sd_con.numpy(), "limit": limit.numpy(), "group_T_loo": T_loo.numpy(),
            "group_kappa": kap.numpy(), "n_rec": st.scatter(torch.ones_like(v)).numpy().astype("i8")}  # fmt: skip


def leave_key_out(rec: Records, fit: FitResult, n_noise: int, n_effect: int, key: np.ndarray,
                  t_lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Per record: the records-only consensus of its star from the records with
    a different ``key`` (e.g. its shared-input group), flat prior, maps
    linearised at ``t_lin``; censored records enter with imputed values."""
    st, coefs, v, eta2b = _setup(rec, Groups.none(rec.n_star), fit, n_noise, n_effect)
    t = torch.as_tensor(np.nan_to_num(t_lin, nan=T0), dtype=DT)
    key = np.asarray(key)
    t_o = np.full(st.R, np.nan)
    v_o = np.full(st.R, np.nan)
    with torch.no_grad():
        f, a, _ = st.evaluate(coefs, t[st.star])
        y, _ = _impute(st, f, v)
        yp = y - f + a * t[st.star]
        for k in np.unique(key):
            sel = torch.as_tensor(key == k)
            pi = torch.where(sel, torch.zeros_like(v), 1.0 / v)
            m, P, _, _, _, _ = _posterior(st, pi, a, yp, eta2b)
            rows = np.flatnonzero(key == k)
            s = st.star[rows]
            ok = (P[s] > 1e-9).numpy()
            t_o[rows[ok]] = m[s].numpy()[ok]
            v_o[rows[ok]] = (1.0 / P[s]).numpy()[ok]
    return t_o, v_o


def record_residuals(rec: Records, fit: FitResult, n_noise: int, n_effect: int,
                     t_other: np.ndarray, var_other: np.ndarray):  # fmt: skip
    """(raw = y - t, mapped = y - f(t), z = mapped / sqrt(v + a^2 (eta^2 + var)))
    of records against a consensus ``t_other`` that excludes them."""
    st, coefs, v, eta2b = _setup(rec, Groups.none(rec.n_star), fit, n_noise, n_effect)
    tt = torch.as_tensor(np.nan_to_num(t_other), dtype=DT)
    with torch.no_grad():
        f, a, _ = st.evaluate(coefs, tt)
    f, a = f.numpy(), a.numpy()
    raw = rec.y - t_other
    mapped = rec.y - f
    z = mapped / np.sqrt(v.numpy() + a * a * (eta2b[st.blk].numpy() + var_other))
    bad = ~np.isfinite(t_other)
    for arr in (raw, mapped, z):
        arr[bad] = np.nan
    return raw, mapped, z


def map_values(fit: FitResult, name: str, t, feh=0.0, cls=0, z=0.0):
    """(f, df/dt, se of f) of one map at latents ``t`` (numpy)."""
    j, local = fit.maps.locate(name)
    fam = fit.maps.families[j]
    t = torch.as_tensor(np.atleast_1d(np.asarray(t, float)), dtype=DT)
    n = len(t)
    x = torch.as_tensor(np.broadcast_to(clip_feh(feh), (n,)).copy(), dtype=DT)
    zz = torch.full((n, 1), float(z), dtype=DT)
    c = torch.full((n,), int(cls), dtype=torch.long)
    loc = torch.full((n,), local, dtype=torch.long)

    def ev(coef_row):
        coef = torch.as_tensor(fit.coefs[j], dtype=DT).clone()
        coef[local] = coef_row
        return fam.evaluate(coef, loc, t, x, zz, c)

    base = torch.as_tensor(fit.coefs[j][local], dtype=DT)
    with torch.no_grad():
        f, d = ev(base)
    se = np.full(n, np.nan)
    if fit.theta_cov is not None:
        pos, cols = fit.maps.theta_slice(name)
        if len(pos):
            G = np.zeros((n, len(pos)))
            for q, col in enumerate(cols):
                e = torch.zeros_like(base)
                e[col] = 1e-6
                with torch.no_grad():
                    f2, _ = ev(base + e)
                G[:, q] = (f2 - f).numpy() / 1e-6
            V = fit.theta_cov[np.ix_(pos, pos)]
            se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", G, V, G), 0, None))
    return f.numpy(), d.numpy(), se
