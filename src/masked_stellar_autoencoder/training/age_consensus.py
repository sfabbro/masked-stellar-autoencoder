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
  pipelines on the same Gaia photometry/parallax), with an optional nested
  sub-family ``u_{i,k}`` (e.g. two pipelines on the same spectra), so
  correlated catalogues are not counted as independent votes. They cancel in
  contrasts within a (sub-)family, so the etas are learned only from
  independent constraints (another family or a co-natal partner).
* records at a catalogue cap (floor) are right- (left-) censored; a record
  that only bounds the value to an interval [y, y2] has a flat-top
  (interval-censored) likelihood;
* co-natal groups (clusters, wide binaries) share a fitted age latent T_g:
  ``t_i - T_g ~ Student-t(member_nu, s_g)`` weighted by a fixed external
  membership probability (HR23, VB21, 1 - R_chance); no field density, so the
  likelihood is covariant under any common change of the latent scale and a
  change of gauge is an exact reparametrisation. T_g has a flat prior; a
  literature group age enters as a record of a pseudo-member (the group node)
  with its own map and noise, never as a fixed value.
* ``noise_identified`` says which units have an identified noise variance
  (triads, or pairs with an identified unit): only those may carry slope or
  [Fe/H] map terms, since the errors-in-variables slope needs the variance
  ratio.

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
    censor: (
        np.ndarray
    )  # (R,) int8: +1 lower limit (cap), -1 upper limit, 2 interval [y, y2]
    n_star: int
    effect: np.ndarray | None = None  # (R,) int64 shared-effect family, -1 none
    extra: np.ndarray | None = None  # (R, q) further covariates (e.g. colour)
    subeffect: np.ndarray | None = None  # (R,) int64 nested sub-family, -1 none
    y2: np.ndarray | None = None  # (R,) upper bound of interval records (censor 2)

    def __post_init__(self):
        n = len(self.y)
        if self.effect is None:
            self.effect = np.full(n, -1, "i8")
        if self.extra is None:
            self.extra = np.zeros((n, 0))
        if self.subeffect is None:
            self.subeffect = np.full(n, -1, "i8")
        if self.y2 is None:
            self.y2 = np.asarray(self.y, float).copy()
        for name in ("star", "map_id", "cls", "noise_id", "sig", "feh", "censor", "effect",
                     "extra", "subeffect", "y2"):  # fmt: skip
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"Records.{name} has {len(getattr(self, name))} rows, y has {n}"
                )

    def subset(self, keep: np.ndarray) -> Records:
        return Records(
            self.star[keep], self.map_id[keep], self.cls[keep], self.noise_id[keep],
            self.y[keep], self.sig[keep], self.feh[keep], self.censor[keep], self.n_star,
            self.effect[keep], self.extra[keep], self.subeffect[keep], self.y2[keep],
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
    deltas db_c/dA_c where ``class_terms``, beta where ``use_feh``), FLEX (+ g1,
    g3, g4; the cubic g2 is in the layout but not fitted).
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
        xr = (
            np.tile(FEH_CLIP, (n, 1)) if x_range is None else np.asarray(x_range, float)
        )
        tr = (
            np.tile((-3.0, 3.0), (n, 1))
            if t_range is None
            else np.asarray(t_range, float)
        )
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
        # the cubic term (column 4 + 2c) is kept in the layout but not fitted:
        # it was the one poorly identified direction (slow drift, no gain)
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
        return torch.cat(
            [torch.as_tensor(c, dtype=DT)[m] for c, m in zip(coefs, self.masks)]
        )

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
    log_eta: np.ndarray  # (E + S,) outer families, then nested sub-families
    var_cov: np.ndarray | None  # covariance of (log kappa, log omega, log eta)
    t: np.ndarray  # (n_star,) latent of the fit stars
    var: np.ndarray
    rho: np.ndarray  # (n_star,) co-natal member weight (prior x Student-t weight)
    weight: np.ndarray  # (R,) Student-t weights of the fit records
    history: list = field(default_factory=list)
    nu: float = 5.0
    n_effect: int = 0
    n_sub: int = 0

    def coef_of(self, name: str) -> np.ndarray:
        j, local = self.maps.locate(name)
        return self.coefs[j][local]


# --------------------------------------------------------------------------- #
# Noise identifiability (which units may carry slope / [Fe/H] / class terms)
# --------------------------------------------------------------------------- #
def noise_identified(star, group, unit, n_unit: int, partner=None, min_count: int = 30):
    """Units whose noise variance is identified by the overlap pattern.

    In an errors-in-variables fit the slope of a map depends on the ratio of
    the two error variances; a pair of measurements of the same stars only
    identifies their sum. A unit's own variance is identified (i) by triads:
    stars where it is measured together with at least two other independent
    measurements (records of other shared-input ``group``s, or a co-natal
    partner), or (ii) by pairs with a unit already identified (its variance is
    then the pair variance minus the known one); (ii) is iterated. ``min_count``
    is the same count that the offset tier requires, not a tuned threshold.

    Returns (identified bool[n_unit], triad counts[n_unit], pair-with-identified
    counts[n_unit])."""
    star = np.asarray(star, "i8")
    group = np.asarray(group, "i8")
    unit = np.asarray(unit, "i8")
    n_star = int(star.max()) + 1 if len(star) else 0
    partner = np.zeros(n_star, bool) if partner is None else np.asarray(partner, bool)
    sg = (
        np.unique(star * (int(group.max()) + 2) + group)
        if len(star)
        else np.zeros(0, "i8")
    )
    n_grp = (
        np.bincount(sg // (int(group.max()) + 2), minlength=n_star)
        if len(star)
        else np.zeros(0)
    )
    n_ind = n_grp + partner[: len(n_grp)]
    triad = np.bincount(unit[n_ind[star] >= 3], minlength=n_unit)
    ident = triad >= min_count
    pair = np.zeros(n_unit, "i8")
    for _ in range(n_unit + 1):
        idr = ident[unit].astype("i8")
        tot = np.bincount(star, weights=idr, minlength=n_star)
        key = star * (int(group.max()) + 2) + group
        uk, inv = np.unique(key, return_inverse=True)
        in_grp = np.bincount(inv, weights=idr)[inv]
        has_other = (tot[star] - in_grp) > 0
        pair = np.bincount(unit[has_other], minlength=n_unit)
        new = ident | (pair >= min_count)
        if np.array_equal(new, ident):
            break
        ident = new
    return ident, triad, pair


# --------------------------------------------------------------------------- #
# Internal tensor state
# --------------------------------------------------------------------------- #
class _State:
    def __init__(self, rec: Records, groups: Groups, maps: MapSet, n_noise: int,
                 n_effect: int, n_sub: int = 0):  # fmt: skip
        self.maps = maps
        self.n_star = rec.n_star
        self.R = len(rec.y)
        self.star = torch.as_tensor(rec.star, dtype=torch.long)
        fam = maps.family_of(rec.map_id)
        self.fam_idx = [
            torch.as_tensor(np.flatnonzero(fam == j)) for j in range(len(maps.families))
        ]
        self.local = [torch.as_tensor(rec.map_id[fam == j] - maps.offsets[j])
                      for j in range(len(maps.families))]  # fmt: skip
        self.cls = torch.as_tensor(rec.cls, dtype=torch.long)
        self.noise_id = torch.as_tensor(rec.noise_id, dtype=torch.long)
        self.y_obs = torch.as_tensor(rec.y, dtype=DT)
        self.y_hi = torch.as_tensor(rec.y2, dtype=DT)
        sig = np.where(np.isfinite(rec.sig) & (rec.sig > 0), rec.sig, 0.0)
        self.sig2 = torch.as_tensor(sig, dtype=DT) ** 2
        self.x = torch.as_tensor(clip_feh(rec.feh), dtype=DT)
        self.z = torch.as_tensor(np.nan_to_num(np.asarray(rec.extra, float)), dtype=DT)
        self.censor = torch.as_tensor(rec.censor.astype("i8"))
        self.cens = self.censor != 0
        self.has_sig = np.zeros(n_noise, bool)
        np.logical_or.at(self.has_sig, rec.noise_id, sig > 0)
        self.n_noise, self.n_effect, self.n_sub = n_noise, n_effect, n_sub
        # shared age errors, two nested levels: outer blocks (star, family) and
        # sub-blocks (outer block, sub-family); index -1 means "none" (eta = 0)
        eff = np.asarray(rec.effect, "i8")
        sub = np.asarray(rec.subeffect, "i8")
        if np.any((sub >= 0) & (eff < 0)):
            raise ValueError("a sub-family record must belong to an outer family")
        okey = rec.star.astype("i8") * (n_effect + 1) + (eff + 1)
        uo, blk_o = np.unique(okey, return_inverse=True)
        kkey = blk_o.astype("i8") * (n_sub + 1) + (sub + 1)
        uk, blk_k = np.unique(kkey, return_inverse=True)
        self.blk_o, self.blk_k = torch.as_tensor(blk_o), torch.as_tensor(blk_k)
        self.n_o, self.n_k = len(uo), len(uk)
        self.ostar = torch.as_tensor(uo // (n_effect + 1))
        self.ofam = torch.as_tensor(uo % (n_effect + 1) - 1)
        self.k2o = torch.as_tensor(uk // (n_sub + 1))
        self.kstar = self.ostar[self.k2o]
        self.kfam = torch.as_tensor(uk % (n_sub + 1) - 1)
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

    def osum(self, values):
        return torch.zeros(self.n_o, dtype=DT).index_add(0, self.blk_o, values)

    def ksum(self, values):
        return torch.zeros(self.n_k, dtype=DT).index_add(0, self.blk_k, values)

    def k2osum(self, values):
        return torch.zeros(self.n_o, dtype=DT).index_add(0, self.k2o, values)

    def o2star(self, values):
        return torch.zeros(self.n_star, dtype=DT).index_add(0, self.ostar, values)

    def k2star(self, values):
        return torch.zeros(self.n_star, dtype=DT).index_add(0, self.kstar, values)

    def gscatter(self, values, members):
        return torch.zeros(max(self.n_group, 1), dtype=DT).index_add(0, self.gid[members],
                                                                     values[members])  # fmt: skip

    def evaluate(self, coefs, t_rec):
        """(f, df/dt) for all records at per-record latent ``t_rec``."""
        f = torch.zeros(self.R, dtype=DT)
        a = torch.zeros(self.R, dtype=DT)
        pen = t_rec.new_zeros(())
        for fam, idx, loc, coef in zip(
            self.maps.families, self.fam_idx, self.local, coefs
        ):
            if not len(idx):
                continue
            fj, aj = fam.evaluate(
                coef, loc, t_rec[idx], self.x[idx], self.z[idx], self.cls[idx]
            )
            f = f.index_copy(0, idx, fj)
            a = a.index_copy(0, idx, aj)
            pen = pen + fam.penalty(aj)
        return f, a, pen

    def eta2(self, le):
        """(eta^2 per outer block, eta^2 per sub-block) from log eta (E + S)."""
        E, S = self.n_effect, self.n_sub
        z = torch.zeros(1, dtype=DT)
        e2o = torch.cat([z, torch.exp(2 * le[:E])]) if E else z
        e2k = torch.cat([z, torch.exp(2 * le[E : E + S])]) if S else z
        return e2o[self.ofam + 1], e2k[self.kfam + 1]


def _noise_var(st: _State, lk, lw):
    return torch.exp(2 * lk)[st.noise_id] * st.sig2 + torch.exp(2 * lw)[st.noise_id]


def _posterior(st: _State, pi, a, yp, e2):
    """Records-only flat-prior posterior of the latents with the shared age
    errors integrated out. Per star the covariance of the linearised records
    y' = a (t + u_o + u_k) + e is diag(v) + sum_k eta_k^2 a_k a_k^T +
    eta_o^2 a_o a_o^T (sub-blocks k nested in outer blocks o): two-level
    Sherman-Morrison. Returns a dict with m, P, S, yy (= y'^T Sigma^-1 y'),
    the per-block quantities and the extra log-determinant."""
    e2o, e2k = e2
    saa_k = st.ksum(pi * a * a)
    say_k = st.ksum(pi * a * yp)
    c_k = e2k / (1.0 + e2k * saa_k)
    aoo = st.osum(pi * a * a) - st.k2osum(c_k * saa_k * saa_k)
    aoy = st.osum(pi * a * yp) - st.k2osum(c_k * saa_k * say_k)
    c_o = e2o / (1.0 + e2o * aoo)
    P = (
        st.scatter(pi * a * a)
        - st.k2star(c_k * saa_k * saa_k)
        - st.o2star(c_o * aoo * aoo)
    )
    S = (
        st.scatter(pi * a * yp)
        - st.k2star(c_k * saa_k * say_k)
        - st.o2star(c_o * aoo * aoy)
    )
    yy = (
        st.scatter(pi * yp * yp)
        - st.k2star(c_k * say_k * say_k)
        - st.o2star(c_o * aoy * aoy)
    )
    m = torch.where(P > 0, S / P.clamp_min(1e-300), torch.full_like(P, float("nan")))
    logdet = torch.log1p(e2k * saa_k).sum() + torch.log1p(e2o * aoo).sum()
    return {"m": m, "P": P, "S": S, "yy": yy, "c_k": c_k, "c_o": c_o, "saa_k": saa_k,
            "aoo": aoo, "logdet": logdet}  # fmt: skip


def _shared_modes(st: _State, post, pi, a, r):
    """Posterior modes of the shared errors given residuals r = y' - a t:
    per-record total a-scaled error (u_o + u_k) and the penalty sum u^2/eta^2."""
    c_k, c_o, saa_k = post["c_k"], post["c_o"], post["saa_k"]
    sar_k = st.ksum(pi * a * r)
    aor = st.osum(pi * a * r) - st.k2osum(c_k * saa_k * sar_k)
    u_o = c_o * aor
    u_k = c_k * (sar_k - saa_k * u_o[st.k2o])
    return u_o, u_k


def _shared_penalty(u_o, u_k, e2):
    e2o, e2k = e2
    po, pk = e2o > 0, e2k > 0
    return (u_o[po] ** 2 / e2o[po]).sum() + (u_k[pk] ** 2 / e2k[pk]).sum()


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
    """E[y*] and Var[y*] of censored records given the predicted mean ``mu``:
    +1 lower limit (y* >= y), -1 upper limit (y* <= y), +2 interval
    (y <= y* <= y2, a flat-top likelihood)."""
    y = st.y_obs.clone()
    cvar = torch.zeros_like(y)
    if not bool(st.cens.any()):
        return y, cvar
    sd = v.sqrt()
    up, lo, iv = st.censor == 1, st.censor == -1, st.censor == 2
    a = (st.y_obs - mu) / sd
    lphi = -0.5 * a * a - 0.5 * math.log(2 * math.pi)
    lam = torch.exp(lphi - torch.special.log_ndtr(-a))  # true value >= c
    y = torch.where(up, mu + sd * lam, y)
    cvar = torch.where(up, v * (1 + a * lam - lam * lam).clamp_min(1e-6), cvar)
    lam2 = torch.exp(lphi - torch.special.log_ndtr(a))  # true value <= c
    y = torch.where(lo, mu - sd * lam2, y)
    cvar = torch.where(lo, v * (1 - a * lam2 - lam2 * lam2).clamp_min(1e-6), cvar)
    if bool(iv.any()):
        ey, vy = _interval_moments(a, (st.y_hi - mu) / sd)
        y = torch.where(iv, mu + sd * ey, y)
        cvar = torch.where(iv, (v * vy).clamp_min(1e-8), cvar)
    return y, cvar


def _interval_moments(al, be):
    """Mean and variance of a standard normal truncated to [al, be]."""
    pa, pb = torch.exp(-0.5 * al * al), torch.exp(-0.5 * be * be)
    pa, pb = pa / math.sqrt(2 * math.pi), pb / math.sqrt(2 * math.pi)
    Z = (torch.special.ndtr(be) - torch.special.ndtr(al)).clamp_min(1e-300)
    mean = (pa - pb) / Z
    var = 1.0 + (al * pa - be * pb) / Z - mean * mean
    return mean, var


def _member_weights(st: _State, m, P, rho, nu_m: float):
    """Fixed external membership prior x Student-t member weight.

    The co-natal term of a member is t_i - T_g ~ Student-t(nu_m, s_g) written
    as a Gaussian scale mixture: the EM weight is (nu_m + 1)/(nu_m + d^2), with
    d the records-only latent's deviation from the leave-one-out group mean
    over its predicted spread. No field density is involved, so the term is
    covariant under any common change of the latent scale (gauge)."""
    if st.n_group == 0:
        return rho
    _, _, T_loo, vT_loo, _, _ = _group_solve(st, m, P, rho)
    V = 1.0 / P.clamp_min(1e-300)
    ok = st.member & st.has_rec & torch.isfinite(T_loo) & (P > 0)
    d2 = (m - T_loo) ** 2 / (V + st.s2[st.gid] + vT_loo)
    lam = (nu_m + 1.0) / (nu_m + torch.nan_to_num(d2, nan=0.0))
    base = torch.where(st.member, st.prior, torch.zeros_like(rho))
    return torch.where(ok, base * lam.clamp(max=1.0), base)


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def _map_objective(st: _State, theta, w, v, y, t_lin, rho, e2):
    """Profile objective (latents, shared errors and group ages profiled at
    their joint optimum, which is then held fixed: envelope gradient)."""
    maps = st.maps
    coefs = maps.unpack(theta)
    tl = t_lin[st.star]
    f, a, pen = st.evaluate(coefs, tl)
    yp = y - f + a * tl
    pi = w / v
    with torch.no_grad():
        ad, ypd = a.detach(), yp.detach()
        post = _posterior(st, pi, ad, ypd, e2)
        ts, _, _, _, _, T_hat = _group_solve(st, post["m"], post["P"], rho)
        u_o, u_k = _shared_modes(st, post, pi, ad, ypd - ad * ts[st.star])
        uterm = _shared_penalty(u_o, u_k, e2)
        ushare = u_o[st.blk_o] + u_k[st.blk_k]
        gterm = 0.0
        if st.n_group:
            mem = st.member & st.has_rec
            gterm = (rho / st.s2[st.gid] * (ts - T_hat) ** 2)[mem].sum()
    r = yp - a * (ts[st.star] + ushare)
    F = (pi * r * r).sum() + uterm + gterm
    F = F + (((theta - maps.prior_mean) / maps.prior_sd) ** 2).sum() + pen
    return F


def _var_objective(st: _State, p, w, y, a, f, t_lin, rho, has_sig):
    """-2 log restricted likelihood of (log kappa, log omega, log eta)."""
    U, E, Sn = st.n_noise, st.n_effect, st.n_sub
    lk = torch.where(has_sig, p[:U], torch.zeros_like(p[:U]))
    lw, le = p[U : 2 * U], p[2 * U : 2 * U + E + Sn]
    v = _noise_var(st, lk, lw)
    pi = w / v
    e2 = st.eta2(le)
    yp = y - f + a * t_lin[st.star]
    post = _posterior(st, pi, a, yp, e2)
    m, P, S = post["m"], post["P"], post["S"]
    has = st.has_rec & (P > 0)
    Pc = P.clamp_min(1e-300)
    Q = post["yy"] - torch.where(has, S * S / Pc, 0.0)
    F = torch.log(v).sum() + post["logdet"] + Q[has].sum()
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


def _record_weights(st, f, a, y, t, w, v, e2, nu):
    """Imputed values and Student-t weights. A record is judged against its
    own star's other records (records-only latent); disagreement between a
    star and its co-natal group is the member weight's job."""
    pi = w / v
    yp = y - f + a * t[st.star]
    post = _posterior(st, pi, a, yp, e2)
    u_o, u_k = _shared_modes(st, post, pi, a, yp - a * t[st.star])
    y, _ = _impute(st, f + a * (u_o[st.blk_o] + u_k[st.blk_k]), v)
    yp = y - f + a * t[st.star]
    post = _posterior(st, pi, a, yp, e2)
    t_r = torch.where(post["P"] > 0, post["m"], t)
    u_o, u_k = _shared_modes(st, post, pi, a, yp - a * t_r[st.star])
    resid = yp - a * (t_r[st.star] + u_o[st.blk_o] + u_k[st.blk_k])
    # posterior variance of t + u_o + u_k (anti-correlated): exact for one
    # level, Var(t)/(1 + eta^2 s_aa)^2 + c; the nested form chains the shrinks
    e2o, e2k = e2
    s_k = (1.0 / (1.0 + e2k * post["saa_k"]))[st.blk_k]
    s_o = (1.0 / (1.0 + e2o * post["aoo"]))[st.blk_o]
    var_r = (1.0 / post["P"].clamp_min(1e-300))[st.star]
    var_tu = (
        var_r * (s_o * s_k) ** 2
        + post["c_o"][st.blk_o] * s_k**2
        + post["c_k"][st.blk_k]
    )
    Er = resid**2 + a * a * var_tu
    w = torch.where(st.cens, torch.ones_like(Er), (nu + 1.0) / (nu + Er / v))
    return y, w


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
    n_sub: int = 0,
    nu: float = 5.0,
    member_nu: float = 4.0,
    n_outer: int = 40,
    tol: float = 1e-3,
    inner_iter: int = 80,
    coefs0: list | None = None,
    lk0: np.ndarray | None = None,
    lw0: np.ndarray | None = None,
    le0: np.ndarray | None = None,
    t0: np.ndarray | None = None,
    w0: np.ndarray | None = None,
    rho0: np.ndarray | None = None,
    covariance: bool = True,
    fix_noise: np.ndarray | None = None,
    verbose: bool = False,
    log=print,
) -> FitResult:
    """Alternating maximum likelihood (see the module docstring).

    ``n_effect`` outer and ``n_sub`` nested shared-error families; ``member_nu``
    the Student-t index of the co-natal member term; ``w0``/``rho0`` optional
    starting record and member weights (with ``t0`` and the coefficients, a
    warm start from a transformed solution); ``fix_noise``: boolean per noise
    unit kept at its initial (``lk0``, ``lw0``) values."""
    st = _State(rec, groups, maps, n_noise, n_effect, n_sub)
    R, U, E = st.R, n_noise, n_effect + n_sub
    theta = maps.theta0(coefs0)
    lk = torch.as_tensor(lk0 if lk0 is not None else np.zeros(U), dtype=DT)
    lw = torch.as_tensor(lw0 if lw0 is not None else np.full(U, LOG_OMEGA0), dtype=DT)
    le = torch.as_tensor(le0 if le0 is not None else np.full(E, LOG_OMEGA0), dtype=DT)
    has_sig = torch.as_tensor(st.has_sig)
    fixed = torch.as_tensor(
        np.zeros(U, bool) if fix_noise is None else np.asarray(fix_noise, bool)
    )
    w = torch.as_tensor(w0, dtype=DT) if w0 is not None else torch.ones(R, dtype=DT)
    rho = torch.as_tensor(rho0, dtype=DT) if rho0 is not None else st.prior * st.member
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
    if t0 is not None:  # warm start: impute censored values at the given latents
        with torch.no_grad():
            f, a, _ = st.evaluate(maps.unpack(theta), t[st.star])
            y, _ = _impute(st, f, _noise_var(st, lk, lw))
    history = []
    for it in range(n_outer):
        v = _noise_var(st, lk, lw)
        e2 = st.eta2(le)
        theta_old = theta.clone()
        if theta.numel():
            theta = _lbfgs(lambda th: _map_objective(st, th, w, v, y, t, rho, e2) / R,
                           theta, inner_iter)  # fmt: skip
        coefs = maps.unpack(theta)
        with torch.no_grad():
            pi = w / v
            for _ in range(2):  # Gauss-Newton steps on the latents
                f, a, _ = st.evaluate(coefs, t[st.star])
                post = _posterior(st, pi, a, y - f + a * t[st.star], e2)
                t_new, var, _, _, _, _ = _group_solve(st, post["m"], post["P"], rho)
                t = torch.where(st.has_rec, t_new, t)
            rho = _member_weights(st, post["m"], post["P"], rho, member_nu)
            f, a, _ = st.evaluate(coefs, t[st.star])
        # ---- variances: REML ----------------------------------------------
        p0 = torch.cat([lk, lw, le])
        fixp = torch.cat([fixed, fixed, torch.zeros(E, dtype=torch.bool)])
        p = _lbfgs(lambda q: _var_objective(st, torch.where(fixp, p0, q), w, y, a, f, t, rho,
                                            has_sig) / R, p0, 40)  # fmt: skip
        lk_new = torch.where(
            has_sig, p[:U].clamp(*LOG_KAPPA_RANGE), torch.zeros_like(p[:U])
        )
        lk = torch.where(fixed, lk, lk_new)
        lw = torch.where(fixed, lw, p[U : 2 * U].clamp(*LOG_OMEGA_RANGE))
        le = p[2 * U :].clamp(*LOG_OMEGA_RANGE)
        # ---- imputation and Student-t weights ------------------------------
        with torch.no_grad():
            v = _noise_var(st, lk, lw)
            y, w = _record_weights(st, f, a, y, t, w, v, st.eta2(le), nu)
        dtheta = float((theta - theta_old).abs().max()) if theta.numel() else 0.0
        history.append({"iter": it, "dtheta": dtheta, "eta": torch.exp(le).tolist()})
        if verbose:
            log(
                f"  outer {it:02d} max|dtheta| {dtheta:.2e} eta {np.round(torch.exp(le).numpy(), 4)}"
            )
        if it >= 3 and dtheta < tol:
            break
    coefs = maps.unpack(theta)
    v = _noise_var(st, lk, lw)
    e2 = st.eta2(le)
    with torch.no_grad():
        f, a, _ = st.evaluate(coefs, t[st.star])
        post = _posterior(st, w / v, a, y - f + a * t[st.star], e2)
        t, var, _, _, _, _ = _group_solve(st, post["m"], post["P"], rho)
    cov = vcov = None
    if covariance and theta.numel():
        cov = _fd_covariance(
            lambda th: _map_objective(st, th, w, v, y, t, rho, e2), theta
        )
    if covariance:
        p0 = torch.cat([lk, lw, le])
        H = torch.autograd.functional.hessian(
            lambda q: _var_objective(st, q, w, y, a, f, t, rho, has_sig), p0
        )
        keep = torch.cat(
            [has_sig & ~fixed, ~fixed, torch.ones(E, dtype=torch.bool)]
        ).numpy()
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
        n_effect=n_effect, n_sub=n_sub,
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
def _setup(rec, groups, fit: FitResult, n_noise):
    st = _State(rec, groups, fit.maps, n_noise, fit.n_effect, fit.n_sub)
    coefs = [torch.as_tensor(c, dtype=DT) for c in fit.coefs]
    v = _noise_var(st, torch.as_tensor(fit.log_kappa, dtype=DT),
                   torch.as_tensor(fit.log_omega, dtype=DT))  # fmt: skip
    e2 = st.eta2(torch.as_tensor(fit.log_eta, dtype=DT))
    return st, coefs, v, e2


def solve_stars(rec: Records, groups: Groups, fit: FitResult, n_noise: int, n_effect=None,
                rho: np.ndarray | None = None, iters: int = 25) -> dict:  # fmt: skip
    """Flat-prior consensus for every star: records only (``t_rec``) and with
    the co-natal terms (``t``). Censored records enter through the censored
    likelihood (EM for the mean, observed information for the variance). A
    star whose only records are one-sided limits gets a limit (``limit`` = +1
    lower, -1 upper): ``t`` is where its maps reach the reported values, ``sd``
    NaN. (``n_effect`` is taken from ``fit``; the argument is kept for API
    compatibility.)"""
    st, coefs, v, e2 = _setup(rec, groups, fit, n_noise)
    pi = 1.0 / v
    onesided = (st.censor == 1) | (st.censor == -1)
    unc = (~onesided).to(DT)
    n_unc = st.scatter(unc)
    only_cens = st.has_rec & (n_unc == 0)
    ymid = torch.where(st.censor == 2, 0.5 * (st.y_obs + st.y_hi), st.y_obs)
    with torch.no_grad():
        P0 = st.scatter(pi * unc)
        t = torch.where(P0 > 0, st.scatter(pi * unc * ymid) / P0.clamp_min(1e-300),
                        st.scatter(st.y_obs) / st.scatter(torch.ones_like(v)).clamp_min(1))  # fmt: skip
        for _ in range(iters):
            f, a, _ = st.evaluate(coefs, t[st.star])
            y, _ = _impute(st, f, v)
            yp = y - f + a * t[st.star]
            post = _posterior(st, pi, a, yp, e2)
            m, P = post["m"], post["P"]
            ycl = torch.where(
                onesided, st.y_obs - f + a * t[st.star], torch.zeros_like(f)
            )
            tl = st.scatter(ycl * a) / st.scatter(
                torch.where(onesided, a * a, 0.0)
            ).clamp_min(1e-300)
            t = torch.where(only_cens, tl, torch.where(st.has_rec & (P > 0), m, t))
        f, a, _ = st.evaluate(coefs, t[st.star])
        sd = v.sqrt()
        z = torch.where(st.censor > 0, (f - st.y_obs) / sd, (st.y_obs - f) / sd)
        mills = torch.exp(
            -0.5 * z * z - 0.5 * math.log(2 * math.pi) - torch.special.log_ndtr(z)
        )
        fac = torch.where(onesided, mills * (z + mills), torch.ones_like(z))
        iv = st.censor == 2
        if bool(iv.any()):
            al, be = (st.y_obs - f) / sd, (st.y_hi - f) / sd
            pa = torch.exp(-0.5 * al * al) / math.sqrt(2 * math.pi)
            pb = torch.exp(-0.5 * be * be) / math.sqrt(2 * math.pi)
            Z = (torch.special.ndtr(be) - torch.special.ndtr(al)).clamp_min(1e-300)
            fiv = ((pa - pb) / Z) ** 2 - (al * pa - be * pb) / Z
            fac = torch.where(iv, fiv.clamp_min(0.0), fac)
        y, _ = _impute(st, f, v)
        P = _posterior(st, pi * fac, a, y - f + a * t[st.star], e2)["P"]
        P = torch.where(only_cens, torch.zeros_like(P), P)
        lim_sign = st.scatter(torch.where(onesided, st.censor.to(DT), 0.0)).sign()
        limit = torch.where(only_cens, lim_sign, torch.zeros_like(P)).long()
        rho_t = (
            torch.as_tensor(rho, dtype=DT) if rho is not None else st.prior * st.member
        )
        good = st.has_rec & (P > 0)
        tg, vg, T_loo, _, kap, _ = _group_solve(
            st, torch.where(good, t, 0.0), P.clamp_min(1e-300), rho_t
        )
        nan = torch.full_like(t, float("nan"))
        t_rec = torch.where(st.has_rec, t, nan)
        sd_rec = torch.where(good, P.clamp_min(1e-300).rsqrt(), nan)
        t_con = torch.where(good, tg, t_rec)
        sd_con = torch.where(good, vg.sqrt(), sd_rec)
        use_g = (only_cens & torch.isfinite(T_loo) & (kap > 0)
                 & (torch.nan_to_num(T_loo) * limit >= t * limit))  # fmt: skip
        t_con = torch.where(use_g, T_loo, t_con)
        sd_con = torch.where(use_g, kap.clamp_min(1e-300).rsqrt(), sd_con)
    return {"t_rec": t_rec.numpy(), "sd_rec": sd_rec.numpy(), "t": t_con.numpy(),
            "sd": sd_con.numpy(), "limit": limit.numpy(), "group_T_loo": T_loo.numpy(),
            "group_kappa": kap.numpy(), "n_rec": st.scatter(torch.ones_like(v)).numpy().astype("i8")}  # fmt: skip


def leave_key_out(rec: Records, fit: FitResult, n_noise: int, n_effect, key: np.ndarray,
                  t_lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Per record: the records-only consensus of its star from the records with
    a different ``key`` (e.g. its shared-input group), flat prior, maps
    linearised at ``t_lin``; censored records enter with imputed values."""
    st, coefs, v, e2 = _setup(rec, Groups.none(rec.n_star), fit, n_noise)
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
            post = _posterior(st, pi, a, yp, e2)
            m, P = post["m"], post["P"]
            rows = np.flatnonzero(key == k)
            s = st.star[rows]
            ok = (P[s] > 1e-9).numpy()
            t_o[rows[ok]] = m[s].numpy()[ok]
            v_o[rows[ok]] = (1.0 / P[s]).numpy()[ok]
    return t_o, v_o


def record_residuals(rec: Records, fit: FitResult, n_noise: int, n_effect,
                     t_other: np.ndarray, var_other: np.ndarray):  # fmt: skip
    """(raw = y - t, mapped = y - f(t), z = mapped / sqrt(v + a^2 (eta^2 + var)))
    of records against a consensus ``t_other`` that excludes them."""
    st, coefs, v, e2 = _setup(rec, Groups.none(rec.n_star), fit, n_noise)
    tt = torch.as_tensor(np.nan_to_num(t_other), dtype=DT)
    with torch.no_grad():
        f, a, _ = st.evaluate(coefs, tt)
    f, a = f.numpy(), a.numpy()
    raw = rec.y - t_other
    mapped = rec.y - f
    eta2 = e2[0][st.blk_o].numpy() + e2[1][st.blk_k].numpy()
    z = mapped / np.sqrt(v.numpy() + a * a * (eta2 + var_other))
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
