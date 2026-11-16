"""Survey maps that depend on all stellar parameters, with a local gauge.

Each survey s reports, for common-scale parameters x = (Teff, log g, [Fe/H],
[alpha]) in scaled units,

    y_s(x) = x + delta_s(x),
    delta_s(x) = b_s + L_s x + C_s phi(x).

* ``b_s`` (4) and ``L_s`` (4x4) give every survey an offset and linear trends,
  including cross terms ([Fe/H] offset that grows with Teff, ...).
* ``phi`` is one small MLP shared by all surveys (K basis functions). ``C_s``
  (4xK) picks each survey's mix. Large surveys teach the shapes; a small
  survey only fits its few coefficients, so a few hundred stars suffice.
* All survey coefficients start at zero (identity) and carry an L2 prior. In
  a mean loss that prior counts relatively more for a survey with fewer
  records, so complexity follows the data a survey has.
* Each map should stay increasing in its own parameter: a penalty holds
  1 + d delta_j / d x_j above ``min_slope``.

Gauge (what makes the common scale physical; v4 adds ``hard_targets`` and a
fixed ``precision``, see the end of this docstring). The maps alone leave any
smooth redefinition of the common scale free. It is fixed locally, at every
point x the batch predicts, by a weak penalty on the weighted mean
correction over surveys:

    sum_s w_s,j(x) delta_s,j(x) = 0,
    w_s,j(x) = coverage_s,j(x) * precision_s,j * trust_s,j / group_s,j.

``coverage`` (does survey s have stars near x) comes from a histogram of its
records, so a survey never votes where it has no data. ``precision`` is a
running estimate from each survey's residuals, the iterative reference:
DESI or BOSSNet count less than APOGEE where they scatter more, and a survey
votes once its residuals have been seen. ``trust``
up-weights high-resolution references, and ``group`` makes pipelines run on
the same spectra count once. Records of anchor surveys (asteroseismic log g)
bypass the maps entirely; because the gauge penalty is weak it only breaks
the otherwise flat direction and lets anchors set the scale where they exist.

``hard_targets`` (v4): for parameters without physical anchors (Teff,
[Fe/H], [alpha]) the weighted mean correction is subtracted from every
survey's correction, so the condition holds exactly at every point instead
of as a penalty; a weak penalty cannot hold the common scale once maps have
cross terms (v3: Teff scatter doubled). ``precision`` (v4) fixes the weights
before training from survey overlaps (``overlap_scatter``, three-cornered
hat) instead of from residuals after each survey's own flexible map, which
rewarded small surveys that overfit their few stars.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_PARAM = 4  # Teff, log g, [Fe/H], [alpha] in scaled units
COVERAGE_DIMS = 3  # coverage histogram over Teff, log g, [Fe/H]


def coverage_grid(
    value: np.ndarray,
    survey: np.ndarray,
    row: np.ndarray,
    n_surveys: int,
    bins: int = 16,
    lo: float = -4.0,
    hi: float = 4.0,
    saturation: float = 20.0,
) -> np.ndarray:
    """Coverage in [0, 1) per (survey, target, Teff/log g/[Fe/H] cell).

    ``value`` holds scaled record values; a record missing a coordinate takes
    the mean of that star's other records. Coverage is n / (n + saturation),
    with n the survey's records in the cell that have the target.
    """
    value = np.asarray(value, dtype="f8")
    coords = value[:, :COVERAGE_DIMS].copy()
    n_rows = int(row.max()) + 1 if len(row) else 0
    for j in range(COVERAGE_DIMS):
        ok = np.isfinite(coords[:, j])
        total = np.bincount(row[ok], coords[ok, j], minlength=n_rows)
        count = np.bincount(row[ok], minlength=n_rows)
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        coords[~ok, j] = mean[row[~ok]]
    cell = np.floor((coords - lo) / (hi - lo) * bins).clip(0, bins - 1)
    placed = np.isfinite(coords).all(axis=1)
    flat = np.zeros(len(value), dtype="i8")
    flat[placed] = np.ravel_multi_index(
        cell[placed].astype("i8").T, (bins,) * COVERAGE_DIMS
    )
    out = np.zeros((n_surveys, N_PARAM, bins**COVERAGE_DIMS), dtype="f4")
    for j in range(N_PARAM):
        use = placed & np.isfinite(value[:, j]) & (survey >= 0)
        key = survey[use] * bins**COVERAGE_DIMS + flat[use]
        n = np.bincount(key, minlength=n_surveys * bins**COVERAGE_DIMS)
        out[:, j] = (n / (n + saturation)).reshape(n_surveys, -1)
    return out


class ConditionalSurveyMaps(nn.Module):
    """Per-survey corrections delta_s(x) on a shared learned basis."""

    def __init__(
        self,
        support,
        coverage,
        *,
        trust=None,
        groups=None,
        basis: int = 8,
        hidden: int = 64,
        bins: int = 16,
        lo: float = -4.0,
        hi: float = 4.0,
        min_slope: float = 0.25,
        precision_momentum: float = 0.99,
        hard_targets=(),
        precision=None,
    ):
        super().__init__()
        support = torch.as_tensor(np.asarray(support), dtype=torch.bool)
        support = support[:, :N_PARAM]
        m = len(support)
        coverage = torch.as_tensor(np.asarray(coverage), dtype=torch.float32)
        if coverage.shape != (m, N_PARAM, bins**COVERAGE_DIMS):
            raise ValueError("coverage must be (surveys, 4, bins**3)")
        trust = (
            torch.ones(m, N_PARAM)
            if trust is None
            else torch.as_tensor(np.asarray(trust), dtype=torch.float32)
        )
        groups = (
            torch.arange(m)
            if groups is None
            else torch.as_tensor(np.asarray(groups), dtype=torch.long)
        )
        share = torch.ones(m, N_PARAM)
        for j in range(N_PARAM):
            g = groups[support[:, j]]
            share[support[:, j], j] = torch.stack(
                [(g == x).sum() for x in g]
            ).float() if len(g) else share[support[:, j], j]  # fmt: skip
        self.register_buffer("support", support)
        self.register_buffer("coverage", coverage * support[:, :, None])
        self.register_buffer("static_weight", trust * support / share)
        # Fixed precision (e.g. from overlap_scatter) or a running estimate.
        self.fixed_precision = precision is not None
        self.register_buffer(
            "precision",
            torch.zeros(m, N_PARAM)  # 0 = unseen
            if precision is None
            else torch.as_tensor(np.asarray(precision), dtype=torch.float32),
        )
        hard = torch.zeros(N_PARAM, dtype=torch.bool)
        hard[list(hard_targets)] = True
        self.register_buffer("hard", hard)
        self.bins, self.lo, self.hi = bins, lo, hi
        self.min_slope = float(min_slope)
        self.momentum = float(precision_momentum)
        self.bias = nn.Parameter(torch.zeros(m, N_PARAM))
        self.linear = nn.Parameter(torch.zeros(m, N_PARAM, N_PARAM))
        self.mix = nn.Parameter(torch.zeros(m, N_PARAM, basis))
        self.phi = nn.Sequential(
            nn.Linear(N_PARAM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, basis),
            nn.Tanh(),  # bounded basis: corrections cannot run away off-data
        )

    # Corrections ------------------------------------------------------------
    def _raw_all(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        out = (
            self.bias[:, :, None]
            + torch.einsum("mjk,nk->mjn", self.linear, x)
            + torch.einsum("mjk,nk->mjn", self.mix, phi)
        )
        return out * self.support[:, :, None]

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        """Gauge weight of each survey at points ``x``: (surveys, 4, N)."""
        w = self.static_weight * self.precision
        return self.coverage_at(x.detach()) * w[:, :, None]

    def _consensus(self, raw_all: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Weighted mean raw correction over surveys at ``x``: (4, N)."""
        w = self._weights(x)
        total = w.sum(0)
        return (w * raw_all).sum(0) / total.clamp_min(1e-12) * (total > 0)

    def delta(self, x: torch.Tensor, survey: torch.Tensor) -> torch.Tensor:
        """delta_s(x) for points ``x`` (N, 4) observed by ``survey`` (N,).

        For ``hard_targets`` the weighted mean correction over surveys at x
        is subtracted, so that mean is exactly zero everywhere.
        """
        phi = self.phi(x)
        out = (
            self.bias[survey]
            + torch.einsum("njk,nk->nj", self.linear[survey], x)
            + torch.einsum("njk,nk->nj", self.mix[survey], phi)
        ) * self.support[survey]
        if self.hard.any():
            mean = self._consensus(self._raw_all(x, phi), x).T  # (N, 4)
            out = out - mean * (self.hard[None] & self.support[survey])
        return out

    def delta_all(self, x: torch.Tensor) -> torch.Tensor:
        """delta of every survey at points ``x`` (N, 4): (surveys, 4, N)."""
        raw = self._raw_all(x, self.phi(x))
        if self.hard.any():
            mean = self._consensus(raw, x)
            raw = raw - mean[None] * (self.hard[:, None] & self.support[:, :, None])
        return raw

    def forward(self, quantiles: torch.Tensor, survey: torch.Tensor) -> torch.Tensor:
        """Map common-scale quantiles (R, 4, Q) to each record's survey scale.

        The median vector is mapped as a point. For any other quantile of
        parameter j, the input is the median vector with component j set to
        that quantile, so a map increasing in its own parameter sends
        quantiles to quantiles while cross terms use the other medians.
        """
        r, p, nq = quantiles.shape
        mid = nq // 2
        med = quantiles[:, :, mid]
        points = [med]
        for k in range(nq):
            if k == mid:
                continue
            for j in range(p):
                x = med.clone()
                x[:, j] = quantiles[:, j, k]
                points.append(x)
        x = torch.cat(points)
        d = self.delta(x, survey.repeat(len(points)))
        out = torch.empty_like(quantiles)
        out[:, :, mid] = med + d[:r]
        i = 1
        for k in range(nq):
            if k == mid:
                continue
            for j in range(p):
                out[:, j, k] = quantiles[:, j, k] + d[i * r : (i + 1) * r, j]
                i += 1
        return out

    # Gauge, priors, precision ----------------------------------------------
    def coverage_at(self, x: torch.Tensor) -> torch.Tensor:
        """Coverage of each survey at points ``x``: (surveys, 4, N)."""
        cell = (x[:, :COVERAGE_DIMS] - self.lo) / (self.hi - self.lo) * self.bins
        cell = cell.floor().long().clamp(0, self.bins - 1)
        flat = (cell[:, 0] * self.bins + cell[:, 1]) * self.bins + cell[:, 2]
        return self.coverage[:, :, flat]

    def gauge_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squared weighted-mean correction at points ``x`` (N, 4).

        Only for targets not in ``hard_targets`` (those hold exactly).
        """
        x = x.detach()
        covered = (self._weights(x).sum(0) > 0) & ~self.hard[:, None]
        if not covered.any():
            return self.bias.sum() * 0
        mean = self._consensus(self._raw_all(x, self.phi(x)), x)
        return mean[covered].square().mean()

    def regularization(self) -> torch.Tensor:
        """L2 prior on the survey coefficients (zero = identity)."""
        s = self.support
        return (
            self.bias[s].square().mean()
            + self.linear[s].square().mean()
            + self.mix[s].square().mean()
        )

    def monotone_penalty(
        self, x: torch.Tensor, survey: torch.Tensor, h: float = 0.05
    ) -> torch.Tensor:
        """Squared shortfall of 1 + d delta_j / d x_j below ``min_slope``."""
        x = x.detach()
        base = self.delta(x, survey)
        shortfall = []
        for j in range(N_PARAM):
            step = x.clone()
            step[:, j] = step[:, j] + h
            slope = 1 + (self.delta(step, survey)[:, j] - base[:, j]) / h
            shortfall.append((self.min_slope - slope).clamp_min(0))
        short = torch.stack(shortfall, 1) * self.support[survey]
        return short.square().mean()

    @torch.no_grad()
    def update_precision(
        self, residual: torch.Tensor, survey: torch.Tensor, valid: torch.Tensor
    ) -> None:
        """Running 1 / (1.25 mean |residual|)^2 per (survey, parameter)."""
        if self.fixed_precision:
            return
        m = len(self.support)
        a = residual.abs().nan_to_num(0.0) * valid
        total = torch.zeros(m, N_PARAM, device=a.device).index_add_(0, survey, a)
        count = torch.zeros(m, N_PARAM, device=a.device).index_add_(
            0, survey, valid.to(a.dtype)
        )
        seen = count > 0
        prec = 1.0 / (1.25 * total / count.clamp_min(1.0)).clamp_min(1e-3).square()
        first = seen & (self.precision == 0)
        later = seen & ~first
        self.precision[first] = prec[first]
        mix = self.momentum
        self.precision[later] = mix * self.precision[later] + (1 - mix) * prec[later]


def overlap_scatter(
    value: np.ndarray,
    survey: np.ndarray,
    row: np.ndarray,
    n_surveys: int,
    groups=None,
    min_overlap: int = 50,
) -> np.ndarray:
    """Per-survey scatter sigma (surveys, 4) from overlaps (three-cornered hat).

    For each pair of surveys with >= ``min_overlap`` common stars, the robust
    variance of their difference is sigma_a^2 + sigma_b^2. The sigma^2 solve
    all pairs at once (non-negative least squares on relative residuals).
    Pairs in the same ``group`` (pipelines on the same spectra, whose errors
    correlate) are skipped. A survey with no usable pair gets NaN.
    """
    from scipy.optimize import nnls

    groups = np.arange(n_surveys) if groups is None else np.asarray(groups)
    out = np.full((n_surveys, N_PARAM), np.nan)
    for j in range(N_PARAM):
        per = {}
        for s in range(n_surveys):
            k = (survey == s) & np.isfinite(value[:, j])
            if k.sum() >= min_overlap:
                r, first = np.unique(row[k], return_index=True)
                per[s] = (r, value[k, j][first])
        rows, rhs, weight = [], [], []
        keys = sorted(per)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if groups[a] == groups[b]:
                    continue
                _, ia, ib = np.intersect1d(
                    per[a][0], per[b][0], assume_unique=True, return_indices=True
                )
                if len(ia) < min_overlap:
                    continue
                d = per[a][1][ia] - per[b][1][ib]
                v = (1.4826 * np.median(np.abs(d - np.median(d)))) ** 2
                if v <= 0:
                    continue
                e = np.zeros(n_surveys)
                e[[a, b]] = 1.0
                rows.append(e)
                rhs.append(v)
                weight.append(np.sqrt(len(ia)) / v)
        if not rows:
            continue
        A = np.array(rows) * np.array(weight)[:, None]
        used = A.any(axis=0)
        var, _ = nnls(A[:, used], np.array(rhs) * np.array(weight))
        out[np.flatnonzero(used), j] = np.sqrt(var)
    return out


def overlap_precision(sigma: np.ndarray, support, floor_quantile: float = 0.25):
    """1 / sigma^2 per (survey, parameter) from ``overlap_scatter``.

    sigma is floored at the ``floor_quantile`` of that parameter's positive
    estimates: the solve drives poorly constrained surveys to ~0, and the
    best surveys should count equally, so only clearly noisier ones lose
    weight. Unmeasured surveys get the largest measured sigma.
    """
    sigma = np.asarray(sigma, dtype="f8").copy()
    for j in range(sigma.shape[1]):
        known = np.isfinite(sigma[:, j])
        positive = sigma[known, j][sigma[known, j] > 0]
        if not len(positive):
            sigma[:, j] = 1.0
            continue
        sigma[~known, j] = positive.max()
        sigma[:, j] = np.maximum(sigma[:, j], np.quantile(positive, floor_quantile))
    prec = 1.0 / sigma**2
    return (prec * np.asarray(support)[:, :N_PARAM]).astype("f4")
