"""Prior-free [Fe/H] and tail probabilities from the anchored head.

The head's [Fe/H] marginal is a posterior under the training population
(mostly near-solar stars), so a metal-poor star's location is pulled toward
the bulk. Two extra outputs for metal-poor work:

* ``p_below``: P([Fe/H] < limit) under the posterior (calibrated for stars
  drawn like the training population; a ranking elsewhere);
* ``prior_free``: the posterior divided by the metal-poor side of the
  training [Fe/H] prior (flat above its peak, so near-solar and metal-rich
  stars are unchanged), as 16/50/84 % quantiles. On held-out high-resolution
  references (v7, 7.7k stars) the median offset goes from +0.60 to +0.07 dex
  below -3 and from +0.29 to -0.04 for -3 to -2.5, but it over-corrects by
  about -0.14 dex between -2.5 and -1, because the head's marginal is a
  symmetric Student-t (dividing it by a steep prior is not exact; a
  flexible, skewed marginal would be). Dividing by the whole prior also
  moves metal-rich stars up by 0.3-0.5 dex (the prior falls steeply above
  +0.3), hence the flat top. Use it to select very metal-poor stars, not as
  the default estimate.

The training prior is a smoothed histogram of the fine-tune table's [Fe/H]
(per star, the median of its survey records), saved by
``scripts/make_feh_prior.py``. Prior reweighting uses torchregress'
label-shift correction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t
from torchregress.test_time.label_shift import apply_label_shift_correction

EDGES = np.arange(-5.0, 1.0001, 0.02)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])


def smoothed_histogram(values, edges=EDGES, smooth_bins=5):
    """Normalised histogram on ``edges``, Gaussian-smoothed, never zero."""
    v = np.asarray(values, float)
    h = np.histogram(v[np.isfinite(v)], bins=edges)[0].astype(float)
    k = np.exp(
        -0.5 * (np.arange(-4 * smooth_bins, 4 * smooth_bins + 1) / smooth_bins) ** 2
    )
    h = np.convolve(h + 1e-3, k / k.sum(), mode="same").clip(1e-12)
    return h / h.sum()


def gridded_posterior(loc, scale, nu, edges=EDGES):
    """(N, bins) probabilities of t_nu(loc, scale) on ``edges`` (nu: scalar or (N,))."""
    z = (edges[None, :] - np.asarray(loc)[:, None]) / np.asarray(scale)[:, None]
    nu = np.asarray(nu, float)
    p = np.diff(student_t.cdf(z, nu[:, None] if nu.ndim else nu), axis=1).clip(1e-12)
    return p / p.sum(1, keepdims=True)


def _quantiles(p, qs=(0.16, 0.5, 0.84), centres=CENTRES):
    cdf = np.cumsum(p, 1)
    return np.stack([centres[np.argmax(cdf >= q, axis=1)] for q in qs], axis=1)


def feh_extras(loc, scale, nu, prior, *, limit=-2.5, batch=50_000):
    """(prior-free 16/50/84 quantiles (N, 3), P([Fe/H] < limit) (N,)).

    ``loc``/``scale``: physical [Fe/H] location and Student-t scale (not the
    standard deviation); ``nu``: scalar or per row. Rows with a non-finite
    location give NaN.
    """
    loc, scale = np.asarray(loc, float), np.asarray(scale, float)
    nu = np.broadcast_to(np.asarray(nu, float), loc.shape)
    ok = np.isfinite(loc) & np.isfinite(scale) & (scale > 0)
    q = np.full((len(loc), 3), np.nan)
    pb = np.full(len(loc), np.nan)
    centres = CENTRES
    prior = np.asarray(prior, float).copy()
    prior[centres >= centres[np.argmax(prior)]] = prior.max()  # metal-poor side only
    prior /= prior.sum()
    flat = np.full(len(prior), 1.0 / len(prior))
    below = centres < limit
    idx = np.flatnonzero(ok)
    for i in range(0, len(idx), batch):
        j = idx[i : i + batch]
        post = gridded_posterior(loc[j], scale[j], nu[j])
        pb[j] = post[:, below].sum(1)
        # eps far below the grid's probability floor (1e-12): the default
        # 1e-8 clip would put spurious mass in empty tail bins.
        free = apply_label_shift_correction(
            post, source_prior=prior, target_prior=flat, eps=1e-300
        )
        q[j] = _quantiles(free)
    return q, pb
