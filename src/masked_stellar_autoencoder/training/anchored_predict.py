"""Inference for anchored (Student-t head) checkpoints.

One path for training-time checks, evaluation and prediction: inputs are the
clean scaled features (NaN = missing), with the checkpoint's dropped input
features hidden from the encoder, optional head context, and optional
conditioning of the joint Student-t on a measured label (Gaia parallax).
"""

from __future__ import annotations

import numpy as np
import torch

from .anchored_survey_map import condition_mixture


def encoder_input(xb: torch.Tensor, drop_idx=()) -> torch.Tensor:
    """Encoder tensor: dropped features hidden, NaN as the -9999 token."""
    if len(drop_idx):
        xb = xb.clone()
        xb[:, list(drop_idx)] = float("nan")
    return torch.nan_to_num(xb, nan=-9999.0)


def mixture_quantiles(logw, loc, scale, nu, qs=(0.16, 0.5, 0.84), iters=45):
    """Marginal quantiles (N, P, len(qs)) of a Student-t mixture (numpy).

    ``logw`` (N, K), ``loc``/``scale`` (N, K, P), ``nu`` (N,) degrees of
    freedom. One component: closed form; otherwise bisection on the mixture
    CDF (``iters`` halvings of a +-30 scale bracket).
    """
    from scipy.stats import t as student_t

    nu = np.asarray(nu, float)
    if loc.shape[1] == 1:
        k = np.stack([student_t.ppf(q, nu) for q in qs], axis=-1)  # (N, Q)
        return loc[:, 0, :, None] + scale[:, 0, :, None] * k[:, None, :]
    w = np.exp(logw)[:, :, None]
    out = np.empty((*loc.shape[::2], len(qs)))
    for i, q in enumerate(qs):
        lo = (loc - 30 * scale).min(1)
        hi = (loc + 30 * scale).max(1)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            cdf = (
                w * student_t.cdf((mid[:, None] - loc) / scale, nu[:, None, None])
            ).sum(1)
            below = cdf < q
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out[..., i] = 0.5 * (lo + hi)
    return out


@torch.no_grad()
def anchored_predict(
    model, head, X, *, device, E=None, context=None, drop_idx=(), condition=None,
    Y=None, eY=None, batch: int = 4096, full: bool = False,
):  # fmt: skip
    """Locations and marginal scales (N, P) in scaled label units.

    ``E``: scaled feature errors (needed by a context that uses errors);
    ``Y``/``eY``: scaled labels and errors (a context with label inputs).
    ``condition``: (label index j, z (N,), var (N,)) in scaled label units;
    rows with a finite z are conditioned on it (``condition_on_observation``,
    ``condition_mixture``); conditioned rows have nu + 1 degrees of freedom.

    Single-component head: (location, t scale). Mixture head: (median,
    equivalent scale = (q84 - q16) / (2 t_nu(0.84))). ``full=True`` returns a
    dict with the exact marginal 16/50/84 % quantiles ``q`` (N, P, 3) and
    the mixture (``logw``, ``loc``, ``scale``, ``nu`` per row).
    """
    from scipy.stats import t as student_t

    model.eval()
    head.eval()
    if context is not None:
        context.eval()
    X = np.asarray(X, dtype=np.float32)
    lws, locs, sds, oks = [], [], [], []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i : i + batch], device=device)
        z = model.encoder(encoder_input(xb, drop_idx))
        ctx = None
        if context is not None:

            def t(a):
                return (
                    None
                    if a is None
                    else torch.as_tensor(
                        np.asarray(a[i : i + batch], dtype=np.float32), device=device
                    )
                )

            ctx, _ = context(xb, t(E), t(Y), t(eY))
        logw, mu, L = head.mixture(z, ctx)
        ok = torch.zeros(len(z), dtype=torch.bool, device=z.device)
        if condition is not None:
            j, zc, vc = condition
            sl = slice(i, i + batch)
            zt = torch.as_tensor(zc[sl], device=device)
            vt = torch.as_tensor(vc[sl], device=device)
            logw, mu, sd = condition_mixture(logw, mu, L, j, zt, vt, head.nu)
            ok = torch.isfinite(zt) & torch.isfinite(vt) & (vt >= 0)
        else:
            sd = L.square().sum(-1).sqrt()
        lws.append(logw.cpu())
        locs.append(mu.cpu())
        sds.append(sd.cpu())
        oks.append(ok.cpu())
    logw = torch.cat(lws).numpy().astype("f8")
    loc = torch.cat(locs).numpy().astype("f8")
    scale = torch.cat(sds).numpy().astype("f8")
    nu = head.nu + torch.cat(oks).numpy().astype("f8")
    if loc.shape[1] == 1 and not full:
        return loc[:, 0].astype("f4"), scale[:, 0].astype("f4")
    q = mixture_quantiles(logw, loc, scale, nu)
    if full:
        return dict(q=q, logw=logw, loc=loc, scale=scale, nu=nu)
    k = student_t.ppf(0.84, nu)[:, None]
    return q[..., 1].astype("f4"), ((q[..., 2] - q[..., 0]) / (2 * k)).astype("f4")
