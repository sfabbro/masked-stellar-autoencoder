"""Likelihood (ratio) head on frozen latents and its frequentist inference.

The head s(theta, z) is a contrastive ratio classifier
(``torchregress.losses.NeuralRatioLoss(criterion="cnre")``): at the optimum
s = log p(z | theta) - log p(z), so the training population cancels and
argmax_theta s is the maximum-likelihood estimate. Negatives are the
parameters of other training stars (the natural population; re-weighting
only adds variance).

For cheap evaluation at many theta the latent is embedded once per star,
e = E(z), and the score is a small smooth MLP on [standardised theta, e]
(``score_net``). Inference (``torchregress.inference``) receives e as the
observation, so the multistart Newton search and the profile scans only run
the small MLP.

``EnsembleScore`` combines K heads (the mean of their log-ratios, i.e. the
geometric mean of the ratios) and optionally a parallax nuisance parameter:
the heads then take theta' = (Teff, log g, [Fe/H], u = log10 parallax/mas),
and the score adds the Gaussian measurement term of the observed
(zero-point-corrected) parallax, so parallax enters once, with its error, and
is profiled together with theta.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn


class Support:
    """Support of the training labels: a dilated occupancy grid in theta.

    A ratio estimator learns log p(z | theta) only where training labels
    exist; outside, its score is unconstrained (in S1 ~11 % of MLEs ran to
    the upper [Fe/H] edge of the box). ``inside(theta)`` marks grid cells
    holding at least ``min_count`` training labels, grown by ``dilate``
    cells in every direction. A hard, data-defined constraint, not a prior.

    Any number of parameters P (e.g. 4 with log10 parallax as the last one);
    the grid has prod(ceil(range / width)) cells. ``on_edge(theta)`` marks
    support cells with a neighbour (3^P neighbourhood) outside the support or
    the box: an MLE there was stopped by the support.
    """

    def __init__(self, theta, bounds, widths, device="cpu", *, dilate=1, min_count=1):
        from scipy.ndimage import binary_dilation, binary_erosion

        theta = np.asarray(theta, dtype="f8").reshape(-1, len(bounds))
        self.lo = torch.tensor(
            [b[0] for b in bounds], dtype=torch.float64, device=device
        )
        self.w = torch.tensor(widths, dtype=torch.float64, device=device)
        self.n = [
            int(np.ceil((b[1] - b[0]) / w)) for b, w in zip(bounds, widths, strict=True)
        ]
        idx = np.floor((theta - self.lo.cpu().numpy()) / np.asarray(widths)).astype(int)
        idx = np.clip(idx, 0, np.asarray(self.n) - 1)
        counts = np.zeros(self.n, dtype=np.int64)
        np.add.at(counts, tuple(idx.T), 1)
        occ = counts >= min_count
        cube = np.ones((3,) * occ.ndim, bool)  # full 3^P neighbourhood
        if dilate:
            # No wrap-around at the box edges.
            occ = binary_dilation(occ, structure=cube, iterations=int(dilate))
        edge = occ & ~binary_erosion(occ, structure=cube, border_value=0)
        self.grid = torch.as_tensor(occ, device=device)
        self.edge = torch.as_tensor(edge, device=device)
        self.fraction = float(occ.mean())

    def _cells(self, theta):
        idx = torch.floor((theta.double() - self.lo) / self.w).long()
        n = torch.tensor(self.n, device=theta.device)
        ok = ((idx >= 0) & (idx < n)).all(-1)
        return ok, tuple(torch.minimum(idx.clamp_min(0), n - 1).T)

    def inside(self, theta):
        ok, cell = self._cells(theta)
        return ok & self.grid[cell]

    def on_edge(self, theta):
        """Inside the support, in a cell next to its boundary (or the box's)."""
        ok, cell = self._cells(theta)
        return ok & self.edge[cell]

    def mask(self, theta, score):
        """``score`` where ``theta`` is inside the support, NaN elsewhere."""
        return torch.where(self.inside(theta), score, torch.full_like(score, torch.nan))


class RatioHead(nn.Module):
    """s(theta, z) = score_net([(theta - c) / s, E((z - m) / sd)])."""

    def __init__(self, latent_dim, n_params=3, *, embed_dims=(512, 256),
                 embed_out=128, hidden=(256, 256)):  # fmt: skip
        super().__init__()
        from torchregress.losses import build_ratio_classifier

        self.config = dict(
            latent_dim=int(latent_dim), n_params=int(n_params),
            embed_dims=tuple(int(d) for d in embed_dims), embed_out=int(embed_out),
            hidden=tuple(int(d) for d in hidden),
        )  # fmt: skip
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_scale", torch.ones(latent_dim))
        self.register_buffer("theta_center", torch.zeros(n_params))
        self.register_buffer("theta_scale", torch.ones(n_params))
        layers, d = [], int(latent_dim)
        for w in embed_dims:
            layers += [nn.Linear(d, int(w)), nn.SiLU()]
            d = int(w)
        layers.append(nn.Linear(d, int(embed_out)))
        self.embedder = nn.Sequential(*layers)
        self.score_net = build_ratio_classifier(
            int(n_params), int(embed_out), hidden_dims=tuple(hidden), activation="silu"
        )
        self.support = None  # optional Support: searches stay inside it

    def set_standardisation(self, z_mean, z_scale, theta_center, theta_scale):
        for name, v in (("z_mean", z_mean), ("z_scale", z_scale),
                        ("theta_center", theta_center), ("theta_scale", theta_scale)):  # fmt: skip
            buf = getattr(self, name)
            buf.copy_(torch.as_tensor(np.asarray(v), dtype=buf.dtype))

    def embed(self, z):
        z = z.to(self.z_mean.dtype)
        return self.embedder((z - self.z_mean) / self.z_scale)

    def standardise(self, theta):
        return (theta - self.theta_center.to(theta.dtype)) / self.theta_scale.to(
            theta.dtype
        )

    def score_embedded(self, theta, e):
        """Score from physical theta [M, P] and embeddings [M, E] -> [M]."""
        inp = torch.cat([self.standardise(theta), e.to(theta.dtype)], dim=-1)
        return self.score_net(inp).squeeze(-1)

    def forward(self, theta, z):
        return self.score_embedded(theta, self.embed(z))

    def score_fn(self, masked=True):
        """``score(theta, e)`` for ``torchregress.inference`` (x = embeddings).

        With a ``support`` set (and ``masked``), the score is NaN outside it,
        so maximisation and profiles stay where the ratio was learned.
        """

        def score(theta, e):
            s = self.score_embedded(theta, e)
            if masked and self.support is not None:
                s = self.support.mask(theta, s)
            return s

        return score


def save_head(head, path, **extra):
    torch.save(dict(config=head.config, state_dict=head.state_dict(), **extra), path)


def load_head(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    head = RatioHead(**blob["config"])
    head.load_state_dict(blob["state_dict"])
    return head.to(device).eval(), blob


# ── ensemble and parallax nuisance ──────────────────────────────────────────

LN10 = math.log(10.0)
N_PARALLAX_COLUMNS = 4  # observation columns after the embeddings: a, b, plx, w


def parallax_window(parallax, error, u_bounds, k=8.0):
    """Per-row search window of u = log10(parallax / mas) and measurement terms.

    ``parallax`` (zero-point corrected) and ``error`` in mas. Returns numpy
    (a, b, plx, w), each [N]: the search coordinate t in [0, 1] maps to
    u = a + b t; the measurement term is -0.5 ((plx - 10**u) w)^2.

    * A measured parallax (finite, error > 0): w = 1 / error, and the window
      holds the parallaxes within ``k`` errors of the measurement (clipped to
      the box ``u_bounds``), i.e. wherever the term is above -k^2/2. Without
      the window, in box units, a parallax with parallax / error ~ 1000 is
      ~1e5 times more curved than the ratio in theta: the saddle-free Newton
      eigenvalue floor (sqrt(eps) of the largest, float32) would then stall
      the theta directions. A
      window missing the box (e.g. parallax < -k error) falls back to the
      whole box with the term kept (the MLE then sits on the window edge).
    * No measurement: w = 0 (flat term, the learned coupling alone) and the
      whole box.
    """
    plx = np.asarray(parallax, dtype="f8")
    err = np.asarray(error, dtype="f8")
    lo, hi = (float(v) for v in u_bounds)
    meas = np.isfinite(plx) & np.isfinite(err) & (err > 0)
    w = np.where(meas, 1.0 / np.where(meas, err, 1.0), 0.0)
    p0 = np.where(meas, plx, 0.0)
    e0 = np.where(meas, err, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ulo = np.log10(np.maximum(p0 - k * e0, 10.0**lo))
        uhi = np.log10(np.minimum(p0 + k * e0, 10.0**hi))
    ok = meas & (p0 + k * e0 > 10.0**lo) & (p0 - k * e0 < 10.0**hi) & (uhi > ulo)
    a = np.where(ok, ulo, lo)
    b = np.where(ok, uhi - ulo, hi - lo)
    return a, b, p0, w


class EnsembleScore:
    """Score of K ratio heads, optionally with a parallax nuisance parameter.

    s(theta, x) = (1/K) sum_k s_k(theta', e_k(z))            (no parallax)
                + -0.5 ((plx - 10**u) / sigma)^2              (parallax)

    The ensemble score is the mean of the heads' log-ratios (the geometric
    mean of the ratios): each s_k estimates log p(z | theta') - log p(z), so
    the mean estimates the same quantity and averages the members' (partly
    independent: different initialisation and batches, same data) errors.
    With ``parallax`` the last search coordinate is t in
    [0, 1], mapped per row to u = log10(parallax / mas) = a + b t
    (``parallax_window``), theta' = (theta, u), and the observation x is
    [e_1, ..., e_K, a, b, plx, w] (``observation``); constant terms of the
    Gaussian (log sigma, log 2 pi) do not depend on theta and are dropped.
    A ``support`` (in theta' space) masks the score with NaN outside it.
    """

    def __init__(self, heads, *, support=None, parallax=False):
        self.heads = list(heads)
        if not self.heads:
            raise ValueError("EnsembleScore needs at least one head")
        dims = [int(h.config["embed_out"]) for h in self.heads]
        self.offsets = [0, *np.cumsum(dims).tolist()]
        self.embed_dim = int(self.offsets[-1])
        self.support = support
        self.parallax = bool(parallax)
        n_params = {int(h.config["n_params"]) for h in self.heads}
        if len(n_params) != 1:
            raise ValueError(f"Heads disagree on the parameter count: {n_params}")
        self.n_params = n_params.pop()

    def __len__(self):
        return len(self.heads)

    def member(self, k):
        """The single-head score of member ``k`` (same support and parallax term)."""
        return EnsembleScore([self.heads[k]], support=self.support,
                             parallax=self.parallax)  # fmt: skip

    def member_observation(self, x, k):
        """Columns of ``x`` (this ensemble's observation) for ``member(k)``."""
        lo, hi = self.offsets[k], self.offsets[k + 1]
        return torch.cat([x[:, lo:hi], x[:, self.embed_dim :]], dim=1)

    @torch.no_grad()
    def embed(self, z):
        return torch.cat([h.embed(z) for h in self.heads], dim=-1)

    def observation(self, emb, window=None):
        """x for ``torchregress.inference``: embeddings [+ the parallax window].

        ``emb`` [N, E] (copied) or [N, E + 4] from ``embed_rows(...,
        extra=N_PARALLAX_COLUMNS)`` (its last columns are filled in place).
        """
        if not self.parallax:
            return emb
        if window is None:
            raise ValueError("A parallax score needs the parallax window")
        extra = torch.as_tensor(np.column_stack(window), dtype=emb.dtype,
                                device=emb.device)  # fmt: skip
        if emb.shape[1] == self.embed_dim + N_PARALLAX_COLUMNS:
            emb[:, self.embed_dim :] = extra
            return emb
        return torch.cat([emb, extra], dim=1)

    def physical(self, theta, x):
        """theta' (heads' parameters) from the search coordinates."""
        if not self.parallax:
            return theta
        e = self.embed_dim
        u = x[:, e] + x[:, e + 1] * theta[:, -1]
        return torch.cat([theta[:, :-1], u[:, None]], dim=1)

    def log_ratio(self, theta_phys, x):
        s = 0.0
        for h, lo, hi in zip(self.heads, self.offsets[:-1], self.offsets[1:],
                             strict=True):  # fmt: skip
            s = s + h.score_embedded(theta_phys, x[:, lo:hi])
        return s / len(self.heads)

    def measurement(self, theta_phys, x):
        """-0.5 ((plx - 10**u) w)^2; 0 where w = 0 (no parallax)."""
        e = self.embed_dim
        plx, w = x[:, e + 2], x[:, e + 3]
        return -0.5 * ((plx - torch.exp(LN10 * theta_phys[:, -1])) * w).square()

    def score_fn(self, masked=True, measurement=True):
        def score(theta, x):
            tp = self.physical(theta, x)
            s = self.log_ratio(tp, x)
            if self.parallax and measurement:
                s = s + self.measurement(tp, x)
            if masked and self.support is not None:
                s = self.support.mask(tp, s)
            return s

        return score


@torch.no_grad()
def window_start(score, x, theta, *, n_grid=17, chunk=1 << 17):
    """Best window coordinate t per row on a grid, at fixed ``theta`` [N, P-1].

    The start of the parallax coordinate for ``maximize``: the grid point
    with the highest (masked) score; rows with no finite score start at the
    measured parallax (inside the window) or the window centre. numpy [N].
    """
    fn = score.score_fn()
    n = len(x)
    th = torch.as_tensor(np.asarray(theta), dtype=x.dtype, device=x.device)
    grid = torch.linspace(0.0, 1.0, n_grid, dtype=x.dtype, device=x.device)
    out = np.full(n, 0.5)
    step = max(1, chunk // n_grid)
    e = score.embed_dim
    for i in range(0, n, step):
        xb, tb = x[i : i + step], th[i : i + step]
        m = len(xb)
        tt = torch.cat([tb.repeat_interleave(n_grid, 0), grid.repeat(m)[:, None]], 1)
        s = fn(tt, xb.repeat_interleave(n_grid, 0)).view(m, n_grid)
        s = torch.nan_to_num(s, nan=-math.inf)
        best = s.argmax(1)
        found = torch.isfinite(s.max(1).values)
        a, b, plx = xb[:, e], xb[:, e + 1], xb[:, e + 2]
        t_obs = (torch.log10(plx.clamp_min(1e-30)) - a) / b
        t_obs = torch.where((plx > 0) & (xb[:, e + 3] > 0), t_obs.clamp(0, 1), 0.5)
        out[i : i + m] = torch.where(found, grid[best], t_obs).cpu().numpy()
    return out


def _robust_stats(z, rows, chunk=262144):
    """Mean and standard deviation of latent rows (float64 accumulation)."""
    s = s2 = None
    for i in range(0, len(rows), chunk):
        b = z[rows[i : i + chunk]].double()
        s = b.sum(0) if s is None else s + b.sum(0)
        s2 = (b * b).sum(0) if s2 is None else s2 + (b * b).sum(0)
    n = len(rows)
    mean = s / n
    sd = (s2 / n - mean * mean).clamp_min(0).sqrt() + 1e-6
    return mean.float(), sd.float()


def train_ratio_head(head, z, theta, train, valid, *, negatives=8, max_epochs=60,
                     patience=8, batch_size=None, lr=1e-3, weight_decay=1e-4,
                     max_valid=200_000, seed=0, log=print):  # fmt: skip
    """Train ``head`` with CNRE; early stopping on the validation loss.

    ``z`` (N, D) latents (any float dtype, on the training device), ``theta``
    (N, P) physical labels (float32, same device), ``train``/``valid`` index
    tensors. Negatives are the labels of random training stars. Returns the
    history (per-epoch train and validation loss); ``head`` ends at its best
    validation state.
    """
    from torchregress.losses import NeuralRatioLoss

    dev = z.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    zm, zs = _robust_stats(z, train)
    tt = theta[train].double()
    head.set_standardisation(
        zm.cpu(), zs.cpu(), tt.mean(0).float().cpu(), tt.std(0).float().cpu()
    )
    head.to(dev)
    loss_fn = NeuralRatioLoss(head.score_net, criterion="cnre")
    theta_std = head.standardise(theta)
    bank = theta_std[train]
    n_train = len(train)
    if batch_size is None:
        batch_size = int(min(4096, max(256, n_train // 100)))
    if len(valid) > max_valid:
        pick = torch.randperm(len(valid), generator=gen)[:max_valid]
        valid = valid[pick.to(valid.device)]
    # Fixed validation negatives: the loss is comparable between epochs.
    vneg = torch.randint(n_train, (len(valid), negatives), generator=gen).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    params = list(head.parameters())
    best, best_state, since = math.inf, None, 0
    history = dict(train=[], valid=[], batch_size=batch_size, n_train=n_train,
                   n_valid=len(valid))  # fmt: skip
    t0 = time.time()
    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(n_train, generator=gen).to(dev)
        tot, nb = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = train[perm[i : i + batch_size]]
            neg = bank[torch.randint(n_train, (len(idx), negatives), device=dev)]
            loss = loss_fn(theta_std[idx], head.embed(z[idx]), neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        head.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(valid), 16384):
                idx = valid[i : i + 16384]
                neg = bank[vneg[i : i + 16384]]
                vl.append(
                    float(loss_fn(theta_std[idx], head.embed(z[idx]), neg)) * len(idx)
                )
            v = sum(vl) / max(len(valid), 1)
        history["train"].append(tot / max(nb, 1))
        history["valid"].append(v)
        log(f"  epoch {epoch:3d} train {tot / max(nb, 1):.4f} valid {v:.4f}"
            f" ({time.time() - t0:.0f} s)")  # fmt: skip
        if v < best - 1e-5:
            best, since = v, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    history["best_valid"] = best
    history["epochs"] = len(history["valid"])
    return history


@torch.no_grad()
def embed_rows(head, z, rows, *, device, batch=65536, extra=0):
    """Embeddings (len(rows), E [+ extra]) float32 on ``device``.

    ``head``: a ``RatioHead`` or an ``EnsembleScore`` (E = sum of the
    members'); ``z`` may be a numpy memmap. ``extra`` zero columns are
    appended (filled in place by ``EnsembleScore.observation``), so the
    observation is allocated once (K = 5 heads: ~2.6 kB per row).
    """
    dim = getattr(head, "embed_dim", None) or head.config["embed_out"]
    out = torch.zeros(len(rows), dim + int(extra), device=device)
    for i in range(0, len(rows), batch):
        r = rows[i : i + batch]
        if isinstance(z, torch.Tensor):
            zb = z[torch.as_tensor(r, device=z.device)].to(device)
        else:
            zb = torch.as_tensor(np.asarray(z[r]), device=device)
        out[i : i + len(r), :dim] = head.embed(zb).float()
    return out


def maximize(head, emb, bounds, *, init=None, block=200_000, n_candidates=256,
             n_starts=4, chunk_size=None, seed=0, log=print):  # fmt: skip
    """MLE per row: (theta [N, P], s_max [N], at_bound [N, P]) as numpy."""
    from torchregress.inference import maximize_score

    score = head.score_fn()
    box = torch.as_tensor(bounds, dtype=emb.dtype, device=emb.device)
    thetas, smax, atb = [], [], []
    t0 = time.time()
    for i in range(0, len(emb), block):
        sl = slice(i, i + block)
        ini = None
        if init is not None:
            ini = torch.as_tensor(
                np.asarray(init[sl]), dtype=emb.dtype, device=emb.device
            )
            ini = torch.where(torch.isfinite(ini), ini, box.mean(1))
        res = maximize_score(
            score, emb[sl], box, n_candidates=n_candidates, n_starts=n_starts,
            init=ini, chunk_size=chunk_size,
            generator=torch.Generator().manual_seed(seed + i),
        )  # fmt: skip
        thetas.append(res.theta.cpu())
        smax.append(res.score.cpu())
        atb.append(res.at_bound.cpu())
        log(
            f"  maximize: {min(i + block, len(emb))}/{len(emb)} ({time.time() - t0:.0f} s)"
        )
    if not thetas:
        P = len(bounds)
        return np.empty((0, P)), np.empty(0), np.empty((0, P), bool)
    return (torch.cat(thetas).double().numpy(), torch.cat(smax).double().numpy(),
            torch.cat(atb).numpy())  # fmt: skip


def profiles(head, emb, theta_hat, s_max, threshold, bounds, *, params=None,
             block=200_000, chunk_size=None, n_scan=8, log=print, label=""):  # fmt: skip
    """Profile intervals of parameters ``params`` (default: all) at ``threshold``.

    ``threshold``: scalar, [N] (one per row) or [N, J] (one per row and
    profiled parameter, e.g. per-parameter Neyman critical values). The
    inner maximisation runs over every other parameter of the box (e.g. the
    parallax coordinate of an ``EnsembleScore``). Returns (lower [N, J],
    upper [N, J], lower_at_bound [N, J], upper_at_bound [N, J]) as numpy,
    J = len(params).
    """
    from torchregress.inference import profile_interval

    score = head.score_fn()
    dt, dev = emb.dtype, emb.device
    box = torch.as_tensor(bounds, dtype=dt, device=dev)
    N = len(emb)
    params = list(range(len(bounds))) if params is None else [int(j) for j in params]
    J = len(params)
    lo, hi = np.full((N, J), np.nan), np.full((N, J), np.nan)
    lob, hib = np.zeros((N, J), bool), np.zeros((N, J), bool)
    thr_all = np.asarray(threshold, dtype="f8")
    if thr_all.ndim == 1:
        thr_all = thr_all[:, None]
    thr_all = np.broadcast_to(thr_all, (N, J))
    t0 = time.time()
    for i in range(0, N, block):
        sl = slice(i, min(N, i + block))
        th = torch.as_tensor(theta_hat[sl], dtype=dt, device=dev)
        sm = torch.as_tensor(s_max[sl], dtype=dt, device=dev)
        for c, j in enumerate(params):
            thr = torch.as_tensor(
                np.ascontiguousarray(thr_all[sl, c]), dtype=dt, device=dev
            )
            r = profile_interval(
                score, emb[sl], th, sm, j, thr, box, n_scan=n_scan,
                chunk_size=chunk_size,
            )  # fmt: skip
            lo[sl, c], hi[sl, c] = r.lower.cpu().numpy(), r.upper.cpu().numpy()
            lob[sl, c] = r.lower_at_bound.cpu().numpy()
            hib[sl, c] = r.upper_at_bound.cpu().numpy()
        log(f"  profiles {label}: {sl.stop}/{N} ({time.time() - t0:.0f} s)")
    return lo, hi, lob, hib


def profile_statistic(head, emb, index, value, s_max, theta_init, bounds, *,
                      block=200_000, n_candidates=64, n_starts=2, chunk_size=None,
                      seed=0):  # fmt: skip
    """Profile statistic s_max - max_{theta: theta_J = value} s(theta) per row.

    ``index``: one parameter j (``value`` [N]) or several, J (``value``
    [N, len(J)]); e.g. the three atmospheric parameters of a parallax score,
    whose statistic then profiles the parallax only. The constrained maximum
    over the other parameters uses ``maximize_score`` on the reduced box,
    started from ``theta_init`` (e.g. the MLE). Returns a numpy [N] array,
    clamped at 0 (a constrained maximum above ``s_max`` means ``s_max`` was
    not the global maximum).
    """
    from torchregress.inference import maximize_score

    dt, dev = emb.dtype, emb.device
    P = len(bounds)
    fixed = [int(index)] if np.ndim(index) == 0 else [int(i) for i in index]
    keep = [k for k in range(P) if k not in fixed]
    if not keep:
        raise ValueError("profile_statistic needs at least one free parameter")
    nf = len(fixed)
    box = torch.as_tensor(bounds, dtype=dt, device=dev)[keep]
    value = np.asarray(value, dtype="f8").reshape(len(emb), nf)
    base = head.score_fn()

    def score(free, xa):
        t = xa[:, -nf:]
        cols = [t[:, fixed.index(k)] if k in fixed else free[:, keep.index(k)]
                for k in range(P)]  # fmt: skip
        return base(torch.stack(cols, dim=1), xa[:, :-nf])

    out = []
    for i in range(0, len(emb), block):
        sl = slice(i, i + block)
        t = torch.as_tensor(value[sl], dtype=dt, device=dev)
        xa = torch.cat([emb[sl], t], dim=1)
        ini = torch.as_tensor(np.asarray(theta_init[sl])[:, keep], dtype=dt, device=dev)
        res = maximize_score(
            score, xa, box, n_candidates=n_candidates,
            n_starts=min(n_starts, n_candidates), init=ini, chunk_size=chunk_size,
            generator=torch.Generator().manual_seed(seed + i),
        )  # fmt: skip
        sm = torch.as_tensor(np.asarray(s_max[sl]), dtype=dt, device=dev)
        out.append((sm - res.score).clamp_min(0).cpu())
    if not out:
        return np.empty(0)
    return torch.cat(out).double().numpy()


def lr_statistic(head, emb, theta, s_max, *, chunk_size=None):
    """lambda = s_max - s(theta) per row (numpy), clamped at 0."""
    from torchregress.inference import likelihood_ratio_statistic

    dt, dev = emb.dtype, emb.device
    lam = likelihood_ratio_statistic(
        head.score_fn(masked=False), emb, torch.as_tensor(theta, dtype=dt, device=dev),
        torch.as_tensor(s_max, dtype=dt, device=dev), chunk_size=chunk_size,
    )  # fmt: skip
    return lam.double().cpu().numpy()
