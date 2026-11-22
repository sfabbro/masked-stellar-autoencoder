"""Distributional heads on frozen latents (design note 2026-11-22, product B).

Conditional densities p(Teff, log g, [Fe/H] | z) on a frozen latent z, each run
as a K-member ensemble whose density is the equal-weight mixture of
independently trained members (different seeds, same data). Member kinds:

* ``flow``: an MLP trunk z -> context and a conditional rational-quadratic
  spline flow on the standardized 3-D target (torchregress
  ``create_flow_model(flow_type="nsf")`` / ``NormalizingFlowLoss``): exact
  densities, sampling, multimodality.
* ``tmix``: an MLP trunk z -> the parameters of a mixture of M multivariate
  Student-t with full scale matrices (torchregress ``MixtureDensityLoss``,
  precision-Cholesky parameterisation, one learned df per member); closed-form
  marginal quantiles (``StudentTMixture.marginal_quantile``).
* ``t``: the same head with M = 1 (a single multivariate Student-t).

Errors in variables (``eiv=True``): members are fitted to the noisy label by
p(y_obs | z) = int p(theta | z) N(y_obs | theta, diag(sigma_label^2)) dtheta, so
that p(theta | z) is a density of the parameter, not of the label.

* flows: torchregress ``NoiseAwareDensityLoss`` (Monte Carlo over the label
  noise, log-mean-exp of ``n_mc`` draws; a consistent lower bound);
* mixtures: closed form per component
  (``StudentTMixture.component_observation_log_prob`` with an identity design),
  exact for Gaussian components and the "Gaussian noise inside the t"
  approximation for finite df.

All distribution math (densities, sampling, quantiles, convolution, conformal
thresholds, scores) comes from torchregress; this module adds the trunks, the
training loop, the ensemble bookkeeping (member contexts, stratified sampling,
the concatenated mixture of mixture members), a gradient-refined mode and a
sample-based count of separated modes (``secondary_mode``).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

PARAMS = ("teff", "logg", "fe_h")
KINDS = ("flow", "tmix", "t")
# Label-error floor and cap (physical units) of the errors-in-variables term:
# quoted common-scale errors below the floor are not credible, above the cap a
# record is not a measurement of the parameter.
ERROR_FLOOR = (10.0, 0.01, 0.01)
ERROR_CAP = (500.0, 0.5, 0.5)
# Mode finding: k nearest draws (standardized units) and the density dip a
# secondary mode needs along the segment to the primary one.
MODE_NEIGHBOURS = 10
MODE_DIP = 0.8
MODE_PATH_POINTS = 9


def gather_rows(array, rows, *, chunk_rows=262_144, dtype=None):
    """Rows of a (memory-mapped) 2-D array, in the order given.

    The file is read in contiguous blocks of ``chunk_rows`` rows (sparse blocks
    by fancy indexing), so a large ``np.load(..., mmap_mode="r")`` array is
    streamed once instead of paged randomly.
    """
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((len(rows), array.shape[1]), dtype=dtype or array.dtype)
    if not len(rows):
        return out
    order = np.argsort(rows, kind="stable")
    srt = rows[order]
    if srt[0] < 0 or srt[-1] >= array.shape[0]:
        raise IndexError("row index out of range")
    pos = 0
    for start in range(0, array.shape[0], chunk_rows):
        stop = min(start + chunk_rows, array.shape[0])
        hi = int(np.searchsorted(srt, stop, side="left"))
        if hi > pos:
            want = srt[pos:hi]
            if hi - pos < (stop - start) // 64:
                out[order[pos:hi]] = np.asarray(array[want])
            else:
                out[order[pos:hi]] = np.asarray(array[start:stop])[want - start]
            pos = hi
        if pos == len(srt):
            break
    return out


@dataclass
class Standardizer:
    """Latent and label standardization (physical <-> standardized units)."""

    lat_mean: np.ndarray
    lat_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @classmethod
    def fit(cls, latents, labels):
        z = np.asarray(latents, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        return cls(
            z.mean(0), np.maximum(z.std(0), 1e-6), y.mean(0), np.maximum(y.std(0), 1e-6)
        )

    def state(self):
        return {k: np.asarray(getattr(self, k)) for k in self.__dataclass_fields__}

    @classmethod
    def from_state(cls, state):
        return cls(
            **{
                k: np.asarray(state[k], dtype=np.float64)
                for k in cls.__dataclass_fields__
            }
        )

    def latent(self, z, device=None):
        """Standardized float32 latents (numpy or torch input, e.g. float16)."""
        t = torch.as_tensor(z if isinstance(z, torch.Tensor) else np.asarray(z))
        t = t.to(device=device if device is not None else t.device)
        m = torch.as_tensor(self.lat_mean, dtype=torch.float32, device=t.device)
        s = torch.as_tensor(self.lat_std, dtype=torch.float32, device=t.device)
        return (t.float() - m) / s

    def y(self, y):
        return (np.asarray(y, dtype=np.float64) - self.y_mean) / self.y_std

    def y_phys(self, y):
        return np.asarray(y, dtype=np.float64) * self.y_std + self.y_mean

    def sigma(self, s):
        return np.asarray(s, dtype=np.float64) / self.y_std


def label_errors(err, reference=None):
    """Label errors (N, 3) for the EIV term: non-finite or non-positive values
    get the median of the ``reference`` rows, then the floor and cap apply."""
    err = np.array(err, dtype=np.float64, copy=True)
    ref = np.ones(len(err), bool) if reference is None else np.asarray(reference, bool)
    for j in range(3):
        e = err[:, j]
        good = np.isfinite(e) & (e > 0)
        pool = e[good & ref]
        e[~good] = np.median(pool) if len(pool) else 10 * ERROR_FLOOR[j]
        err[:, j] = np.clip(e, ERROR_FLOOR[j], ERROR_CAP[j])
    return err


def mlp(in_dim, hidden, out_dim):
    layers, d = [], int(in_dim)
    for h in hidden:
        layers += [nn.Linear(d, int(h)), nn.GELU()]
        d = int(h)
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


def _identity_design(like):
    return torch.eye(3, dtype=like.dtype, device=like.device).reshape(1, 1, 3, 3)


class Member(nn.Module):
    """One ensemble member: an MLP trunk on the standardized latent and a
    torchregress density head on the standardized (Teff, log g, [Fe/H])."""

    def __init__(self, kind, in_dim, *, hidden=(256, 256), context=64, transforms=3,
                 flow_hidden=(128, 128), bins=16, tail_bound=5.0, components=4,
                 df=5.0):  # fmt: skip
        super().__init__()
        from torchregress.losses import MixtureDensityLoss, NormalizingFlowLoss
        from torchregress.losses.nflows import create_flow_model

        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.config = dict(kind=kind, in_dim=int(in_dim), hidden=list(hidden),
                           context=int(context), transforms=int(transforms),
                           flow_hidden=list(flow_hidden), bins=int(bins),
                           tail_bound=float(tail_bound), components=int(components),
                           df=float(df))  # fmt: skip
        if kind == "flow":
            flow = create_flow_model(
                3, context, "nsf", n_transforms=transforms,
                hidden_features=list(flow_hidden), bins=bins, bound=tail_bound,
            )  # fmt: skip
            self.density = NormalizingFlowLoss(flow, reduction="none")
            self.trunk = mlp(in_dim, hidden, context)
        else:
            m = 1 if kind == "t" else int(components)
            self.density = MixtureDensityLoss(
                m, 3, "full", full_parameterization="precision", df=df, learn_df=True,
                reduction="none",
            )  # fmt: skip
            self.trunk = mlp(in_dim, hidden, self.density.expected_output_size)

    def forward(self, x):
        """Context (flow) or mixture parameters (t, tmix) of standardized latents."""
        return self.trunk(x)

    def mixture(self, ctx):
        return self.density.distribution(ctx)

    def log_prob(self, ctx, y):
        if self.kind == "flow":
            return self.density.log_prob(ctx, y).reshape(-1)
        return self.mixture(ctx).log_prob(y)

    def label_log_prob(self, ctx, y, sigma, n_mc=8):
        """log p(y_obs | z) with Gaussian label noise ``sigma`` (standardized)."""
        if self.kind == "flow":
            from torchregress.losses import NoiseAwareDensityLoss

            nll = NoiseAwareDensityLoss(
                self.density, mode="mc", n_samples=int(n_mc), reduction="none"
            )(ctx, y, sigma=sigma)
            return -nll.reshape(-1)
        mix = self.mixture(ctx)
        comp = mix.component_observation_log_prob(
            y, _identity_design(y), sigma.square()
        )
        return mix.mixture_log_prob(comp)

    def sample(self, ctx, n):
        """Draws ``[N, n, 3]``."""
        if self.kind == "flow":
            s = self.density.sample(ctx, n_samples=int(n))
            return s.unsqueeze(1) if s.dim() == 2 else s
        return self.mixture(ctx).sample((int(n),)).transpose(0, 1)


def train_member(member, std, z_train, y_train, s_train, z_stop, y_stop, s_stop, *,
                 eiv=True, eiv_samples=8, stop_mc=16, weights=None, batch=4096,
                 max_epochs=30, patience=4, lr=1e-3, weight_decay=1e-4, seed=0,
                 eval_batch=16384, log=print):  # fmt: skip
    """Fit one member by (EIV) maximum likelihood; early stopping on the stop set.

    ``z_*``: raw latents (torch, e.g. float16, on the training device); ``y_*``,
    ``s_*``: standardized labels and label errors (float32, same device);
    ``weights``: optional per-row importance weights (training rows). The stop
    loss is the mean noisy-label NLL (EIV with ``stop_mc`` draws for a flow).
    Returns the history; ``member`` holds the best state.
    """
    dev = z_train.device
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    raw_df = [p for n, p in member.named_parameters() if n.endswith("raw_df")]
    rest = [p for n, p in member.named_parameters() if not n.endswith("raw_df")]
    groups = [dict(params=rest, lr=lr, weight_decay=weight_decay)]
    if raw_df:  # torchregress: give the log-df its own group at ~1e-2
        groups.append(dict(params=raw_df, lr=1e-2, weight_decay=0.0))
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    w_all = None if weights is None else torch.as_tensor(weights, dtype=torch.float32,
                                                         device=dev)  # fmt: skip

    def rows_loss(z, y, s, n_mc):
        ctx = member(std.latent(z))
        if eiv:
            return -member.label_log_prob(ctx, y, s, n_mc)
        return -member.log_prob(ctx, y)

    def stop_loss():
        member.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(z_stop), eval_batch):
                sl = slice(i, i + eval_batch)
                nll = rows_loss(z_stop[sl], y_stop[sl], s_stop[sl], stop_mc)
                ok = torch.isfinite(nll)
                tot += float(nll[ok].sum())
                cnt += int(ok.sum())
        member.train()
        return tot / max(cnt, 1)

    n = len(z_train)
    best, best_state, bad = math.inf, None, 0
    hist = dict(train=[], valid=[], lr=[], skipped=[], seconds=[])
    t_start = time.time()
    member.train()
    for epoch in range(max_epochs):
        t0 = time.time()
        perm = torch.randperm(n, generator=gen).to(dev)
        tot, cnt, skipped = 0.0, 0, 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            nll = rows_loss(z_train[idx], y_train[idx], s_train[idx], eiv_samples)
            if w_all is None:
                loss = nll.mean()
            else:
                w = w_all[idx]
                loss = (w * nll).sum() / w.sum()
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
    hist.update(best_valid=best, epochs=len(hist["valid"]), n_train=int(n),
                n_valid=int(len(z_stop)), batch_size=int(batch), seed=int(seed),
                eiv=bool(eiv), eiv_samples=int(eiv_samples),
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


def concat_mixtures(mixtures):
    """Equal-weight mixture of ``StudentTMixture`` members (one per row batch)."""
    from torchregress.losses import StudentTMixture

    k = len(mixtures)
    logits = torch.cat([m.log_weights for m in mixtures], -1) - math.log(k)
    loc = torch.cat([m.loc for m in mixtures], -2)
    tril = torch.cat([m.scale_tril for m in mixtures], -3)
    df = torch.cat([m.df for m in mixtures], -1)
    return StudentTMixture(logits, loc, scale_tril=tril, df=df)


def sample_quantiles(samples, levels):
    """Per-row, per-coordinate quantiles ``[N, D, Q]`` of draws ``[N, S, D]``
    (linear interpolation between order statistics)."""
    srt = torch.sort(samples, dim=1).values
    n = srt.shape[1]
    lv = torch.as_tensor(levels, dtype=srt.dtype, device=srt.device).reshape(-1)
    pos = lv * (n - 1)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    w = (pos - lo.to(pos.dtype)).reshape(1, -1, 1)
    q = srt[:, lo] + (srt[:, hi] - srt[:, lo]) * w  # [N, Q, D]
    return q.transpose(1, 2)


class FlowPrediction:
    """The flow ensemble's predictive distribution for a batch of rows."""

    is_mixture = False

    def __init__(self, members, ctxs, eiv):
        self.members, self.ctxs, self.eiv = members, ctxs, bool(eiv)

    def _mix(self, per_member):
        return torch.logsumexp(torch.stack(per_member), 0) - math.log(len(per_member))

    def log_prob(self, y):
        return self._mix([m.log_prob(c, y) for m, c in zip(self.members, self.ctxs,
                                                           strict=True)])  # fmt: skip

    def log_prob_many(self, y):
        """``y [N, S, 3]`` -> ``[N, S]``."""
        n, s, _ = y.shape
        flat = y.reshape(n * s, 3)
        return self._mix([m.log_prob(c.repeat_interleave(s, 0), flat)
                          for m, c in zip(self.members, self.ctxs, strict=True)]
                         ).reshape(n, s)  # fmt: skip

    def label_log_prob(self, y, sigma, n_mc=32):
        """Predictive of the noisy label (EIV convolution) at ``y [N, 3]`` or
        ``[N, S, 3]``; the clean density for a non-EIV ensemble."""
        many = y.dim() == 3
        if not self.eiv:
            return self.log_prob_many(y) if many else self.log_prob(y)
        if not many:
            return self._mix([m.label_log_prob(c, y, sigma, n_mc) for m, c in
                              zip(self.members, self.ctxs, strict=True)])  # fmt: skip
        n, s, _ = y.shape
        flat, sf = y.reshape(n * s, 3), sigma.repeat_interleave(s, 0)
        return self._mix([m.label_log_prob(c.repeat_interleave(s, 0), flat, sf, n_mc)
                          for m, c in zip(self.members, self.ctxs, strict=True)]
                         ).reshape(n, s)  # fmt: skip

    def sample(self, n):
        """``[N, n, 3]``: n // K draws from every member (stratified over the
        equal-weight members; the remainder from the first members)."""
        k = len(self.members)
        counts = [n // k + (1 if i < n % k else 0) for i in range(k)]
        parts = [m.sample(c, cnt) for m, c, cnt in
                 zip(self.members, self.ctxs, counts, strict=True) if cnt]  # fmt: skip
        return torch.cat(parts, 1)

    def quantiles(self, levels, n_samples=128, samples=None):
        s = self.sample(n_samples) if samples is None else samples
        return sample_quantiles(s, levels), s

    def mode_starts(self, samples):
        return samples

    def prob_below(self, j, value, samples):
        return (samples[..., j] < value).float().mean(1)


class MixturePrediction:
    """A ``StudentTMixture`` predictive (mixture ensembles, the posterior head)."""

    is_mixture = True

    def __init__(self, mixture, eiv):
        self.mixture, self.eiv = mixture, bool(eiv)

    def log_prob(self, y):
        return self.mixture.log_prob(y)

    def log_prob_many(self, y):
        return self.mixture.log_prob(y.transpose(0, 1)).transpose(0, 1)

    def label_log_prob(self, y, sigma, n_mc=None):
        many = y.dim() == 3
        if not self.eiv:
            return self.log_prob_many(y) if many else self.log_prob(y)
        mix = self.mixture
        if many:
            v = y.transpose(0, 1)
            comp = mix.component_observation_log_prob(
                v, _identity_design(v), sigma.square().expand_as(v)
            )
            return mix.mixture_log_prob(comp).transpose(0, 1)
        comp = mix.component_observation_log_prob(
            y, _identity_design(y), sigma.square()
        )
        return mix.mixture_log_prob(comp)

    def sample(self, n):
        return self.mixture.sample((int(n),)).transpose(0, 1)

    def quantiles(self, levels, n_samples=None, samples=None):
        return self.mixture.marginal_quantile(list(levels)), samples

    def mode_starts(self, samples):
        return self.mixture.loc

    def prob_below(self, j, value, samples=None):
        x = self.mixture.loc.new_zeros(self.mixture.batch_shape + (3,))
        x[..., j] = value
        return self.mixture.marginal_cdf(x)[..., j]


class EnsemblePredictor:
    """K members of one kind on the frozen latent (equal-weight mixture)."""

    def __init__(self, members, std, eiv):
        kinds = {m.kind for m in members}
        if len(kinds) != 1:
            raise ValueError(f"members of one kind expected, got {kinds}")
        self.members, self.std, self.eiv = list(members), std, bool(eiv)
        self.kind = kinds.pop()

    def predict(self, z, device):
        """Prediction object for raw latents ``z`` (numpy or torch)."""
        x = self.std.latent(z, device)
        ctxs = [m(x) for m in self.members]
        if self.kind == "flow":
            return FlowPrediction(self.members, ctxs, self.eiv)
        return MixturePrediction(
            concat_mixtures([m.mixture(c) for m, c in zip(self.members, ctxs,
                                                          strict=True)]), self.eiv
        )  # fmt: skip

    def to(self, device):
        for m in self.members:
            m.to(device)
        return self


def posterior_prediction(median, q16, q84, nu, std, device):
    """The posterior head as independent Student-t marginals (standardized).

    Only per-parameter median and 16/84 % quantiles are stored, so the joint
    density is the product of the marginals (the head's correlations are lost).
    """
    from scipy.stats import t as student_t
    from torchregress.losses import StudentTMixture

    nu = np.array(np.broadcast_to(np.asarray(nu, dtype=np.float64), (len(median),)))
    half = 0.5 * (np.asarray(q84, "f8") - np.asarray(q16, "f8"))
    scale = half / student_t.ppf(0.84, nu)[:, None]
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, std.y_std * 0.05)
    loc = std.y(median)
    loc = np.where(np.isfinite(loc), loc, 0.0)
    f32 = dict(dtype=torch.float32, device=device)
    mix = StudentTMixture(
        torch.zeros(len(loc), 1, **f32), torch.as_tensor(loc, **f32)[:, None],
        scale_diag=torch.as_tensor(scale / std.y_std, **f32)[:, None],
        df=torch.as_tensor(nu, **f32)[:, None],
    )  # fmt: skip
    return MixturePrediction(mix, eiv=False)


def find_mode(pred, starts, *, steps=25, lr=0.05):
    """Mode of a predictive: the densest of ``starts [N, C, 3]``, refined by
    ``steps`` Adam steps on log p (kept only where it increases log p)."""
    with torch.no_grad():
        lp = pred.log_prob_many(starts)
        lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -math.inf))
        best = lp.argmax(1)
        th0 = starts[torch.arange(len(starts), device=starts.device), best]
        lp0 = lp.max(1).values
    if steps <= 0:
        return th0, lp0
    th = th0.clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            out = pred.log_prob(th)
            (
                -torch.where(torch.isfinite(out), out, torch.zeros_like(out)).sum()
            ).backward()
            opt.step()
    with torch.no_grad():
        th = th.detach()
        lp1 = pred.log_prob(th)
        better = torch.isfinite(lp1) & (lp1 > lp0)
        return torch.where(better[:, None], th, th0), torch.where(better, lp1, lp0)


def secondary_mode(samples, log_prob, path_log_prob, *, k=MODE_NEIGHBOURS,
                   dip=MODE_DIP, n_path=MODE_PATH_POINTS, min_mass=0.10):  # fmt: skip
    """Mass and location of the largest separated secondary mode, per row.

    ``samples [N, S, 3]`` (standardized) with their exact log densities
    ``log_prob [N, S]``; ``path_log_prob(points [N, P, 3]) -> [N, P]``.
    k-nearest-neighbour mode seeking: every draw links to the densest of its
    ``k`` nearest draws when that one is denser; following the links ends at
    local maxima, whose basins (fractions of draws) estimate the modes'
    masses. A basin with mass >= ``min_mass`` counts as a separate mode when
    the density along the segment from the primary (largest) basin's peak to
    its own dips below ``dip`` times the lower of the two peaks. Returns
    (secondary mass [N], 0 without one; its peak [N, 3], NaN without one).
    """
    n, s, _ = samples.shape
    rows = torch.arange(n, device=samples.device)
    dist = torch.cdist(samples, samples)
    kk = min(k + 1, s)
    nbr = dist.topk(kk, dim=-1, largest=False).indices[..., 1:]  # [N, S, k]
    lp_nb = log_prob.unsqueeze(1).expand(n, s, s).gather(2, nbr)
    best_val, best_pos = lp_nb.max(-1)
    cand = nbr.gather(2, best_pos.unsqueeze(-1)).squeeze(-1)
    own = torch.arange(s, device=samples.device).expand(n, s)
    parent = torch.where(best_val > log_prob, cand, own)
    for _ in range(int(math.ceil(math.log2(max(s, 2)))) + 2):
        parent = parent.gather(1, parent)
    mass = (
        torch.zeros(n, s, device=samples.device).scatter_add_(
            1, parent, torch.ones(n, s, device=samples.device)
        )
        / s
    )
    top_mass, top_idx = mass.topk(min(4, s), dim=1)
    primary = samples[rows, top_idx[:, 0]]
    lp_primary = log_prob[rows, top_idx[:, 0]]
    t = torch.linspace(0.1, 0.9, n_path, device=samples.device).reshape(1, -1, 1)
    second = torch.zeros(n, device=samples.device)
    peak = torch.full((n, 3), math.nan, device=samples.device)
    for c in range(1, top_mass.shape[1]):
        m = top_mass[:, c]
        pos = samples[rows, top_idx[:, c]]
        lp_c = log_prob[rows, top_idx[:, c]]
        path = primary.unsqueeze(1) + t * (pos - primary).unsqueeze(1)
        low = path_log_prob(path).min(1).values
        sep = (m >= min_mass) & (low < torch.minimum(lp_primary, lp_c) + math.log(dip))
        take = sep & (m > second)
        second = torch.where(take, m, second)
        peak = torch.where(take[:, None], pos, peak)
    return second, peak


def bin_index(values, edges):
    """Bin of each value for ``edges`` (b_i < v <= b_{i+1}); clipped to the
    first and last bins; NaN goes to the last bin."""
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(e[1:-1], v, side="left")
    return np.where(np.isfinite(v), idx, len(e) - 2).astype(np.int64)


def mondrian_groups(primary, secondary, calibration, min_count):
    """Mondrian groups: a partition of the rows into unions of cells, each
    holding calibration rows.

    1. a (primary, secondary) cell with >= ``min_count`` calibration rows is a
       group (id 100 * primary + secondary; ``secondary=None``: the primary bin
       is the cell, id 1000 + primary);
    2. the remaining cells of a primary bin pool into one group (id 1000 +
       primary) when they hold >= ``min_count`` calibration rows together;
    3. rows still unassigned join the largest group (most calibration rows) of
       their primary bin, else one pooled group (-1), which itself joins the
       largest group overall when it holds no calibration row.
    """
    p = np.asarray(primary, dtype=np.int64)
    cal = np.asarray(calibration, bool)
    out = np.full(len(p), -1, dtype=np.int64)
    if secondary is None:
        cell = 1000 + p
    else:
        cell = p * 100 + np.asarray(secondary, dtype=np.int64)
    uc, inv = np.unique(cell, return_inverse=True)
    dense = np.bincount(inv[cal], minlength=len(uc))[inv] >= min_count
    out[dense] = cell[dense]
    for b in np.unique(p):
        rest = (p == b) & (out == -1)
        if secondary is not None and (rest & cal).sum() >= min_count:
            out[rest] = 1000 + b

    def largest(mask):
        g, c = np.unique(out[mask & cal & (out != -1)], return_counts=True)
        return int(g[np.argmax(c)]) if len(g) else None

    for b in np.unique(p[out == -1]):
        target = largest(p == b)
        if target is not None:
            out[(p == b) & (out == -1)] = target
    left = out == -1
    if left.any() and not (left & cal).any():
        target = largest(np.ones(len(p), bool))
        if target is not None:
            out[left] = target
    return out


def conformal_level_sets(lp_cal, lp_test, level, groups_cal=None, groups_test=None):
    """Split-conformal level sets of -log p (torchregress ``ConformalizedDensity``).

    ``lp_*``: log densities of the calibration / test targets. Returns
    (membership of the test targets [N_test] bool, thresholds {group: q} on
    -log p; key None without groups).
    """
    from torchregress.losses import ConformalizedDensity

    cp = ConformalizedDensity(alpha=1.0 - float(level))
    lc = torch.as_tensor(np.asarray(lp_cal, dtype=np.float64))[:, None]
    lt = torch.as_tensor(np.asarray(lp_test, dtype=np.float64))[:, None]
    gc = None if groups_cal is None else torch.as_tensor(np.asarray(groups_cal))
    gt = None if groups_test is None else torch.as_tensor(np.asarray(groups_test))
    cp.calibrate_log_prob(lambda c, y: c[:, 0], lc, lc, groups=gc)
    inside = cp.contains(lambda c, y: -c[:, 0], lt, lt, groups=gt).numpy()
    q = cp.cti.q_hat
    thr = ({int(g): float(v) for g, v in q.items()} if isinstance(q, dict)
           else {None: float(q)})  # fmt: skip
    return inside, thr
