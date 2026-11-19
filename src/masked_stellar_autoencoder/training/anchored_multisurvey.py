"""Anchored multi-survey fine-tune (v6): records, wrapper, per-epoch checks.

See ``anchored_survey_map`` for the model. Differences from the v1-v5
trainer (``multisurvey_finetune``):

* the head is a multivariate Student-t over all labels (``StudentTHead``);
* survey maps are anchored by construction (identity anchors, an [Fe/H]
  reference core with zero-sum offsets, capacity tiers) - no consensus
  weights, trust, precision or coverage;
* each star's records and own labels (e.g. E(B-V), parallax) enter one joint
  likelihood with learned survey noise and quoted errors;
* pipelines on the same spectra (``spectra_groups``) give one value per star
  and parameter (the first pipeline in the group's list that has it);
* after every epoch, held-out anchor metrics (``*_anchor_metrics.jsonl``)
  measure the common scale directly; pick the epoch by their ``score``.

v7 options (all off by default, so v6 configs behave as before):

* ``exclude_surveys``: drop sources entirely (e.g. redundant pipelines);
* ``record_cuts``: drop records outside a survey's valid domain (e.g. SLAM,
  an M-dwarf model, above 3800 K);
* ``censor``: a survey that cannot report [Fe/H] below its grid edge gives
  an upper limit for records below ``floor + margin`` (censored likelihood,
  ``censored_upper_nll``) instead of a value;
* ``min_records``: drop non-anchor surveys with fewer records (optionally
  keeping the [Fe/H] core);
* ``label_input: clean``: the label likelihood uses the encoder on clean
  inputs; masking and feature noise stay on the reconstruction branch;
* ``drop_input_features``: hide inputs (e.g. proper motions, RA/Dec) from
  the encoder;
* ``context``: per-star metadata for the head (``context_inputs``).

Metal-poor / scale experiments (off by default):

* ``balance``: oversample training rows by their table [Fe/H] (weights
  ``balance_weights``); the head then learns under the sampling prior, whose
  ratio to the training prior is saved (``sampling_ratio``) so predictions
  can be moved back to the training prior;
* ``reference_pool``: extra stars outside the table (inputs plus one
  reference value per star, e.g. a wide-binary companion inheriting its FGK
  primary's [Fe/H]) that enter through an offset-only map with a learned
  noise, so the common scale cannot depend on the stellar type there.
"""

from __future__ import annotations

import json
import logging
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from masked_stellar_autoencoder.models.model import (
    _atomic_torch_save,
    _capture_rng_state,
)

from .anchored_predict import anchored_predict, encoder_input
from .anchored_survey_map import (
    FULL,
    AnchoredSurveyMaps,
    StudentTHead,
    assign_tiers,
    censored_upper_nll,
    student_t_nll,
)
from .multisurvey_finetune import MultiSurveyWrapper

MAPPED = ("teff", "logg", "fe_h")
# Scale of each held-out check in the epoch score (physical units).
SCORE_UNITS = {
    "teff_anchor": 50.0,
    "logg_anchor": 0.05,
    "fe_h_core": 0.05,
    "ebv_star": 0.02,
    "fe_h_core_mp": 0.1,  # core references below [Fe/H] = -2
}
MP, VMP = -2.0, -2.5


def balance_weights(feh, *, bin=0.1, smooth=0.2, power=0.5, cap=10.0, lo=-5.0, hi=1.0):
    """Per-row sampling weights (max density / density)**power, capped.

    ``feh``: physical [Fe/H] proxy per training row (NaN: weight 1). Returns
    (weights, bin centres, ratio): ``ratio`` is the sampling-prior over
    training-prior factor at each centre (the binned weight, smoothed by
    ``smooth`` dex), used to move a prediction back to the training prior.
    """

    def smooth_bins(h, width):
        k = np.exp(-0.5 * (np.arange(-4 * width, 4 * width + 1) / width) ** 2)
        return np.convolve(h, k / k.sum(), mode="same")

    feh = np.asarray(feh, float)
    ok = np.isfinite(feh)
    edges = np.arange(lo, hi + 1e-9, bin)
    width = max(1, int(round(smooth / bin)))
    dens = smooth_bins(np.histogram(feh[ok], bins=edges)[0].astype(float), width)
    w_bin = np.clip((dens.max() / (dens + 1e-9)) ** power, 1.0, cap)
    idx = np.clip(np.digitize(feh[ok], edges) - 1, 0, len(w_bin) - 1)
    w = np.ones(len(feh))
    w[ok] = w_bin[idx]
    centres = 0.5 * (edges[1:] + edges[:-1])
    return w, centres, smooth_bins(w_bin, width)


def _starts(name, prefixes):
    return any(name.startswith(p) for p in prefixes)


def dedupe_spectra_groups(row, survey, value, names, groups):
    """NaN out values that a higher-priority pipeline of the same group gives.

    ``groups`` maps a group name to its pipelines in priority order. For each
    (star, group, parameter) only the first pipeline with a finite value keeps
    it. Returns the edited value array (a copy).
    """
    value = value.copy()
    names = list(names)
    for members in (groups or {}).values():
        rank = {names.index(m): k for k, m in enumerate(members) if m in names}
        if len(rank) < 2:
            continue
        in_group = np.isin(survey, list(rank))
        idx = np.flatnonzero(in_group)
        r = np.array([rank[s] for s in survey[idx]])
        for j in range(value.shape[1]):
            ok = np.isfinite(value[idx, j])
            sel = idx[ok]
            order = np.lexsort((r[ok], row[sel]))
            first = np.ones(len(sel), bool)
            first[1:] = row[sel][order][1:] != row[sel][order][:-1]
            drop = sel[order][~first]
            value[drop, j] = np.nan
    return value


class AnchoredRecords:
    """Survey records of the table's stars, scaled, grouped by row."""

    def __init__(
        self, rec, scalers, label_names, n_rows, device, *, core=(),
        spectra_groups=None, linear_min=5000, full_min=50000, cuts=None,
        censor=None, min_records=0, min_records_keep_core=True, exclude=(),
    ):  # fmt: skip
        params = [str(p) for p in rec["parameters"]]
        cols = [params.index(p) for p in MAPPED]
        self.label_index = [label_names.index(p) for p in MAPPED]
        self.names = [str(n) for n in rec["survey_names"]]
        row = np.asarray(rec["row"], dtype="i8")
        survey = np.asarray(rec["survey"], dtype="i8")
        value = np.asarray(rec["value"], dtype="f8")[:, cols]
        error = np.asarray(rec["error"], dtype="f8")[:, cols]
        is_anchor = np.asarray(rec["survey_is_anchor"], dtype=bool)
        is_core = np.array([_starts(n, core) for n in self.names])
        value = self._apply_cuts(survey, value, cuts)
        gone = [self.names.index(n) for n in exclude or () if n in self.names]
        value[np.isin(survey, gone)] = np.nan
        value = dedupe_spectra_groups(row, survey, value, self.names, spectra_groups)
        self.dropped = []
        if min_records:
            n = np.bincount(
                survey[np.isfinite(value).any(1)], minlength=len(self.names)
            )
            small = (n < int(min_records)) & ~is_anchor
            if min_records_keep_core:
                small &= ~is_core
            self.dropped = [self.names[i] for i in np.flatnonzero(small & (n > 0))]
            value[small[survey]] = np.nan
        censored = np.zeros(value.shape, bool)
        upper = np.full(value.shape, np.nan)
        for name, lims in (censor or {}).items():
            if name not in self.names:
                continue
            here = survey == self.names.index(name)
            for p, u in lims.items():
                k = MAPPED.index(p)
                c = here & (value[:, k] < float(u))
                censored[c, k] = True
                upper[c, k] = float(u)
        for k, li in enumerate(self.label_index):
            mean, scale = scalers[li].mean_[0], scalers[li].scale_[0]
            value[:, k] = (value[:, k] - mean) / scale
            error[:, k] = error[:, k] / scale
            upper[:, k] = (upper[:, k] - mean) / scale
        keep = np.isfinite(value).any(axis=1) & (row < n_rows)
        row, survey, value, error = row[keep], survey[keep], value[keep], error[keep]
        censored, upper = censored[keep], upper[keep]
        error[~np.isfinite(error) | (error < 0)] = 0.0
        order = np.argsort(row, kind="stable")
        row, survey, value, error = (
            row[order],
            survey[order],
            value[order],
            error[order],
        )
        censored, upper = censored[order], upper[order]
        m = len(self.names)
        self.counts = np.zeros((m, len(MAPPED)), dtype="i8")
        for k in range(len(MAPPED)):
            ok = np.isfinite(value[:, k])
            self.counts[:, k] = np.bincount(survey[ok], minlength=m)
        self.anchor = np.repeat(is_anchor[:, None], len(MAPPED), axis=1)
        self.core = np.zeros((m, len(MAPPED)), dtype=bool)
        self.core[:, MAPPED.index("fe_h")] = is_core
        self.core &= self.counts > 0
        self.tier = assign_tiers(
            self.counts, self.anchor, self.core, linear_min=linear_min,
            full_min=full_min,
        )  # fmt: skip
        ptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=n_rows))])
        t = lambda a, dt: torch.as_tensor(a, device=device, dtype=dt)  # noqa: E731
        self.ptr = t(ptr, torch.long)
        self.value = t(value, torch.float32)
        self.error = t(error, torch.float32)
        self.survey = t(survey, torch.long)
        self.censored = t(censored, torch.bool)
        self.upper = t(np.nan_to_num(upper), torch.float32)
        self.n = len(row)
        self.n_censored = int(censored.sum())

    def _apply_cuts(self, survey, value, cuts):
        """NaN out whole records outside ``{survey: {param: [lo, hi]}}`` (raw units)."""
        value = value.copy()
        for name, lims in (cuts or {}).items():
            if name not in self.names:
                continue
            here = survey == self.names.index(name)
            out = np.zeros(len(value), bool)
            for p, (lo, hi) in lims.items():
                v = value[:, MAPPED.index(p)]
                lo = -np.inf if lo is None else float(lo)
                hi = np.inf if hi is None else float(hi)
                out |= here & np.isfinite(v) & ((v < lo) | (v >= hi))
            value[out] = np.nan
        return value

    def gather(self, rows: torch.Tensor):
        """Record indices, their batch positions and slot within the star."""
        start = self.ptr[rows]
        count = self.ptr[rows + 1] - start
        pos = torch.repeat_interleave(
            torch.arange(len(rows), device=rows.device), count
        )
        slot = torch.arange(len(pos), device=rows.device) - torch.repeat_interleave(
            torch.cumsum(count, 0) - count, count
        )
        return start[pos] + slot, pos, slot


class AnchoredMultiSurveyWrapper(MultiSurveyWrapper):
    """Joint Student-t likelihood over survey records and own labels."""

    def configure_anchored(
        self, records: AnchoredRecords, label_names, *, nu=5.0, basis=8, hidden=64,
        map_reg_weight=1e-3, slope_weight=0.01, monotone_weight=1.0,
        error_fill=None, label_input="masked", drop_input_features=(),
        context=None, dust_weight=1.0,
    ):  # fmt: skip
        if label_input not in ("masked", "clean"):
            raise ValueError(
                f"label_input must be masked or clean, not {label_input!r}"
            )
        self.label_input = label_input
        self.drop_input_features = list(drop_input_features)
        self.drop_idx = [self.feature_cols.index(c) for c in self.drop_input_features]
        self.context = context
        self.dust_weight = float(dust_weight)
        self.records = records
        self.label_names = list(label_names)
        self.mapped = list(records.label_index)
        self.own = [i for i in range(len(label_names)) if i not in self.mapped]
        self.nu = float(nu)
        self.survey_maps = AnchoredSurveyMaps(
            records.tier, records.core, records.counts, basis=int(basis),
            hidden=int(hidden),
        ).to(self.device)  # fmt: skip
        self.survey_maps.own_log_sigma = nn.Parameter(
            torch.full((len(self.own),), math.log(0.05), device=self.device)
        )
        if context is not None:
            # Rides with the maps: optimizer group, clipping, checkpoint.
            self.survey_maps.context_inputs = context.to(self.device)
        self.map_reg_weight = float(map_reg_weight)
        self.slope_weight = float(slope_weight)
        self.monotone_weight = float(monotone_weight)
        self.error_fill = error_fill
        self.map_kind = "anchored"
        self._val_epoch = 0
        tiers = {k: int((records.tier == k).sum()) for k in range(4)}
        print(f"anchored maps: tiers (identity/offset/linear/full) {tiers}")
        print(
            f"label input {label_input}; dropped inputs {self.drop_input_features};"
            f" context dim {context.dim if context is not None else 0};"
            f" censored records {records.n_censored}; dropped surveys {records.dropped}"
        )
        for n, c, tr in zip(records.names, records.counts, records.tier, strict=True):
            print(f"  {n:30s} stars {c.tolist()} tier {tr.tolist()}")

    # Balance and reference pool --------------------------------------------------
    def configure_balance(self, weights, centres, ratio):
        """Sample training rows with ``weights`` (``balance_weights``)."""
        self.balance_w = torch.as_tensor(np.asarray(weights), dtype=torch.double)
        self.sampling_ratio = {
            "centres": np.asarray(centres, "f8"),
            "ratio": np.asarray(ratio, "f8"),
        }
        ess = float(self.balance_w.sum() ** 2 / self.balance_w.square().sum())
        print(f"balance: effective sample size {ess / len(self.balance_w):.2f} N")

    def _prepare_finetune_loader(
        self, X_train, eX_train, y_train, e_y_train, mini_batch
    ):
        w = getattr(self, "balance_w", None)
        if w is None:
            return super()._prepare_finetune_loader(
                X_train, eX_train, y_train, e_y_train, mini_batch
            )
        if len(w) != len(X_train):
            raise ValueError("Balance weights do not match the training rows")
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        tensors = [
            torch.as_tensor(arr, device=self.device, dtype=torch.float32)
            for arr in (X_train, eX_train, y_train, e_y_train)
        ]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(
            TensorDataset(*tensors), batch_size=mini_batch, sampler=sampler
        )

    def configure_reference_pool(
        self, X, E, value, error, *, label="fe_h", mean=0.0, scale=1.0, ratio=0.25,
        weight=1.0, sigma_init=0.05,
    ):  # fmt: skip
        """Stars outside the table with one reference value each (physical).

        Their likelihood uses an offset-only map (one learned offset and
        noise for the whole pool), so it constrains the common scale the
        same way for every stellar type the pool covers.
        """
        if getattr(self, "context", None) is not None:
            raise ValueError("The reference pool does not support a head context")
        self.ref_X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.ref_value = torch.as_tensor((np.asarray(value) - mean) / scale).float()
        self.ref_error = torch.as_tensor(np.asarray(error) / scale).float()
        self.ref_j = self.label_names.index(label)
        self.ref_ratio, self.ref_weight = float(ratio), float(weight)
        self.ref_scale = float(scale)
        maps = self.survey_maps
        maps.ref_bias = nn.Parameter(torch.zeros((), device=self.device))
        maps.ref_log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma_init / scale), device=self.device)
        )
        self._ref_steps, self._ref_sum = 0, 0.0
        print(f"reference pool: {len(self.ref_X)} stars ({label})")

    def _reference_pool_loss(self, n):
        idx = torch.randint(len(self.ref_X), (max(1, int(n)),))
        x = self.ref_X[idx].to(self.device, non_blocking=True)
        v = self.ref_value[idx].to(self.device)
        e = self.ref_error[idx].to(self.device)
        mu, L = self.ft.params(self.model.encoder(encoder_input(x, self.drop_idx)))
        maps = self.survey_maps
        resid = (v - mu[:, self.ref_j] - maps.ref_bias)[:, None]
        A = torch.zeros(len(x), 1, mu.shape[1], device=self.device)
        A[:, 0, self.ref_j] = 1.0
        noise = (maps.ref_log_sigma.exp() ** 2 + e * e)[:, None]
        valid = torch.isfinite(resid)
        nll, _ = student_t_nll(torch.nan_to_num(resid), A, L, noise, valid, self.nu)
        loss = nll.mean()
        self._ref_steps += 1
        self._ref_sum += float(loss.detach())
        if self._ref_steps % 1000 == 0:
            logging.info(
                f"Reference pool: nll {self._ref_sum / 1000:.4f} offset "
                f"{float(maps.ref_bias) * self.ref_scale:+.4f} sigma "
                f"{float(maps.ref_log_sigma.exp()) * self.ref_scale:.4f}"
            )
            self._ref_sum = 0.0
        return loss

    # Head ---------------------------------------------------------------------
    def _init_finetune_head(self, linearprobe, ftlabeldim, ftact):
        if linearprobe:
            raise ValueError("The anchored fine-tune needs the full head")
        act = {"relu": nn.ReLU(), "elu": nn.ELU(), "gelu": nn.GELU()}[ftact]
        self.lp = None
        dim = self.context.dim if getattr(self, "context", None) is not None else 0
        self.ft = StudentTHead(
            self.latent_size, ftlabeldim, act, self.nu, context_dim=dim
        ).to(self.device)

    # Loss ---------------------------------------------------------------------
    def _joint_loss(self, mu, L, y, e_y, rows):
        B, P = mu.shape
        dev = mu.device
        recs = self.records
        rec, pos, slot = recs.gather(rows)
        maps = self.survey_maps
        mapped = torch.as_tensor(self.mapped, device=dev)
        n_own = len(self.own)
        k_max = int(slot.max()) + 1 if len(slot) else 0
        D = n_own + len(self.mapped) * k_max
        resid = mu.new_zeros(B, D)
        A = mu.new_zeros(B, D, P)
        noise = mu.new_ones(B, D)
        valid = torch.zeros(B, D, dtype=torch.bool, device=dev)
        own_sigma = maps.own_log_sigma.exp()
        for o, li in enumerate(self.own):
            z = y[:, li]
            ok = torch.isfinite(z)
            resid[:, o] = torch.where(ok, z - mu[:, li], 0.0)
            A[:, o, li] = 1.0
            e = torch.nan_to_num(e_y[:, li], nan=0.0) if e_y is not None else 0.0
            noise[:, o] = e * e + own_sigma[o] ** 2
            valid[:, o] = ok
        jac = None
        if len(rec):
            s = recs.survey[rec]
            ym, jac = maps(mu[pos][:, mapped], s)
            v = recs.value[rec]
            cens = recs.censored[rec]
            ok = torch.isfinite(v) & ~cens
            sig = maps.sigma(s)
            err = recs.error[rec]
            d = (
                n_own
                + len(self.mapped) * slot[:, None]
                + torch.arange(len(self.mapped), device=dev)
            )  # (R, 3)
            b = pos[:, None].expand_as(d)
            resid = resid.index_put((b, d), torch.where(ok, v - ym, 0.0))
            noise = noise.index_put((b, d), sig * sig + err * err)
            valid = valid.index_put((b, d), ok)
            A = A.index_put(
                (b[:, :, None].expand(-1, -1, len(self.mapped)),
                 d[:, :, None].expand(-1, -1, len(self.mapped)),
                 mapped[None, None, :].expand(len(rec), len(self.mapped), -1)),
                jac,
            )  # fmt: skip
        nll, dims = student_t_nll(resid, A, L, noise, valid, self.nu)
        per_star = torch.where(dims > 0, nll, torch.zeros_like(nll))
        has = dims > 0
        if jac is not None and bool(cens.any()):
            # Censored records: P(mapped value <= upper limit), marginal of the
            # linearised joint (head scale through the map, plus survey noise).
            jl = torch.einsum("rjk,rkp->rjp", jac, L[pos][:, mapped, :])
            var = jl.square().sum(-1) + sig * sig + err * err
            term = censored_upper_nll(recs.upper[rec], ym, var)
            term = torch.where(cens, term, torch.zeros_like(term)).sum(-1)
            per_star = per_star.index_add(0, pos, term)
            has = has.index_fill(0, pos[cens.any(-1)], True)
        loss = per_star[has].sum() / has.sum().clamp_min(1)
        loss = loss + self.map_reg_weight * maps.regularization()
        if jac is not None:
            log_slope, short = maps.slope_penalties(jac)
            loss = loss + self.slope_weight * log_slope + self.monotone_weight * short
        return loss

    def _compute_finetune_batch_loss(self, batch, ctx):
        X_batch, eX_batch, y_ext, e_y_ext = batch
        rows = y_ext[:, -1].round().long()
        y_batch = y_ext[:, :-1]  # the last column is the table row
        e_y = e_y_ext[:, : y_batch.shape[1]]
        X_in = X_batch
        if self.drop_idx:
            X_in = X_batch.clone()
            X_in[:, self.drop_idx] = float("nan")
        X_masked, mask, nanmask = self._apply_batch_masking(X_in, eX_batch, ctx)
        if ctx.multitask:
            X_reconstructed, encoded = self.model(X_masked)
        else:
            encoded = self.model.encoder(X_masked)
        if self.label_input == "clean":
            encoded = self.model.encoder(encoder_input(X_in))
        context, dust = None, None
        if self.context is not None:
            context, dust = self.context(X_batch, eX_batch, y_batch, e_y)
        mu, L = self.ft.params(encoded, context)
        loss = self._joint_loss(mu, L, y_batch, e_y, rows)
        if dust is not None:
            loss = loss + self.dust_weight * self.context.dust_loss(dust, y_batch, e_y)
        if ctx.multitask:
            keep = mask[:, : -self.diff] & nanmask[:, : -self.diff]
            e = eX_batch[:, : -self.diff]
            rec = self._rec_loss_fn(
                X_batch[:, : -self.diff], X_reconstructed, keep, 1.0 / (e * e + 1e-8)
            )
            loss = ctx.ft_lambda_pred * loss + ctx.ft_lambda_rec * rec
        if getattr(self, "unl_X", None) is not None and torch.is_grad_enabled():
            n = max(2, int(round(self.unl_ratio * len(X_batch))))
            rec_u, cons = self._unlabelled_loss(
                n, ctx, consistency=self.unl_cons_weight > 0
            )
            loss = loss + self.unl_rec_weight * rec_u + self.unl_cons_weight * cons
            self._log_unlabelled(rec_u, cons)
        if getattr(self, "ref_X", None) is not None and torch.is_grad_enabled():
            n = round(self.ref_ratio * len(X_batch))
            loss = loss + self.ref_weight * self._reference_pool_loss(n)
        return loss

    # Per-epoch checks of the common scale ---------------------------------------
    @torch.no_grad()
    def predict_scaled(self, X, E=None, Y=None, eY=None):
        """Unmasked common-scale locations and scales (N, P), scaled units."""
        mu, sd = anchored_predict(
            self.model, self.ft, X, device=self.device, E=E, context=self.context,
            drop_idx=self.drop_idx, Y=Y, eY=eY,
        )  # fmt: skip
        return torch.as_tensor(mu, device=self.device), torch.as_tensor(
            sd, device=self.device
        )

    @torch.no_grad()
    def anchor_metrics(self, X_val, y_val, eX_val=None, e_y_val=None) -> dict:
        """Held-out residuals against anchors, the [Fe/H] core and own labels.

        ``cov``: fraction of the references inside the model's 16-84 %
        interval (reference errors not included). ``fe_h_core_mp``/``_vmp``:
        core references below -2 / -2.5; ``fe_h_mp_predbin``: median
        (reference - prediction) for stars predicted below -2 (a calibrated
        posterior gives ~0 there; bias in reference bins is expected).
        """
        from scipy.stats import t as student_t

        y_val = torch.as_tensor(y_val, dtype=torch.float32)
        rows = y_val[:, -1].round().long().to(self.device)
        P = len(self.label_names)
        Y = y_val[:, :P].numpy()
        eY = None if e_y_val is None else np.asarray(e_y_val)[:, :P]
        mu, sd = self.predict_scaled(X_val, eX_val, Y, eY)
        q84 = float(student_t.ppf(0.84, self.nu))
        scale = [float(s.scale_[0]) for s in self.label_scalers]
        mean = [float(s.mean_[0]) for s in self.label_scalers]
        recs = self.records
        rec, pos, _ = recs.gather(rows)
        s = recs.survey[rec]
        out = {}

        def summary(key, d, half=None):
            ok = torch.isfinite(d)
            if int(ok.sum()) < 10:
                return
            d0 = d[ok]
            med = d0.median()
            out[key] = dict(
                n=int(len(d0)), bias=float(med),
                nmad=float(1.4826 * (d0 - med).abs().median()),
            )  # fmt: skip
            if half is not None:
                out[key]["cov"] = round(
                    float((d[ok].abs() <= half[ok]).float().mean()), 3
                )

        names = recs.names
        anchor_s = torch.as_tensor(recs.anchor[:, 0], device=self.device)[s]
        core_s = torch.as_tensor(recs.core[:, 2], device=self.device)[s]
        for k, name in enumerate(MAPPED):
            li = self.mapped[k]
            d = (mu[pos, li] - recs.value[rec, k]) * scale[li]
            half = q84 * sd[pos, li] * scale[li]
            if name == "fe_h":
                summary("fe_h_core", d[core_s], half[core_s])
                ref = recs.value[rec, k] * scale[li] + mean[li]
                pred = mu[pos, li] * scale[li] + mean[li]
                for key, lim in (("fe_h_core_mp", MP), ("fe_h_core_vmp", VMP)):
                    m = core_s & (ref < lim)
                    summary(key, d[m], half[m])
                m = core_s & (pred < MP)
                if int(m.sum()) >= 10:
                    out["fe_h_mp_predbin"] = dict(
                        n=int(m.sum()), bias=float((ref[m] - pred[m]).median())
                    )
            else:
                summary(f"{name}_anchor", d[anchor_s], half[anchor_s])
        # Inference-time conditioning on the measured parallax (parallax/error
        # >= 5), as predict_fits_rows --condition-parallax 5.
        if "parallax" in self.label_names and eY is not None:
            j = self.label_names.index("parallax")
            e_log = eY[:, j] * scale[j]
            ok = np.isfinite(Y[:, j]) & (e_log <= 1 / (5 * math.log(10)))
            z = np.where(ok, Y[:, j], np.nan).astype("f4")
            var = np.where(ok, eY[:, j] ** 2, np.nan).astype("f4")
            mu_c, _ = anchored_predict(
                self.model, self.ft, X_val, device=self.device, E=eX_val,
                context=self.context, drop_idx=self.drop_idx, Y=Y, eY=eY,
                condition=(j, z, var),
            )  # fmt: skip
            mu_c = torch.as_tensor(mu_c, device=self.device)
            for k, name in enumerate(MAPPED[:2]):
                li = self.mapped[k]
                d = (mu_c[pos, li] - recs.value[rec, k]) * scale[li]
                summary(f"{name}_anchor_plx", d[anchor_s])
        for li in self.own:
            name = self.label_names[li]
            d = (mu[:, li].cpu() - y_val[:, li]) * scale[li]
            summary(name, d, (q84 * sd[:, li] * scale[li]).cpu())
        if self.context is not None and self.context.dust is not None:
            with torch.no_grad():
                _, dust = self.context(
                    torch.as_tensor(np.asarray(X_val, "f4"), device=self.device),
                    None if eX_val is None else torch.as_tensor(
                        np.asarray(eX_val, "f4"), device=self.device),
                    torch.as_tensor(Y, device=self.device), None,
                )  # fmt: skip
            j = self.label_names.index("ebv_star")
            target = y_val[:, j] * scale[j] + mean[j]
            summary("dust_branch_ebv", dust[0].cpu() - target)
        # Map slopes of the large surveys at the held-out stars they observe.
        slopes = {}
        full = np.flatnonzero((recs.tier == FULL).any(axis=1))
        mapped = torch.as_tensor(self.mapped, device=self.device)
        for si in full:
            k = s == int(si)
            if k.sum() < 50:
                continue
            _, jac = self.survey_maps(mu[pos[k]][:, mapped], s[k])
            diag = torch.diagonal(jac, dim1=1, dim2=2).median(0).values
            slopes[names[si]] = [round(float(x), 3) for x in diag]
        out["map_slopes"] = slopes
        out["score"] = float(
            sum(
                (abs(out[k]["bias"]) + out[k]["nmad"]) / unit
                for k, unit in SCORE_UNITS.items()
                if k in out
            )
        )
        return out

    def _maybe_validate_finetune(self, **kw):
        super()._maybe_validate_finetune(**kw)
        if kw.get("X_val") is None:
            return
        self._val_epoch += 1
        m = self.anchor_metrics(
            kw["X_val"], kw["y_val"], kw.get("eX_val"), kw.get("e_y_val")
        )
        m["epoch"] = self._val_epoch
        # Peak resident memory of this process (diagnoses killed sessions).
        m["max_rss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1
        )
        logging.info("Anchor metrics: " + json.dumps(m))
        print("Anchor metrics:", json.dumps(m), flush=True)
        base = self.ft_save_str.rsplit(".", 1)[0]
        path = Path(f"{base}_anchor_metrics.jsonl")
        if self._val_epoch == 1 and path.exists():
            # A rerun starts from scratch: keep the old epochs aside, unmixed.
            path.rename(path.with_name(f"{path.name}.{int(time.time())}.old"))
        with open(path, "a") as f:
            f.write(json.dumps(m) + "\n")

    # Checkpoints ------------------------------------------------------------------
    def _save_finetune_checkpoint(self, linearprobe, epoch, optimizer, scheduler):
        recs = self.records
        payload = {
            "autoencoder_state_dict": self.model.state_dict(),
            "prediction_head_state_dict": self.ft.state_dict(),
            "linear_probe": False,
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng_state(),
            "featurescaler": self.featurescaler,
            "label_scalers": getattr(self, "label_scalers", None),
            "label_names": self.label_names,
            "head": "student_t",
            "nu": self.nu,
            "survey_maps_state_dict": self.survey_maps.state_dict(),
            "survey_names": recs.names,
            "map_tier": recs.tier,
            "map_core": recs.core,
            "map_counts": recs.counts,
            "map_kind": "anchored",
            "error_fill": self.error_fill,
            "label_input": self.label_input,
            "drop_input_features": self.drop_input_features,
            "context_dim": self.context.dim if self.context is not None else 0,
            "context_spec": self.context.spec if self.context is not None else None,
            "context_state_dict": (
                self.context.state_dict() if self.context is not None else None
            ),
            "dropped_surveys": recs.dropped,
            "sampling_ratio": getattr(self, "sampling_ratio", None),
            "gauge": "identity anchors (seismic log g, IRFM Teff) + [Fe/H] core",
        }
        _atomic_torch_save(payload, self.ft_save_str)
        if (
            self.checkpoint_interval is not None
            and (epoch + 1) % self.checkpoint_interval == 0
        ):
            base = self.ft_save_str.rsplit(".", 1)[0]
            _atomic_torch_save(payload, f"{base}_checkpoint_{epoch + 1}.pth")
