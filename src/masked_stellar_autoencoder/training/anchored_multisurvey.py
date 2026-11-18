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
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import torch
from torch import nn

from masked_stellar_autoencoder.models.model import (
    _atomic_torch_save,
    _capture_rng_state,
)

from .anchored_survey_map import (
    FULL,
    AnchoredSurveyMaps,
    StudentTHead,
    assign_tiers,
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
}


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
        spectra_groups=None, linear_min=5000, full_min=50000,
    ):  # fmt: skip
        params = [str(p) for p in rec["parameters"]]
        cols = [params.index(p) for p in MAPPED]
        self.label_index = [label_names.index(p) for p in MAPPED]
        self.names = [str(n) for n in rec["survey_names"]]
        row = np.asarray(rec["row"], dtype="i8")
        survey = np.asarray(rec["survey"], dtype="i8")
        value = np.asarray(rec["value"], dtype="f8")[:, cols]
        error = np.asarray(rec["error"], dtype="f8")[:, cols]
        value = dedupe_spectra_groups(row, survey, value, self.names, spectra_groups)
        for k, li in enumerate(self.label_index):
            mean, scale = scalers[li].mean_[0], scalers[li].scale_[0]
            value[:, k] = (value[:, k] - mean) / scale
            error[:, k] = error[:, k] / scale
        keep = np.isfinite(value).any(axis=1) & (row < n_rows)
        row, survey, value, error = row[keep], survey[keep], value[keep], error[keep]
        error[~np.isfinite(error) | (error < 0)] = 0.0
        order = np.argsort(row, kind="stable")
        row, survey, value, error = (
            row[order],
            survey[order],
            value[order],
            error[order],
        )
        m = len(self.names)
        self.counts = np.zeros((m, len(MAPPED)), dtype="i8")
        for k in range(len(MAPPED)):
            ok = np.isfinite(value[:, k])
            self.counts[:, k] = np.bincount(survey[ok], minlength=m)
        is_anchor = np.asarray(rec["survey_is_anchor"], dtype=bool)
        self.anchor = np.repeat(is_anchor[:, None], len(MAPPED), axis=1)
        self.core = np.zeros((m, len(MAPPED)), dtype=bool)
        self.core[:, MAPPED.index("fe_h")] = [_starts(n, core) for n in self.names]
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
        self.n = len(row)

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
        error_fill=None,
    ):  # fmt: skip
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
        self.map_reg_weight = float(map_reg_weight)
        self.slope_weight = float(slope_weight)
        self.monotone_weight = float(monotone_weight)
        self.error_fill = error_fill
        self.map_kind = "anchored"
        self._val_epoch = 0
        tiers = {k: int((records.tier == k).sum()) for k in range(4)}
        print(f"anchored maps: tiers (identity/offset/linear/full) {tiers}")
        for n, c, tr in zip(records.names, records.counts, records.tier, strict=True):
            print(f"  {n:30s} stars {c.tolist()} tier {tr.tolist()}")

    # Head ---------------------------------------------------------------------
    def _init_finetune_head(self, linearprobe, ftlabeldim, ftact):
        if linearprobe:
            raise ValueError("The anchored fine-tune needs the full head")
        act = {"relu": nn.ReLU(), "elu": nn.ELU(), "gelu": nn.GELU()}[ftact]
        self.lp = None
        self.ft = StudentTHead(self.latent_size, ftlabeldim, act, self.nu).to(
            self.device
        )

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
            ok = torch.isfinite(v)
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
        has = dims > 0
        loss = nll[has].sum() / has.sum().clamp_min(1)
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
        X_masked, mask, nanmask = self._apply_batch_masking(X_batch, eX_batch, ctx)
        if ctx.multitask:
            X_reconstructed, encoded = self.model(X_masked)
        else:
            encoded = self.model.encoder(X_masked)
        mu, L = self.ft.params(encoded)
        loss = self._joint_loss(mu, L, y_batch, e_y, rows)
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
        return loss

    # Per-epoch checks of the common scale ---------------------------------------
    @torch.no_grad()
    def predict_scaled(self, X, batch=8192):
        """Unmasked common-scale locations (N, P), scaled units, as inference."""
        from .eval_ensemble import _feature_batch_tensor

        self.model.eval()
        self.ft.eval()
        X = np.asarray(X)
        out = []
        for i in range(0, len(X), batch):
            xb = _feature_batch_tensor(X[i : i + batch], self.device)
            mu, _ = self.ft.params(self.model.encoder(xb))
            out.append(mu)
        return torch.cat(out)

    @torch.no_grad()
    def anchor_metrics(self, X_val, y_val) -> dict:
        """Held-out residuals against anchors, the [Fe/H] core and own labels."""
        y_val = torch.as_tensor(y_val, dtype=torch.float32)
        rows = y_val[:, -1].round().long().to(self.device)
        mu = self.predict_scaled(X_val)
        scale = [float(s.scale_[0]) for s in self.label_scalers]
        recs = self.records
        rec, pos, _ = recs.gather(rows)
        s = recs.survey[rec]
        out = {}

        def summary(key, d):
            d = d[torch.isfinite(d)]
            if len(d) < 10:
                return
            med = d.median()
            out[key] = dict(
                n=int(len(d)), bias=float(med),
                nmad=float(1.4826 * (d - med).abs().median()),
            )  # fmt: skip

        names = recs.names
        anchor_s = torch.as_tensor(recs.anchor[:, 0], device=self.device)[s]
        core_s = torch.as_tensor(recs.core[:, 2], device=self.device)[s]
        for k, name in enumerate(MAPPED):
            li = self.mapped[k]
            d = (mu[pos, li] - recs.value[rec, k]) * scale[li]
            if name == "fe_h":
                summary("fe_h_core", d[core_s])
            else:
                summary(f"{name}_anchor", d[anchor_s])
        for li in self.own:
            name = self.label_names[li]
            d = (mu[:, li].cpu() - y_val[:, li]) * scale[li]
            summary(name, d)
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
        m = self.anchor_metrics(kw["X_val"], kw["y_val"])
        m["epoch"] = self._val_epoch
        logging.info("Anchor metrics: " + json.dumps(m))
        print("Anchor metrics:", json.dumps(m), flush=True)
        base = self.ft_save_str.rsplit(".", 1)[0]
        with open(f"{base}_anchor_metrics.jsonl", "a") as f:
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
            "gauge": "identity anchors (seismic log g, IRFM Teff) + [Fe/H] core",
        }
        _atomic_torch_save(payload, self.ft_save_str)
        if (
            self.checkpoint_interval is not None
            and (epoch + 1) % self.checkpoint_interval == 0
        ):
            base = self.ft_save_str.rsplit(".", 1)[0]
            _atomic_torch_save(payload, f"{base}_checkpoint_{epoch + 1}.pth")
