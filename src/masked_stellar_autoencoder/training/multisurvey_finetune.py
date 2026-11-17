"""Multi-survey fine-tune: one common-scale prediction, one map per survey.

The stopgap fine-tune scores each star against one chosen label. Here every
survey record of a star is scored: the head's common-scale quantiles for
Teff, log g, [Fe/H] and [alpha] pass through that survey's monotone map
before the pinball loss. A monotone increasing map sends quantiles to
quantiles, so mapping the 16/50/84 % predictions is exact.

Gauge (which common scale the maps leave free):

* Teff, [Fe/H], [alpha]: survey consensus. Every survey has the same map
  family (monotone linear spline, 8 bins on [-4, 4] in scaled units), and
  the arithmetic mean over surveys of offsets and bin slopes is fixed to
  the identity (SixSurveyMap ``zero_sum``).
* log g: physical anchors. Seismic records (APOKASC-3, APO-K2, APOKASC
  dwarfs, Kepler LEGACY) bypass the maps, so they pin the scale where they
  exist (mostly giants). A soft consensus penalty on the survey log g maps
  keeps the scale defined elsewhere (dwarfs, hot and metal-poor stars).

That is ``map_kind="spline"`` (v1, v2). ``map_kind="conditional"`` replaces
both with maps that depend on all four parameters and a local, coverage-
and precision-weighted gauge (conditional_survey_map).

Each (star, label) carries total weight 1 shared by its records, so a star
counts as much as in the stopgap however many surveys observed it. Gaia
benchmark stars are not in the training table (validation only).
"""

from __future__ import annotations

import numpy as np
import torch

from masked_stellar_autoencoder.models.model import (
    TabResnetWrapper,
    _atomic_torch_save,
    _capture_rng_state,
)

from .conditional_survey_map import (
    ConditionalSurveyMaps,
    coverage_grid,
    overlap_precision,
    overlap_scatter,
)
from .six_survey_map import SixSurveyMap

SURVEY_LABELS = 4  # Teff, log g, [Fe/H], [alpha]; age and parallax are not mapped
LOGG = 1
LABEL_INDEX = {"teff": 0, "logg": 1, "fe_h": 2, "alpha": 3}


def scale_records(value: np.ndarray, scalers) -> np.ndarray:
    """Apply the fine-tune label scalers to record values (NaN stays NaN)."""
    out = np.full(value.shape, np.nan, dtype=np.float32)
    for j in range(value.shape[1]):
        out[:, j] = scalers[j].transform(value[:, j : j + 1].astype("f8"))[:, 0]
    return out


class SurveyRecords:
    """Records grouped by table row, on the training device."""

    def __init__(self, rec: dict, scalers, n_rows: int, device):
        order = np.argsort(rec["row"], kind="stable")
        row = rec["row"][order]
        value = scale_records(rec["value"][order], scalers)
        survey = rec["survey"][order].astype("i8")
        self.names = [str(n) for n in rec["survey_names"]]
        self.is_anchor = np.asarray(rec["survey_is_anchor"], dtype=bool)
        mapped = np.flatnonzero(~self.is_anchor)
        map_index = np.full(len(self.names), -1, dtype="i8")
        map_index[mapped] = np.arange(len(mapped))
        self.mapped_names = [self.names[i] for i in mapped]
        finite = np.isfinite(value)
        support = np.zeros((len(mapped), 6), dtype=bool)
        for s_all, s_map in zip(mapped, range(len(mapped)), strict=True):
            support[s_map, :SURVEY_LABELS] = finite[survey == s_all].any(axis=0)
        self.support = support
        # Finite records per (mapped survey, label): decides consensus members.
        self.counts = np.zeros((len(mapped), 6), dtype="i8")
        for s_all, s_map in zip(mapped, range(len(mapped)), strict=True):
            self.counts[s_map, :SURVEY_LABELS] = finite[survey == s_all].sum(axis=0)
        counts = np.bincount(row, minlength=n_rows)
        ptr = np.concatenate([[0], np.cumsum(counts)])
        t = lambda a, dtype: torch.as_tensor(a, device=device, dtype=dtype)  # noqa: E731
        self.ptr = t(ptr, torch.long)
        self.value = t(value, torch.float32)
        self.survey_map = t(map_index[survey], torch.long)
        self.n = len(row)
        # Where each mapped survey has stars (for the conditional maps' gauge).
        self.coverage = coverage_grid(value, map_index[survey], row, len(mapped))
        self._host = (row, value, map_index[survey])  # for overlap_scatter

    def gather(self, rows: torch.Tensor):
        """Record indices and their batch positions for table rows ``rows``."""
        start = self.ptr[rows]
        count = self.ptr[rows + 1] - start
        pos = torch.repeat_interleave(
            torch.arange(len(rows), device=rows.device), count
        )
        offset = torch.arange(len(pos), device=rows.device) - torch.repeat_interleave(
            torch.cumsum(count, 0) - count, count
        )
        return start[pos] + offset, pos


class ConsensusSurveyMap(SixSurveyMap):
    """Survey maps whose consensus runs over ``members`` only.

    ``members[s, j]`` marks the surveys that define target j's common scale
    (the large surveys). For each target in ``consensus`` their arithmetic mean
    offset is zero and mean bin slope one; other surveys keep free maps,
    pulled toward the identity by ``regularization``. With every supported
    survey as a member this is SixSurveyMap's ``zero_sum`` gauge.
    """

    def __init__(self, support, members, consensus, *, bins=8, groups=None):
        reference = torch.full((6,), -1, dtype=torch.long)  # no reference survey
        super().__init__(support, reference, kind="spline", bins=bins)
        members = torch.as_tensor(members, dtype=torch.bool) & self.support
        hard = torch.zeros(6, dtype=torch.bool)
        hard[list(consensus)] = True
        # A target no survey reaches the threshold for uses all its surveys.
        empty = ~members.any(0)
        members[:, empty] = self.support[:, empty]
        # Member weights: each spectra group (e.g. all pipelines run on APOGEE
        # spectra) counts once per target, shared by its member surveys.
        groups = (
            torch.arange(len(members))
            if groups is None
            else torch.as_tensor(np.asarray(groups), dtype=torch.long)
        )
        weight = torch.zeros(members.shape)
        for j in range(6):
            m = members[:, j]
            if not m.any():
                continue
            g = groups[m]
            share = torch.stack([(g == x).sum() for x in g]).float()
            n_groups = len(torch.unique(g))
            weight[m, j] = 1.0 / (share * n_groups)
        self.register_buffer("member_weight", weight)
        self.register_buffer("members", members)
        self.register_buffer("consensus", hard)

    def coefficients(self):
        bias, slopes = super().coefficients()
        bias, slopes = bias.clone(), slopes.clone()
        for j in torch.nonzero(self.consensus, as_tuple=False).flatten().tolist():
            m = self.members[:, j]
            if m.any():
                w = self.member_weight[m, j]
                bias[m, j] = bias[m, j] - (w * bias[m, j]).sum()
                mean_slope = (w[:, None] * slopes[m, j]).sum(0, keepdim=True)
                slopes[m, j] = slopes[m, j] / mean_slope
        return bias, slopes


def survey_groups(names, prefixes=None) -> np.ndarray:
    """Group index per survey; ``prefixes`` maps a group name to name prefixes."""
    labels = []
    for name in names:
        hit = [g for g, pre in (prefixes or {}).items() if name.startswith(tuple(pre))]
        labels.append(hit[0] if hit else name)
    _, index = np.unique(labels, return_inverse=True)
    return index


def build_survey_maps(
    records: SurveyRecords, bins: int = 8, min_records: int = 0, groups=None
) -> ConsensusSurveyMap:
    """Teff/[Fe/H]/alpha consensus over surveys with >= ``min_records``."""
    members = records.support & (records.counts >= min_records)
    consensus = [j for j in range(SURVEY_LABELS) if j != LOGG]
    return ConsensusSurveyMap(
        records.support, members, consensus, bins=bins, groups=groups
    )


def trust_matrix(names, trust) -> np.ndarray | None:
    """(surveys, 4) consensus trust: ``factor`` for ``prefixes`` on ``targets``."""
    if not trust:
        return None
    out = np.ones((len(names), SURVEY_LABELS), dtype="f4")
    cols = [LABEL_INDEX[t] for t in trust.get("targets", list(LABEL_INDEX))]
    hit = np.array([n.startswith(tuple(trust["prefixes"])) for n in names])
    out[np.ix_(hit, cols)] = float(trust["factor"])
    return out


def soft_consensus(maps: SixSurveyMap, target: int = LOGG) -> torch.Tensor:
    """Squared mean (over member surveys) offset and log bin slope of a target."""
    bias, slopes = maps.coefficients()
    active = getattr(maps, "members", maps.support)[:, target]
    if not active.any():
        return bias.sum() * 0
    w = getattr(maps, "member_weight", None)
    if w is None:
        w = active.float() / active.sum()
    w = w[active, target]
    mean_bias = (w * bias[active, target]).sum()
    mean_log_slope = (w[:, None] * torch.log(slopes[active, target])).sum(0)
    return mean_bias.square() + mean_log_slope.square().mean()


class MultiSurveyWrapper(TabResnetWrapper):
    """TabResnetWrapper whose label term scores every survey record.

    ``y`` arrays carry one extra last column: the star's table row. Both the
    training loader and ``validate_fit`` pass it through to this loss.
    """

    def configure_multisurvey(
        self,
        records: SurveyRecords,
        *,
        consensus_weight=1.0,
        map_reg_weight=1e-3,
        consensus_min_records=0,
        anchor_weight=None,
        consensus_groups=None,
        map_kind="spline",
        gauge_weight=0.01,
        monotone_weight=1.0,
        trust=None,
        basis=8,
        hidden=64,
        hard_targets=(),
        precision="running",
    ):
        """``anchor_weight=None`` shares weight 1 across all records (v1);
        a number gives each anchor record that full weight instead.

        ``map_kind="spline"``: per-parameter monotone splines with a global
        consensus (v1, v2). ``"conditional"``: maps on all four parameters
        with the local precision-weighted gauge (ConditionalSurveyMaps);
        ``trust`` is ``{"factor": f, "prefixes": [...], "targets": [...]}``.
        ``hard_targets`` (names) hold the consensus exactly; ``precision`` is
        ``"running"`` (from residuals, v3) or ``"overlap"`` (fixed, v4).
        """
        self.records = records
        self.map_kind = map_kind
        groups = survey_groups(records.mapped_names, consensus_groups)
        if map_kind == "conditional":
            fixed = None
            if precision == "overlap":
                row, value, smap = records._host
                sigma = overlap_scatter(
                    value, smap, row, len(records.mapped_names), groups=groups
                )
                fixed = overlap_precision(sigma, records.support)
                self.overlap_sigma = sigma
                print("overlap scatter (scaled units): teff logg fe_h alpha")
                for name, sg in zip(records.mapped_names, sigma, strict=True):
                    print(f"  {name:30s} " + " ".join(f"{x:6.3f}" for x in sg))
            elif precision != "running":
                raise ValueError(f"unknown precision {precision!r}")
            self.survey_maps = ConditionalSurveyMaps(
                records.support,
                records.coverage,
                trust=trust_matrix(records.mapped_names, trust),
                groups=groups,
                basis=int(basis),
                hidden=int(hidden),
                hard_targets=[LABEL_INDEX[t] for t in hard_targets],
                precision=fixed,
            ).to(self.device)
        elif map_kind == "spline":
            self.survey_maps = build_survey_maps(
                records, min_records=int(consensus_min_records), groups=groups
            ).to(self.device)
        else:
            raise ValueError(f"unknown map_kind {map_kind!r}")
        self.gauge_weight = float(gauge_weight)
        self.monotone_weight = float(monotone_weight)
        self.consensus_weight = float(consensus_weight)
        self.map_reg_weight = float(map_reg_weight)
        self.anchor_weight = None if anchor_weight is None else float(anchor_weight)

    def configure_semisupervised(
        self, X, E, *, ratio=1.0, rec_weight=0.2, consistency_weight=0.1
    ):
        """Unlabelled stars (scaled like the training features) for each step.

        Every training step also draws ``ratio`` x batch unlabelled stars:
        masked reconstruction (``rec_weight``) keeps the encoder fitting the
        whole Gaia population, not only the labelled sample, and a
        consistency term (``consistency_weight``) asks two independent
        maskings of a star for the same common-scale Teff, log g, [Fe/H] and
        [alpha], which regularises the head where no label exists.
        """
        self.unl_X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.unl_E = torch.as_tensor(np.asarray(E), dtype=torch.float32)
        if self.unl_X.device.type == "cpu" and torch.cuda.is_available():
            self.unl_X, self.unl_E = self.unl_X.pin_memory(), self.unl_E.pin_memory()
        self.unl_ratio = float(ratio)
        self.unl_rec_weight = float(rec_weight)
        self.unl_cons_weight = float(consistency_weight)
        self._unl_steps = 0
        self._unl_sums = np.zeros(2)

    def _unlabelled_loss(self, n, ctx):
        """(reconstruction, consistency) on ``n`` random unlabelled stars."""
        idx = torch.randint(len(self.unl_X), (n,))
        xu = self.unl_X[idx].to(self.device, non_blocking=True)
        eu = self.unl_E[idx].to(self.device, non_blocking=True)
        xm1, mask1, nan1 = self._apply_batch_masking(xu, eu, ctx)
        xm2, _, _ = self._apply_batch_masking(xu, eu, ctx)
        if ctx.multitask:
            recon, enc1 = self.model(xm1)
            keep = mask1[:, : -self.diff] & nan1[:, : -self.diff]
            e = eu[:, : -self.diff]
            rec = self._rec_loss_fn(
                xu[:, : -self.diff], recon, keep, 1.0 / (e * e + 1e-8)
            )
        else:
            enc1 = self.model.encoder(xm1)
            rec = xu.new_zeros(())
        y1, y2 = self.ft(enc1), self.ft(self.model.encoder(xm2))
        mid = y1.shape[-1] // 2
        cons = (y1[:, :SURVEY_LABELS, mid] - y2[:, :SURVEY_LABELS, mid]).abs().mean()
        return rec, cons

    # The loss -------------------------------------------------------------
    def _survey_label_loss(self, y_head, y_rest, rows):
        """Pinball over survey records (labels 0-3) and own labels (4-5)."""
        q = self._quantiles_tensor.view(1, 1, -1)
        rec, pos = self.records.gather(rows)
        target = self.records.value[rec]  # (R, 4)
        pred = y_head[pos, :SURVEY_LABELS, :]  # (R, 4, 3) common scale
        smap = self.records.survey_map[rec]
        mapped = smap >= 0
        conditional = self.map_kind == "conditional"
        if mapped.any():
            if conditional:
                out = self.survey_maps(pred[mapped], smap[mapped])
            else:
                draws = torch.zeros(3, int(mapped.sum()), 6, device=pred.device)
                draws[:, :, :SURVEY_LABELS] = pred[mapped].permute(2, 0, 1)
                out = self.survey_maps(draws, smap[mapped])[:, :, :SURVEY_LABELS]
                out = out.permute(1, 2, 0)
            pred = pred.clone()
            pred[mapped] = out
        valid = torch.isfinite(target)
        if conditional and mapped.any() and torch.is_grad_enabled():
            mid = pred.shape[-1] // 2
            self.survey_maps.update_precision(
                target[mapped] - pred[mapped][:, :, mid].detach(),
                smap[mapped],
                valid[mapped],
            )
        if self.anchor_weight is None:
            # Each (star, label) shares weight 1 across all its records.
            shared = valid
        else:
            # Survey records share weight 1; each anchor record has its own.
            shared = valid & mapped[:, None]
        per_star = torch.zeros(len(rows), SURVEY_LABELS, device=pred.device)
        per_star.index_add_(0, pos, shared.to(pred.dtype))
        weight = shared.to(pred.dtype) / per_star[pos].clamp_min(1.0)
        n_anchor = pred.new_zeros(())
        if self.anchor_weight is not None:
            own = valid & ~mapped[:, None]
            weight = weight + self.anchor_weight * own.to(pred.dtype)
            n_anchor = self.anchor_weight * own.sum().to(pred.dtype)
        err = target.nan_to_num(0.0).unsqueeze(2) - pred
        pin = (err * q - err.clamp_max(0.0)).sum(2)  # (R, 4)
        num = (pin * weight).sum()
        den = (per_star > 0).sum().to(pred.dtype) + n_anchor
        # Age and parallax keep their own labels (parallax from the input).
        own = y_rest[:, SURVEY_LABELS:]
        own_valid = ~torch.isnan(own)
        err2 = own.nan_to_num(0.0).unsqueeze(2) - y_head[:, SURVEY_LABELS:, :]
        pin2 = (err2 * q - err2.clamp_max(0.0)).sum(2)
        num = num + pin2.masked_fill(~own_valid, 0.0).sum()
        den = den + own_valid.sum().to(pred.dtype)
        loss = num / (den.clamp_min(1.0) * q.shape[-1])
        maps = self.survey_maps
        if conditional:
            mid = y_head.shape[-1] // 2
            loss = loss + self.gauge_weight * maps.gauge_penalty(
                y_head[:, :SURVEY_LABELS, mid]
            )
            if mapped.any():
                loss = loss + self.monotone_weight * maps.monotone_penalty(
                    y_head[pos[mapped], :SURVEY_LABELS, mid], smap[mapped]
                )
        else:
            loss = loss + self.consensus_weight * soft_consensus(maps)
        return loss + self.map_reg_weight * maps.regularization()

    def _compute_finetune_batch_loss(self, batch, ctx):
        X_batch, eX_batch, y_ext, e_y_batch = batch
        rows = y_ext[:, -1].round().long()
        y_batch = y_ext[:, :-1]
        X_masked, mask, nanmask = self._apply_batch_masking(X_batch, eX_batch, ctx)
        if ctx.ftlf != "quantile" or ctx.linearprobe or ctx.rncloss:
            raise ValueError("Multi-survey fine-tune needs the quantile head")
        if ctx.multitask:
            X_reconstructed, encoded = self.model(X_masked)
        else:
            encoded = self.model.encoder(X_masked)
            X_reconstructed = None
        y_raw = self.ft(encoded)
        y_raw = self._apply_parallax_masked_forward(X_masked, y_batch, y_raw, ctx)
        loss = self._survey_label_loss(y_raw, y_batch, rows)
        if ctx.multitask:
            reconstruction_mask = mask[:, : -self.diff] & nanmask[:, : -self.diff]
            reconstruction_w = 1.0 / (
                (eX_batch[:, : -self.diff] * eX_batch[:, : -self.diff]) + 1e-8
            )
            rec = self._rec_loss_fn(
                X_batch[:, : -self.diff],
                X_reconstructed,
                reconstruction_mask,
                reconstruction_w,
            )
            loss = ctx.ft_lambda_pred * loss + ctx.ft_lambda_rec * rec
        if getattr(self, "unl_X", None) is not None and torch.is_grad_enabled():
            n = max(1, int(round(self.unl_ratio * len(X_batch))))
            rec_u, cons = self._unlabelled_loss(n, ctx)
            loss = loss + self.unl_rec_weight * rec_u + self.unl_cons_weight * cons
            self._unl_steps += 1
            self._unl_sums += [float(rec_u.detach()), float(cons.detach())]
            if self._unl_steps % 1000 == 0:
                import logging

                rec_m, cons_m = self._unl_sums / 1000
                logging.info(f"Unlabelled: recon {rec_m:.4f} consistency {cons_m:.4f}")
                self._unl_sums[:] = 0
        return loss

    # Maps join the optimizer, clipping and checkpoints ---------------------
    def _setup_finetune_optimizer(
        self, linearprobe, ftopt, ftlr, ftl2, enc_lr, head_lambda, encoder_lambda
    ):
        optimizer, scheduler = super()._setup_finetune_optimizer(
            linearprobe, ftopt, ftlr, ftl2, enc_lr, head_lambda, encoder_lambda
        )
        optimizer.add_param_group(
            {"params": self.survey_maps.parameters(), "lr": ftlr, "weight_decay": 0.0}
        )
        lambdas = list(scheduler.lr_lambdas) + [head_lambda]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        return optimizer, scheduler

    def _finetune_parameters(self, linearprobe: bool):
        return super()._finetune_parameters(linearprobe) + list(
            self.survey_maps.parameters()
        )

    def _set_finetune_train_mode(self, linearprobe: bool) -> None:
        super()._set_finetune_train_mode(linearprobe)
        self.survey_maps.train()

    def _save_finetune_checkpoint(self, linearprobe, epoch, optimizer, scheduler):
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
            "survey_maps_state_dict": self.survey_maps.state_dict(),
            "survey_names": self.records.mapped_names,
            "anchor_surveys": [
                n
                for n, a in zip(self.records.names, self.records.is_anchor, strict=True)
                if a
            ],
            "map_kind": self.map_kind,
            "gauge": (
                "local coverage x precision x trust weighted consensus + anchors"
                if self.map_kind == "conditional"
                else "consensus (Teff, [Fe/H], alpha) + seismic anchors (log g)"
            ),
        }
        _atomic_torch_save(payload, self.ft_save_str)
        if (
            self.checkpoint_interval is not None
            and (epoch + 1) % self.checkpoint_interval == 0
        ):
            base = self.ft_save_str.rsplit(".", 1)[0]
            _atomic_torch_save(payload, f"{base}_checkpoint_{epoch + 1}.pth")
