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

from .six_survey_map import SixSurveyMap

SURVEY_LABELS = 4  # Teff, log g, [Fe/H], [alpha]; age and parallax are not mapped
LOGG = 1


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
        counts = np.bincount(row, minlength=n_rows)
        ptr = np.concatenate([[0], np.cumsum(counts)])
        t = lambda a, dtype: torch.as_tensor(a, device=device, dtype=dtype)  # noqa: E731
        self.ptr = t(ptr, torch.long)
        self.value = t(value, torch.float32)
        self.survey_map = t(map_index[survey], torch.long)
        self.n = len(row)

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


def build_survey_maps(records: SurveyRecords, bins: int = 8) -> SixSurveyMap:
    reference = torch.full((6,), -1, dtype=torch.long)  # no reference survey
    zero_sum = [j for j in range(SURVEY_LABELS) if j != LOGG]
    return SixSurveyMap(
        records.support, reference, zero_sum=zero_sum, kind="spline", bins=bins
    )


def soft_consensus(maps: SixSurveyMap, target: int = LOGG) -> torch.Tensor:
    """Squared mean (over surveys) offset and log bin slope of one target."""
    bias, slopes = maps.coefficients()
    active = maps.support[:, target]
    if not active.any():
        return bias.sum() * 0
    mean_bias = bias[active, target].mean()
    mean_log_slope = torch.log(slopes[active, target]).mean(0)
    return mean_bias.square() + mean_log_slope.square().mean()


class MultiSurveyWrapper(TabResnetWrapper):
    """TabResnetWrapper whose label term scores every survey record.

    ``y`` arrays carry one extra last column: the star's table row. Both the
    training loader and ``validate_fit`` pass it through to this loss.
    """

    def configure_multisurvey(
        self, records: SurveyRecords, *, consensus_weight=1.0, map_reg_weight=1e-3
    ):
        self.records = records
        self.survey_maps = build_survey_maps(records).to(self.device)
        self.consensus_weight = float(consensus_weight)
        self.map_reg_weight = float(map_reg_weight)

    # The loss -------------------------------------------------------------
    def _survey_label_loss(self, y_head, y_rest, rows):
        """Pinball over survey records (labels 0-3) and own labels (4-5)."""
        q = self._quantiles_tensor.view(1, 1, -1)
        rec, pos = self.records.gather(rows)
        target = self.records.value[rec]  # (R, 4)
        pred = y_head[pos, :SURVEY_LABELS, :]  # (R, 4, 3) common scale
        smap = self.records.survey_map[rec]
        mapped = smap >= 0
        if mapped.any():
            draws = torch.zeros(3, int(mapped.sum()), 6, device=pred.device)
            draws[:, :, :SURVEY_LABELS] = pred[mapped].permute(2, 0, 1)
            out = self.survey_maps(draws, smap[mapped])[:, :, :SURVEY_LABELS]
            pred = pred.clone()
            pred[mapped] = out.permute(1, 2, 0)
        valid = torch.isfinite(target)
        # Each (star, label) shares weight 1 across its records.
        per_star = torch.zeros(len(rows), SURVEY_LABELS, device=pred.device)
        per_star.index_add_(0, pos, valid.to(pred.dtype))
        weight = valid.to(pred.dtype) / per_star[pos].clamp_min(1.0)
        err = target.nan_to_num(0.0).unsqueeze(2) - pred
        pin = (err * q - err.clamp_max(0.0)).sum(2)  # (R, 4)
        num = (pin * weight).sum()
        den = (per_star > 0).sum().to(pred.dtype)
        # Age and parallax keep their own labels (parallax from the input).
        own = y_rest[:, SURVEY_LABELS:]
        own_valid = ~torch.isnan(own)
        err2 = own.nan_to_num(0.0).unsqueeze(2) - y_head[:, SURVEY_LABELS:, :]
        pin2 = (err2 * q - err2.clamp_max(0.0)).sum(2)
        num = num + pin2.masked_fill(~own_valid, 0.0).sum()
        den = den + own_valid.sum().to(pred.dtype)
        loss = num / (den.clamp_min(1.0) * q.shape[-1])
        loss = loss + self.consensus_weight * soft_consensus(self.survey_maps)
        return loss + self.map_reg_weight * self.survey_maps.regularization()

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
            "gauge": "consensus (Teff, [Fe/H], alpha) + seismic anchors (log g)",
        }
        _atomic_torch_save(payload, self.ft_save_str)
        if (
            self.checkpoint_interval is not None
            and (epoch + 1) % self.checkpoint_interval == 0
        ):
            base = self.ft_save_str.rsplit(".", 1)[0]
            _atomic_torch_save(payload, f"{base}_checkpoint_{epoch + 1}.pth")
