"""Pretrain, fine-tune, CQR, and the joint head. Losses come from torchregress."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.stats
import torch
from torchregress.losses.conformal import CQR
from torchregress.losses.gaussian import LowRankGaussianLoss, MultivariateGaussianLoss
from torchregress.losses.quantile import MultiQuantileLoss, QuantileCrossoverLoss
from torchregress.metrics.point import NormalizedMedianAbsoluteDeviation

from masked_stellar_autoencoder.pipeline.batch import StellarBatch
from masked_stellar_autoencoder.pipeline.model import StellarNet
from masked_stellar_autoencoder.pipeline.normalize import (
    OrderScaler,
    VectorScaler,
    clipped_snr,
    diagonal_covariance,
    transform_coefficients,
    transform_vector,
)

QUANTILES = (0.16, 0.5, 0.84)
# Nominal central interval of the 0.16/0.84 pair. CQR widens it on the calibrate split.
CQR_ALPHA = 0.32


class Prepared(NamedTuple):
    z_bp: np.ndarray
    cov_bp: np.ndarray | None
    missing_bp: np.ndarray
    z_rp: np.ndarray
    cov_rp: np.ndarray | None
    missing_rp: np.ndarray
    ancillary: np.ndarray
    missing_ancillary: np.ndarray
    labels: np.ndarray
    label_missing: np.ndarray


def _covariance(group) -> np.ndarray | None:
    if group.covariance is not None:
        return group.covariance
    if group.errors is None:
        return None
    return diagonal_covariance(group.errors)


def prepare(
    batch: StellarBatch,
    bp_scaler: OrderScaler,
    rp_scaler: OrderScaler,
    photo_scaler: VectorScaler,
    label_scaler: VectorScaler,
) -> Prepared:
    """Scaled inputs. Missing photometry stays NaN; astrometry is clipped S/N."""

    def side(name: str, scaler: OrderScaler, n_relevant: np.ndarray | None):
        group = batch.groups[name]
        missing = group.missing | structural_missing(
            group.values.shape[1], n_relevant, len(batch)
        )
        if group.errors is not None:
            errors = np.asarray(group.errors)
            missing = missing | ~np.isfinite(errors) | (errors <= 0)
        cov = _covariance(group)
        if cov is not None:
            variance = np.diagonal(cov, axis1=-2, axis2=-1)
            # A non-positive published variance is not a measurement.
            missing = missing | ~np.isfinite(variance) | (variance <= 0)
        if batch.flux_state == "raw":
            # No G means the Pogson factor is undefined. Those orders are not inputs.
            missing = missing | ~np.isfinite(batch.g_mag)[:, None]
        z, cov_z, _ = transform_coefficients(
            group.values,
            cov,
            batch.g_mag,
            scaler,
            flux_state=batch.flux_state,
        )
        missing = missing | ~np.isfinite(z)
        return z, cov_z, missing

    z_bp, cov_bp, miss_bp = side("xp_bp", bp_scaler, batch.n_relevant_bp)
    z_rp, cov_rp, miss_rp = side("xp_rp", rp_scaler, batch.n_relevant_rp)
    photo = batch.groups["photometry"]
    astro = batch.groups["astrometry"]
    if astro.errors is None:
        snr = np.full(astro.values.shape, np.nan, dtype=np.float64)
        astro_missing = np.ones(astro.missing.shape, dtype=bool)
    else:
        bad_sigma = ~np.isfinite(astro.errors) | (np.asarray(astro.errors) <= 0)
        astro_missing = astro.missing | bad_sigma | ~np.isfinite(astro.values)
        snr = np.where(astro_missing, np.nan, clipped_snr(astro.values, astro.errors))
    labels = batch.groups["labels"]
    labels_z = transform_vector(labels.values, labels.missing, label_scaler)
    photo_z = transform_vector(photo.values, photo.missing, photo_scaler)
    photo_missing = photo.missing | ~np.isfinite(photo_z)
    return Prepared(
        z_bp=z_bp,
        cov_bp=cov_bp,
        missing_bp=miss_bp,
        z_rp=z_rp,
        cov_rp=cov_rp,
        missing_rp=miss_rp,
        ancillary=np.concatenate([photo_z, snr], axis=1),
        missing_ancillary=np.concatenate([photo_missing, astro_missing], axis=1),
        labels=labels_z,
        label_missing=labels.missing | ~np.isfinite(labels_z),
    )


def structural_missing(
    n_order: int, n_relevant: np.ndarray | None, n_rows: int
) -> np.ndarray:
    """Coefficients past n_relevant_bases. The whole side is missing when n_relevant is 0."""
    if n_relevant is None:
        return np.zeros((n_rows, n_order), dtype=bool)
    relevant = np.asarray(n_relevant)
    if relevant.shape != (n_rows,):
        raise ValueError(f"n_relevant shape {relevant.shape} != ({n_rows},)")
    return np.arange(n_order)[None, :] >= relevant[:, None]


def artificial_xp_masks(
    observed_bp: np.ndarray,
    observed_rp: np.ndarray,
    rng: np.random.Generator,
    *,
    span: int,
    p_span: float,
    p_drop_bp: float,
    p_drop_rp: float,
    p_drop_both: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous spans, plus full BP, RP, or both drops.

    Returns hidden_bp, hidden_rp, loss_bp, loss_rp. Loss is only orders that
    exist and were hidden. Stars with no XP are left alone: their mask is structural.

    ponytail: one Python loop over rows. Vectorize if a profile says this is the step.
    """
    total = p_drop_both + p_drop_bp + p_drop_rp + p_span
    if min(p_drop_both, p_drop_bp, p_drop_rp, p_span) < 0 or total > 1.0:
        raise ValueError(
            f"mask probabilities must be non-negative and sum to <= 1, got {total}"
        )
    hidden_bp = np.zeros(observed_bp.shape, dtype=bool)
    hidden_rp = np.zeros(observed_rp.shape, dtype=bool)
    has = observed_bp.any(axis=1) | observed_rp.any(axis=1)
    for i in np.flatnonzero(has):
        draw = float(rng.random())
        if draw < p_drop_both:
            hidden_bp[i] = observed_bp[i]
            hidden_rp[i] = observed_rp[i]
        elif draw < p_drop_both + p_drop_bp:
            hidden_bp[i] = observed_bp[i]
        elif draw < p_drop_both + p_drop_bp + p_drop_rp:
            hidden_rp[i] = observed_rp[i]
        elif draw < total:
            _hide_span(hidden_bp, observed_bp, i, span, rng)
            _hide_span(hidden_rp, observed_rp, i, span, rng)
    return hidden_bp, hidden_rp, observed_bp & hidden_bp, observed_rp & hidden_rp


def _hide_span(
    hidden: np.ndarray,
    observed: np.ndarray,
    row: int,
    span: int,
    rng: np.random.Generator,
) -> None:
    idx = np.flatnonzero(observed[row])
    if idx.size == 0:
        return
    width = min(span, int(idx.size))
    start = int(rng.integers(0, idx.size - width + 1))
    hidden[row, idx[start : start + width]] = True


def covariance_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    cov: torch.Tensor,
    coeff_mask: torch.Tensor,
) -> torch.Tensor:
    """MultivariateGaussianLoss on the masked observed orders only.

    Rows that share a mask are one call. The submatrix is Sigma of those orders.

    ponytail: groups masks on CPU. Fine at D=55; a packed kernel replaces it if this step shows up.
    """
    if coeff_mask.dtype != torch.bool:
        coeff_mask = coeff_mask.to(torch.bool)
    loss_fn = MultivariateGaussianLoss()
    groups: dict[bytes, list[int]] = {}
    mask_cpu = coeff_mask.detach().cpu().numpy()
    for i, row in enumerate(mask_cpu):
        groups.setdefault(row.tobytes(), []).append(i)
    pieces: list[torch.Tensor] = []
    counts: list[int] = []
    for idxs in groups.values():
        take = coeff_mask[idxs[0]]
        if not bool(take.any()):
            continue
        cols = take.nonzero(as_tuple=False).flatten()
        index = torch.tensor(idxs, device=pred.device)
        nll = loss_fn(
            pred[index][:, cols],
            target[index][:, cols],
            cov[index][:, cols][:, :, cols],
        )
        pieces.append(nll)
        counts.append(len(idxs))
    if not pieces:
        return pred.sum() * 0.0
    weights = pred.new_tensor(counts, dtype=pred.dtype)
    return (torch.stack(pieces) * weights).sum() / weights.sum()


def diagonal_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    coeff_mask: torch.Tensor,
) -> torch.Tensor:
    """Fallback when the file has no correlation matrix. Weight is 1/sigma."""
    err = (pred - target).abs() / sigma.clamp_min(1e-6)
    err = torch.where(coeff_mask, err, torch.zeros_like(err))
    return err.sum() / coeff_mask.sum().clamp_min(1)


def pretrain_loss(
    model: StellarNet,
    *,
    z_bp: torch.Tensor,
    cov_bp: torch.Tensor | None,
    input_missing_bp: torch.Tensor,
    loss_bp: torch.Tensor,
    z_rp: torch.Tensor,
    cov_rp: torch.Tensor | None,
    input_missing_rp: torch.Tensor,
    loss_rp: torch.Tensor,
    ancillary: torch.Tensor,
    miss_anc: torch.Tensor,
    sigma_bp: torch.Tensor | None = None,
    sigma_rp: torch.Tensor | None = None,
) -> torch.Tensor:
    """Covariance NLL on masked orders. No second noise draw on top of Sigma."""
    seen_bp = z_bp.masked_fill(input_missing_bp, 0.0)
    seen_rp = z_rp.masked_fill(input_missing_rp, 0.0)
    out = model(
        seen_bp, input_missing_bp, seen_rp, input_missing_rp, ancillary, miss_anc
    )
    return _side_loss(out.recon_bp, z_bp, cov_bp, sigma_bp, loss_bp) + _side_loss(
        out.recon_rp, z_rp, cov_rp, sigma_rp, loss_rp
    )


def _side_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cov: torch.Tensor | None,
    sigma: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    if cov is not None:
        return covariance_nll(pred, target, cov, mask)
    if sigma is None:
        sigma = torch.ones_like(pred)
    return diagonal_weighted_l1(pred, target, sigma, mask)


def _tensor(values: np.ndarray, *, boolean: bool = False) -> torch.Tensor:
    array = np.asarray(values)
    if boolean:
        return torch.tensor(array, dtype=torch.bool)
    return torch.tensor(array, dtype=torch.float32)


def hide_observed(
    missing: np.ndarray, rng: np.random.Generator, p: float
) -> np.ndarray:
    """Hide present ancillary columns so the missing bit is learned on those stars."""
    if p < 0 or p > 1:
        raise ValueError(f"p_ancillary must be in [0, 1], got {p}")
    if p == 0:
        return missing
    drop = (rng.random(missing.shape) < p) & ~missing
    return missing | drop


def pretrain_epoch(
    model: StellarNet,
    reader,
    optimizer: torch.optim.Optimizer,
    bp_scaler: OrderScaler,
    rp_scaler: OrderScaler,
    photo_scaler: VectorScaler,
    label_scaler: VectorScaler,
    rng: np.random.Generator,
    *,
    span: int = 4,
    p_span: float = 0.8,
    p_drop_bp: float = 0.05,
    p_drop_rp: float = 0.05,
    p_drop_both: float = 0.05,
    p_ancillary: float = 0.1,
) -> float:
    """One pass over partitions. The logged mean is per scored star, not per row in the batch."""
    model.train()
    total = 0.0
    seen = 0
    for batch in reader:
        prepared = prepare(batch, bp_scaler, rp_scaler, photo_scaler, label_scaler)
        hidden_bp, hidden_rp, loss_bp, loss_rp = artificial_xp_masks(
            ~prepared.missing_bp,
            ~prepared.missing_rp,
            rng,
            span=span,
            p_span=p_span,
            p_drop_bp=p_drop_bp,
            p_drop_rp=p_drop_rp,
            p_drop_both=p_drop_both,
        )
        n_scored = int((loss_bp.any(axis=1) | loss_rp.any(axis=1)).sum())
        if n_scored == 0:
            continue
        cov_bp = None if prepared.cov_bp is None else _tensor(prepared.cov_bp)
        cov_rp = None if prepared.cov_rp is None else _tensor(prepared.cov_rp)
        loss = pretrain_loss(
            model,
            z_bp=_tensor(prepared.z_bp),
            cov_bp=cov_bp,
            input_missing_bp=_tensor(prepared.missing_bp | hidden_bp, boolean=True),
            loss_bp=_tensor(loss_bp, boolean=True),
            z_rp=_tensor(prepared.z_rp),
            cov_rp=cov_rp,
            input_missing_rp=_tensor(prepared.missing_rp | hidden_rp, boolean=True),
            loss_rp=_tensor(loss_rp, boolean=True),
            ancillary=_tensor(prepared.ancillary),
            miss_anc=_tensor(
                hide_observed(prepared.missing_ancillary, rng, p_ancillary),
                boolean=True,
            ),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # covariance_nll is already a mean over scored stars. Weight by that count.
        total += float(loss.detach()) * n_scored
        seen += n_scored
    if seen == 0:
        raise ValueError("pretrain epoch saw no scored stars")
    return total / seen


def _regime_nll(
    model: StellarNet,
    prepared: Prepared,
    rng: np.random.Generator,
    **mask_kw: float,
) -> float:
    hidden_bp, hidden_rp, loss_bp, loss_rp = artificial_xp_masks(
        ~prepared.missing_bp,
        ~prepared.missing_rp,
        rng,
        **mask_kw,
    )
    if not (loss_bp.any() or loss_rp.any()):
        return float("nan")
    cov_bp = None if prepared.cov_bp is None else _tensor(prepared.cov_bp)
    cov_rp = None if prepared.cov_rp is None else _tensor(prepared.cov_rp)
    with torch.no_grad():
        loss = pretrain_loss(
            model,
            z_bp=_tensor(prepared.z_bp),
            cov_bp=cov_bp,
            input_missing_bp=_tensor(prepared.missing_bp | hidden_bp, boolean=True),
            loss_bp=_tensor(loss_bp, boolean=True),
            z_rp=_tensor(prepared.z_rp),
            cov_rp=cov_rp,
            input_missing_rp=_tensor(prepared.missing_rp | hidden_rp, boolean=True),
            loss_rp=_tensor(loss_rp, boolean=True),
            ancillary=_tensor(prepared.ancillary),
            miss_anc=_tensor(prepared.missing_ancillary, boolean=True),
        )
    return float(loss)


def regime_nll(
    model: StellarNet,
    prepared: Prepared,
    rng: np.random.Generator,
    *,
    span: int = 4,
) -> tuple[float, float]:
    """Holdout NLL for a span mask and for a full XP drop. Eval mode, no optimizer step."""
    training = model.training
    model.eval()
    try:
        span_nll = _regime_nll(
            model,
            prepared,
            rng,
            span=span,
            p_span=1.0,
            p_drop_bp=0.0,
            p_drop_rp=0.0,
            p_drop_both=0.0,
        )
        drop_nll = _regime_nll(
            model,
            prepared,
            rng,
            span=span,
            p_span=0.0,
            p_drop_bp=0.0,
            p_drop_rp=0.0,
            p_drop_both=1.0,
        )
    finally:
        model.train(training)
    return span_nll, drop_nll


def eval_forward(model: StellarNet, prepared: Prepared, *, drop_xp: bool = False):
    """Eval forward. Structural missing only. drop_xp is the XP-off pass, not a training mask."""
    training = model.training
    model.eval()
    miss_bp = np.ones_like(prepared.missing_bp) if drop_xp else prepared.missing_bp
    miss_rp = np.ones_like(prepared.missing_rp) if drop_xp else prepared.missing_rp
    try:
        with torch.no_grad():
            return model(
                _tensor(np.where(miss_bp, 0.0, prepared.z_bp)),
                _tensor(miss_bp, boolean=True),
                _tensor(np.where(miss_rp, 0.0, prepared.z_rp)),
                _tensor(miss_rp, boolean=True),
                _tensor(np.where(prepared.missing_ancillary, 0.0, prepared.ancillary)),
                _tensor(prepared.missing_ancillary, boolean=True),
            )
    finally:
        model.train(training)


def rare_label_weights(
    labels: torch.Tensor, column: int, n_bins: int = 5
) -> torch.Tensor:
    """Inverse histogram frequency on one label, after scaling. Other columns stay at 1."""
    weights = torch.ones_like(labels)
    column_values = labels[:, column]
    finite = torch.isfinite(column_values)
    if int(finite.sum()) < 2:
        return weights
    values = column_values[finite]
    lo = values.min()
    hi = values.max()
    if hi == lo:
        return weights
    edges = torch.linspace(lo, hi, n_bins + 1, device=labels.device, dtype=labels.dtype)
    bins = torch.bucketize(values, edges[1:-1])
    counts = torch.bincount(bins, minlength=n_bins).to(labels.dtype).clamp_min(1)
    inverse = counts.sum() / counts
    column_w = torch.ones_like(column_values)
    column_w[finite] = inverse[bins]
    column_w = column_w / column_w[finite].mean()
    weights[:, column] = column_w
    return weights


class Finetune(NamedTuple):
    total: torch.Tensor
    recon: torch.Tensor
    joint: torch.Tensor


def finetune_loss(
    model: StellarNet,
    *,
    z_bp: torch.Tensor,
    cov_bp: torch.Tensor | None,
    input_missing_bp: torch.Tensor,
    loss_bp: torch.Tensor,
    z_rp: torch.Tensor,
    cov_rp: torch.Tensor | None,
    input_missing_rp: torch.Tensor,
    loss_rp: torch.Tensor,
    ancillary: torch.Tensor,
    miss_anc: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    label_weights: torch.Tensor,
    recon_weight: float,
) -> Finetune:
    """Quantile loss, a small reconstruction term, and the joint head.

    The checkpoint score is the quantile NMAD from eval_forward, not this total.
    """
    seen_bp = z_bp.masked_fill(input_missing_bp, 0.0)
    seen_rp = z_rp.masked_fill(input_missing_rp, 0.0)
    out = model(
        seen_bp, input_missing_bp, seen_rp, input_missing_rp, ancillary, miss_anc
    )
    # A label that became NaN in the scaler (Teff <= 0) is not a target.
    usable = label_mask & torch.isfinite(labels)
    pinball = MultiQuantileLoss(list(QUANTILES))
    crossover = QuantileCrossoverLoss(
        list(QUANTILES), base_loss=0.0, crossover_penalty=1.0
    )
    supervised = pinball(out.quantiles, labels, mask=usable, weights=label_weights)
    supervised = supervised + crossover(
        out.quantiles, labels, mask=usable, weights=label_weights
    )
    recon = _side_loss(out.recon_bp, z_bp, cov_bp, None, loss_bp)
    recon = recon + _side_loss(out.recon_rp, z_rp, cov_rp, None, loss_rp)
    # log_prob rejects NaN. The zero is not a target: a row with any missing
    # label is dropped, because the joint density needs every label.
    observed = torch.where(usable, labels, torch.zeros_like(labels))
    joint = joint_nll(
        out.joint_mean, out.joint_factor, out.joint_diag, observed, mask=usable
    )
    if model.flow is not None:
        joint = joint + model.flow(out.latent, observed, mask=usable)
    total = supervised + recon_weight * recon + joint
    return Finetune(total, recon.detach(), joint.detach())


def _nmad(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("inf")
    metric = NormalizedMedianAbsoluteDeviation()
    metric.update(pred.detach().view(-1), target.detach().view(-1))
    return float(metric.compute())


def checkpoint_score(
    pred_metal_poor: torch.Tensor,
    true_metal_poor: torch.Tensor,
    pred_xp_off: torch.Tensor,
    true_xp_off: torch.Tensor,
) -> float:
    """NMAD of [Fe/H] from eval_forward, metal-poor bin plus the XP-off pass."""
    if pred_metal_poor.numel() == 0 or pred_xp_off.numel() == 0:
        return float("inf")
    return _nmad(pred_metal_poor, true_metal_poor) + _nmad(pred_xp_off, true_xp_off)


def selection_score(
    model: StellarNet,
    prepared: Prepared,
    feh_index: int,
    feh_physical: np.ndarray,
) -> float:
    """NMAD of [Fe/H] from the two eval forwards. Artificial masks stay off."""
    feh = np.asarray(feh_physical, dtype=np.float64).reshape(-1)
    if feh.shape[0] != prepared.labels.shape[0]:
        raise ValueError(
            f"feh_physical length {feh.shape[0]} != {prepared.labels.shape[0]} stars"
        )
    on = eval_forward(model, prepared, drop_xp=False)
    off = eval_forward(model, prepared, drop_xp=True)
    true = prepared.labels[:, feh_index]
    finite = np.isfinite(true)
    poor = metal_poor_mask(feh) & finite
    pred_on = on.quantiles[:, 1, feh_index]
    pred_off = off.quantiles[:, 1, feh_index]
    return checkpoint_score(
        pred_on[torch.as_tensor(poor)],
        torch.as_tensor(true[poor], dtype=pred_on.dtype),
        pred_off[torch.as_tensor(finite)],
        torch.as_tensor(true[finite], dtype=pred_off.dtype),
    )


def metal_poor_mask(fe_h: np.ndarray, limit: float = -1.0) -> np.ndarray:
    values = np.asarray(fe_h, dtype=np.float64)
    return np.isfinite(values) & (values < limit)


def fit_cqr(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = CQR_ALPHA,
) -> list[CQR]:
    """One CQR per label, in scaled space. The score is max(q_lo - y, y - q_hi)."""
    if lower.shape != upper.shape or lower.shape != target.shape:
        raise ValueError("lower, upper, and target must share a shape")
    predictors: list[CQR] = []
    for j in range(target.shape[-1]):
        finite = (
            torch.isfinite(target[:, j])
            & torch.isfinite(lower[:, j])
            & torch.isfinite(upper[:, j])
        )
        if not bool(finite.any()):
            raise ValueError(f"label {j} has no finite calibrate rows")
        pred = torch.cat([lower[:, j : j + 1], upper[:, j : j + 1]], dim=-1)
        cqr = CQR(alpha=alpha)
        cqr.calibrate(pred, target[:, j : j + 1], mask=finite)
        predictors.append(cqr)
    return predictors


def cqr_intervals(
    predictors: list[CQR], lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lows = []
    highs = []
    for j, cqr in enumerate(predictors):
        pred = torch.cat([lower[:, j : j + 1], upper[:, j : j + 1]], dim=-1)
        lo, hi = cqr.predict_interval(pred)
        lows.append(lo)
        highs.append(hi)
    return torch.cat(lows, dim=-1), torch.cat(highs, dim=-1)


def coverage_after_inverse(
    lower: torch.Tensor,
    upper: torch.Tensor,
    y_physical: torch.Tensor,
    inverse,
) -> float:
    """Quote coverage in physical units. The conformal guarantee stays in scaled space."""
    lo = inverse(lower)
    hi = inverse(upper)
    if not torch.is_tensor(lo):
        lo = torch.as_tensor(lo, dtype=y_physical.dtype)
        hi = torch.as_tensor(hi, dtype=y_physical.dtype)
    finite = torch.isfinite(y_physical) & torch.isfinite(lo) & torch.isfinite(hi)
    if not bool(finite.any()):
        return float("nan")
    inside = (y_physical >= lo) & (y_physical <= hi) & finite
    return float(inside.sum() / finite.sum())


def set_encoder_lr(
    optimizer: torch.optim.Optimizer, epoch: int, freeze_epochs: int, encoder_lr: float
) -> float:
    """Epoch 0..freeze_epochs-1 keeps the encoder at lr 0. Group 0 is the encoder."""
    lr = 0.0 if epoch < freeze_epochs else encoder_lr
    optimizer.param_groups[0]["lr"] = lr
    return lr


def build_optimizer(
    model: StellarNet,
    *,
    encoder_lr: float,
    head_lr: float,
    encoder_weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    """Encoder weight decay is explicit. AdamW's default 0.01 is not used."""
    encoder = []
    head = []
    for name, param in model.named_parameters():
        if name.startswith(("quantile", "joint_", "flow")):
            head.append(param)
        else:
            encoder.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder, "lr": encoder_lr, "weight_decay": encoder_weight_decay},
            {"params": head, "lr": head_lr, "weight_decay": 0.0},
        ]
    )


def joint_nll(
    mean: torch.Tensor,
    factor: torch.Tensor,
    diag: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return LowRankGaussianLoss()(mean, target, factor, diag, mask=mask)


def gaussian_joint_coverage(
    mean: torch.Tensor,
    factor: torch.Tensor,
    diag: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.1,
) -> float:
    """Fraction of stars inside the chi-square ball of cov = factor factor^T + diag."""
    cov = factor @ factor.transpose(-1, -2)
    cov = cov + torch.diag_embed(diag.clamp_min(1e-8))
    diff = (target - mean).unsqueeze(-1)
    solved = torch.linalg.solve(cov, diff)
    distance = (diff * solved).sum(dim=(-2, -1))
    row = torch.isfinite(target).all(dim=-1) & torch.isfinite(distance)
    if not bool(row.any()):
        return float("nan")
    threshold = float(scipy.stats.chi2.ppf(1.0 - alpha, int(mean.shape[-1])))
    return float((distance.detach()[row] <= threshold).to(torch.float32).mean())


def choose_joint(
    flow_coverage: float | None, low_rank_coverage: float, minimum: float = 0.9
) -> str:
    """The flow ships only when its joint coverage holds. Otherwise the low-rank head does."""
    if flow_coverage is not None and flow_coverage >= minimum:
        return "flow"
    return "low_rank"
