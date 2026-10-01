# loading the packages
import logging
import math
import os
import random
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.base import BaseEstimator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..training.hdf5_io import prefetch_batches
from .blocks import TabDenseNet, TabResnet
from .checkpoint_load import torch_load_finetune_checkpoint, torch_load_trusted


def _atomic_torch_save(payload, path):
    temporary_path = f"{path}.tmp"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["torch_cuda"]])


def _summarize_feature_residuals(errors: np.ndarray, feature_names: list[str]) -> dict:
    """Summarize sampled masked residuals by output feature for QA plots."""
    result = {
        "feature_names": list(feature_names),
        "feature_valid_count": [],
        "feature_valid_fraction": [],
        "feature_mae": [],
        "feature_p50": [],
        "feature_p84": [],
        "feature_p95": [],
    }
    sample_count = errors.shape[0]
    for index in range(len(feature_names)):
        values = errors[:, index]
        values = values[np.isfinite(values)]
        values = values.astype(np.float64, copy=False)
        count = int(values.size)
        result["feature_valid_count"].append(count)
        result["feature_valid_fraction"].append(
            round(count / sample_count, 8) if sample_count else 0.0
        )
        if count:
            summaries = {
                "feature_mae": float(values.mean()),
                "feature_p50": float(np.quantile(values, 0.50)),
                "feature_p84": float(np.quantile(values, 0.84)),
                "feature_p95": float(np.quantile(values, 0.95)),
            }
        else:
            summaries = {
                "feature_mae": None,
                "feature_p50": None,
                "feature_p84": None,
                "feature_p95": None,
            }
        for name, value in summaries.items():
            result[name].append(round(value, 8) if value is not None else None)
    return result


class MaskedGaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_mean, target, pred_var, target_var):
        # Entries of var must be non-negative
        if isinstance(target_var, float):
            if target_var < 0:
                raise ValueError("var has negative entry/entries")
        elif torch.any(target_var < 0):
            raise ValueError("var has negative entry/entries")

        mask = (~torch.isnan(target)) & (~torch.isnan(target_var))

        if self.reduction == "none":
            pred_mean = pred_mean[mask]
            pred_var = pred_var[mask]
            target = target[mask]
            target_var = target_var[mask]

            var = pred_var.clamp(min=self.eps)
            obs_var = target_var.clamp(min=self.eps)
            err = var + obs_var
            diff = pred_mean - target
            diff_squared = diff * diff
            return 0.5 * (torch.log(err) + (diff_squared / err)) + 0.5 * math.log(
                2 * math.pi
            )

        inv_mask = ~mask
        # ⚡ Bolt: Use .masked_fill instead of boolean indexing for performance
        safe_target = target.masked_fill(inv_mask, 0.0)

        var = pred_var.clamp(min=self.eps).masked_fill_(inv_mask, 1.0)
        obs_var = target_var.clamp(min=self.eps).masked_fill_(inv_mask, 1.0)

        err = var + obs_var
        diff = pred_mean - safe_target
        diff.masked_fill_(inv_mask, 0.0)
        # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
        diff_squared = diff * diff

        # Compute Gaussian NLL
        nll = 0.5 * (torch.log(err) + (diff_squared / err)) + 0.5 * math.log(
            2 * math.pi
        )
        nll = nll.masked_fill_(inv_mask, 0.0)

        if self.reduction == "mean":
            return nll.sum() / mask.sum().to(dtype=nll.dtype).clamp_min(1.0)
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll


class WeightedMaskedMSELoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps  # To avoid divide-by-zero if all values are masked

    def forward(self, target, input, weight):
        # Create mask for non-NaN targets
        mask = (~torch.isnan(target)) & (~torch.isnan(weight))

        if self.reduction == "none":
            masked_input = input[mask]
            masked_target = target[mask]
            masked_weights = weight[mask]
            # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
            diff_w = masked_input - masked_target
            return (diff_w * diff_w) * masked_weights

        inv_mask = ~mask
        # ⚡ Bolt: Use .masked_fill instead of boolean indexing for performance
        safe_target = target.masked_fill(inv_mask, 0.0)
        safe_weights = weight.masked_fill(inv_mask, 0.0)

        diff = input - safe_target
        diff.masked_fill_(inv_mask, 0.0)

        # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
        masked_error = (diff * diff) * safe_weights

        if self.reduction == "mean":
            return masked_error.sum() / (safe_weights.sum() + self.eps)
        elif self.reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, target, input):
        # Create a mask for non-NaN targets
        mask = ~torch.isnan(target)

        if self.reduction == "none":
            masked_input = input[mask]
            masked_target = target[mask]
            diff = masked_input - masked_target
            return diff * diff

        inv_mask = ~mask
        # ⚡ Bolt: Use .masked_fill instead of boolean indexing for performance
        safe_target = target.masked_fill(inv_mask, 0.0)

        # Compute squared error only where target is not NaN
        diff = input - safe_target
        diff.masked_fill_(inv_mask, 0.0)
        # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
        masked_error = diff * diff

        if self.reduction == "mean":
            return masked_error.sum() / mask.sum().to(
                dtype=masked_error.dtype
            ).clamp_min(1.0)
        elif self.reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error


class MaskedMAELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, target, input):
        # Create a mask for non-NaN targets
        mask = ~torch.isnan(target)

        if self.reduction == "none":
            masked_input = input[mask]
            masked_target = target[mask]
            return torch.abs(masked_input - masked_target)

        inv_mask = ~mask
        # ⚡ Bolt: Use .masked_fill instead of boolean indexing for performance
        safe_target = target.masked_fill(inv_mask, 0.0)

        # Compute absolute error only where target is not NaN
        diff = input - safe_target
        masked_error = torch.abs(diff.masked_fill_(inv_mask, 0.0))

        if self.reduction == "mean":
            return masked_error.sum() / mask.sum().to(
                dtype=masked_error.dtype
            ).clamp_min(1.0)
        elif self.reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error


class LabelDifference(nn.Module):
    """
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    """

    def __init__(self, distance_type="l1"):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == "l1":
            return torch.cdist(labels, labels, p=1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    """
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    """

    def __init__(self, similarity_type="l2"):
        super().__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == "l2":
            return -torch.cdist(features, features, p=2)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    """
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    """

    def __init__(self, temperature=2, label_diff="l1", feature_sim="l2"):
        super().__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # ⚡ Bolt: Compute off-diagonal mask once and use boolean indexing to reduce memory allocations and speed up by ~2x
        mask = ~torch.eye(n, dtype=torch.bool, device=logits.device)

        # remove diagonal
        logits = logits[mask].view(n, n - 1)
        exp_logits = exp_logits[mask].view(n, n - 1)
        label_diffs = label_diffs[mask].view(n, n - 1)

        loss = 0.0
        for i in range(n):
            row_label_diffs = label_diffs[i]
            # ⚡ Bolt: Use torch.mv with boolean mask matrix instead of masked_fill with expand_as for ~2x faster execution
            row_neg_mask = (
                row_label_diffs.unsqueeze(0) >= row_label_diffs.unsqueeze(1)
            ).to(exp_logits.dtype)
            row_log_sum_exp = torch.log(torch.mv(row_neg_mask, exp_logits[i]))
            # ⚡ Bolt: Compute sum of difference as difference of sums to avoid intermediate tensor allocation
            loss = loss - (logits[i].sum() - row_log_sum_exp.sum()) / (n * (n - 1))

        return loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path  # Filepath to save the model
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss, model):
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.save_checkpoint(
                model
            )  # Save the model when the best validation loss is found
        elif validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(
                    f"Validation loss improved to {self.best_loss:.6f}, saving model."
                )
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class EncoderDecoderLoss(nn.Module):
    r"""
    From pytorch-widedeep with some of my own modifications:
    '_Standard_' Encoder Decoder Loss. Loss applied during the Endoder-Decoder
     Self-Supervised Pre-Training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    The implementation of this lost is based on that at the
    [tabnet repo](https://github.com/dreamquark-ai/tabnet), which is in itself an
    adaptation of that in the original paper [TabNet: Attentive
    Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442).

    Parameters
    ----------
    eps: float
        Simply a small number to avoid dividing by zero
    """

    def __init__(self, eps: float = 1e-9, lf="mse"):
        super().__init__()
        self.eps = eps
        self.cost = lf

    def forward(
        self,
        x_true: Tensor,
        x_pred: Tensor,
        mask: Tensor,
        w: Tensor,
        logvar: Tensor | None = None,
    ) -> Tensor:
        r"""
        Parameters
        ----------
        x_true: Tensor
            Embeddings of the input data
        x_pred: Tensor
            Reconstructed embeddings
        mask: Tensor
            Mask with 1s indicated that the reconstruction, and therefore the
            loss, is based on those features.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import EncoderDecoderLoss
        >>> x_true = torch.rand(3, 3)
        >>> x_pred = torch.rand(3, 3)
        >>> mask = torch.empty(3, 3).random_(2)
        >>> loss = EncoderDecoderLoss()
        >>> res = loss(x_true, x_pred, mask)
        """

        inv_mask = ~mask.bool()
        # Correctly apply mask to errors before squaring
        # ⚡ Bolt: Replaced torch.where with .masked_fill_ for ~50% faster in-place execution and lower memory usage
        errors = (x_pred - x_true).masked_fill_(inv_mask, 0.0)
        if self.cost == "mse":
            # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
            reconstruction_errors = errors * errors
        elif self.cost == "mae":
            reconstruction_errors = abs(errors)
        elif self.cost == "wmse":
            if w is None:
                raise ValueError(
                    "Weight tensor w is required for wmse loss but got None"
                )
            # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
            reconstruction_errors = w * (errors * errors)
        elif self.cost == "wmae":
            if w is None:
                raise ValueError(
                    "Weight tensor w is required for wmae loss but got None"
                )
            reconstruction_errors = w * abs(errors)
        elif self.cost == "gnll":
            if logvar is None:
                raise ValueError("logvar required for gnll loss")
            lv = logvar.masked_fill_(inv_mask, 0.0)
            # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
            reconstruction_errors = 0.5 * (lv + errors * errors / (lv.exp() + self.eps))

        # Mean squared (or absolute) error over masked elements only — avoids
        # per-column divisors that up-weight rarely masked features in a batch.
        # ⚡ Bolt: Avoid allocating a new float tensor for the mask before summing, but cast to float before clamp_min
        denom = mask.sum().to(dtype=reconstruction_errors.dtype).clamp_min(self.eps)
        loss = reconstruction_errors.sum() / denom

        return loss


class PredictionHead(nn.Module):
    def __init__(self, latent_size, ft_label_dim, ft_activ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(latent_size, 2048),
            ft_activ,
            nn.Linear(2048, 2048),
            ft_activ,
            nn.Linear(2048, 1024),
            ft_activ,
            nn.Linear(1024, 512),
            ft_activ,
            nn.Linear(512, 256),
            ft_activ,
        )
        self.output_y = nn.Linear(256, ft_label_dim)
        self.output_upper = nn.Linear(256, ft_label_dim)
        self.output_lower = nn.Linear(256, ft_label_dim)

    def forward(self, x):
        h = self.shared(x)
        y_median = self.output_y(h)

        # Predict offsets from median to ensure monotonicity: lower ≤ median ≤ upper
        # Use softplus to ensure positive offsets
        lower_offset = torch.nn.functional.softplus(self.output_lower(h))
        upper_offset = torch.nn.functional.softplus(self.output_upper(h))

        y_lower = y_median - lower_offset
        y_upper = y_median + upper_offset

        return torch.stack([y_lower, y_median, y_upper], dim=2)


def quantile_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    label_weights: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> torch.Tensor:
    """
    Pinball / quantile loss. Optionally up-weight rare labels (e.g. [Fe/H]) so
    solar-metallicity stars do not dominate the gradient.

    ``sample_weight`` (B, L) scales each label per example, e.g. inverse variance
    from scaled label uncertainties; combined multiplicatively with ``label_weights``.
    """
    mask = ~torch.isnan(target)
    quantiles = quantiles.view(1, 1, -1)

    # ⚡ Bolt: Exploit automatic broadcasting to prevent allocating full-shape intermediate tensors for target and mask
    # ⚡ Bolt: Sanitize NaN targets to 0.0 out-of-place to prevent NaN propagation to gradients
    inv_mask = ~mask
    safe_target = target.masked_fill(inv_mask, 0.0).unsqueeze(2)
    inv_mask_unsq = inv_mask.unsqueeze(2)

    error = safe_target - preds
    # ⚡ Bolt: pinball as err*q - err.clamp_max(0) — fewer intermediates than torch.max
    loss = error * quantiles - error.clamp_max(0.0)

    if label_weights is None and sample_weight is None:
        # ⚡ Bolt: Replace dynamic boolean indexing with out-of-place masked_fill for ~2x faster execution and lower memory usage
        return loss.masked_fill(inv_mask_unsq, 0.0).sum() / (
            mask.sum().to(dtype=loss.dtype).clamp_min(1.0) * preds.shape[2]
        )

    # ⚡ Bolt: Delay allocation of float mask tensor until after unweighted fast-path
    # ⚡ Bolt: Avoid allocating full-shape w_eff if possible and apply masking directly
    w_eff = None
    if label_weights is not None:
        w_lab = label_weights.to(device=loss.device, dtype=loss.dtype).view(1, -1, 1)
        w_eff = w_lab
    if sample_weight is not None:
        w_s = sample_weight.to(device=loss.device, dtype=loss.dtype).unsqueeze(2)
        w_eff = w_s if w_eff is None else w_eff * w_s

    if w_eff is None:
        # Fallback if unweighted path is somehow skipped
        w_eff_sum = mask.sum().to(dtype=loss.dtype) * preds.shape[2]
        return loss.masked_fill(inv_mask_unsq, 0.0).sum() / w_eff_sum.clamp_min(1e-8)
    else:
        # Mask the weights so invalid entries don't contribute to the denominator
        # ⚡ Bolt: Exploit automatic broadcasting and multiply the denominator by the broadcast dimension (preds.shape[2]) to avoid allocating a full-shape intermediate tensor for w_eff
        w_eff = w_eff.masked_fill(inv_mask_unsq, 0.0)

    return (loss.masked_fill(inv_mask_unsq, 0.0) * w_eff).sum() / (
        w_eff.sum() * preds.shape[2]
    ).clamp_min(1e-8)


def _sigma_pinball_weights(
    sigma_scaled: Tensor,
    y: Tensor,
    floor: float,
    max_w: float,
    normalize_batch: bool,
) -> Tensor:
    """Inverse-variance style weights (B, L) in scaled label-error space."""
    sig = torch.nan_to_num(sigma_scaled, nan=1.0, posinf=1.0, neginf=1.0)
    # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
    w = 1.0 / (sig * sig + float(floor) * float(floor))
    w = w.clamp(max=float(max_w))
    if normalize_batch:
        w = w / (w.mean(dim=0, keepdim=True).clamp_min(1e-8))
    # ⚡ Bolt: Replaced torch.where with .masked_fill_ to reduce memory allocation overhead
    w = w.masked_fill_(torch.isnan(y), 0.0)
    return w


def _reduce_finetune_prediction(y_raw: Tensor, ftlf: str, linearprobe: bool):
    """
    Non-quantile losses need a single (B, L) prediction. Quantile head returns (B, L, 3).
    Legacy code paths may return a (mean, err) tuple for Gaussian NLL.
    """
    if ftlf == "quantile":
        return y_raw, None
    if linearprobe:
        return y_raw, None
    if isinstance(y_raw, Tensor) and y_raw.dim() == 3:
        return y_raw[..., 1], None
    if isinstance(y_raw, tuple | list) and len(y_raw) >= 2:
        return y_raw[0], y_raw[1]
    return y_raw, None


# creating a training wrapper for the algorithm
@dataclass
class FinetuneContext:
    linearprobe: bool
    maskft: bool
    multitask: bool
    ftlf: str
    rncloss: bool
    pert_features: bool
    pert_labels: bool
    parallax_use_masked_pred: bool
    parallax_label_idx: int | None
    ft_use_sigma_quantile_weights: bool
    ft_sigma_weight_floor: float
    ft_sigma_weight_max: float
    ft_sigma_weight_normalize_batch: bool
    q_weight_t: torch.Tensor | None
    criterion: torch.nn.Module | None
    criterion2: torch.nn.Module | None
    rnc: torch.nn.Module | None
    parallax_mle_weight: float
    m_consistency: torch.Tensor | None
    c_consistency: torch.Tensor | None
    parallax_sigma_scale: float
    parallax_sigma_floor: float
    ft_lambda_pred: float
    ft_lambda_rec: float


class TabResnetWrapper(BaseEstimator):
    @staticmethod
    def _open_datafile(datafile):
        if hasattr(datafile, "keys"):
            return datafile
        if isinstance(datafile, str):
            import h5py

            try:
                return h5py.File(datafile, "r")
            except Exception as e:
                raise ValueError(f"Could not open datafile '{datafile}': {e}") from e
        raise ValueError("datafile must be an open HDF5 file or file path")

    @staticmethod
    def _pert_channel_scale_array(
        feature_cols, pert_channel_scale: np.ndarray | None
    ) -> np.ndarray:
        nfeat = len(feature_cols)
        if pert_channel_scale is None:
            return np.ones(nfeat, dtype=np.float32)
        pc = np.asarray(pert_channel_scale, dtype=np.float32).reshape(-1)
        if pc.shape[0] != nfeat:
            raise ValueError(
                f"pert_channel_scale length {pc.shape[0]} != len(feature_cols)={nfeat}"
            )
        return pc

    def __init__(
        self,
        model,
        datafile,
        scaler,
        feature_cols,
        error_cols,
        recon_cols,
        label_scalers=None,
        latent_size=256,
        xp_masking_ratio=0.9,
        m_masking_ratio=0.9,
        lr=1e-3,
        optimizer="adam",
        wd=0,
        lasso=0,
        lf="mse",
        pt_save_str="pt_model.pth",
        ft_save_str="ft_model.pth",
        pt_log_file="pt_loss.log",
        ft_log_file="ft_loss.log",
        checkpoint_interval=None,
        pert_features=False,
        pert_scale=1.0,
        mask_mixture_xp_full_frac: float = 0.0,
        pert_channel_scale: np.ndarray | None = None,
        scheduler_cosine_t0: int = 10,
        scheduler_cosine_t_mult: int = 2,
        scheduler_eta_min_factor: float = 0.01,
        max_rows_per_key: int | None = None,
        data_store=None,
        micro_batch_size: int | None = None,
    ):
        """
        Changes to the original that can predict ages are the following:
        periodic embeddings
        scaling the coefficients with the RobustScaler
        changing the mask value to -9999
        cosine LR schedule with warm restarts (see ``scheduler_cosine_*`` on the wrapper)
        different masking ratios

        """
        self.model = model
        self.datafile = self._open_datafile(datafile)
        self.featurescaler = scaler
        self.label_scalers = label_scalers
        if (
            hasattr(self.featurescaler, "scale_")
            and self.featurescaler.scale_ is not None
        ):
            self.scale_factors = (
                self.featurescaler.scale_
            )  # This is the IQR used by RobustScaler for each feature
        else:
            raise ValueError(
                "Scaler must be fitted and have scale_ attribute before initializing wrapper"
            )
        self.feature_cols = feature_cols
        self.error_cols = error_cols
        if len(self.error_cols) != len(self.feature_cols):
            raise ValueError("error_cols must align one-to-one with feature_cols")
        if any(
            error is not None and not isinstance(error, str)
            for error in self.error_cols
        ):
            raise TypeError("error_cols entries must be column names or None")
        self.recon_cols = recon_cols
        if max_rows_per_key is not None and max_rows_per_key < 1:
            raise ValueError("max_rows_per_key must be positive when set")
        self.max_rows_per_key = max_rows_per_key
        self.data_store = data_store
        if micro_batch_size is not None and micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive when set")
        self._pretrain_micro_batch_size = micro_batch_size
        self._pretrain_prefetch_depth = 2
        self.diff = len(feature_cols) - len(recon_cols)
        self.xp_masking_ratio = xp_masking_ratio
        self.m_masking_ratio = m_masking_ratio
        self.mask_mixture_xp_full_frac = float(mask_mixture_xp_full_frac)
        self.lr = lr
        self.opt = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = EncoderDecoderLoss(lf=lf)
        # Non-gnll loss for multitask reconstruction (gnll requires logvar which
        # the finetune multitask path does not provide).
        self._rec_loss_fn = EncoderDecoderLoss(lf=lf if lf != "gnll" else "mse")
        self.latent_size = latent_size
        self.lasso = lasso
        self.wd = wd

        self.pt_save_str = pt_save_str
        self.ft_save_str = ft_save_str
        self.pt_log_file = pt_log_file
        self.ft_log_file = ft_log_file
        self.checkpoint_interval = checkpoint_interval
        self.scheduler_cosine_t0 = int(scheduler_cosine_t0)
        self.scheduler_cosine_t_mult = int(scheduler_cosine_t_mult)
        self.scheduler_eta_min_factor = float(scheduler_eta_min_factor)
        self.pert_features = pert_features
        self.pert_scale = pert_scale
        self._pert_channel_scale_np = self._pert_channel_scale_array(
            feature_cols, pert_channel_scale
        )
        self._error_available_np = np.asarray(
            [error is not None for error in error_cols], dtype=np.float32
        )
        self._pert_channel_scale_np *= self._error_available_np
        self.lp: nn.Linear | None = None
        self.ft: PredictionHead | None = None

        try:
            self.parallax_feature_idx = feature_cols.index("PARALLAX")
        except ValueError:
            self.parallax_feature_idx = None

        # Derive XP column indices from feature_cols for masking
        xp_indices = [
            idx
            for idx, c in enumerate(feature_cols)
            if c.startswith("bp_") or c.startswith("rp_")
        ]
        if xp_indices:
            self.xp_col_start = min(xp_indices)
            self.xp_col_end = max(xp_indices) + 1
        else:
            self.xp_col_start = 5
            self.xp_col_end = 115

        # CANFAR output defaults (no-op unless configured via _configure_canfar_output)
        self._metrics_file = None
        self._residual_stats_file = None
        self._residual_latest_file = None
        self._progress_file = None
        self._arc_checkpoint_dir = None
        self._arc_sync_interval = 5
        self._epoch_start = None
        self._pretrain_monitor_interval_size = 0
        self._pretrain_monitor_interval_rows = 0
        self._pretrain_monitor_interval_count = 0
        self._pretrain_monitor_epoch_rows_seen = 0
        self._pretrain_monitor_interval_loss = None
        self._pretrain_monitor_sample = None
        self._pretrain_rows_per_epoch = 0
        self._pretrain_rows_seen_total = 0
        self._run_id = (
            f"pretrain-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
        )

    @property
    def _pert_channel_scale_tensor(self):
        # ⚡ Bolt: Cache static tensor to prevent repetitive host-to-device transfers and CPU-GPU syncs
        if not hasattr(self, "_pert_channel_scale_cached"):
            self._pert_channel_scale_cached = torch.as_tensor(
                self._pert_channel_scale_np,
                device=self.device,
                dtype=torch.float32,
            )
        return self._pert_channel_scale_cached

    def _pert_noise(self, X_batch: Tensor, eX_batch: Tensor) -> Tensor:
        """Gaussian noise scaled by per-feature errors and ``pert_channel_scale``."""
        noise = torch.randn_like(X_batch) * eX_batch * self.pert_scale
        w = self._pert_channel_scale_tensor.to(device=X_batch.device, dtype=noise.dtype)
        if w.dim() != 1 or w.shape[0] != X_batch.shape[1]:
            raise RuntimeError("pert_channel_scale length must match feature dimension")
        return noise * w.unsqueeze(0)

    def _apply_mask(self, X):
        """
        Apply masking strategies to the input tensor while tracking NaN locations:
        1. Mask XP coefficient columns for a random subset of rows.
        2. Mask photometric band columns randomly per element.

        Args:
            X (Tensor): Input data tensor.

        Returns:
            X_masked (Tensor): Tensor with masking applied.
            mask (Tensor): Boolean mask indicating where the mask was applied.
            nan_mask (Tensor): Boolean mask indicating original NaN locations.
        """
        col_start_fixed = self.xp_col_start
        col_end_fixed = self.xp_col_end
        col_start_random = self.xp_col_end
        # ⚡ Bolt: Use .detach().clone() instead of .clone().detach() and allocate tensors directly on target device (using device=self.device) to prevent host-to-device transfers and CPU-GPU synchronization overhead.
        X_masked = X.detach().clone().to(self.device)

        # get NaN locations
        nan_mask = ~torch.isnan(X_masked)

        # ⚡ Bolt: Consolidate multiple zeros_like allocations into a single combined_mask tensor
        combined_mask = torch.zeros_like(X, dtype=torch.bool, device=self.device)

        # row-wise masking for cols [5:115] - XP coeffs
        num_rows_to_mask = int(self.xp_masking_ratio * X.shape[0])
        row_indices = torch.randperm(X.shape[0], device=self.device)[:num_rows_to_mask]
        combined_mask[row_indices, col_start_fixed:col_end_fixed] = True

        # Extra rows with XP fully masked (mixture component toward XP-off at inference).
        mf = getattr(self, "mask_mixture_xp_full_frac", 0.0) or 0.0
        if mf > 0.0:
            n_add = int(mf * X.shape[0])
            if n_add > 0:
                add_idx = torch.randperm(X.shape[0], device=self.device)[:n_add]
                combined_mask[add_idx, col_start_fixed:col_end_fixed] = True

        # random element-wise masking for cols [0:5] and [115:] - phot bands
        # ⚡ Bolt: Assign rand values directly to slices of combined_mask to avoid allocating intermediate mask_random tensor
        combined_mask[:, :col_start_fixed] = (
            torch.rand(X.shape[0], col_start_fixed, device=self.device)
            < self.m_masking_ratio
        )
        combined_mask[:, col_start_random:] = (
            torch.rand(X.shape[0], X.shape[1] - col_start_random, device=self.device)
            < self.m_masking_ratio
        )

        # apply masks sequentially to avoid allocating ~nan_mask | combined_mask
        X_masked.masked_fill_(~nan_mask, -9999).masked_fill_(combined_mask, -9999)

        return X_masked, combined_mask, nan_mask

    def _load_data(self, key):
        """Load and validate data with proper error handling"""
        try:
            if key not in self.datafile:
                raise KeyError(f"Key '{key}' not found in datafile")

            dataset = self.datafile[key]
            data = (
                dataset[: self.max_rows_per_key]
                if self.max_rows_per_key is not None
                else dataset[:]
            )
            if len(data) == 0:
                raise ValueError(f"Dataset '{key}' is empty")

            # Validate required columns exist
            missing_features = [
                col for col in self.feature_cols if col not in data.dtype.names
            ]
            missing_errors = [
                col
                for col in self.error_cols
                if col is not None and col not in data.dtype.names
            ]

            if missing_features:
                raise ValueError(
                    f"Missing feature columns in '{key}': {missing_features}"
                )
            if missing_errors:
                raise ValueError(f"Missing error columns in '{key}': {missing_errors}")

            X = np.column_stack(
                [self._clean_column(col, data[col]) for col in self.feature_cols]
            )
            eX = np.column_stack(
                [
                    self._clean_column(error, data[error])
                    if error is not None
                    else np.full(len(data), self.scale_factors[index])
                    for index, error in enumerate(self.error_cols)
                ]
            )

            # Validate data shapes
            if X.shape[0] != eX.shape[0]:
                raise ValueError(
                    f"Feature and error arrays have mismatched lengths: {X.shape[0]} vs {eX.shape[0]}"
                )

            # Keep invalid measurements masked and replace unusable uncertainties
            # with the largest valid error in that feature, as before.
            X[~np.isfinite(X)] = np.nan
            valid_errors = np.isfinite(eX) & (eX > 0)
            col_maxes = np.max(np.where(valid_errors, eX, -np.inf), axis=0)
            col_maxes[~np.isfinite(col_maxes)] = 1.0
            eX = np.where(valid_errors, eX, col_maxes[None, :])

            # Apply scaling with validation
            X = self.featurescaler.transform(X)
            eX = eX / self.scale_factors

            # Final validation
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print(f"Warning: Invalid values in features for key '{key}'")
            if np.any(np.isnan(eX)) or np.any(np.isinf(eX)):
                print(f"Warning: Invalid values in errors for key '{key}'")

            return torch.as_tensor(
                X, device=self.device, dtype=torch.float32
            ), torch.as_tensor(eX, device=self.device, dtype=torch.float32)

        except Exception as e:
            raise RuntimeError(f"Error loading data for key '{key}': {e}")

    @staticmethod
    def _clean_column(col, col_data):
        """Convert byte strings to NaN and stack columns"""
        try:
            if col_data.dtype.kind in {
                "S",
                "U",
            }:  # If the column contains byte strings or unicode
                return np.array(
                    [np.nan if v in {b"", ""} else float(v) for v in col_data],
                    dtype=np.float32,
                )
            return col_data.astype(np.float32)  # Convert other numeric types to float32
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error processing column {col}: {e}")

    def init_weights_gelu(self, m):
        if isinstance(m, nn.Linear | nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _setup_pretrain_optimizer(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        param_groups = [
            {"params": decay, "weight_decay": self.wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if self.opt == "adam":
            optimizer = optim.Adam(param_groups, lr=self.lr)
        elif self.opt == "adamw":
            optimizer = optim.AdamW(param_groups, lr=self.lr)
        elif self.opt == "sgd":
            optimizer = optim.SGD(param_groups, lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown pretrain optimizer {self.opt!r}")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.scheduler_cosine_t0,
            T_mult=self.scheduler_cosine_t_mult,
            eta_min=self.lr * self.scheduler_eta_min_factor,
        )
        return optimizer, scheduler

    def _configure_pretrain_logging(self) -> None:
        log_dir = os.path.dirname(self.pt_log_file) or "."
        os.makedirs(log_dir, exist_ok=True)
        save_dir = os.path.dirname(self.pt_save_str)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(
            filename=self.pt_log_file,
            level=logging.INFO,
            format=f"%(asctime)s - {self._run_id} - Sub-Epoch: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode="a",
            force=True,
        )

    def _configure_canfar_output(
        self,
        metrics_file=None,
        residual_stats_file=None,
        residual_latest_file=None,
        progress_file=None,
        arc_checkpoint_dir=None,
        arc_sync_interval=5,
        run_id=None,
    ):
        self._metrics_file = metrics_file
        self._residual_stats_file = residual_stats_file
        self._residual_latest_file = residual_latest_file
        self._progress_file = progress_file
        self._arc_checkpoint_dir = arc_checkpoint_dir
        self._arc_sync_interval = arc_sync_interval
        self._epoch_start = None
        self._run_id = (
            run_id
            or os.environ.get("MSA_RUN_ID")
            or (
                f"pretrain-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
                f"{os.getpid()}"
            )
        )

    def _write_progress(self, stage, **fields):
        progress_file = getattr(self, "_progress_file", None)
        if not progress_file:
            return
        import json

        entry = {
            "run_id": getattr(self, "_run_id", None),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            **fields,
        }
        progress_path = Path(progress_file)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    def _resource_metrics(self):
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics = {
            "peak_host_rss_bytes": int(
                peak_rss if sys.platform == "darwin" else peak_rss * 1024
            ),
            "current_host_rss_bytes": None,
            "scratch_free_bytes": None,
            "output_free_bytes": None,
        }
        try:
            with open("/proc/self/statm") as statm:
                resident_pages = int(statm.read().split()[1])
            metrics["current_host_rss_bytes"] = resident_pages * os.sysconf(
                "SC_PAGE_SIZE"
            )
        except (OSError, IndexError, ValueError):
            pass

        data_store = getattr(self, "data_store", None)
        if data_store is not None:
            metrics["loader_mode"] = data_store.mode
            metrics["projected_cache_bytes"] = data_store.cache_bytes
            try:
                metrics["scratch_free_bytes"] = shutil.disk_usage(
                    data_store.scratch_dir
                ).free
            except OSError:
                pass
        metrics["optimizer_batch_size"] = getattr(self, "_pretrain_batch_size", None)
        metrics["micro_batch_size"] = getattr(self, "_pretrain_micro_batch_size", None)
        output_dir = os.path.dirname(self.pt_save_str) or "."
        try:
            metrics["output_free_bytes"] = shutil.disk_usage(output_dir).free
        except OSError:
            pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            metrics["current_gpu_allocated_bytes"] = torch.cuda.memory_allocated()
            metrics["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            metrics["current_gpu_reserved_bytes"] = torch.cuda.memory_reserved()
            metrics["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return metrics

    def _log_epoch_metrics(
        self,
        epoch,
        total_epochs,
        mean_loss,
        val_loss,
        optimizer,
        residual_stats=None,
        rows_seen_total=None,
    ):
        if not self._metrics_file:
            return
        import json
        import time

        entry = {
            "run_id": self._run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "record_type": "epoch",
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "train_loss": round(float(mean_loss), 8),
            "val_loss": (round(float(val_loss), 8) if val_loss is not None else None),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "wall_time_s": (
                round(time.time() - self._epoch_start, 1) if self._epoch_start else None
            ),
        }
        if rows_seen_total is not None:
            entry["rows_seen_total"] = int(rows_seen_total)
        entry.update(self._resource_metrics())
        entry.update(
            {
                f"residual_{key}": value
                for key, value in (residual_stats or {}).items()
                if key
                not in {
                    "run_id",
                    "timestamp_utc",
                    "epoch",
                    "record_type",
                    "rows_seen_total",
                    "epoch_rows_seen",
                    "interval_rows",
                    "interval_index",
                    "interval_partial",
                    "train_loss",
                }
                and isinstance(value, int | float)
            }
        )
        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_residual_stats(
        self,
        val_keys,
        epoch,
        mini_batch=32768,
        *,
        sample_data=None,
        record_type=None,
        rows_seen_total=None,
        epoch_rows_seen=None,
        interval_rows=None,
        train_loss=None,
        interval_index=None,
        interval_partial=None,
        snapshot_only=False,
        report_progress=True,
    ):
        if (
            not (
                getattr(self, "_residual_stats_file", None)
                or getattr(self, "_residual_latest_file", None)
                or sample_data is not None
            )
            or not val_keys
        ):
            return {}
        import json

        def progress(stage, **fields):
            if report_progress:
                self._write_progress(stage, **fields)

        self.model.eval()
        stats: dict = {
            "run_id": self._run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epoch": epoch + 1,
        }
        sample_rows = 10_000
        progress(
            "residual_sampling_started",
            epoch=epoch + 1,
            validation_shard_count=len(val_keys),
            target_rows=sample_rows,
        )
        with torch.no_grad():
            data_store = getattr(self, "data_store", None)
            if sample_data is not None:
                X_sample, eX_sample = sample_data
                batch_rows = min(
                    mini_batch,
                    getattr(self, "_pretrain_micro_batch_size", None) or mini_batch,
                )
                batches = (
                    (
                        X_sample[start : start + batch_rows],
                        eX_sample[start : start + batch_rows],
                    )
                    for start in range(0, len(X_sample), batch_rows)
                )
                batch_sources = (("validation_sample", batches),)
            elif data_store is not None:
                sample_rows_per_key = max(1, sample_rows // len(val_keys))
                batch_rows = min(
                    mini_batch,
                    getattr(self, "_pretrain_micro_batch_size", None) or mini_batch,
                )
                batch_sources = (
                    (
                        key,
                        prefetch_batches(
                            data_store.sample_batches(
                                key,
                                self.featurescaler,
                                self.scale_factors,
                                sample_rows=sample_rows_per_key,
                                seed=epoch * len(val_keys) + key_index,
                                batch_rows=batch_rows,
                            ),
                            depth=self._pretrain_prefetch_depth,
                        ),
                    )
                    for key_index, key in enumerate(val_keys)
                )
            else:
                X_val, eX_val = self._load_data(val_keys[0])
                n = min(sample_rows, X_val.shape[0])
                idx = torch.randperm(X_val.shape[0], device=self.device)[:n]
                X_sub, eX_sub = X_val[idx], eX_val[idx]
                batches = (
                    (
                        X_sub[start : start + mini_batch],
                        eX_sub[start : start + mini_batch],
                    )
                    for start in range(0, n, mini_batch)
                )
                batch_sources = ((val_keys[0], batches),)

            error_batches = []
            sampled_rows = 0
            sampled_shards = 0
            validation_loss_sum = torch.zeros((), device=self.device)
            photo_idx = list(range(self.xp_col_start)) + list(
                range(self.xp_col_end, len(self.recon_cols))
            )
            for key, batches in batch_sources:
                shard_sampled = False
                shard_sampled_rows = 0
                for X_batch, eX_batch in batches:
                    sampled_rows += len(X_batch)
                    shard_sampled_rows += len(X_batch)
                    shard_sampled = True
                    if not isinstance(X_batch, torch.Tensor):
                        X_batch = torch.as_tensor(
                            np.ascontiguousarray(X_batch), device=self.device
                        )
                        eX_batch = torch.as_tensor(
                            np.ascontiguousarray(eX_batch), device=self.device
                        )
                    X_masked, mask, nanmask = self._apply_mask(X_batch)
                    X_recon, _ = self.model(X_masked)
                    recon_mask = mask[:, : -self.diff] & nanmask[:, : -self.diff]
                    logvar = getattr(self.model, "_last_logvar", None)
                    batch_loss = self.loss_fn(
                        X_batch[:, : -self.diff],
                        X_recon,
                        recon_mask,
                        1.0 / (eX_batch[:, : -self.diff].square() + 1e-8),
                        logvar=logvar,
                    )
                    validation_loss_sum.add_(batch_loss * len(X_batch))
                    errors = (X_recon - X_batch[:, : -self.diff]).abs()
                    errors = errors.masked_fill(~recon_mask, float("nan"))
                    error_batches.append(errors)
                sampled_shards += int(shard_sampled)
                progress(
                    "residual_shard_sampled",
                    epoch=epoch + 1,
                    key=key,
                    rows_sampled=shard_sampled_rows,
                    sampled=shard_sampled,
                )

            if error_batches:
                all_errors = torch.cat(error_batches)
                xp_errors = all_errors[:, self.xp_col_start : self.xp_col_end]
                photo_errors = all_errors[:, photo_idx]
                xp_valid = xp_errors[torch.isfinite(xp_errors)]
                photo_valid = photo_errors[torch.isfinite(photo_errors)]
                all_valid = all_errors[torch.isfinite(all_errors)]
                feature_errors = all_errors.detach().cpu().numpy()
            else:
                all_valid = xp_valid = photo_valid = torch.empty(0)
                feature_errors = np.empty((0, len(self.recon_cols)), dtype=np.float32)
            if xp_valid.numel() > 0:
                stats["xp_mae"] = round(xp_valid.mean().item(), 8)
                stats["xp_p84"] = round(xp_valid.quantile(0.84).item(), 8)
            if photo_valid.numel() > 0:
                stats["photo_mae"] = round(photo_valid.mean().item(), 8)
            if all_valid.numel() > 0:
                stats["overall_mae"] = round(all_valid.mean().item(), 8)
            stats.update(_summarize_feature_residuals(feature_errors, self.recon_cols))
        stats["sampled_rows"] = sampled_rows
        stats["sampled_validation_shards"] = (
            len(val_keys) if sample_data is not None else sampled_shards
        )
        stats["sampled_val_loss"] = (
            round(float((validation_loss_sum / sampled_rows).item()), 8)
            if sampled_rows
            else None
        )
        if record_type is not None:
            stats["record_type"] = record_type
        if rows_seen_total is not None:
            stats["rows_seen_total"] = int(rows_seen_total)
        if epoch_rows_seen is not None:
            stats["epoch_rows_seen"] = int(epoch_rows_seen)
        if interval_rows is not None:
            stats["interval_rows"] = int(interval_rows)
        if train_loss is not None:
            stats["train_loss"] = round(float(train_loss), 8)
        if interval_index is not None:
            stats["interval_index"] = int(interval_index)
        if interval_partial is not None:
            stats["interval_partial"] = bool(interval_partial)
        if snapshot_only:
            self._write_pretrain_residual_snapshot(
                {
                    "run_id": self._run_id,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    **stats,
                }
            )
        else:
            self._write_pretrain_residual_record(stats)
        progress(
            "residual_sampling_finished",
            epoch=epoch + 1,
            sampled_rows=sampled_rows,
            sampled_validation_shards=sampled_shards,
        )
        residual_metrics = {
            key: stats[key]
            for key in ("xp_mae", "xp_p84", "photo_mae", "overall_mae")
            if key in stats
        }
        if residual_metrics:
            print(
                "Validation residuals: "
                f"{json.dumps(residual_metrics, sort_keys=True)}, "
                f"sampled_rows={sampled_rows}, sampled_shards={sampled_shards}"
            )
        return stats

    def _prepare_pretrain_monitor_sample(
        self, val_keys, sample_rows, mini_batch, *, seed=42
    ):
        data_store = getattr(self, "data_store", None)
        if data_store is None or not val_keys or sample_rows < 1:
            return None

        batch_rows = min(
            mini_batch,
            getattr(self, "_pretrain_micro_batch_size", None) or mini_batch,
        )
        x_parts, e_parts = [], []
        base, extra = divmod(sample_rows, len(val_keys))
        for key_index, key in enumerate(val_keys):
            rows_for_key = base + int(key_index < extra)
            if rows_for_key < 1:
                continue
            for X, eX in data_store.sample_batches(
                key,
                self.featurescaler,
                self.scale_factors,
                sample_rows=rows_for_key,
                seed=seed + key_index,
                batch_rows=batch_rows,
            ):
                x_parts.append(X)
                e_parts.append(eX)
        if not x_parts:
            return None
        return np.concatenate(x_parts), np.concatenate(e_parts)

    def _evaluate_pretrain_monitor_sample(
        self,
        epoch,
        mini_batch,
        *,
        record_type,
        rows_seen_total,
        epoch_rows_seen,
        interval_rows=None,
        train_loss=None,
        interval_index=None,
        interval_partial=None,
        snapshot_only=False,
        seed=42,
    ):
        sample = getattr(self, "_pretrain_monitor_sample", None)
        if sample is None:
            return {}
        was_training = self.model.training
        rng_state = _capture_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            return self._log_residual_stats(
                self._pretrain_valid_keys,
                epoch,
                mini_batch,
                sample_data=sample,
                record_type=record_type,
                rows_seen_total=rows_seen_total,
                epoch_rows_seen=epoch_rows_seen,
                interval_rows=interval_rows,
                train_loss=train_loss,
                interval_index=interval_index,
                interval_partial=interval_partial,
                snapshot_only=snapshot_only,
                report_progress=False,
            )
        finally:
            self.model.train(was_training)
            _restore_rng_state(rng_state)

    def _write_pretrain_residual_record(self, stats):
        if not stats:
            return
        import json

        entry = {
            "run_id": self._run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **stats,
        }
        if self._residual_stats_file:
            path = Path(self._residual_stats_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
        self._write_pretrain_residual_snapshot(entry)

    def _write_pretrain_residual_snapshot(self, entry):
        path = getattr(self, "_residual_latest_file", None)
        if not path or not entry:
            return
        import json

        snapshot = Path(path)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot.with_name(snapshot.name + ".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(entry) + "\n")
        os.replace(temporary, snapshot)

    def _emit_pretrain_interval(
        self, interval_rows, interval_loss, mini_batch, *, partial
    ):
        if interval_rows < 1:
            return
        import json

        epoch = int(getattr(self, "_active_epoch", 1))
        total_epochs = int(getattr(self, "_active_total_epochs", epoch))
        rows_seen = int(self._pretrain_monitor_epoch_rows_seen)
        rows_seen_total = int(self._pretrain_rows_seen_total)
        train_loss = float((interval_loss / interval_rows).item())
        interval_index = int(self._pretrain_monitor_interval_count) + 1
        residual_stats = self._evaluate_pretrain_monitor_sample(
            epoch - 1,
            mini_batch,
            record_type="interval",
            rows_seen_total=rows_seen_total,
            epoch_rows_seen=rows_seen,
            interval_rows=interval_rows,
            train_loss=train_loss,
            interval_index=interval_index,
            interval_partial=partial,
            snapshot_only=True,
        )

        if self._metrics_file:
            entry = {
                "run_id": self._run_id,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "record_type": "interval",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "interval_index": interval_index,
                "interval_rows": int(interval_rows),
                "epoch_rows_seen": rows_seen,
                "rows_seen_total": rows_seen_total,
                "train_loss": round(train_loss, 8),
                "sampled_val_loss": residual_stats.get("sampled_val_loss"),
                "interval_partial": bool(partial),
            }
            entry.update(
                {
                    f"residual_{key}": residual_stats[key]
                    for key in ("xp_mae", "xp_p84", "photo_mae", "overall_mae")
                    if key in residual_stats
                }
            )
            entry.update(self._resource_metrics())
            with open(self._metrics_file, "a") as stream:
                stream.write(json.dumps(entry) + "\n")

        self._pretrain_monitor_interval_count = interval_index
        self._write_progress(
            "training_interval_finished",
            epoch=epoch,
            total_epochs=total_epochs,
            interval_index=interval_index,
            interval_rows=int(interval_rows),
            epoch_rows_seen=rows_seen,
            rows_seen_total=rows_seen_total,
            train_loss=round(train_loss, 8),
            sampled_val_loss=residual_stats.get("sampled_val_loss"),
            interval_partial=bool(partial),
        )

    def _accumulate_pretrain_monitor_batch(self, batch_loss, rows, mini_batch):
        self._pretrain_rows_seen_total = (
            int(getattr(self, "_pretrain_rows_seen_total", 0)) + rows
        )
        interval = int(getattr(self, "_pretrain_monitor_interval_size", 0))
        if interval < 1:
            return
        self._pretrain_monitor_epoch_rows_seen += rows
        self._pretrain_monitor_interval_rows += rows
        self._pretrain_monitor_interval_loss.add_(batch_loss.detach() * rows)
        if self._pretrain_monitor_interval_rows >= interval:
            interval_rows = self._pretrain_monitor_interval_rows
            interval_loss = self._pretrain_monitor_interval_loss
            self._pretrain_monitor_interval_rows = 0
            self._pretrain_monitor_interval_loss = torch.zeros((), device=self.device)
            self._emit_pretrain_interval(
                interval_rows, interval_loss, mini_batch, partial=False
            )

    def _flush_pretrain_monitor_interval(self, mini_batch):
        interval_rows = int(getattr(self, "_pretrain_monitor_interval_rows", 0))
        if interval_rows:
            interval_loss = self._pretrain_monitor_interval_loss
            self._pretrain_monitor_interval_rows = 0
            self._pretrain_monitor_interval_loss = torch.zeros((), device=self.device)
            self._emit_pretrain_interval(
                interval_rows, interval_loss, mini_batch, partial=True
            )

    def _sync_checkpoint_to_arc(self):
        if not self._arc_checkpoint_dir:
            return
        import os
        import shutil

        os.makedirs(self._arc_checkpoint_dir, exist_ok=True)
        src = self.pt_save_str
        dst = os.path.join(self._arc_checkpoint_dir, os.path.basename(src))
        tmp = dst + ".tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    def _pretrain_run_signature(self) -> dict:
        return {
            "feature_cols": list(self.feature_cols),
            "error_cols": list(self.error_cols),
            "recon_cols": list(self.recon_cols),
            "scaler_center": np.asarray(self.featurescaler.center_).tolist(),
            "scaler_scale": np.asarray(self.featurescaler.scale_).tolist(),
            "train_keys": list(getattr(self, "_pretrain_train_keys", [])),
            "loader_mode": getattr(
                getattr(self, "data_store", None), "mode", "legacy-full-shard"
            ),
            "loader_policy": getattr(
                getattr(self, "data_store", None),
                "shuffle_policy",
                "full_shard_shuffle_v0",
            ),
            "train_rows_per_epoch": getattr(self, "_pretrain_rows_per_epoch", None),
            "train_rows_by_key": {
                key: int(self.data_store._row_counts.get(key, 0))
                for key in getattr(self, "_pretrain_train_keys", [])
            }
            if getattr(self, "data_store", None) is not None
            else None,
            "cache_max_bytes": getattr(
                getattr(self, "data_store", None), "cache_max_bytes", None
            ),
            "cache_seed": getattr(
                getattr(self, "data_store", None), "cache_seed", None
            ),
            "micro_batch_size": getattr(self, "_pretrain_micro_batch_size", None),
        }

    def _load_pretrain_resume(self, pretrained, optimizer, scheduler):
        epoch_loss = 0.0
        loss_div = 0.0
        pretrained_epoch = 0
        if pretrained is None:
            return epoch_loss, loss_div, pretrained_epoch
        checkpoint = torch_load_trusted(pretrained)
        if checkpoint.get("experiment_only"):
            raise ValueError(
                "This checkpoint is from a bounded experiment; use explicit experiment "
                "handling to preserve its subset, mask RNG and embedding policy"
            )
        signature = checkpoint.get("run_signature")
        if signature is not None:
            current_signature = self._pretrain_run_signature()
            saved_rows_by_key = signature.get("train_rows_by_key")
            current_rows_by_key = current_signature.get("train_rows_by_key")
            expanded_training_rows = (
                signature.get("train_keys") == current_signature.get("train_keys")
                and isinstance(saved_rows_by_key, dict)
                and isinstance(current_rows_by_key, dict)
                and saved_rows_by_key.keys() == current_rows_by_key.keys()
                and all(
                    int(current_rows_by_key[key]) >= int(saved_rows_by_key[key])
                    for key in saved_rows_by_key
                )
                and sum(map(int, current_rows_by_key.values()))
                > sum(map(int, saved_rows_by_key.values()))
            )
            mismatches = [
                key
                for key, current_value in current_signature.items()
                if key in signature
                and key not in {"loader_mode", "loader_policy"}
                and not (
                    expanded_training_rows
                    and key in {"train_rows_per_epoch", "train_rows_by_key"}
                )
                and signature.get(key) != current_value
                and not (
                    key == "error_cols"
                    and not self.pert_features
                    and self.loss_fn.cost not in {"wmse", "wmae"}
                )
            ]
            if mismatches:
                raise ValueError(
                    "Pretraining checkpoint does not match current run settings: "
                    + ", ".join(mismatches)
                )
            if expanded_training_rows:
                print(
                    "Resuming with expanded training data coverage: "
                    f"{sum(map(int, saved_rows_by_key.values()))} -> "
                    f"{sum(map(int, current_rows_by_key.values()))} rows per epoch"
                )
            if signature.get("loader_policy") not in {
                None,
                current_signature.get("loader_policy"),
            }:
                print(
                    "Resuming with updated input shuffle policy: "
                    f"{signature.get('loader_policy')} -> "
                    f"{current_signature.get('loader_policy')}"
                )
        else:
            print(
                "Warning: legacy pretraining checkpoint has no run signature; "
                "feature/scaler compatibility cannot be verified"
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch_loss = checkpoint["epoch_loss"]
        loss_div = checkpoint["loss_div"]
        pretrained_epoch = checkpoint["epoch"]
        if "rows_seen_total" in checkpoint:
            self._pretrain_rows_seen_total = int(checkpoint["rows_seen_total"])
        else:
            data_store = getattr(self, "data_store", None)
            if data_store is None:
                rows_per_epoch = int(getattr(self, "_pretrain_rows_per_epoch", 0))
            else:
                rows_per_epoch = sum(
                    int(
                        data_store._source_row_counts.get(
                            key, data_store._row_counts.get(key, 0)
                        )
                    )
                    for key in getattr(self, "_pretrain_train_keys", [])
                )
            self._pretrain_rows_seen_total = pretrained_epoch * rows_per_epoch
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
        print("Picking up pre-training from epoch", pretrained_epoch)
        return epoch_loss, loss_div, pretrained_epoch

    def _pretrain_checkpoint_payload(
        self, epoch, optimizer, scheduler, epoch_loss, loss_div
    ) -> dict:
        return {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch_loss": epoch_loss,
            "loss_div": loss_div,
            "rows_seen_total": int(getattr(self, "_pretrain_rows_seen_total", 0)),
            "rng_state": _capture_rng_state(),
            "run_signature": self._pretrain_run_signature(),
        }

    def _save_pretrain_checkpoints(
        self, epoch, optimizer, scheduler, epoch_loss, loss_div
    ) -> None:
        payload = self._pretrain_checkpoint_payload(
            epoch, optimizer, scheduler, epoch_loss, loss_div
        )
        _atomic_torch_save(payload, self.pt_save_str)
        if (
            self.checkpoint_interval is not None
            and (epoch + 1) % self.checkpoint_interval == 0
        ):
            interval_path = (
                f"{os.path.splitext(self.pt_save_str)[0]}_checkpoint_{epoch + 1}.pth"
            )
            _atomic_torch_save(payload, interval_path)

    def _pretrain_reconstruction_loss(
        self,
        X_batch,
        eX_batch,
        X_reconstructed,
        z,
        mask,
        nanmask,
        global_mask_count=None,
    ):
        reconstruction_mask = mask[:, : -self.diff] & nanmask[:, : -self.diff]
        # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
        reconstruction_w = 1.0 / (
            (eX_batch[:, : -self.diff] * eX_batch[:, : -self.diff]) + 1e-8
        )

        logvar = getattr(self.model, "_last_logvar", None)
        reconstruction_loss = self.loss_fn(
            X_batch[:, : -self.diff],
            X_reconstructed,
            reconstruction_mask,
            reconstruction_w,
            logvar=logvar,
        )
        if global_mask_count is not None:
            local_count = reconstruction_mask.sum().to(reconstruction_loss.dtype)
            reconstruction_loss = (
                reconstruction_loss
                * local_count
                / global_mask_count.clamp_min(self.loss_fn.eps)
            )
        return reconstruction_loss + self.lasso * z.abs().sum()

    def _train_pretrain_key(
        self, key, optimizer, mini_batch, epoch_loss, loss_div, subkeynum
    ):
        started = time.perf_counter()
        self._write_progress(
            "train_shard_started",
            epoch=getattr(self, "_active_epoch", None),
            key=key,
            shard_index=subkeynum + 1,
            shard_count=getattr(self, "_active_train_shard_count", None),
        )
        store_times = None
        data_store = getattr(self, "data_store", None)
        if data_store is not None:
            store_times = (
                data_store.read_seconds,
                data_store.conversion_seconds,
                data_store.transform_seconds,
            )
        micro_batch = min(mini_batch, self._pretrain_micro_batch_size or mini_batch)
        if micro_batch < mini_batch and any(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in self.model.modules()
        ):
            raise ValueError(
                "Microbatch gradient accumulation changes BatchNorm statistics; "
                "use LayerNorm or set micro_batch_size equal to mini_batch_size"
            )
        n_rows = 0
        shard_loss = torch.zeros((), device=self.device)
        for X_batch, eX_batch in self._iter_pretrain_batches(
            key, mini_batch, shuffle=True
        ):
            n_rows += len(X_batch)
            X_target = X_batch
            if self.pert_features:
                X_input = X_target + self._pert_noise(X_target, eX_batch)
            else:
                X_input = X_target
            X_masked, mask, nanmask = self._apply_mask(X_input)
            reconstruction_mask = mask[:, : -self.diff] & nanmask[:, : -self.diff]
            global_mask_count = reconstruction_mask.sum().to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=self.device)
            for start in range(0, len(X_batch), micro_batch):
                stop = min(start + micro_batch, len(X_batch))
                X_reconstructed, z = self.model(X_masked[start:stop])
                loss = self._pretrain_reconstruction_loss(
                    X_target[start:stop],
                    eX_batch[start:stop],
                    X_reconstructed,
                    z,
                    mask[start:stop],
                    nanmask[start:stop],
                    global_mask_count=global_mask_count,
                )
                loss.backward()
                batch_loss.add_(loss.detach())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            self._accumulate_pretrain_monitor_batch(
                batch_loss, len(X_batch), mini_batch
            )
            shard_loss.add_(batch_loss * len(X_batch))
        if n_rows:
            epoch_loss += float(shard_loss.item())
            loss_div += n_rows
        wall_seconds = time.perf_counter() - started
        if store_times is None:
            mean_loss = epoch_loss / loss_div if loss_div else 0.0
            logging.info(f"{subkeynum + 1}, Loss: {mean_loss}")
        else:
            read_seconds = data_store.read_seconds - store_times[0]
            conversion_seconds = (data_store.conversion_seconds - store_times[1]) + (
                data_store.transform_seconds - store_times[2]
            )
            logging.info(
                "shard=%s rows=%d mode=%s wall_s=%.1f read_s=%.1f "
                "conversion_s=%.1f rows_per_s=%.1f",
                key,
                n_rows,
                data_store.mode,
                wall_seconds,
                read_seconds,
                conversion_seconds,
                n_rows / max(wall_seconds, 1e-9),
            )
            self._write_progress(
                "train_shard_finished",
                epoch=getattr(self, "_active_epoch", None),
                key=key,
                shard_index=subkeynum + 1,
                shard_count=getattr(self, "_active_train_shard_count", None),
                rows_completed=n_rows,
                rows_total=data_store._row_counts.get(key, n_rows),
                wall_seconds=round(wall_seconds, 3),
                source_read_seconds=round(read_seconds, 3),
                conversion_seconds=round(conversion_seconds, 3),
                rows_per_second=round(n_rows / max(wall_seconds, 1e-9), 2),
            )
        return epoch_loss, loss_div

    def _iter_pretrain_batches(self, key, mini_batch, *, shuffle):
        data_store = getattr(self, "data_store", None)
        if data_store is None:
            X, eX = self._load_data(key)
            loader = DataLoader(
                TensorDataset(X, eX), batch_size=mini_batch, shuffle=shuffle
            )
            yield from loader
            return

        seed = int(np.random.randint(0, np.iinfo(np.uint32).max)) if shuffle else 0
        batches = data_store.iter_batches(
            key,
            self.featurescaler,
            self.scale_factors,
            batch_rows=mini_batch,
            shuffle=shuffle,
            seed=seed,
            max_rows=self.max_rows_per_key,
        )
        for X, eX in prefetch_batches(batches, depth=self._pretrain_prefetch_depth):
            yield (
                torch.as_tensor(np.ascontiguousarray(X), device=self.device),
                torch.as_tensor(np.ascontiguousarray(eX), device=self.device),
            )

    def _run_pretrain_epoch(
        self,
        epoch,
        total_epochs,
        train_keys,
        val_keys,
        optimizer,
        scheduler,
        mini_batch,
        epoch_loss,
        loss_div,
        running_pt_loss,
        running_pt_validation_loss,
    ):
        self._epoch_start = time.time()
        self._active_epoch = epoch + 1
        self._active_train_shard_count = len(train_keys)
        self._active_total_epochs = total_epochs
        self._pretrain_monitor_epoch_rows_seen = 0
        self._pretrain_monitor_interval_rows = 0
        self._pretrain_monitor_interval_count = 0
        self._pretrain_monitor_interval_loss = torch.zeros((), device=self.device)
        self._write_progress(
            "epoch_started", epoch=epoch + 1, total_epochs=total_epochs
        )
        epoch_loss = 0.0
        loss_div = 0.0
        random.shuffle(train_keys)
        self.model.train()
        pbar = tqdm.tqdm(
            enumerate(train_keys),
            total=len(train_keys),
            desc="Iterating Training Files",
        )
        for subkeynum, key in pbar:
            epoch_loss, loss_div = self._train_pretrain_key(
                key, optimizer, mini_batch, epoch_loss, loss_div, subkeynum
            )

        self._flush_pretrain_monitor_interval(mini_batch)
        scheduler.step()
        mean_loss = epoch_loss / loss_div if loss_div else 0.0
        print(f"Pre-training Epoch [{epoch + 1}/{total_epochs}], Loss: {mean_loss}")
        running_pt_loss.append(mean_loss)

        if val_keys is not None:
            validation_loss = self.validate(val_keys, self.loss_fn, mini_batch)
            logging.info(f"{epoch + 1}, Validation Loss: {validation_loss}")
            running_pt_validation_loss.append(validation_loss)

        self._save_pretrain_checkpoints(
            epoch, optimizer, scheduler, epoch_loss, loss_div
        )
        self._write_progress(
            "checkpoint_written",
            epoch=epoch + 1,
            path=os.path.basename(self.pt_save_str),
        )
        # CANFAR monitoring: record resources after validation and residual stats.
        validation = (
            running_pt_validation_loss[-1] if running_pt_validation_loss else None
        )
        rows_seen_total = int(getattr(self, "_pretrain_rows_seen_total", 0))
        if getattr(self, "_pretrain_monitor_sample", None) is not None:
            residual_stats = self._evaluate_pretrain_monitor_sample(
                epoch,
                mini_batch,
                record_type="epoch",
                rows_seen_total=rows_seen_total,
                epoch_rows_seen=int(getattr(self, "_pretrain_rows_per_epoch", 0)),
            )
        else:
            residual_stats = self._log_residual_stats(val_keys, epoch, mini_batch)
        self._log_epoch_metrics(
            epoch,
            total_epochs,
            mean_loss,
            validation,
            optimizer,
            residual_stats=residual_stats,
            rows_seen_total=rows_seen_total,
        )
        self._write_progress(
            "epoch_finished",
            epoch=epoch + 1,
            total_epochs=total_epochs,
            rows_seen_total=rows_seen_total,
            epoch_rows_seen=int(getattr(self, "_pretrain_rows_per_epoch", 0)),
            train_loss=round(float(mean_loss), 8),
            val_loss=(round(float(validation), 8) if validation is not None else None),
            wall_time_seconds=(
                round(time.time() - self._epoch_start, 3) if self._epoch_start else None
            ),
        )
        if (epoch + 1) % self._arc_sync_interval == 0:
            self._sync_checkpoint_to_arc()
            self._write_progress("checkpoint_synced", epoch=epoch + 1)
        return epoch_loss, loss_div

    def _chain_finetune_after_pretrain(self, ft_stuff) -> None:
        if ft_stuff is None:
            return
        self.fit(
            ft_stuff[0],
            ft_stuff[1],
            ft_stuff[2],
            e_y_train=ft_stuff[3],
            X_val=ft_stuff[4],
            eX_val=ft_stuff[5],
            y_val=ft_stuff[6],
            e_y_val=ft_stuff[7],
            num_epochs=ft_stuff[8],
            mini_batch=ft_stuff[9],
            linearprobe=ft_stuff[10],
            maskft=ft_stuff[11],
            multitask=ft_stuff[12],
            rncloss=ft_stuff[13],
            last=True,
        )

    def pretrain_hdf(
        self,
        train_keys,
        num_epochs=10,
        val_keys=None,
        ft_stuff=None,
        mini_batch=32,
        pretrained=None,
        monitor_interval_rows=0,
        monitor_sample_rows=10_000,
    ):
        """
        Pre-trains the model on the training dataset with optional validation.

        Args:
            train_keys: Training dataset files in the large h5 (features).
            num_epochs: Total target epoch; a resumed run stops at this epoch.
            val_keys: Optional validation dataset files in the large h5 (features).
            ft_stuff:
            mini_batch: Mini-batch size for pretraining.
        """
        if not train_keys:
            raise ValueError("pretrain_hdf requires at least one training shard")
        if mini_batch < 1:
            raise ValueError("mini_batch must be positive")
        if val_keys is not None and not val_keys:
            raise ValueError("val_keys was provided but contains no validation shards")
        if num_epochs < 0:
            raise ValueError("num_epochs must be non-negative")
        if monitor_interval_rows < 0:
            raise ValueError("monitor_interval_rows must be non-negative")
        if monitor_sample_rows < 1:
            raise ValueError("monitor_sample_rows must be positive")
        self._pretrain_batch_size = mini_batch
        self._pretrain_rows_seen_total = 0
        self._pretrain_train_keys = sorted(train_keys)
        self._pretrain_valid_keys = list(val_keys or [])
        data_store = getattr(self, "data_store", None)
        if data_store is not None:
            self._pretrain_rows_per_epoch = sum(
                int(data_store._row_counts.get(key, 0)) for key in train_keys
            )
        elif hasattr(self, "datafile"):
            max_rows_per_key = getattr(self, "max_rows_per_key", None)
            self._pretrain_rows_per_epoch = sum(
                min(
                    len(self.datafile[key]),
                    max_rows_per_key or len(self.datafile[key]),
                )
                for key in train_keys
            )
        else:
            self._pretrain_rows_per_epoch = 0

        monitoring_configured = any(
            getattr(self, field, None)
            for field in ("_metrics_file", "_residual_stats_file", "_progress_file")
        )
        self._pretrain_monitor_interval_size = (
            int(monitor_interval_rows) if monitoring_configured else 0
        )
        self._pretrain_monitor_sample = (
            self._prepare_pretrain_monitor_sample(
                self._pretrain_valid_keys,
                int(monitor_sample_rows),
                mini_batch,
            )
            if self._pretrain_monitor_interval_size and val_keys
            else None
        )

        optimizer, scheduler = self._setup_pretrain_optimizer()
        self._configure_pretrain_logging()

        running_pt_loss = []
        running_pt_validation_loss = []
        epoch_loss, loss_div, pretrained_epoch = self._load_pretrain_resume(
            pretrained, optimizer, scheduler
        )

        for epoch in range(pretrained_epoch, num_epochs):
            epoch_loss, loss_div = self._run_pretrain_epoch(
                epoch,
                num_epochs,
                train_keys,
                val_keys,
                optimizer,
                scheduler,
                mini_batch,
                epoch_loss,
                loss_div,
                running_pt_loss,
                running_pt_validation_loss,
            )

        self._chain_finetune_after_pretrain(ft_stuff)

    def validate(self, val_keys, criterion, mini_batch=32):
        """
        Validates the model on a validation dataset during pretraining.

        Args:
            X_val: Validation dataset (features).
            criterion: Loss function used for validation (MSE).
            mini_batch: Mini-batch size for validation.

        """
        self.model.eval()
        if not val_keys:
            raise ValueError("validate requires at least one validation shard")
        self._write_progress(
            "validation_started",
            epoch=getattr(self, "_active_epoch", None),
            shard_count=len(val_keys),
        )
        with torch.no_grad():
            n_keys = len(val_keys)
            pbar = tqdm.tqdm(
                val_keys, total=n_keys, desc="Iterating Over Validation Keys"
            )
            loss_sum = torch.zeros((), device=self.device)
            row_count = 0
            data_store = getattr(self, "data_store", None)
            for shard_index, key in enumerate(pbar, start=1):
                shard_started = time.perf_counter()
                rows_before = row_count
                read_before = data_store.read_seconds if data_store is not None else 0.0
                conversion_before = (
                    data_store.conversion_seconds + data_store.transform_seconds
                    if data_store is not None
                    else 0.0
                )
                for X_batch, eX_batch in self._iter_pretrain_batches(
                    key, mini_batch, shuffle=False
                ):
                    # Apply masking to validation data
                    X_masked, mask, nanmask = self._apply_mask(X_batch)

                    micro_batch = min(
                        mini_batch,
                        getattr(self, "_pretrain_micro_batch_size", None) or mini_batch,
                    )
                    for start in range(0, len(X_batch), micro_batch):
                        stop = min(start + micro_batch, len(X_batch))
                        X_reconstructed, _ = self.model(X_masked[start:stop])
                        reconstruction_mask = (
                            mask[start:stop, : -self.diff]
                            & nanmask[start:stop, : -self.diff]
                        )
                        logvar = getattr(self.model, "_last_logvar", None)
                        batch_loss = self.loss_fn(
                            X_batch[start:stop, : -self.diff],
                            X_reconstructed,
                            reconstruction_mask,
                            1.0
                            / (
                                (
                                    eX_batch[start:stop, : -self.diff]
                                    * eX_batch[start:stop, : -self.diff]
                                )
                                + 1e-8
                            ),
                            logvar=logvar,
                        )
                        loss_sum.add_(batch_loss * (stop - start))
                        row_count += stop - start
                self._write_progress(
                    "validation_shard_finished",
                    epoch=getattr(self, "_active_epoch", None),
                    key=key,
                    shard_index=shard_index,
                    shard_count=n_keys,
                    rows_completed=row_count - rows_before,
                    rows_total=(
                        data_store._row_counts.get(key, row_count - rows_before)
                        if data_store is not None
                        else row_count - rows_before
                    ),
                    wall_seconds=round(time.perf_counter() - shard_started, 3),
                    source_read_seconds=round(data_store.read_seconds - read_before, 3)
                    if data_store is not None
                    else None,
                    conversion_seconds=round(
                        data_store.conversion_seconds
                        + data_store.transform_seconds
                        - conversion_before,
                        3,
                    )
                    if data_store is not None
                    else None,
                )

            if not row_count:
                raise ValueError("Validation shards contain no rows")
            val_loss = float((loss_sum / row_count).item())
            print(f"Validation Loss: {val_loss}")
            self._write_progress(
                "validation_finished",
                epoch=getattr(self, "_active_epoch", None),
                rows_completed=row_count,
                val_loss=round(val_loss, 8),
            )
            return val_loss

    def _setup_finetune_optimizer(
        self, linearprobe, ftopt, ftlr, ftl2, enc_lr, head_lambda, encoder_lambda
    ):
        if linearprobe:
            for p in self.model.parameters():
                p.requires_grad = False
            if ftopt == "adam":
                optimizer = optim.Adam(self.lp.parameters(), lr=ftlr, weight_decay=ftl2)
            elif ftopt == "sgd":
                optimizer = optim.SGD(
                    self.lp.parameters(), lr=ftlr, momentum=0.9, weight_decay=ftl2
                )
            elif ftopt == "adamw":
                optimizer = optim.AdamW(
                    self.lp.parameters(), lr=ftlr, weight_decay=ftl2
                )
            else:
                raise ValueError(f"Unknown ftopt {ftopt!r}")
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=head_lambda)
        else:
            if ftopt == "adam":
                optimizer = optim.Adam(
                    [
                        {"params": self.model.parameters(), "lr": enc_lr},
                        {
                            "params": self.ft.parameters(),
                            "lr": ftlr,
                            "weight_decay": ftl2,
                        },
                    ]
                )
            elif ftopt == "sgd":
                optimizer = optim.SGD(
                    [
                        {"params": self.model.parameters(), "lr": enc_lr},
                        {
                            "params": self.ft.parameters(),
                            "lr": ftlr,
                            "momentum": 0.9,
                            "weight_decay": ftl2,
                        },
                    ]
                )
            elif ftopt == "adamw":
                optimizer = optim.AdamW(
                    [
                        {"params": self.model.parameters(), "lr": enc_lr},
                        {
                            "params": self.ft.parameters(),
                            "lr": ftlr,
                            "weight_decay": ftl2,
                        },
                    ]
                )
            else:
                raise ValueError(f"Unknown ftopt {ftopt!r}")
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[encoder_lambda, head_lambda]
            )
        return optimizer, scheduler

    def _setup_finetune_criteria(self, ftlf, rncloss):
        criterion, criterion2, rnc = None, None, None
        if ftlf in ("wmse", "wgnll"):
            criterion = WeightedMaskedMSELoss()
        elif ftlf == "mse":
            criterion = MaskedMSELoss()
        elif ftlf == "mae":
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = RnCLoss(temperature=2, label_diff="l1", feature_sim="l2")

        if ftlf in ("gnll", "wgnll"):
            criterion2 = MaskedGaussianNLLLoss()

        return criterion, criterion2, rnc

    def _apply_batch_masking(self, X_batch, eX_batch, ctx: FinetuneContext):
        if ctx.maskft and ctx.pert_features:
            return self._apply_mask(X_batch + self._pert_noise(X_batch, eX_batch))
        elif ctx.pert_features and not ctx.maskft:
            X_masked = X_batch + self._pert_noise(X_batch, eX_batch)
            mask = torch.zeros_like(X_batch, dtype=torch.bool, device=X_batch.device)
            return X_masked, mask, ~torch.isnan(X_batch)
        elif ctx.maskft and not ctx.pert_features:
            return self._apply_mask(X_batch)
        else:
            mask = torch.zeros_like(X_batch, dtype=torch.bool, device=X_batch.device)
            return X_batch.clone(), mask, ~torch.isnan(X_batch)

    def _forward_pass(self, X_masked, linearprobe):
        encoded = self.model.encoder(X_masked)
        return self.lp(encoded) if linearprobe else self.ft(encoded), encoded

    def _apply_parallax_mask(self, X_masked, parallax_feature_idx):
        parallax_masked = X_masked.clone()
        parallax_masked[:, parallax_feature_idx] = -9999
        return parallax_masked

    @property
    def _quantiles_tensor(self):
        # ⚡ Bolt: Cache static tensor to prevent repetitive host-to-device transfers and CPU-GPU syncs
        if not hasattr(self, "_quantiles_cached"):
            self._quantiles_cached = torch.tensor([0.16, 0.5, 0.84], device=self.device)
        return self._quantiles_cached

    def _compute_base_loss(self, y_batch, y_head, batch, ctx: FinetuneContext):
        if ctx.ftlf in ("wmse", "wgnll"):
            # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
            return ctx.criterion(
                y_batch, y_head, 1 / ((batch[3] + 1e-5) * (batch[3] + 1e-5))
            )
        elif ctx.ftlf in ("mse", "mae"):
            return ctx.criterion(y_batch, y_head)
        elif ctx.ftlf == "quantile":
            quantiles = self._quantiles_tensor
            sw = (
                _sigma_pinball_weights(
                    batch[3],
                    y_batch,
                    ctx.ft_sigma_weight_floor,
                    ctx.ft_sigma_weight_max,
                    ctx.ft_sigma_weight_normalize_batch,
                )
                if ctx.ft_use_sigma_quantile_weights
                else None
            )
            return quantile_loss(
                y_head, y_batch, quantiles, ctx.q_weight_t, sample_weight=sw
            )
        return 0

    def _compute_parallax_mle(
        self, y_raw, y_head, X_batch, eX_batch, p_idx, ctx: FinetuneContext
    ):
        pi_gaia = (
            ctx.m_consistency * X_batch[:, self.parallax_feature_idx]
            + ctx.c_consistency
        )
        sigma_gaia = (
            ctx.m_consistency
            * eX_batch[:, self.parallax_feature_idx]
            * ctx.parallax_sigma_scale
        )

        if y_raw.dim() == 3:
            mu_phot = y_head[:, p_idx, 1]
            sigma_phot = 0.5 * (y_head[:, p_idx, 2] - y_head[:, p_idx, 0])
        else:
            mu_phot = y_head[:, p_idx]
            sigma_phot = None

        var = (
            # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
            sigma_gaia * sigma_gaia
            + (sigma_phot * sigma_phot if sigma_phot is not None else 0)
            + (
                (ctx.parallax_sigma_floor * ctx.parallax_sigma_floor)
                if ctx.parallax_sigma_floor > 0
                else 0
            )
        )

        mle_mask = (
            (~torch.isnan(mu_phot)) & (~torch.isnan(pi_gaia)) & (~torch.isnan(var))
        )

        # ⚡ Bolt: Replace dynamic boolean indexing with nan_to_num and masked_fill to avoid CPU-GPU syncs
        safe_mu_phot = torch.nan_to_num(mu_phot, nan=0.0)
        safe_pi_gaia = torch.nan_to_num(pi_gaia, nan=0.0)
        safe_var = torch.nan_to_num(var, nan=1.0)

        # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
        diff_mle = safe_mu_phot - safe_pi_gaia
        return ((diff_mle * diff_mle) / (safe_var + 1e-8)).masked_fill(
            ~mle_mask, 0.0
        ).sum() / mle_mask.sum().to(dtype=diff_mle.dtype).clamp_min(1.0)

    def _apply_parallax_masked_forward(
        self, X_masked, y_batch, y_raw, ctx: FinetuneContext
    ):
        if ctx.parallax_use_masked_pred and self.parallax_feature_idx is not None:
            p_idx = (
                ctx.parallax_label_idx
                if ctx.parallax_label_idx is not None
                else y_batch.shape[1] - 1
            )
            parallax_masked = self._apply_parallax_mask(
                X_masked, self.parallax_feature_idx
            )
            y_raw_masked, _ = self._forward_pass(parallax_masked, ctx.linearprobe)
            if y_raw.dim() == 3:
                y_new = y_raw.clone()
                y_new[:, p_idx, :] = y_raw_masked[:, p_idx, :]
                y_raw = y_new
            else:
                y_new = y_raw.clone()
                y_new[:, p_idx] = y_raw_masked[:, p_idx]
                y_raw = y_new
        return y_raw

    def _compute_finetune_batch_loss(self, batch, ctx: FinetuneContext):
        X_batch, eX_batch, y_batch, e_y_batch = batch

        X_masked, mask, nanmask = self._apply_batch_masking(X_batch, eX_batch, ctx)

        if ctx.pert_labels:
            y_batch = (
                y_batch + torch.randn_like(y_batch, device=y_batch.device) * e_y_batch
            )

        # When multitask, run full autoencoder once and reuse encoded for the head
        if ctx.multitask:
            X_reconstructed, encoded = self.model(X_masked)
            y_raw = self.lp(encoded) if ctx.linearprobe else self.ft(encoded)
        else:
            y_raw, encoded = self._forward_pass(X_masked, ctx.linearprobe)
            X_reconstructed = None

        y_raw = self._apply_parallax_masked_forward(X_masked, y_batch, y_raw, ctx)

        if ctx.ftlf == "quantile":
            y_head, y_pred_err = y_raw, None
        else:
            y_head, y_pred_err = _reduce_finetune_prediction(
                y_raw, ctx.ftlf, ctx.linearprobe
            )

        loss = self._compute_base_loss(y_batch, y_head, batch, ctx)

        if (
            ctx.parallax_mle_weight > 0
            and self.parallax_feature_idx is not None
            and ctx.m_consistency is not None
        ):
            p_idx = (
                ctx.parallax_label_idx
                if ctx.parallax_label_idx is not None
                else y_batch.shape[1] - 1
            )
            loss += ctx.parallax_mle_weight * self._compute_parallax_mle(
                y_raw, y_head, X_batch, eX_batch, p_idx, ctx
            )

        if ctx.multitask:
            reconstruction_mask = mask[:, : -self.diff] & nanmask[:, : -self.diff]
            # ⚡ Bolt: Replace ** 2 with explicit multiplication for faster execution
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

        if ctx.rncloss:
            try:
                X_m_2, _, _ = self._apply_batch_masking(X_batch, eX_batch, ctx)
                _, encoded_2 = self._forward_pass(X_m_2, False)
                loss += ctx.rnc(torch.stack((encoded, encoded_2), dim=1), y_batch)
            except RuntimeError as e:
                print(e)

        if ctx.ftlf in ("gnll", "wgnll"):
            if y_pred_err is None:
                raise RuntimeError(
                    "Gaussian NLL path requires a (mean, logvar) tuple head; not supported for quantile head"
                )
            loss += ctx.criterion2(
                y_head, y_batch, y_pred_err, torch.ones_like(e_y_batch)
            )

        return loss

    def _check_linearprobe_compatibility(self, linearprobe, ftlf, multitask, rncloss):
        if linearprobe:
            if ftlf == "quantile":
                raise ValueError(
                    "linearprobe requires finetuning lf 'mse' or 'mae', not 'quantile'"
                )
            if ftlf in ("gnll", "wgnll", "wmse"):
                raise ValueError(f"linearprobe does not support loss type {ftlf!r}")
            if multitask:
                raise ValueError("linearprobe with multitask is unsupported")
            if rncloss:
                raise ValueError("linearprobe with rncloss is unsupported")

    def _init_finetune_head(self, linearprobe, ftlabeldim, ftact):
        if ftact == "relu":
            ftactivationfunc = nn.ReLU()
        elif ftact == "elu":
            ftactivationfunc = nn.ELU()
        elif ftact == "gelu":
            ftactivationfunc = nn.GELU()
        else:
            raise ValueError(
                f"Unknown ftact {ftact!r}; expected 'relu', 'elu', or 'gelu'"
            )

        self.lp = None
        if linearprobe:
            self.lp = nn.Linear(self.latent_size, ftlabeldim).to(self.device)
            nn.init.xavier_uniform_(self.lp.weight)
            nn.init.zeros_(self.lp.bias)
            self.ft = None
        else:
            self.ft = PredictionHead(self.latent_size, ftlabeldim, ftactivationfunc).to(
                self.device
            )

    def _load_finetune_checkpoint(self, ensemblepath, linearprobe):
        self._finetune_resume_state = None
        if not ensemblepath:
            if not linearprobe:
                self.ft.apply(self.init_weights_gelu)
            return

        try:
            state_dict = torch_load_finetune_checkpoint(
                ensemblepath, map_location=self.device
            )
        except FileNotFoundError as e:
            print(
                f"Fine-tune checkpoint not found ({e}); using the loaded encoder "
                "and a fresh head"
            )
            if not linearprobe:
                self.ft.apply(self.init_weights_gelu)
            return

        self._finetune_resume_state = state_dict
        if "linear_probe" in state_dict and bool(state_dict["linear_probe"]) != bool(
            linearprobe
        ):
            raise ValueError(
                "Fine-tune checkpoint linear_probe setting does not match this run"
            )

        if "autoencoder_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["autoencoder_state_dict"])
            if linearprobe and "prediction_head_state_dict" in state_dict:
                self.lp.load_state_dict(state_dict["prediction_head_state_dict"])
                print("Loaded fine-tune checkpoint")
            elif not linearprobe and "prediction_head_state_dict" in state_dict:
                self.ft.load_state_dict(state_dict["prediction_head_state_dict"])
                print("Loaded fine-tune checkpoint")
            elif not linearprobe:
                self.ft.apply(self.init_weights_gelu)
                print("Loaded encoder checkpoint; initialized a fresh fine-tune head")
        elif "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
            if not linearprobe:
                self.ft.apply(self.init_weights_gelu)
            print("Loaded pretraining checkpoint; initialized a fresh fine-tune head")
        else:
            raise ValueError(
                f"Unsupported fine-tune checkpoint format at {ensemblepath!r}"
            )

    def _restore_finetune_training_state(self, optimizer, scheduler) -> int:
        state = getattr(self, "_finetune_resume_state", None) or {}
        if "optimizer_state_dict" not in state or "scheduler_state_dict" not in state:
            return 0
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if "rng_state" in state:
            _restore_rng_state(state["rng_state"])
        return int(state.get("epoch", 0))

    def _build_finetune_context(
        self,
        linearprobe,
        maskft,
        multitask,
        ftlf,
        rncloss,
        pert_features,
        pert_labels,
        parallax_use_masked_pred,
        parallax_label_idx,
        ft_use_sigma_quantile_weights,
        ft_sigma_weight_floor,
        ft_sigma_weight_max,
        ft_sigma_weight_normalize_batch,
        ft_quantile_label_weights,
        parallax_mle_weight,
        consistency_params,
        parallax_sigma_scale,
        parallax_sigma_floor,
        ft_lambda_pred,
        ft_lambda_rec,
    ) -> FinetuneContext:
        criterion, criterion2, rnc = self._setup_finetune_criteria(ftlf, rncloss)
        consistency_params = consistency_params or {}
        m_consistency = (
            torch.tensor(consistency_params["m"], device=self.device)
            if parallax_mle_weight > 0 and "m" in consistency_params
            else None
        )
        c_consistency = (
            torch.tensor(consistency_params["c"], device=self.device)
            if parallax_mle_weight > 0 and "c" in consistency_params
            else None
        )
        q_weight_t = (
            torch.tensor(
                ft_quantile_label_weights, dtype=torch.float32, device=self.device
            )
            if ft_quantile_label_weights is not None
            else None
        )

        return FinetuneContext(
            linearprobe=linearprobe,
            maskft=maskft,
            multitask=multitask,
            ftlf=ftlf,
            rncloss=rncloss,
            pert_features=pert_features,
            pert_labels=pert_labels,
            parallax_use_masked_pred=parallax_use_masked_pred,
            parallax_label_idx=parallax_label_idx,
            ft_use_sigma_quantile_weights=ft_use_sigma_quantile_weights,
            ft_sigma_weight_floor=ft_sigma_weight_floor,
            ft_sigma_weight_max=ft_sigma_weight_max,
            ft_sigma_weight_normalize_batch=ft_sigma_weight_normalize_batch,
            q_weight_t=q_weight_t,
            criterion=criterion,
            criterion2=criterion2,
            rnc=rnc,
            parallax_mle_weight=parallax_mle_weight,
            m_consistency=m_consistency,
            c_consistency=c_consistency,
            parallax_sigma_scale=parallax_sigma_scale,
            parallax_sigma_floor=parallax_sigma_floor,
            ft_lambda_pred=ft_lambda_pred,
            ft_lambda_rec=ft_lambda_rec,
        )

    def _prepare_finetune_loader(
        self, X_train, eX_train, y_train, e_y_train, mini_batch
    ):
        tensors = [
            torch.as_tensor(arr, device=self.device, dtype=torch.float32)
            for arr in (X_train, eX_train, y_train, e_y_train)
        ]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=mini_batch, shuffle=True)

    def _configure_finetune_logging(self) -> None:
        log_dir = os.path.dirname(self.ft_log_file) or "."
        os.makedirs(log_dir, exist_ok=True)
        if save_dir := os.path.dirname(self.ft_save_str):
            os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(
            filename=self.ft_log_file,
            level=logging.INFO,
            format="%(asctime)s - Sub-Epoch: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode="a",
            force=True,
        )

    def _set_finetune_train_mode(self, linearprobe: bool) -> None:
        if linearprobe:
            self.model.eval()
            self.lp.train()
        else:
            self.model.train()
            self.ft.train()

    def _finetune_parameters(self, linearprobe: bool):
        if linearprobe:
            return list(self.lp.parameters())
        return list(self.model.parameters()) + list(self.ft.parameters())

    def _save_finetune_checkpoint(
        self, linearprobe: bool, epoch: int, optimizer, scheduler
    ) -> None:
        head_sd = self.lp.state_dict() if linearprobe else self.ft.state_dict()
        payload = {
            "autoencoder_state_dict": self.model.state_dict(),
            "prediction_head_state_dict": head_sd,
            "linear_probe": bool(linearprobe),
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng_state(),
            "featurescaler": self.featurescaler,
            "label_scalers": getattr(self, "label_scalers", None),
        }
        _atomic_torch_save(payload, self.ft_save_str)
        if (
            self.checkpoint_interval is not None
            and (epoch + 1) % self.checkpoint_interval == 0
        ):
            interval_path = (
                f"{os.path.splitext(self.ft_save_str)[0]}_checkpoint_{epoch + 1}.pth"
            )
            _atomic_torch_save(payload, interval_path)

    def _run_finetune_epoch(
        self, train_loader, optimizer, scheduler, ctx, linearprobe, epoch, num_epochs
    ):
        self._set_finetune_train_mode(linearprobe)
        epoch_loss = 0.0
        for batch in train_loader:
            loss = self._compute_finetune_batch_loss(batch, ctx)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._finetune_parameters(linearprobe), max_norm=1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        mean_loss = epoch_loss / len(train_loader)
        print(f"Training Epoch [{epoch + 1}/{num_epochs}], Loss: {mean_loss}")
        logging.info(f"Training Loss: {mean_loss}")
        return mean_loss

    def _maybe_validate_finetune(
        self,
        *,
        X_val,
        eX_val,
        y_val,
        e_y_val,
        mini_batch,
        linearprobe,
        maskft,
        multitask,
        ftlf,
        rncloss,
        ftlabeldim,
        ft_lambda_pred,
        ft_lambda_rec,
        ft_quantile_label_weights,
        ft_use_sigma_quantile_weights,
        ft_sigma_weight_floor,
        ft_sigma_weight_normalize_batch,
        parallax_mle_weight,
        parallax_use_masked_pred,
        parallax_label_idx,
        parallax_sigma_floor,
        parallax_sigma_scale,
        consistency_params,
    ):
        if X_val is None or y_val is None:
            return
        validation_loss = self.validate_fit(
            X_val,
            eX_val,
            y_val,
            e_y_val=e_y_val,
            mini_batch=mini_batch,
            linearprobe=linearprobe,
            maskft=maskft,
            multitask=multitask,
            ftlf=ftlf,
            rncloss=rncloss,
            ftlabeldim=ftlabeldim,
            ft_lambda_pred=ft_lambda_pred,
            ft_lambda_rec=ft_lambda_rec,
            ft_quantile_label_weights=ft_quantile_label_weights,
            ft_use_sigma_quantile_weights=ft_use_sigma_quantile_weights,
            ft_sigma_weight_floor=ft_sigma_weight_floor,
            ft_sigma_weight_normalize_batch=ft_sigma_weight_normalize_batch,
            parallax_mle_weight=parallax_mle_weight,
            parallax_use_masked_pred=parallax_use_masked_pred,
            parallax_label_idx=parallax_label_idx,
            parallax_sigma_floor=parallax_sigma_floor,
            parallax_sigma_scale=parallax_sigma_scale,
            consistency_params=consistency_params,
        )
        logging.info(f"Validation Loss: {validation_loss}")

    def fit(
        self,
        X_train,
        eX_train,
        y_train,
        e_y_train=None,
        X_val=None,
        eX_val=None,
        y_val=None,
        e_y_val=None,
        num_epochs=10,
        mini_batch=32,
        linearprobe=False,
        maskft=False,
        multitask=False,
        rncloss=False,
        last=False,
        ftlr=1e-3,
        ftopt="adam",
        ftact="relu",
        ftl2=0.0,
        ftlf="mse",
        ftdim="1layer512",
        ftlabeldim=5,
        pt_epoch=0,
        pert_features=False,
        pert_labels=False,
        feature_seed=42,
        ensemblepath=None,
        ft_lambda_pred=0.8,
        ft_lambda_rec=0.2,
        ft_quantile_label_weights: list | None = None,
        ft_use_sigma_quantile_weights: bool = False,
        ft_sigma_weight_floor: float = 1e-6,
        ft_sigma_weight_max: float = 1e6,
        ft_sigma_weight_normalize_batch: bool = True,
        ft_encoder_lr: float | None = None,
        ft_scheduler_encoder_decay: float = 0.95,
        ft_scheduler_head_decay: float = 0.5,
        ft_scheduler_head_step_epochs: int = 10,
        parallax_mle_weight: float = 0.0,
        parallax_use_masked_pred: bool = False,
        parallax_label_idx: int | None = None,
        parallax_sigma_floor: float = 0.0,
        parallax_sigma_scale: float = 1.0,
        consistency_params: dict | None = None,
        ft_encoder_warmup_epochs: int = 0,
        resume_training: bool = True,
    ):
        train_loader = self._prepare_finetune_loader(
            X_train, eX_train, y_train, e_y_train, mini_batch
        )

        self._check_linearprobe_compatibility(linearprobe, ftlf, multitask, rncloss)
        self._init_finetune_head(linearprobe, ftlabeldim, ftact)
        self._load_finetune_checkpoint(ensemblepath, linearprobe)

        ctx = self._build_finetune_context(
            linearprobe,
            maskft,
            multitask,
            ftlf,
            rncloss,
            pert_features,
            pert_labels,
            parallax_use_masked_pred,
            parallax_label_idx,
            ft_use_sigma_quantile_weights,
            ft_sigma_weight_floor,
            ft_sigma_weight_max,
            ft_sigma_weight_normalize_batch,
            ft_quantile_label_weights,
            parallax_mle_weight,
            consistency_params,
            parallax_sigma_scale,
            parallax_sigma_floor,
            ft_lambda_pred,
            ft_lambda_rec,
        )

        enc_lr = float(ft_encoder_lr) if ft_encoder_lr is not None else float(self.lr)
        head_step = max(1, int(ft_scheduler_head_step_epochs))
        head_lambda = lambda epoch, h=ft_scheduler_head_decay, s=head_step: (
            h ** (epoch // s)
        )
        encoder_lambda = lambda epoch, b=ft_scheduler_encoder_decay: b**epoch

        optimizer, scheduler = self._setup_finetune_optimizer(
            linearprobe, ftopt, ftlr, ftl2, enc_lr, head_lambda, encoder_lambda
        )

        self._configure_finetune_logging()

        if pert_features or pert_labels:
            random.seed(feature_seed)
            torch.manual_seed(feature_seed)

        start_epoch = (
            self._restore_finetune_training_state(optimizer, scheduler)
            if resume_training
            else 0
        )

        # Encoder warmup: freeze encoder for first N epochs, then unfreeze
        if ft_encoder_warmup_epochs > start_epoch and not linearprobe:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            print(f"Encoder frozen for warmup ({ft_encoder_warmup_epochs} epochs)")

        for epoch in range(start_epoch, num_epochs):
            # Unfreeze encoder at warmup boundary and rebuild optimizer
            if (
                ft_encoder_warmup_epochs > 0
                and not linearprobe
                and epoch == ft_encoder_warmup_epochs
            ):
                for p in self.model.encoder.parameters():
                    p.requires_grad = True
                optimizer, scheduler = self._setup_finetune_optimizer(
                    linearprobe, ftopt, ftlr, ftl2, enc_lr, head_lambda, encoder_lambda
                )
                print("Encoder unfrozen after warmup, optimizer rebuilt")
            self._run_finetune_epoch(
                train_loader, optimizer, scheduler, ctx, linearprobe, epoch, num_epochs
            )
            self._maybe_validate_finetune(
                X_val=X_val,
                eX_val=eX_val,
                y_val=y_val,
                e_y_val=e_y_val,
                mini_batch=mini_batch,
                linearprobe=linearprobe,
                maskft=maskft,
                multitask=multitask,
                ftlf=ftlf,
                rncloss=rncloss,
                ftlabeldim=ftlabeldim,
                ft_lambda_pred=ft_lambda_pred,
                ft_lambda_rec=ft_lambda_rec,
                ft_quantile_label_weights=ft_quantile_label_weights,
                ft_use_sigma_quantile_weights=ft_use_sigma_quantile_weights,
                ft_sigma_weight_floor=ft_sigma_weight_floor,
                ft_sigma_weight_normalize_batch=ft_sigma_weight_normalize_batch,
                parallax_mle_weight=parallax_mle_weight,
                parallax_use_masked_pred=parallax_use_masked_pred,
                parallax_label_idx=parallax_label_idx,
                parallax_sigma_floor=parallax_sigma_floor,
                parallax_sigma_scale=parallax_sigma_scale,
                consistency_params=consistency_params,
            )
            self._save_finetune_checkpoint(linearprobe, epoch, optimizer, scheduler)

    def validate_fit(
        self,
        X_val,
        eX_val,
        y_val,
        e_y_val=None,
        mini_batch=32,
        linearprobe=False,
        maskft=False,
        multitask=False,
        ftlf="mse",
        rncloss=False,
        ftlabeldim=5,
        ft_lambda_pred=0.8,
        ft_lambda_rec=0.2,
        ft_quantile_label_weights: list | None = None,
        ft_use_sigma_quantile_weights: bool = False,
        ft_sigma_weight_floor: float = 1e-6,
        ft_sigma_weight_max: float = 1e6,
        ft_sigma_weight_normalize_batch: bool = True,
        parallax_mle_weight: float = 0.0,
        parallax_use_masked_pred: bool = False,
        parallax_label_idx: int | None = None,
        parallax_sigma_floor: float = 0.0,
        parallax_sigma_scale: float = 1.0,
        consistency_params: dict | None = None,
    ):
        self.model.eval()
        if linearprobe:
            self.lp.eval()
        else:
            self.ft.eval()

        val_loss = 0
        X_val, eX_val = (
            torch.as_tensor(X_val, device=self.device, dtype=torch.float32),
            torch.as_tensor(eX_val, device=self.device, dtype=torch.float32),
        )
        y_val, e_y_val = (
            torch.as_tensor(y_val, device=self.device, dtype=torch.float32),
            torch.as_tensor(e_y_val, device=self.device, dtype=torch.float32),
        )
        rdataset = TensorDataset(X_val, eX_val, y_val, e_y_val)
        val_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=False)

        ctx = self._build_finetune_context(
            linearprobe,
            maskft,
            multitask,
            ftlf,
            rncloss,
            False,
            False,
            parallax_use_masked_pred,
            parallax_label_idx,
            ft_use_sigma_quantile_weights,
            ft_sigma_weight_floor,
            ft_sigma_weight_max,
            ft_sigma_weight_normalize_batch,
            ft_quantile_label_weights,
            parallax_mle_weight,
            consistency_params,
            parallax_sigma_scale,
            parallax_sigma_floor,
            ft_lambda_pred,
            ft_lambda_rec,
        )

        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_finetune_batch_loss(batch, ctx)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")
        return val_loss / len(val_loader)


def make_model(
    input_dim,
    layer_dims,
    output_dim,
    active,
    rtdl_embed_dim,
    norm,
    decoder_dims=None,
    encoder_type="resnet",
    growth_rate=64,
    num_dense_layers=8,
    cosine_latent=False,
    heteroscedastic=False,
):
    """
    Helper function to make the MSA in the same file as the wrapper

    input_dim :: int
        length of the input features including positional information not reconstructed.
    layer_dims :: list
        Residual block dimensions. The list is discretized, being the specific widths for each individual layer.
    output_dim :: int
        Length of the output features, those features that are reconstructed.
    active :: string
        String of the possible activation functions. Must be one of ('elu', 'relu', or 'gelu').
    rtdl_embed_dim :: int
        Embedding dimension the input data is blown up to.
    norm :: string
        String of the possible normalization options. Must be one of ('layer', or 'batch')
    decoder_dims :: list, optional
        Decoder dimensions. If None, uses symmetric (mirrored) encoder dimensions.
        For asymmetric decoder, specify custom dimensions (e.g., [256, 512, 1024])
    encoder_type : str
        'resnet' (default) or 'dense' for DenseNet with concatenation skip connections.
    growth_rate : int
        DenseNet growth rate (only used when encoder_type='dense').
    num_dense_layers : int
        Number of dense layers per block (only used when encoder_type='dense').
    cosine_latent : bool
        L2-normalize the latent space.
    heteroscedastic : bool
        Decoder outputs (mean, logvar) for gnll pretraining loss.
    """
    latent_size = layer_dims[-1]

    if encoder_type == "dense":
        model = TabDenseNet(
            continuous_cols=input_dim,
            latent_size=latent_size,
            output_cols=output_dim,
            growth_rate=growth_rate,
            num_layers=num_dense_layers,
            d_embedding=rtdl_embed_dim,
            active=active,
            norm=norm,
            cosine_latent=cosine_latent,
            heteroscedastic=heteroscedastic,
        )
    elif encoder_type == "resnet":
        model = TabResnet(
            continuous_cols=input_dim,
            blocks_dims=layer_dims,
            output_cols=output_dim,
            active=active,
            d_embedding=rtdl_embed_dim,
            norm=norm,
            decoder_dims=decoder_dims,
            cosine_latent=cosine_latent,
            heteroscedastic=heteroscedastic,
        )
    else:
        raise ValueError(
            f"Unknown encoder_type {encoder_type!r}; expected 'resnet' or 'dense'"
        )

    return model
