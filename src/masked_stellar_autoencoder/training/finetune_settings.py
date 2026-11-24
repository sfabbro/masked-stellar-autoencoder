"""Effective fine-tune settings: config keys that used to be silently ignored.

Audit 2026-11-24 (``docs/programme/README.md`` section 5, item 1). The
fine-tune configs (v1-v10, the stopgap and the v7 ablation grid) set the
masking ratios under ``training:``, but ``finetune_msa`` read
``finetuning.xp_masking_ratio`` / ``m_masking_ratio`` with defaults 0.3 / 0.3:
every v10 run used 0.3 / 0.3 instead of 0.9 / 0.6. The autoencoder parameter
group of the fine-tune optimiser had no ``weight_decay`` entry either, so it
got the optimiser default (AdamW 0.01, Adam/SGD 0), not ``training.weight_decay``.

This module resolves those settings from either location, fails when two
locations disagree, and lists the keys a fine-tune config carries that the
fine-tune does not read (``inert_finetune_keys``), so they are printed with
the effective values instead of passing silently.

To reproduce a v10-era run exactly, set the values it actually used:
``finetuning.xp_masking_ratio: 0.3``, ``finetuning.m_masking_ratio: 0.3`` (and
drop the ``training:`` copies), ``finetuning.encoder_weight_decay: 0.01``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

# Defaults applied when a config sets a ratio nowhere (pre-2026-11-24 behaviour).
LEGACY_MASKING_DEFAULTS = {"xp_masking_ratio": 0.3, "m_masking_ratio": 0.3}
# AdamW's own default: what the autoencoder group got before the fix.
ADAMW_DEFAULT_WEIGHT_DECAY = 0.01

# Keys of a fine-tune config that ``finetune_msa`` never reads. The training:
# entries belong to pretraining (pretrain_msa reads them); data.datafile and
# valid_keys name the pretraining HDF5.
_PRETRAIN_ONLY = {
    "training": (
        "lasso",
        "optimizer",
        "mini_batch_size",
        "micro_batch_size",
        "epochs",
        "presaved",
        "io_cache_fraction",
        "io_cache_max_bytes",
        "io_cache_seed",
        "io_shuffle_buffer_bytes",
        "monitor_interval_rows",
        "monitor_sample_rows",
        "pert_features",
        "pert_scale",
        "pert_ebv_scale",
        # passed to the wrapper, used only by the pretraining cosine schedule
        "scheduler_cosine_t0",
        "scheduler_cosine_t_mult",
        "scheduler_eta_min_factor",
    ),
    # data.datafile is opened by the wrapper but no row of it is read
    "data": ("valid_keys",),
}
# Read by the base (quantile/MSE) fine-tune, ignored by the anchored
# multi-survey wrapper, whose loss is the Student-t joint likelihood.
_IGNORED_BY_ANCHORED = {
    "finetuning": (
        "lf",
        "rncloss",
        "pert_labels",
        "quantile_label_weights",
        "quantile_use_label_errors",
        "quantile_sigma_weight_floor",
        "quantile_sigma_weight_max",
        "quantile_sigma_weight_normalize_batch",
    ),
}


def _section(config: dict, name: str) -> dict:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def finetune_masking_ratios(config: dict) -> dict:
    """Fine-tune ``xp_masking_ratio`` and ``m_masking_ratio`` with their source.

    Accepts ``finetuning.<key>`` and ``training.<key>``. When both are set they
    must agree, otherwise ``ValueError``. When neither is set the legacy
    default 0.3 applies, with a warning.
    """
    ft, tr = _section(config, "finetuning"), _section(config, "training")
    out, source = {}, {}
    for key, default in LEGACY_MASKING_DEFAULTS.items():
        a, b = ft.get(key), tr.get(key)
        if a is not None and b is not None and float(a) != float(b):
            raise ValueError(
                f"finetuning.{key}={a} disagrees with training.{key}={b}; "
                "set the fine-tune masking ratio in one place"
            )
        if a is not None:
            out[key], source[key] = float(a), "finetuning"
        elif b is not None:
            out[key], source[key] = float(b), "training"
        else:
            warnings.warn(
                f"{key} is set neither under finetuning: nor training:; using the "
                f"legacy default {default}",
                stacklevel=2,
            )
            out[key], source[key] = float(default), "default"
        if not 0.0 <= out[key] <= 1.0:
            raise ValueError(f"{key}={out[key]} must lie in [0, 1]")
    out["source"] = source
    return out


def finetune_encoder_weight_decay(config: dict) -> tuple[float, str]:
    """Weight decay of the autoencoder group in the fine-tune optimiser.

    ``finetuning.encoder_weight_decay`` when set, else ``training.weight_decay``
    (the value the configs document). Before 2026-11-24 neither was applied.
    """
    ft, tr = _section(config, "finetuning"), _section(config, "training")
    if ft.get("encoder_weight_decay") is not None:
        return float(ft["encoder_weight_decay"]), "finetuning.encoder_weight_decay"
    if tr.get("weight_decay") is not None:
        return float(tr["weight_decay"]), "training.weight_decay"
    return ADAMW_DEFAULT_WEIGHT_DECAY, "default (AdamW)"


def inert_finetune_keys(config: dict) -> list[str]:
    """Keys present in a fine-tune config that the fine-tune does not read."""
    groups = [_PRETRAIN_ONLY]
    ms = _section(config, "multisurvey")
    if ms.get("formulation") == "anchored":
        groups.append(_IGNORED_BY_ANCHORED)
    out = []
    for group in groups:
        for section, keys in group.items():
            present = _section(config, section)
            out += [f"{section}.{k}" for k in keys if k in present]
    if not _section(config, "finetuning").get("ensemble", False):
        out += [
            f"finetuning.{k}"
            for k in ("ensemble_seed", "ensemble_size")
            if k in _section(config, "finetuning")
        ]
    return sorted(out)


def effective_finetune_settings(config: dict) -> dict:
    """Everything the audit found, as one JSON-serialisable record."""
    masking = finetune_masking_ratios(config)
    wd, wd_source = finetune_encoder_weight_decay(config)
    return {
        "xp_masking_ratio": masking["xp_masking_ratio"],
        "m_masking_ratio": masking["m_masking_ratio"],
        "masking_source": masking["source"],
        "encoder_weight_decay": wd,
        "encoder_weight_decay_source": wd_source,
        "inert_keys": inert_finetune_keys(config),
    }


def log_effective_settings(settings: dict, log_file: str | None = None) -> None:
    """Print the effective settings; also write them next to the fine-tune log."""
    print(
        "fine-tune effective settings: "
        f"xp_masking_ratio={settings['xp_masking_ratio']} "
        f"(from {settings['masking_source']['xp_masking_ratio']}), "
        f"m_masking_ratio={settings['m_masking_ratio']} "
        f"(from {settings['masking_source']['m_masking_ratio']}), "
        f"encoder_weight_decay={settings['encoder_weight_decay']} "
        f"(from {settings['encoder_weight_decay_source']})"
    )
    if settings["inert_keys"]:
        print("fine-tune config keys not read: " + ", ".join(settings["inert_keys"]))
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = path.with_name(path.stem + "_effective_settings.json")
        out.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
