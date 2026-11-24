"""Fine-tune config keys that were silently ignored (training/finetune_settings)."""

from pathlib import Path

import pytest
import torch
import yaml

from masked_stellar_autoencoder.models.model import TabResnetWrapper
from masked_stellar_autoencoder.training.finetune_settings import (
    effective_finetune_settings,
    finetune_encoder_weight_decay,
    finetune_masking_ratios,
    inert_finetune_keys,
    log_effective_settings,
)

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "configs" / "finetune-multisurvey-20261121-v10.yaml"


def test_v10_masking_ratios_come_from_training_section():
    config = yaml.safe_load(V10.read_text())
    m = finetune_masking_ratios(config)
    # v10 ran 0.3 / 0.3 because finetune_msa read finetuning: only
    assert (m["xp_masking_ratio"], m["m_masking_ratio"]) == (0.9, 0.6)
    assert m["source"] == {
        "xp_masking_ratio": "training",
        "m_masking_ratio": "training",
    }


def test_v10_encoder_weight_decay_and_inert_keys():
    config = yaml.safe_load(V10.read_text())
    assert finetune_encoder_weight_decay(config) == (1e-5, "training.weight_decay")
    inert = inert_finetune_keys(config)
    for key in (
        "training.lasso",
        "training.optimizer",
        "training.mini_batch_size",
        "training.epochs",
        "data.valid_keys",
        "finetuning.lf",
        "finetuning.rncloss",
        "finetuning.pert_labels",
        "finetuning.quantile_label_weights",
        "finetuning.ensemble_seed",
        "finetuning.ensemble_size",
    ):
        assert key in inert
    assert "training.xp_masking_ratio" not in inert
    assert "training.weight_decay" not in inert


def test_disagreeing_locations_fail_loudly():
    config = {
        "training": {"xp_masking_ratio": 0.9, "m_masking_ratio": 0.6},
        "finetuning": {"xp_masking_ratio": 0.3},
    }
    with pytest.raises(ValueError, match="disagrees"):
        finetune_masking_ratios(config)


def test_agreeing_locations_and_finetuning_only():
    both = {
        "training": {"xp_masking_ratio": 0.5, "m_masking_ratio": 0.2},
        "finetuning": {"xp_masking_ratio": 0.5, "m_masking_ratio": 0.2},
    }
    assert finetune_masking_ratios(both)["source"]["xp_masking_ratio"] == "finetuning"
    ft = {"finetuning": {"xp_masking_ratio": 0.7, "m_masking_ratio": 0.1}}
    m = finetune_masking_ratios(ft)
    assert (m["xp_masking_ratio"], m["m_masking_ratio"]) == (0.7, 0.1)


def test_missing_ratios_warn_and_use_legacy_default():
    with pytest.warns(UserWarning, match="legacy default"):
        m = finetune_masking_ratios({"training": {}, "finetuning": {}})
    assert (m["xp_masking_ratio"], m["m_masking_ratio"]) == (0.3, 0.3)


def test_out_of_range_ratio_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        finetune_masking_ratios(
            {"finetuning": {"xp_masking_ratio": 1.5, "m_masking_ratio": 0}}
        )


def test_every_finetune_config_resolves():
    paths = sorted((ROOT / "configs").glob("finetune*.yaml"))
    paths += sorted((ROOT / "configs" / "ablation-20261118-v7").glob("*.yaml"))
    assert paths
    for path in paths:
        config = yaml.safe_load(path.read_text())
        if "finetuning" not in config:
            continue
        effective_finetune_settings(config)  # raises on a disagreement


def test_effective_settings_are_logged(tmp_path, capsys):
    config = yaml.safe_load(V10.read_text())
    settings = effective_finetune_settings(config)
    log_effective_settings(settings, str(tmp_path / "finetune.log"))
    out = capsys.readouterr().out
    assert "xp_masking_ratio=0.9 (from training)" in out
    assert "encoder_weight_decay=1e-05" in out
    assert (tmp_path / "finetune_effective_settings.json").exists()


@pytest.mark.parametrize(
    ("ftopt", "configured", "expected"),
    [
        ("adamw", 1e-5, 1e-5),
        ("adamw", None, 0.01),
        ("adam", 3e-4, 3e-4),
        ("sgd", None, 0.0),
    ],
)
def test_finetune_optimizer_applies_encoder_weight_decay(ftopt, configured, expected):
    wrapper = TabResnetWrapper.__new__(TabResnetWrapper)
    wrapper.model = torch.nn.Linear(2, 2)
    wrapper.ft = torch.nn.Linear(2, 1)
    if configured is not None:
        wrapper.ft_encoder_weight_decay = configured
    optimizer, _ = wrapper._setup_finetune_optimizer(
        False, ftopt, 1e-4, 0.0, 1e-5, lambda e: 1.0, lambda e: 1.0
    )
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(expected)
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
