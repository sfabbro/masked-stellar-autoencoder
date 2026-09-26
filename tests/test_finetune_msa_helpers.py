from argparse import Namespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.preprocessing import RobustScaler

pytest.importorskip("torch")

from masked_stellar_autoencoder.models.model import TabResnetWrapper
from masked_stellar_autoencoder.training.finetune_msa import (
    _configure_pilot,
    _print_parallax_consistency,
)


def test_print_parallax_consistency_legacy(capsys):
    pack = {
        "astrometry_input_policy": "legacy_raw",
        "parallax_target_space": "linear_mas",
        "featurescaler": MagicMock(center_=[0.0, 1.0], scale_=[1.0, 2.0]),
    }
    scalers = [
        MagicMock(),
        MagicMock(mean_=np.array([[0.5]]), scale_=np.array([[2.0]])),
    ]

    _print_parallax_consistency(pack, ["G", "PARALLAX"], scalers)

    assert "Consistency Params for Parallax" in capsys.readouterr().out


def test_print_parallax_consistency_skips_non_legacy(capsys):
    pack = {
        "astrometry_input_policy": "snr",
        "parallax_target_space": "log10_mas",
        "featurescaler": MagicMock(center_=[0.0], scale_=[1.0]),
    }

    _print_parallax_consistency(pack, ["PARALLAX"], [MagicMock()])

    captured = capsys.readouterr().out
    assert "Skipping parallax feature/label consistency check" in captured


def test_pretrain_checkpoint_initializes_a_fresh_prediction_head(tmp_path, capsys):
    checkpoint_path = tmp_path / "pretrain.pth"
    torch.save({"model_state_dict": {"encoder": torch.tensor([1.0])}}, checkpoint_path)

    wrapper = TabResnetWrapper.__new__(TabResnetWrapper)
    wrapper.model = MagicMock()
    wrapper.ft = MagicMock()
    wrapper.device = torch.device("cpu")
    wrapper._load_finetune_checkpoint(str(checkpoint_path), linearprobe=False)

    wrapper.model.load_state_dict.assert_called_once()
    wrapper.ft.apply.assert_called_once_with(wrapper.init_weights_gelu)
    assert "Loaded pretraining checkpoint" in capsys.readouterr().out


def test_unrecognized_finetune_checkpoint_fails_visibly(tmp_path):
    checkpoint_path = tmp_path / "bad.pth"
    torch.save({"unknown": {}}, checkpoint_path)

    wrapper = TabResnetWrapper.__new__(TabResnetWrapper)
    wrapper.model = MagicMock()
    wrapper.ft = MagicMock()
    wrapper.device = torch.device("cpu")
    with pytest.raises(ValueError, match="Unsupported fine-tune checkpoint format"):
        wrapper._load_finetune_checkpoint(str(checkpoint_path), linearprobe=False)


def test_finetune_resume_loads_trusted_scaler_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "finetune.pth"
    encoder = torch.nn.Linear(2, 2)
    probe = torch.nn.Linear(2, 1)
    torch.save(
        {
            "autoencoder_state_dict": encoder.state_dict(),
            "prediction_head_state_dict": probe.state_dict(),
            "featurescaler": RobustScaler().fit([[0.0, 1.0], [1.0, 2.0]]),
            "label_scalers": [],
            "linear_probe": True,
        },
        checkpoint_path,
    )

    wrapper = TabResnetWrapper.__new__(TabResnetWrapper)
    wrapper.model = torch.nn.Linear(2, 2)
    wrapper.lp = torch.nn.Linear(2, 1)
    wrapper.device = torch.device("cpu")
    wrapper._load_finetune_checkpoint(str(checkpoint_path), linearprobe=True)

    for name, value in probe.state_dict().items():
        torch.testing.assert_close(wrapper.lp.state_dict()[name], value)


def test_finetune_checkpoint_restores_optimizer_scheduler_and_epoch(tmp_path):
    wrapper = TabResnetWrapper.__new__(TabResnetWrapper)
    wrapper.model = torch.nn.Linear(2, 2)
    wrapper.ft = torch.nn.Linear(2, 1)
    wrapper.lp = None
    wrapper.featurescaler = RobustScaler().fit([[0.0, 1.0], [1.0, 2.0]])
    wrapper.label_scalers = []
    wrapper.ft_save_str = str(tmp_path / "finetune.pth")
    wrapper.checkpoint_interval = None

    optimizer = torch.optim.SGD(
        list(wrapper.model.parameters()) + list(wrapper.ft.parameters()),
        lr=0.1,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss = wrapper.model.weight.sum() + wrapper.ft.weight.sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    wrapper._save_finetune_checkpoint(False, 1, optimizer, scheduler)

    resumed = TabResnetWrapper.__new__(TabResnetWrapper)
    resumed.model = torch.nn.Linear(2, 2)
    resumed.ft = torch.nn.Linear(2, 1)
    resumed.lp = None
    resumed.device = torch.device("cpu")
    resumed._load_finetune_checkpoint(wrapper.ft_save_str, linearprobe=False)
    resumed_optimizer = torch.optim.SGD(
        list(resumed.model.parameters()) + list(resumed.ft.parameters()),
        lr=0.1,
        momentum=0.9,
    )
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=1, gamma=0.5
    )

    assert (
        resumed._restore_finetune_training_state(resumed_optimizer, resumed_scheduler)
        == 2
    )
    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_finetune_pilot_caps_work_and_uses_separate_outputs():
    config = {
        "saving": {"model_str": "/tmp/model.pth", "log_file": "/tmp/train.log"},
        "finetuning": {
            "num_epochs": 100,
            "mini_batch": 4096,
            "ensemble": True,
            "ensemble_path": "/tmp/ensemble",
        },
    }
    args = Namespace(max_train_rows=1000, max_valid_rows=None)

    _configure_pilot(config, args)

    assert config["finetuning"]["num_epochs"] == 1
    assert config["finetuning"]["mini_batch"] == 256
    assert config["finetuning"]["ensemble"] is False
    assert args.max_train_rows == 1000
    assert args.max_valid_rows == 2048
    assert config["saving"]["model_str"] == "/tmp/model_pilot.pth"
    assert config["finetuning"]["ensemble_path"] == "/tmp/ensemble_pilot"
