import random
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch
from sklearn.preprocessing import RobustScaler

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.models.model import (
    EncoderDecoderLoss,
    TabResnetWrapper,
    _capture_rng_state,
    _restore_rng_state,
)
from masked_stellar_autoencoder.training.pretrain_msa import (
    _configure_pilot,
    _validate_error_columns,
    fit_pretrain_scaler,
)


@pytest.fixture
def wrapper_stub():
    model = MagicMock()
    model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    scaler = MagicMock()
    scaler.scale_ = [1.0]
    scaler.center_ = [0.0]
    w = TabResnetWrapper.__new__(TabResnetWrapper)
    w.model = model
    w.featurescaler = scaler
    w.feature_cols = ["a", "b"]
    w.error_cols = ["e_a", "e_b"]
    w.recon_cols = ["a"]
    w.diff = 1
    w.device = torch.device("cpu")
    w.loss_fn = EncoderDecoderLoss(lf="mae")
    w.lasso = 0.0
    w.pert_features = False
    w.pt_save_str = "model.pth"
    w.pt_log_file = "loss.log"
    w.checkpoint_interval = None
    return w


def test_pretrain_checkpoint_payload_shape(wrapper_stub):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    wrapper_stub.model.state_dict = MagicMock(return_value={"w": torch.zeros(1)})
    payload = wrapper_stub._pretrain_checkpoint_payload(
        epoch=0, optimizer=optimizer, scheduler=scheduler, epoch_loss=1.0, loss_div=2.0
    )
    assert payload["epoch"] == 1
    assert payload["epoch_loss"] == 1.0
    assert "model_state_dict" in payload
    assert "rng_state" in payload
    assert "run_signature" in payload


def test_pretrain_resume_rejects_changed_feature_order(wrapper_stub, tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": {"w": torch.zeros(1)},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch_loss": 0.0,
            "loss_div": 0.0,
            "epoch": 2,
            "run_signature": {
                "feature_cols": ["b", "a"],
                "error_cols": ["e_a", "e_b"],
                "recon_cols": ["a"],
                "scaler_center": [0.0, 0.0],
                "scaler_scale": [1.0, 1.0],
                "train_keys": ["train"],
            },
        },
        checkpoint,
    )
    wrapper_stub._pretrain_train_keys = ["train"]

    with pytest.raises(ValueError, match="feature_cols"):
        wrapper_stub._load_pretrain_resume(str(checkpoint), MagicMock(), MagicMock())


def test_unweighted_pretrain_can_resume_with_corrected_error_mapping(
    wrapper_stub, tmp_path
):
    wrapper_stub._pretrain_train_keys = ["train"]
    signature = wrapper_stub._pretrain_run_signature()
    signature["error_cols"] = ["a", "b"]
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch_loss": 0.0,
            "loss_div": 0.0,
            "epoch": 2,
            "run_signature": signature,
        },
        checkpoint,
    )

    result = wrapper_stub._load_pretrain_resume(
        str(checkpoint), MagicMock(), MagicMock()
    )

    assert result == (0.0, 0.0, 2)


def test_weighted_pretrain_rejects_changed_error_mapping(wrapper_stub, tmp_path):
    wrapper_stub._pretrain_train_keys = ["train"]
    wrapper_stub.loss_fn = EncoderDecoderLoss(lf="wmse")
    signature = wrapper_stub._pretrain_run_signature()
    signature["error_cols"] = ["a", "b"]
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch_loss": 0.0,
            "loss_div": 0.0,
            "epoch": 2,
            "run_signature": signature,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="error_cols"):
        wrapper_stub._load_pretrain_resume(str(checkpoint), MagicMock(), MagicMock())


def test_resume_runs_until_total_epoch_target(wrapper_stub):
    wrapper_stub._setup_pretrain_optimizer = MagicMock(return_value=(None, None))
    wrapper_stub._configure_pretrain_logging = MagicMock()
    wrapper_stub._load_pretrain_resume = MagicMock(return_value=(0.0, 0.0, 3))
    wrapper_stub._run_pretrain_epoch = MagicMock(return_value=(1.0, 1.0))
    wrapper_stub._chain_finetune_after_pretrain = MagicMock()

    wrapper_stub.pretrain_hdf(
        train_keys=["train"], num_epochs=5, mini_batch=1, pretrained="checkpoint.pth"
    )

    assert [
        call.args[0] for call in wrapper_stub._run_pretrain_epoch.call_args_list
    ] == [
        3,
        4,
    ]
    assert all(
        call.args[1] == 5 for call in wrapper_stub._run_pretrain_epoch.call_args_list
    )


def test_pretrain_rng_state_roundtrips_through_weights_only_checkpoint(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = _capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))

    checkpoint = tmp_path / "rng.pth"
    torch.save({"rng_state": state}, checkpoint)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    _restore_rng_state(torch_load_trusted(checkpoint)["rng_state"])

    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_pretrain_reconstruction_loss_masked_mean(wrapper_stub):
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    eX = torch.ones_like(X)
    X_reconstructed = torch.tensor([[1.5, 0.0], [3.5, 0.0]])
    z = torch.zeros(2, 4)
    mask = torch.tensor([[False, True], [False, True]])
    nanmask = torch.ones_like(X, dtype=torch.bool)
    loss = wrapper_stub._pretrain_reconstruction_loss(
        X, eX, X_reconstructed, z, mask, nanmask
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_pretrain_scaler_samples_all_training_shards(tmp_path):
    data_path = tmp_path / "shards.h5"
    dtype = np.dtype([("feature", "f4"), ("xp", "f4")])
    with h5py.File(data_path, "w") as datafile:
        datafile.create_dataset(
            "near", data=np.array([(1.0, 2.0), (2.0, 3.0)], dtype=dtype)
        )
        datafile.create_dataset(
            "far", data=np.array([(100.0, 4.0), (101.0, 5.0)], dtype=dtype)
        )
        scaler = fit_pretrain_scaler(
            datafile,
            ["near", "far"],
            ["feature", "xp"],
            {"scaler_max_rows": 10, "scaler_seed": 0},
        )

    assert isinstance(scaler, RobustScaler)
    assert scaler.center_[0] == pytest.approx(51.0)


def test_pretrain_scaler_rejects_validation_shard(tmp_path):
    dtype = np.dtype([("feature", "f4")])
    with h5py.File(tmp_path / "shards.h5", "w") as datafile:
        datafile.create_dataset("train", data=np.array([(1.0,), (2.0,)], dtype=dtype))
        datafile.create_dataset("valid", data=np.array([(3.0,), (4.0,)], dtype=dtype))
        with pytest.raises(ValueError, match="scaler_keys must be training shards"):
            fit_pretrain_scaler(
                datafile,
                ["train"],
                ["feature"],
                {"scaler_keys": ["valid"]},
            )


def test_pretrain_rejects_feature_values_as_uncertainties():
    with pytest.raises(ValueError, match="duplicates data.feature_cols"):
        _validate_error_columns(["G", "bp_1"], ["G", "bp_1"])
    with pytest.raises(ValueError, match="duplicates data.feature_cols"):
        _validate_error_columns(["G", "bp_1"], ["bp_1", None])


def test_pretrain_accepts_explicitly_unavailable_uncertainties():
    _validate_error_columns(["W1", "bp_1"], [None, "bpe_1"])


def test_pretrain_pilot_caps_work_and_uses_separate_outputs():
    config = {
        "training": {"epochs": 100, "mini_batch_size": 32768},
        "saving": {
            "model_str": "/tmp/model.pth",
            "log_file": "/tmp/train.log",
            "arc_checkpoint_dir": "/tmp/checkpoints",
        },
    }

    _configure_pilot(config)

    assert config["training"]["epochs"] == 1
    assert config["training"]["mini_batch_size"] == 128
    assert config["training"]["max_rows_per_shard"] == 512
    assert config["saving"]["model_str"] == "/tmp/model_pilot.pth"
    assert config["saving"]["arc_checkpoint_dir"] == "/tmp/checkpoints_pilot"


def test_load_data_masks_nonfinite_features_and_repairs_invalid_errors(tmp_path):
    dtype = np.dtype([("feature", "f4"), ("error", "f4")])
    values = np.array([(1.0, 0.0), (np.inf, np.inf), (np.nan, np.nan)], dtype=dtype)
    with h5py.File(tmp_path / "data.h5", "w") as datafile:
        datafile.create_dataset("train", data=values)
        wrapper = TabResnetWrapper(
            model=torch.nn.Identity(),
            datafile=datafile,
            scaler=RobustScaler().fit(np.array([[1.0], [2.0]])),
            feature_cols=["feature"],
            error_cols=["error"],
            recon_cols=["feature"],
        )
        features, errors = wrapper._load_data("train")

    assert torch.isfinite(features[0]).all()
    assert torch.isnan(features[1:]).all()
    assert torch.isfinite(errors).all()
    assert (errors > 0).all()


def test_load_data_uses_neutral_uncertainty_for_unavailable_feature(tmp_path):
    dtype = np.dtype([("feature", "f4")])
    values = np.array([(1.0,), (2.0,)], dtype=dtype)
    with h5py.File(tmp_path / "data.h5", "w") as datafile:
        datafile.create_dataset("train", data=values)
        scaler = RobustScaler().fit(np.array([[1.0], [3.0]]))
        wrapper = TabResnetWrapper(
            model=torch.nn.Identity(),
            datafile=datafile,
            scaler=scaler,
            feature_cols=["feature"],
            error_cols=[None],
            recon_cols=["feature"],
        )
        _, errors = wrapper._load_data("train")
        noise = wrapper._pert_noise(torch.ones((2, 1)), torch.ones((2, 1)))

    torch.testing.assert_close(errors, torch.ones_like(errors))
    torch.testing.assert_close(noise, torch.zeros_like(noise))


def test_load_data_respects_pilot_row_limit(tmp_path):
    dtype = np.dtype([("feature", "f4"), ("error", "f4")])
    values = np.array([(i, 1.0) for i in range(5)], dtype=dtype)
    with h5py.File(tmp_path / "data.h5", "w") as datafile:
        datafile.create_dataset("train", data=values)
        wrapper = TabResnetWrapper(
            model=torch.nn.Identity(),
            datafile=datafile,
            scaler=RobustScaler().fit(np.array([[1.0], [2.0]])),
            feature_cols=["feature"],
            error_cols=["error"],
            recon_cols=["feature"],
            max_rows_per_key=2,
        )
        features, _ = wrapper._load_data("train")

    assert features.shape == (2, 1)


def test_pretrain_epoch_resets_loss_counters(wrapper_stub):
    wrapper_stub._arc_sync_interval = 5
    wrapper_stub._metrics_file = None
    wrapper_stub._residual_stats_file = None
    wrapper_stub._train_pretrain_key = MagicMock(return_value=(2.0, 1.0))
    wrapper_stub._save_pretrain_checkpoints = MagicMock()

    wrapper_stub._run_pretrain_epoch(
        epoch=0,
        total_epochs=1,
        train_keys=["train"],
        val_keys=None,
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        mini_batch=2,
        epoch_loss=100.0,
        loss_div=10.0,
        running_pt_loss=[],
        running_pt_validation_loss=[],
    )

    assert wrapper_stub._save_pretrain_checkpoints.call_args.args[-2:] == (2.0, 1.0)


def test_pretrain_epoch_does_not_hide_shard_failure(wrapper_stub):
    wrapper_stub._metrics_file = None
    wrapper_stub._train_pretrain_key = MagicMock(side_effect=RuntimeError("bad shard"))

    with pytest.raises(RuntimeError, match="bad shard"):
        wrapper_stub._run_pretrain_epoch(
            epoch=0,
            total_epochs=1,
            train_keys=["broken"],
            val_keys=None,
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            mini_batch=2,
            epoch_loss=0.0,
            loss_div=0.0,
            running_pt_loss=[],
            running_pt_validation_loss=[],
        )
