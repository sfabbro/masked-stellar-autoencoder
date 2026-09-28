import json
import random
import time
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
    _summarize_feature_residuals,
)
from masked_stellar_autoencoder.training.pretrain_msa import (
    _configure_batch_pilot,
    _configure_pilot,
    _limit_batch_pilot_shards,
    _limit_pilot_shards,
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


def test_feature_residual_summary_handles_masked_and_unavailable_columns():
    summary = _summarize_feature_residuals(
        np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 2.0]], dtype=np.float32),
        ["xp_0", "photo_g"],
    )

    assert summary["feature_names"] == ["xp_0", "photo_g"]
    assert summary["feature_valid_count"] == [3, 1]
    assert summary["feature_valid_fraction"] == [1.0, 0.33333333]
    assert summary["feature_mae"] == [3.0, 2.0]
    assert summary["feature_p84"] == [4.36, 2.0]
    assert summary["feature_p95"] == [4.8, 2.0]


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
        "training": {
            "epochs": 100,
            "mini_batch_size": 32768,
            "micro_batch_size": 8192,
        },
        "saving": {
            "model_str": "/tmp/model.pth",
            "log_file": "/tmp/train.log",
            "arc_checkpoint_dir": "/tmp/checkpoints",
        },
    }

    _configure_pilot(config)

    assert config["training"]["epochs"] == 1
    assert config["training"]["mini_batch_size"] == 128
    assert config["training"]["micro_batch_size"] == 128
    assert config["training"]["max_rows_per_shard"] == 512
    assert config["saving"]["model_str"] == "/tmp/model_pilot.pth"
    assert config["saving"]["arc_checkpoint_dir"] == "/tmp/checkpoints_pilot"


def test_pretrain_pilot_limits_training_and_validation_shards():
    train_keys = [f"train_{i}" for i in range(5)]
    valid_keys = [f"valid_{i}" for i in range(3)]

    assert _limit_pilot_shards(train_keys, valid_keys) == (
        ["train_0", "train_1"],
        ["valid_0"],
    )


def test_pretrain_batch_pilot_preserves_full_batch_and_separates_outputs():
    config = {
        "training": {
            "epochs": 100,
            "mini_batch_size": 32768,
            "micro_batch_size": 8192,
        },
        "saving": {
            "model_str": "/tmp/model.pth",
            "log_file": "/tmp/train.log",
            "metrics_file": "/tmp/metrics.jsonl",
            "residual_stats_file": "/tmp/residuals.jsonl",
            "arc_checkpoint_dir": "/tmp/checkpoints",
        },
    }

    _configure_batch_pilot(config)

    assert config["training"]["epochs"] == 1
    assert config["training"]["mini_batch_size"] == 32768
    assert config["training"]["micro_batch_size"] == 8192
    assert config["training"]["max_rows_per_shard"] == 32768
    assert config["saving"]["model_str"] == "/tmp/model_batch_pilot.pth"
    assert config["saving"]["metrics_file"] == "/tmp/metrics_batch_pilot.jsonl"


def test_pretrain_batch_pilot_limits_to_one_train_and_validation_shard():
    assert _limit_batch_pilot_shards(["train_a", "train_b"], ["valid_a"]) == (
        ["train_a"],
        ["valid_a"],
    )


def test_epoch_metrics_include_run_id_memory_and_disk_state(wrapper_stub, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    wrapper_stub._configure_canfar_output(
        metrics_file=str(metrics_path), run_id="run-1"
    )
    wrapper_stub._epoch_start = time.time() - 1
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)

    wrapper_stub._log_epoch_metrics(
        0,
        2,
        0.5,
        0.6,
        optimizer,
        residual_stats={
            "xp_mae": 0.4,
            "xp_p84": 0.8,
            "photo_mae": 0.2,
            "overall_mae": 0.3,
            "sampled_rows": 1000,
            "sampled_validation_shards": 5,
            "feature_names": ["xp_0"],
            "feature_mae": [0.5],
        },
    )

    entry = json.loads(metrics_path.read_text())
    assert entry["run_id"] == "run-1"
    assert entry["peak_host_rss_bytes"] > 0
    assert entry["output_free_bytes"] > 0
    assert entry["epoch"] == 1
    assert entry["residual_xp_mae"] == 0.4
    assert entry["residual_xp_p84"] == 0.8
    assert entry["residual_photo_mae"] == 0.2
    assert entry["residual_overall_mae"] == 0.3
    assert entry["residual_sampled_rows"] == 1000
    assert entry["residual_sampled_validation_shards"] == 5
    assert "residual_feature_mae" not in entry


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


def test_pretrain_monitor_emits_row_weighted_intervals(wrapper_stub):
    wrapper_stub._pretrain_monitor_interval_size = 7
    wrapper_stub._pretrain_monitor_interval_rows = 0
    wrapper_stub._pretrain_monitor_interval_count = 0
    wrapper_stub._pretrain_monitor_interval_loss = torch.zeros(())
    wrapper_stub._pretrain_monitor_epoch_rows_seen = 0
    wrapper_stub._emit_pretrain_interval = MagicMock()

    wrapper_stub._accumulate_pretrain_monitor_batch(torch.tensor(0.5), 4, 4)
    assert not wrapper_stub._emit_pretrain_interval.called
    wrapper_stub._accumulate_pretrain_monitor_batch(torch.tensor(1.5), 4, 4)

    args, kwargs = wrapper_stub._emit_pretrain_interval.call_args
    assert args[0] == 8  # The interval ends at an optimizer batch boundary.
    torch.testing.assert_close(args[1], torch.tensor(8.0))
    assert args[2] == 4
    assert kwargs == {"partial": False}
    assert wrapper_stub._pretrain_monitor_epoch_rows_seen == 8
    assert wrapper_stub._pretrain_monitor_interval_rows == 0


def test_pretrain_monitor_sample_is_bounded_and_seeded(wrapper_stub):
    class SampleStore:
        def __init__(self):
            self.requests = []

        def sample_batches(
            self, key, scaler, scale_factors, *, sample_rows, seed, batch_rows
        ):
            self.requests.append((key, sample_rows, seed, batch_rows))
            values = np.full((sample_rows, 2), seed, dtype=np.float32)
            yield values, values.copy()

    store = SampleStore()
    wrapper_stub.data_store = store
    wrapper_stub._pretrain_micro_batch_size = 2
    wrapper_stub.scale_factors = np.ones(2)

    X_sample, eX_sample = wrapper_stub._prepare_pretrain_monitor_sample(
        ["valid_a", "valid_b"], 5, 8, seed=10
    )

    assert store.requests == [("valid_a", 3, 10, 2), ("valid_b", 2, 11, 2)]
    assert X_sample.shape == eX_sample.shape == (5, 2)
    np.testing.assert_array_equal(X_sample[:, 0], [10, 10, 10, 11, 11])


def test_pretrain_monitor_evaluation_restores_model_mode_and_rng(wrapper_stub):
    class ZeroReconstruction(torch.nn.Module):
        def forward(self, values):
            return torch.zeros((len(values), 1)), torch.zeros((len(values), 1))

    wrapper_stub.model = ZeroReconstruction()
    wrapper_stub.model.train()
    wrapper_stub._run_id = "run-1"
    wrapper_stub.xp_col_start = 0
    wrapper_stub.xp_col_end = 1
    wrapper_stub._pretrain_micro_batch_size = 2
    wrapper_stub._pretrain_valid_keys = ["valid"]
    wrapper_stub._pretrain_monitor_sample = (
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
    )
    wrapper_stub._apply_mask = lambda values: (
        values,
        torch.ones_like(values, dtype=torch.bool),
        torch.ones_like(values, dtype=torch.bool),
    )
    torch.manual_seed(17)
    rng_before = torch.get_rng_state().clone()

    stats = wrapper_stub._evaluate_pretrain_monitor_sample(
        0,
        2,
        record_type="interval",
        rows_seen_total=3,
        epoch_rows_seen=3,
        snapshot_only=True,
    )

    assert wrapper_stub.model.training
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert stats["sampled_rows"] == 3
    assert stats["sampled_validation_shards"] == 1
    assert np.isfinite(stats["sampled_val_loss"])
    assert stats["feature_mae"] == [2.0]


def test_pretrain_monitor_residual_snapshot_is_atomic_json(wrapper_stub, tmp_path):
    snapshot_path = tmp_path / "residual_latest.json"
    wrapper_stub._configure_canfar_output(
        residual_latest_file=str(snapshot_path), run_id="run-1"
    )

    wrapper_stub._write_pretrain_residual_record(
        {"record_type": "epoch", "rows_seen_total": 100, "overall_mae": 0.25}
    )

    record = json.loads(snapshot_path.read_text())
    assert record["run_id"] == "run-1"
    assert record["rows_seen_total"] == 100
    assert record["overall_mae"] == 0.25
    assert not snapshot_path.with_name(snapshot_path.name + ".tmp").exists()


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
