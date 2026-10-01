"""Small checks for paired pilot fairness and masked group normalization."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from sklearn.preprocessing import RobustScaler

from masked_stellar_autoencoder.models.model import (
    TabResnetWrapper,
    _capture_rng_state,
    make_model,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_pretrain_experiments.py"
SPEC = importlib.util.spec_from_file_location("pretrain_experiments", SCRIPT)
experiments = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiments)


def test_balanced_mae_microbatches_match_whole_batch():
    target = torch.tensor([[1.0, float("nan"), 3.0], [2.0, 5.0, 6.0]])
    mask = torch.isfinite(target)
    groups = {"xp": [0, 1], "photo": [2]}
    counts = [mask[:, indices].sum() for indices in groups.values()]
    whole = experiments.balanced_mae(
        torch.zeros_like(target), target, mask, groups, counts
    )
    micro = sum(
        experiments.balanced_mae(
            torch.zeros_like(target[i : i + 1]),
            target[i : i + 1],
            mask[i : i + 1],
            groups,
            counts,
        )
        for i in range(2)
    )
    torch.testing.assert_close(whole, micro)
    assert float(whole) == pytest.approx(((1 + 2 + 5) / 3 + (3 + 6) / 2) / 2)


def test_residual_summary_excludes_invalid_sigma():
    summary = experiments.summarize(
        np.array([1.0, -1.0, 2.0, 3.0]),
        np.zeros(4),
        0.0,
        np.array([1.0, np.nan, -1.0, 0.0]),
    )
    assert summary["uncertainty_normalized"] == {
        "count": 1,
        "bias": 1.0,
        "rmse": 1.0,
        "p50": 1.0,
        "p95": 1.0,
    }
    assert summary["bias"] == 1.25
    assert summary["skill_zero"] is None


def test_mask_draws_do_not_depend_on_dropout_rng():
    target = torch.ones(10, 4)
    generator = torch.Generator().manual_seed(42)
    first = experiments.training_mask(target, [1, 2], 0.5, 0.2, generator)[1]
    torch.rand(1000)
    generator = torch.Generator().manual_seed(42)
    second = experiments.training_mask(target, [1, 2], 0.5, 0.2, generator)[1]
    assert torch.equal(first, second)
    assert int(first[:, 1].sum()) == 5


def test_normal_resume_rejects_pilot_checkpoint(tmp_path):
    path = tmp_path / "pilot.pth"
    torch.save({"experiment_only": True}, path)
    wrapper = TabResnetWrapper.__new__(TabResnetWrapper)
    with pytest.raises(ValueError, match="bounded experiment"):
        wrapper._load_pretrain_resume(path, None, None)


def test_flux_sigma_cannot_be_used_as_magnitude_sigma():
    data = {
        "feature_cols": ["G"],
        "recon_cols": ["G"],
        "error_cols": ["PHOT_G_MEAN_FLUX_ERROR"],
        "valid_keys": ["validation"],
    }
    checkpoint = {
        "run_signature": {
            "feature_cols": ["G"],
            "recon_cols": ["G"],
            "train_keys": ["train"],
        },
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "rng_state": {},
    }
    source = {"train": np.zeros(1), "validation": np.zeros(1)}
    with pytest.raises(ValueError, match="incompatible units"):
        experiments.validate_source(checkpoint, {"data": data}, source)


def test_paired_experiment_smoke(tmp_path, monkeypatch):
    torch.manual_seed(7)
    features = ["G", "bp_1", "bp_2", "rp_1", "PARALLAX", "RA"]
    errors = [None, "bpe_1", "bpe_2", "rpe_1", "e_parallax", None]
    recon = features[:-1]
    source_path = tmp_path / "source.h5"
    rng = np.random.default_rng(8)
    with h5py.File(source_path, "w") as source:
        dtype = [
            (name, "f4")
            for name in [*features, "bpe_1", "bpe_2", "rpe_1", "e_parallax"]
        ]
        for key in ("train1", "train2", "validation"):
            values = np.zeros(64, dtype=dtype)
            for feature in features:
                values[feature] = rng.normal(size=64)
            values["G"] += 15
            for error in errors:
                if error:
                    values[error] = 0.2
            values["bpe_1"][:2] = np.nan
            source.create_dataset(key, data=values)
    config = {
        "data": {
            "datafile": str(source_path),
            "feature_cols": features,
            "error_cols": errors,
            "recon_cols": recon,
            "valid_keys": ["validation"],
        },
        "model": {
            "layer_dims": [8, 4],
            "pt_activ_func": "elu",
            "rtdl_embed": 4,
            "norm": "layer",
            "decoder_dims": [4],
        },
        "training": {
            "xp_masking_ratio": 0.9,
            "m_masking_ratio": 0.6,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "mini_batch_size": 16,
            "micro_batch_size": 8,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    model = make_model(6, [8, 4], 5, "elu", 4, "layer", decoder_dims=[4])
    with h5py.File(source_path, "r") as source:
        scaler = RobustScaler()
        scaler.center_, scaler.scale_ = np.zeros(6), np.ones(6)
        wrapper = TabResnetWrapper(
            model,
            source,
            scaler,
            features,
            errors,
            recon,
            lr=1e-4,
            optimizer="adamw",
            wd=1e-5,
        )
        optimizer, scheduler = wrapper._setup_pretrain_optimizer()
        prediction, _ = model(torch.randn(16, 6))
        prediction.square().mean().backward()
        optimizer.step()
        signature = {
            "feature_cols": features,
            "error_cols": errors,
            "recon_cols": recon,
            "scaler_center": [0.0] * 6,
            "scaler_scale": [1.0] * 6,
            "train_keys": ["train1", "train2"],
            "train_rows_per_epoch": 128,
            "train_rows_by_key": {"train1": 64, "train2": 64},
        }
        checkpoint_path = tmp_path / "parent.pth"
        torch.save(
            {
                "epoch": 2,
                "rows_seen_total": 256,
                "model_state_dict": {
                    name: value
                    for name, value in model.state_dict().items()
                    if "missing_periodic_encoding" not in name
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": _capture_rng_state(),
                "run_signature": signature,
            },
            checkpoint_path,
        )
    parent_hash = experiments.sha256(checkpoint_path)
    output = tmp_path / "suite"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--config",
            str(config_path),
            "--experiments",
            str(SCRIPT.parents[1] / "configs/pretrain.experiments.yaml"),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--train-rows",
            "64",
            "--validation-rows",
            "32",
            "--presentations",
            "32",
            "--log-rows",
            "16",
        ],
    )
    experiments.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sample"]["train_rows"] == 64
    assert experiments.sha256(checkpoint_path) == parent_hash
    initial_maes = []
    for arm in manifest["arms"]:
        assert arm["status"] == "complete"
        metrics = [
            json.loads(line)
            for line in (output / arm["arm_id"] / "metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        assert metrics[0]["phase"] == "init"
        assert metrics[-1]["phase"] == "complete"
        assert metrics[-1]["optimizer_steps"] == 2
        assert metrics[-1]["rows_seen_total"] == 32
        assert metrics[-1]["learning_rate"] == metrics[0]["learning_rate"]
        initial_maes.append(metrics[0]["sampled_validation_mae"])
        qa = json.loads((output / arm["arm_id"] / "residual_latest.json").read_text())
        assert qa["uncertainty_units"] == "unverified"
        assert qa["latent"]["effective_rank"] > 0
        assert qa["snr_units"].startswith("unverified")
        assert "physical_bias" not in qa["regimes"]["common"]["features"][0]
        assert "physical_rmse" not in qa["regimes"]["common"]["features"][0]
        assert (output / arm["arm_id"] / "residual_init.npz").exists()
        assert qa["regimes"]["xp_on"]["blocks"]["xp"]["count"] == 0
        assert qa["regimes"]["xp_off"]["blocks"]["xp"]["count"] > 0
        assert (
            qa["regimes"]["common"]["features"][0]["uncertainty_normalized"]["count"]
            == 0
        )
    assert max(initial_maes) - min(initial_maes) == 0
