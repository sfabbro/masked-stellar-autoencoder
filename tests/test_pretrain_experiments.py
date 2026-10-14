"""Small checks for paired pilot fairness and masked group normalization."""

import importlib.util
import json
import subprocess
import sys
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


def test_encoder_decoder_dropout_are_independent_and_keep_default():
    model = torch.nn.Module()
    model.encoder = torch.nn.Sequential(torch.nn.Dropout(0.1))
    model.decoder = torch.nn.Sequential(torch.nn.Dropout(0.1))
    original = experiments.configure_experiment_dropout(model, {})
    assert original == {"encoder.0": 0.1, "decoder.0": 0.1}
    experiments.configure_experiment_dropout(model, {"decoder_dropout": 0})
    assert model.encoder[0].p == 0.1
    assert model.decoder[0].p == 0
    experiments.configure_experiment_dropout(
        model, {"dropout": 0.05, "encoder_dropout": 0}
    )
    assert model.encoder[0].p == 0
    assert model.decoder[0].p == 0.05
    with pytest.raises(ValueError, match="finite"):
        experiments.configure_experiment_dropout(model, {"dropout": float("nan")})


def test_experiment_seed_changes_row_and_mask_generators():
    seed = experiments.experiment_seed({"seed": 43}, 42)
    assert seed == 43
    assert experiments.experiment_seed({}, 42) == 42
    data = np.arange(90, dtype=np.float32).reshape(30, 3)
    orders = [
        np.concatenate(
            list(experiments.iter_batches(data, 7, np.random.default_rng(value)))
        )
        for value in (42, 43, 43)
    ]
    assert not np.array_equal(orders[0], orders[1])
    np.testing.assert_array_equal(orders[1], orders[2])
    with pytest.raises(ValueError, match="seed"):
        experiments.experiment_seed({"seed": -1}, 42)


def test_second_views_keep_xp_group_and_leave_primary_masks_unchanged():
    target = torch.arange(896, dtype=torch.float32).reshape(128, 7)
    features = ["W1", "bp_1", "rp_1", "G", "PARALLAX", "G_SDSS", "RA"]
    target[3, 1] = float("nan")
    primary = torch.Generator().manual_seed(42)
    baseline = torch.Generator().manual_seed(42)
    secondary = torch.Generator().manual_seed(1_000_042)
    for _ in range(3):
        first, first_mask, _ = experiments.training_mask(
            target, [1, 2], 0.5, 0.2, primary
        )
        second = experiments.paired_training_view(
            target, first, features, 0.2, secondary
        )
        reference, reference_mask, _ = experiments.training_mask(
            target, [1, 2], 0.5, 0.2, baseline
        )
        torch.testing.assert_close(first, reference)
        torch.testing.assert_close(first_mask, reference_mask)
        torch.testing.assert_close(
            second[:, [1, 2, 3, 4, 6]], first[:, [1, 2, 3, 4, 6]]
        )
        assert (second[:, [0, 5]] != first[:, [0, 5]]).any()
        assert second[3, 1] == -9999


def test_auxiliary_dropout_forward_preserves_primary_rng_sequence():
    model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(0.1))
    model.train()
    inputs = torch.ones(128, 5)
    torch.manual_seed(42)
    expected = [model(inputs).detach() for _ in range(3)]
    torch.manual_seed(42)
    actual = []
    for _ in range(3):
        actual.append(model(inputs).detach())
        with torch.random.fork_rng(devices=[]):
            model(inputs * 2)
    for first, second in zip(expected, actual, strict=True):
        torch.testing.assert_close(first, second)


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


@pytest.mark.parametrize("mode", ["coefficients", "block", "channel"])
def test_mixed_xp_mask_keeps_whole_spectrum_task_and_masks_partial_rows(mode):
    target = torch.ones(100, 9)
    features = [
        "G",
        *[f"bp_{i}" for i in range(1, 5)],
        *[f"rp_{i}" for i in range(1, 5)],
    ]
    generator = torch.Generator().manual_seed(17)
    hidden, mask, finite = experiments.training_mask(
        target,
        list(range(1, 9)),
        0.5,
        0.2,
        generator,
        xp_mode=mode,
        features=features,
    )
    count = mask[:, 1:].sum(1)
    assert int((count == 8).sum()) == 25
    assert int(((count > 0) & (count < 8)).sum()) == 50
    assert int(count.sum()) == 400
    assert torch.equal(hidden[mask], torch.full_like(hidden[mask], -9999))
    assert finite.all()


def test_noise_objective_handles_nan_targets_and_normalizes_microbatches():
    prediction = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]], requires_grad=True)
    target = torch.tensor([[0.0, float("nan"), 1.0], [0.0, 0.0, 1.0]])
    sigma = torch.tensor([[1.0, float("nan"), 0.0], [2.0, 1.0, 0.0]])
    mask = torch.isfinite(target)
    options = {"loss": "torchregress_noise", "variance_floor": 1.0}
    whole = experiments.experiment_objective(
        prediction, target, mask, [0, 1], options, mask.sum(), sigma=sigma
    )
    parts = sum(
        experiments.experiment_objective(
            prediction[i : i + 1],
            target[i : i + 1],
            mask[i : i + 1],
            [0, 1],
            options,
            mask.sum(),
            sigma=sigma[i : i + 1],
        )
        for i in range(2)
    )
    torch.testing.assert_close(whole, parts)
    whole.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 1] == 0
    assert float(prediction.grad[0, 0]) == pytest.approx(1 / 2 / 5)
    assert float(prediction.grad[1, 0]) == pytest.approx(2 / 5 / 5)
    assert float(prediction.grad[0, 2]) == pytest.approx(1 / 5)


def test_spectral_projection_unscales_residual_and_never_uses_center():
    residual = np.array([[1.0, -1.0]])
    scale = np.array([2.0, 3.0])
    basis = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
    projected = experiments.project_spectral_residual(residual, scale, basis)
    np.testing.assert_allclose(projected, [[2.0, 1.0, -6.0]])


def test_reliability_weights_do_not_reward_noise_or_use_validation():
    target = np.array([[-1.0, -2.0], [1.0, 2.0]])
    sigma = np.array([[0.5, 4.0], [0.5, 4.0]])
    weights = experiments.noise_reliability(target, sigma)
    np.testing.assert_allclose(weights, [0.75, 0.05])


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


@pytest.mark.parametrize("xp_suite", [False, True])
@pytest.mark.parametrize("evaluation_only", [False, True])
def test_paired_experiment_smoke(tmp_path, monkeypatch, xp_suite, evaluation_only):
    if xp_suite and importlib.util.find_spec("gaiaxpy") is None:
        pytest.skip("The extended suite runs in experiments/xp/pixi.toml")
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
            str(
                SCRIPT.parents[1]
                / (
                    "configs/pretrain.xp-reconstruction.yaml"
                    if xp_suite
                    else "configs/pretrain.experiments.yaml"
                )
            ),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--train-rows",
            "64",
            "--validation-rows",
            "31",
            "--presentations",
            "32",
            "--log-rows",
            "16",
            "--diagnostic-rows",
            "7",
            *(["--xp-diagnostics"] if xp_suite else []),
            *(["--evaluation-only"] if evaluation_only else []),
        ],
    )
    experiments.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sample"]["train_rows"] == 64
    assert experiments.sha256(checkpoint_path) == parent_hash
    if evaluation_only:
        assert manifest["evaluation_only"]
        assert manifest["presentations_per_experiment"] == 0
        for experiment in manifest["experiments"]:
            folder = output / experiment["run_id"]
            verification = json.loads(
                (folder / "evaluation_verification.json").read_text()
            )
            assert verification["checkpoint_sha256"] == parent_hash
            assert verification["optimizer_steps"] == 0
            assert verification["training_presentations"] == 0
            assert verification["reload_predictions_equal"]
            assert not (folder / "checkpoint.pth").exists()
            metrics = [
                json.loads(line)
                for line in (folder / "metrics.jsonl").read_text().splitlines()
            ]
            assert len(metrics) == 1 and metrics[0]["optimizer_steps"] == 0
        return
    initial_maes = []
    for arm in manifest["experiments"]:
        assert arm["status"] == "complete"
        metrics = [
            json.loads(line)
            for line in (output / arm["run_id"] / "metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        assert metrics[0]["phase"] == "init"
        assert metrics[-1]["phase"] == "complete"
        assert metrics[-1]["optimizer_steps"] == 2
        assert metrics[-1]["rows_seen_total"] == 32
        assert metrics[-1]["learning_rate"] == metrics[0]["learning_rate"]
        assert metrics[-1]["loader_mode"] == "experiment-prefix-cache"
        assert metrics[-1]["all_columns_mae"] == metrics[-1]["common_scaled_mae"]
        assert metrics[-1]["xp_only_mae"] is not None
        assert set(metrics[-1]["per_column_mae"]) == set(recon)
        assert metrics[-1]["optimizer_batch_size"] == 16
        assert metrics[-1]["cache_bytes"] > 0
        assert "cgroup_memory_peak_bytes" in metrics[-1]
        initial_maes.append(metrics[0]["sampled_validation_mae"])
        qa = json.loads((output / arm["run_id"] / "residual_latest.json").read_text())
        assert qa["validation_rows_total"] == 31
        assert qa["diagnostic_snapshot"]["rows"] == 7
        assert qa["diagnostic_snapshot"]["validation_rows"] == 31
        assert qa["regimes"]["common"]["overall"]["count"] > 7
        with np.load(output / arm["run_id"] / "residual_latest.npz") as snapshot:
            assert snapshot["targets"].shape[0] == 7
            assert snapshot["validation_row_index"].tolist() == [
                0,
                5,
                10,
                15,
                20,
                25,
                30,
            ]
        assert qa["uncertainty_units"] == (
            "verified only for named features" if xp_suite else "unverified"
        )
        assert qa["latent"]["effective_rank"] > 0
        assert qa["latent"]["sample_rows"] == 7
        assert qa["snr_units"].startswith("unverified")
        assert "physical_bias" not in qa["regimes"]["common"]["features"][0]
        assert "physical_rmse" not in qa["regimes"]["common"]["features"][0]
        assert (output / arm["run_id"] / "residual_init.npz").exists()
        assert qa["regimes"]["xp_on"]["blocks"]["xp"]["count"] == 0
        assert qa["regimes"]["xp_off"]["blocks"]["xp"]["count"] > 0
        assert (
            qa["regimes"]["common"]["features"][0]["uncertainty_normalized"]["count"]
            == 0
        )
        if xp_suite:
            assert qa["regimes"]["partial_coefficients"]["blocks"]["xp"]["count"] > 0
            assert qa["regimes"]["xp_off"]["spectral"]["rows"] == 7
            assert np.isfinite(
                qa["regimes"]["xp_off"]["spectral"]["integrated_absolute_fraction_mean"]
            )
        verification = json.loads(
            (output / arm["run_id"] / "checkpoint_verification.json").read_text()
        )
        assert verification["reload_predictions_equal"]
        assert verification["parent_sha256"] == parent_hash
    assert max(initial_maes) - min(initial_maes) == 0


def test_launch_record_has_separate_bounded_jobs_and_refuses_duplicate_record(tmp_path):
    record = tmp_path / "jobs.json"
    command = [
        sys.executable,
        str(SCRIPT.with_name("launch_xp_experiments.py")),
        "--checkpoint",
        "/arc/immutable.pth",
        "--output-root",
        "/arc/new-suite",
        "--git-ref",
        "codex/xp-tests",
        "--git-sha",
        "a" * 40,
        "--record",
        str(record),
        "--dry-run",
    ]
    subprocess.run(command, cwd=SCRIPT.parents[1], check=True, capture_output=True)
    saved = json.loads(record.read_text())
    assert saved["train_stars"] == 10_000_000
    assert saved["presentations"] == 20_000_000
    assert len(saved["jobs"]) == 7
    assert len({job["output"] for job in saved["jobs"]}) == 7
    assert all(job["status"] == "dry_run" for job in saved["jobs"])
    assert (
        subprocess.run(command, cwd=SCRIPT.parents[1], capture_output=True).returncode
        != 0
    )


def test_photometry_runner_g_is_target_only_and_decoder_detached():
    from masked_stellar_autoencoder.training.xp_photometry import (
        PhotometryCalibration,
        PhotometryProjection,
    )

    matrix = np.zeros((110, 1))
    matrix[0, 0] = 1
    context = {
        "projection": PhotometryProjection(matrix, np.zeros(1), ("band",), ("band",)),
        "calibration": PhotometryCalibration(
            np.ones(1), np.zeros(1), np.full(1, 0.02), np.ones(1)
        ),
        "xp": list(range(110)),
        "photo": [110],
        "g": 111,
        "scale": np.ones(112),
        "center": np.zeros(112),
    }
    target = torch.ones((2, 112))
    target[:, 110] = 0
    target[:, 111] = 8.5
    prediction = target.clone().requires_grad_()
    with torch.no_grad():
        prediction[:, 0] = 2
        prediction[:, 110] = 0.3
        prediction[:, 111] = -1000
    score = torch.ones_like(target, dtype=torch.bool)
    loss = experiments.photometry_terms(prediction, target, score, None, context)
    assert loss.item() == pytest.approx(49.5 * 110 * 2, rel=1e-6)
    loss.backward()
    assert prediction.grad[:, 0].abs().sum() > 0
    assert prediction.grad[:, 110:].abs().sum() == 0
    prediction.grad.zero_()
    decoder = experiments.photometry_terms(
        prediction, target, score, None, context, decoder=True
    )
    decoder.backward()
    assert prediction.grad[:, 0].abs().sum() > 0
    assert prediction.grad[:, 110:].abs().sum() == 0


def test_photometry_diagnostics_allow_unreconstructed_ancillary_columns():
    from masked_stellar_autoencoder.training.xp_photometry import (
        PhotometryCalibration,
        PhotometryProjection,
    )

    matrix = np.zeros((110, 1))
    matrix[0, 0] = 1
    context = {
        "projection": PhotometryProjection(matrix, np.zeros(1), ("band",), ("band",)),
        "calibration": PhotometryCalibration(
            np.ones(1), np.zeros(1), np.full(1, 0.02), np.ones(1)
        ),
        "xp": list(range(110)),
        "photo": [110],
        "g": 111,
        "scale": np.ones(114),
        "center": np.zeros(114),
    }
    target = np.ones((2, 112))
    target[:, 110] = 0
    target[:, 111] = 8.5
    result = experiments.photometry_summary(
        target, target, np.ones_like(target, dtype=bool), context
    )
    assert result["bands"][0]["magnitude_mae"] == 0
    assert result["bands"][0]["count"] == 2
