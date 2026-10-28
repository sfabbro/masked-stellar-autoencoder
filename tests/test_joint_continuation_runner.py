"""Small deterministic interruption gates; synthetic labels, no science claim."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_joint_continuation as runner  # noqa: E402

from masked_stellar_autoencoder.training import (
    continuation_state as state,  # noqa: E402
)


class TinyModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.Dropout(0.2),
            torch.nn.GELU(),
        )
        self.decoder = torch.nn.Linear(256, output_dim)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent


def test_large_sample_requires_completed_matched_pilot(tmp_path):
    cfg = yaml.safe_load(Path("configs/joint-continuation-30e.yaml").read_text())
    identity = dict(
        source_sha="a" * 40,
        parent_sha256="b" * 64,
        cohort_sha256="c" * 64,
        joint_labels_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="pinned sample and pilot"):
        runner.validate_promotion(cfg, identity)
    cfg.update(
        sample_request_sha256="e" * 64,
        pilot_policy_sha256="f" * 64,
        promotion_report=str(tmp_path / "promotion.json"),
    )
    report = dict(
        status="passed",
        training_rows_per_epoch=9000000,
        completed_epochs=3,
        policy_sha256=cfg["pilot_policy_sha256"],
        **identity,
    )
    for name in (
        "native_labels_verified",
        "original_xp_units_verified",
        "source_roles_verified",
        "resume_verified",
        "gpu_preflight_verified",
        "vos_publication_verified",
        "gradient_balance_verified",
        "declared_tuning_checks_passed",
    ):
        report[name] = True
    path = Path(cfg["promotion_report"])
    path.write_text(json.dumps(report))
    runner.validate_promotion(cfg, identity)
    for key, changed in [
        ("completed_epochs", 2),
        ("training_rows_per_epoch", 10000000),
        ("parent_sha256", "0" * 64),
        ("resume_verified", False),
    ]:
        path.write_text(json.dumps({**report, key: changed}))
        with pytest.raises(ValueError):
            runner.validate_promotion(cfg, identity)
    with pytest.raises(ValueError, match="only9M"):
        runner.validate_promotion(
            {
                "reconstruction_source": "benchmark",
                "reconstruction_rows_per_epoch": 150000000,
            },
            identity,
        )


@pytest.fixture
def tiny_runner(tmp_path, monkeypatch):
    model_cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
    features = model_cfg["data"]["feature_cols"]
    width = len(model_cfg["data"]["recon_cols"])
    local_rng = np.random.default_rng(11)
    replay = local_rng.uniform(0.1, 1.0, (8, len(features))).astype("f4")
    validation = local_rng.uniform(0.1, 1.0, (4, len(features))).astype("f4")
    inputs = local_rng.uniform(0.1, 1.0, (20, len(features))).astype("f4")
    values = np.column_stack(
        [np.linspace(4000, 7000, 20), np.linspace(1, 5, 20), np.linspace(-2, 0, 20)]
    )
    fit = np.arange(20) < 12
    observed = np.ones((20, 3), dtype=bool)
    # Ragged target availability is real to the test even though labels are fixtures.
    observed[[1, 3, 6], 1] = False
    values[~observed] = np.nan
    data = dict(
        source_id=np.arange(20, dtype="i8") + 1000,
        input=inputs,
        value=values,
        error=np.broadcast_to([50.0, 0.1, 0.05], (20, 3)).copy(),
        observed=observed,
        survey=np.full(20, b"fixture", dtype="S12"),
        fit=fit,
        tune=(np.arange(20) >= 12) & (np.arange(20) < 16),
        calibration=np.zeros(20, bool),
        test=np.arange(20) >= 16,
        reference_holdout=np.zeros(20, bool),
        benchmark_partition=np.zeros(20, dtype="u1"),
        parallax=np.linspace(-0.2, 0.5, 20),
        parallax_error=np.full(20, 0.2),
        parallax_observed=np.ones(20, bool),
        distance_fit=np.zeros(20, bool),
        zero_point=np.zeros(20),
        fit_rows=np.flatnonzero(fit),
        label_center=np.nanmean(values[fit], axis=0),
        label_scale=np.nanstd(values[fit], axis=0),
    )
    for name in ["cohort.h5", "labels.h5", "benchmark.h5"]:
        with h5py.File(tmp_path / name, "w") as f:
            f.attrs["fixture"] = True
    torch.save({}, tmp_path / "parent.pth")
    cfg = yaml.safe_load(Path("configs/joint-continuation-10m-pilot.yaml").read_text())
    cfg.update(
        epochs=2,
        qa_every_epochs=1,
        checkpoint_every_steps=1,
        head_warmup_steps=1,
        effective_batch=4,
        micro_batch=2,
        supervised_batch=4,
        supervised_micro_batch=2,
        minimum_fit_labels_per_target=1,
        require_cuda=False,
        reconstruction_rows_per_epoch=8,
        validation_rows=4,
        output=str(tmp_path / "run"),
        scratch=str(tmp_path / "scratch"),
        parent=str(tmp_path / "parent.pth"),
        cohort=str(tmp_path / "cohort.h5"),
        joint_labels=str(tmp_path / "labels.h5"),
        benchmark=str(tmp_path / "benchmark.h5"),
        spectral_weight=0.0,
    )
    monkeypatch.setattr(runner, "load_data", lambda *args: copy.deepcopy(data))
    monkeypatch.setattr(
        runner.pt,
        "validate_source",
        lambda *args: (
            ["train"],
            ["validation"],
            np.zeros(len(features)),
            np.ones(len(features)),
        ),
    )
    monkeypatch.setattr(
        runner.pt,
        "read_prefixes",
        lambda source, keys, *args, **kwargs: (
            replay.copy() if keys == ["train"] else validation.copy(),
            None,
            {},
        ),
    )

    def build(*args):
        # All runs start from exactly the same synthetic parent model/Adam state.
        torch.manual_seed(123)
        model = TinyModel(len(features), width)
        return (
            SimpleNamespace(model=model),
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            None,
        )

    monkeypatch.setattr(runner.pt, "build_wrapper", build)
    basis = np.zeros((width, 4), dtype="f8")
    for j, name in enumerate(features[:width]):
        if name.startswith(("bp_", "rp_")):
            basis[j] = [0.1, 0.2, 0.3, 0.4]
    monkeypatch.setattr(
        runner.pt,
        "spectral_basis",
        lambda *args: dict(
            basis=basis,
            wavelength_nm=np.array([400.0, 500.0, 600.0, 700.0]),
            flux_normalization="explicit synthetic unit fixture",
        ),
    )
    return cfg


def _state(cfg):
    directory = Path(cfg["output"]) / "checkpoints"
    pointer = json.loads((directory / "latest.json").read_text())
    return state.load_checkpoint(directory, pointer["identity"])


def _assert_nested_exact(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            _assert_nested_exact(value, actual[key])
    elif isinstance(expected, list | tuple):
        assert len(expected) == len(actual)
        for value, other in zip(expected, actual, strict=True):
            _assert_nested_exact(value, other)
    else:
        assert expected == actual


def _assert_training_equal(expected, actual):
    for section in ["model", "head"]:
        assert expected[section].keys() == actual[section].keys()
        for name, value in expected[section].items():
            torch.testing.assert_close(value, actual[section][name], rtol=0, atol=0)
    assert expected["history"] == actual["history"]
    assert expected["metrics"] == actual["metrics"]
    assert expected["predictions"] == actual["predictions"]
    assert expected["epoch"] == actual["epoch"] == 2
    assert expected["step"] == actual["step"] == 4
    assert expected["next_row"] == actual["next_row"] == 0
    torch.testing.assert_close(expected["mask_rng"], actual["mask_rng"], rtol=0, atol=0)
    for key in ["optimizer", "head_optimizer", "rng", "label_rng"]:
        _assert_nested_exact(expected[key], actual[key])


def test_mid_epoch_stop_resume_matches_uninterrupted_with_dropout(tiny_runner):
    cfg = tiny_runner
    runner.run(cfg, smoke=True)
    expected = _state(cfg)
    interrupted = {
        **cfg,
        "output": cfg["output"] + "-interrupted",
        "scratch": cfg["scratch"] + "-interrupted",
    }
    runner.run(interrupted, smoke=True, stop_after_steps=1)
    assert _state(interrupted)["next_row"] == 4
    runner.run(interrupted, smoke=True, resume=True)
    _assert_training_equal(expected, _state(interrupted))


def test_interruption_before_scheduled_qa_replays_missing_qa(tiny_runner, monkeypatch):
    cfg = tiny_runner
    runner.run(cfg, smoke=True)
    expected = _state(cfg)
    interrupted = {
        **cfg,
        "output": cfg["output"] + "-qa-interrupted",
        "scratch": cfg["scratch"] + "-qa-interrupted",
    }
    original = runner.qa_tables
    once = True

    def fail_before_qa(*args, **kwargs):
        nonlocal once
        if args[4] == 1 and once:
            once = False
            raise RuntimeError("simulated kill before QA")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "qa_tables", fail_before_qa)
    with pytest.raises(RuntimeError, match="simulated kill"):
        runner.run(interrupted, smoke=True)
    assert _state(interrupted)["epoch"] == 1
    runner.run(interrupted, smoke=True, resume=True)
    _assert_training_equal(expected, _state(interrupted))


def test_checkpoint_pointer_remains_valid_if_same_step_rewrite_is_interrupted(
    tmp_path, monkeypatch
):
    identity = {"test": "synthetic"}
    first = state.save_checkpoint(
        tmp_path,
        {"next_row": 0, "value": torch.tensor(1)},
        epoch=1,
        step=2,
        identity=identity,
    )
    original = state.atomic_json

    def fail_pointer(path, content):
        if Path(path).name == "latest.json":
            raise RuntimeError("simulated kill before pointer")
        return original(path, content)

    monkeypatch.setattr(state, "atomic_json", fail_pointer)
    with pytest.raises(RuntimeError, match="simulated kill"):
        state.save_checkpoint(
            tmp_path,
            {"next_row": 0, "value": torch.tensor(2)},
            epoch=1,
            step=2,
            identity=identity,
        )
    restored = state.load_checkpoint(tmp_path, identity)
    assert restored["value"].item() == 1
    assert json.loads((tmp_path / "latest.json").read_text()) == first


@pytest.mark.skipif(
    os.environ.get("MSA_ACTUAL_PARENT_INTEGRATION") != "1",
    reason="Explicit bounded actual-parent integration opt-in; synthetic labels only",
)
def test_actual_epoch72_parent_input_fixture_interrupted_resume_exact(
    tiny_runner, monkeypatch
):
    """Actual parent/inputs, synthetic labels; NOT original-benchmark certificate.

    Run MSA_ACTUAL_PARENT_INTEGRATION=1 pixi run --no-install pytest
    tests/test_joint_continuation_runner.py -k actual_epoch72 -q.
    """
    root = Path(__file__).parents[1]
    parent = (
        root
        / "experiments/xp/results-stellar-improvement/prediction-comparison-e72/parent/checkpoint-72.pth"
    )
    parent = Path(os.environ.get("MSA_JOINT_PARENT_CHECKPOINT", str(parent)))
    benchmark = (
        root
        / "experiments/xp/results-stellar-improvement/prediction-e72-local-integration-v2/actual-values-benchmark-SCHEMA-FIXTURE.h5"
    )
    cohort = (
        root
        / "experiments/xp/results-stellar-improvement/enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    )
    assert parent.exists() and benchmark.exists() and cohort.exists()
    cfg = tiny_runner
    fixture_data = runner.load_data(None, None, None, None)
    # Remove all tiny model/reader patches, retaining only synthetic supervision.
    monkeypatch.undo()
    with h5py.File(cohort) as f:
        center = f["reference_scaler/center"][:]
        scale = f["reference_scaler/scale"][:]
        raw = f["features/values"][:20]
        observed = f["features/feature_observed"][:20]
        fixture_data["source_id"] = f["features/source_id"][:20]
    features = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())["data"][
        "feature_cols"
    ]
    safe, _ = runner.spectrophotometric_inputs(
        torch.from_numpy(((raw - center) / scale).astype("f4")),
        torch.from_numpy(observed),
        features,
        include_reddening=False,
    )
    fixture_data["input"] = safe.numpy()
    monkeypatch.setattr(runner, "load_data", lambda *args: copy.deepcopy(fixture_data))
    cfg.update(
        parent=str(parent), benchmark=str(benchmark), cohort=str(cohort), epochs=1
    )
    runner.run(cfg, smoke=True)
    expected = _state(cfg)
    interrupted = {
        **cfg,
        "output": cfg["output"] + "-actual-resume",
        "scratch": cfg["scratch"] + "-actual-resume",
    }
    runner.run(interrupted, smoke=True, stop_after_steps=1)
    runner.run(interrupted, smoke=True, resume=True)
    actual = _state(interrupted)
    for section in ["model", "head"]:
        for key, value in expected[section].items():
            torch.testing.assert_close(value, actual[section][key], rtol=0, atol=0)
    for section in ["history", "metrics", "predictions"]:
        assert expected[section] == actual[section]
    assert expected["epoch"] == actual["epoch"] == 1
    assert expected["step"] == actual["step"] == 2
    torch.testing.assert_close(expected["mask_rng"], actual["mask_rng"], rtol=0, atol=0)
    for key in ["optimizer", "head_optimizer", "rng", "label_rng"]:
        _assert_nested_exact(expected[key], actual[key])
    report = dict(
        status="passed",
        actual_parent_sha256=state.sha256(parent),
        actual_parent_path=str(parent),
        actual_cohort_input_sha256=state.sha256(cohort),
        actual_encoder_decoder_head_and_adam_resume_exact=True,
        replay_presentations=8,
        epochs=1,
        joint_supervision="SYNTHETIC FIXTURE ONLY; no native Teff/logg association claim",
        benchmark="ACTUAL VALUES IN SCHEMA FIXTURE; NOT ORIGINAL BENCHMARK UNIT CERTIFICATION",
        not_scientific_performance=True,
    )
    destination = Path("/tmp/msa-joint-actual-parent-integration.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
