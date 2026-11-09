"""Exercise saved distribution arrays and launch admission with synthetic data."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


def load_script(name):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_saved_arrays_keep_integer_ids_and_joint_draws(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "survey_training_fixture",
        Path(__file__).with_name("test_multisurvey_training.py"),
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    trainer, dataset = fixture.setup_trainer(tmp_path)
    runner = load_script("run_multisurvey_distribution_pilot")
    try:
        path = tmp_path / "predictions.npz"
        result = runner.evaluate(
            trainer.model,
            trainer.relative_head,
            dataset,
            trainer.source_floor,
            runner.labeled_positions(dataset),
            path,
            draws=128,
        )
        with np.load(path) as data:
            assert data["source_id"].dtype == np.int64
            assert data["record_source_id"].dtype == np.int64
            assert data["joint_draws"].shape[1:] == (128, 3)
            assert np.isfinite(data["observed_pit"]).all()
            assert (data["observed_variance"] > 0).all()
            assert np.isfinite(data["per_star_nll"]).all()
        assert result["scored_stars"] > 0 and result["metrics"]
    finally:
        dataset.close()


def test_runner_preserves_failed_attempt_and_refuses_overwrite(tmp_path):
    runner = load_script("run_multisurvey_distribution_pilot")
    config = yaml.safe_load(
        Path("configs/multisurvey-distribution-20261104.yaml").read_text()
    )["experiments"][0]
    producer = tmp_path / "producer.json"
    parent = dict(config["parent"], overlay_producer_manifest=str(producer))
    config = dict(config, parent=parent)
    producer.write_text(
        json.dumps(
            dict(
                status="multisurvey_desi_overlay_complete",
                source_commit="cf40e37c87d4e724b84fd1dffcac6d9e1363560d",
                result=dict(
                    overlay_manifest_path=parent["overlay_manifest"],
                    overlay_manifest_sha256=parent["overlay_manifest_sha256"],
                ),
            )
        )
    )

    def fail(*args):
        raise RuntimeError("synthetic execution failure")

    output = tmp_path / "pilot"
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.run(config, output, source_commit="a" * 40, execute=fail)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        runner.run(config, output, source_commit="a" * 40, execute=fail)
    with pytest.raises(ValueError, match="bounded"):
        runner.run(
            dict(config, updates=1001), tmp_path / "oversized", source_commit="a" * 40
        )


def test_launcher_requires_gpu_and_distinct_output():
    module = load_script("multisurvey_distribution_launch_policy")
    plans = yaml.safe_load(
        Path("configs/multisurvey-distribution-launch-20261104.yaml").read_text()
    )["jobs"]
    for plan in plans:
        env = {"MSA_EXPERIMENT_OUTPUT": module.ROOT + plan["run_id"]}
        assert (
            module.bind_distribution_policy(plan, dict(env))[
                "MSA_EXPERIMENT_PRESENTATIONS"
            ]
            == "0"
        )
        with pytest.raises(ValueError, match="resource"):
            module.bind_distribution_policy(
                dict(plan, resources={"cores": 4, "ram": 64, "gpu": 0}), env
            )


def test_saved_array_crps_and_interval_width_have_known_normal_limit():
    module = load_script("report_multisurvey_distribution")
    data = {
        "observed_mean": np.array([0.0]),
        "reference": np.array([0.0]),
        "observed_variance": np.array([1.0]),
        "observed_pit": np.array([0.5]),
    }
    result = module.summarize(data, np.array([True]))
    # Closed-form E|X| - E|X-X'|/2 at y=0 for independent unit normals.
    assert result["crps"] == pytest.approx((2**0.5 - 1) / np.pi**0.5)
    assert result["width90"] == pytest.approx(3.2897072539029444)
    assert result["coverage90"] == 1
