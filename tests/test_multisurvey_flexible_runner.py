"""Run the complete density-head lifecycle on a small synthetic frozen cache."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


def runner_module():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "flexible_runner", scripts / "run_multisurvey_flexible.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_four_joint_heads_train_reload_and_predict_without_native_weights(tmp_path):
    from masked_stellar_autoencoder.training.multisurvey_flow import reference_posterior

    torch.manual_seed(4)
    path = tmp_path / "synthetic.h5"
    metadata = dict(
        reference_sha256="a" * 64, library_revision="b" * 40, synthetic=True
    )
    with h5py.File(path, "w") as cache:
        cache.attrs["metadata"] = json.dumps(metadata)
        cache.create_dataset("context_center", data=np.zeros(4))
        cache.create_dataset("context_scale", data=np.ones(4))
        for role, n in (("fit", 128), ("tune", 16)):
            group = cache.create_group(role)
            ids = np.arange(n, dtype="i8") + (1 if role == "fit" else 1001)
            group.create_dataset("source_id", data=ids)
            group.attrs["identity_sha256"] = hashlib.sha256(
                ids.astype("<i8").tobytes()
            ).hexdigest()
            design = torch.zeros(n, 9, 3, dtype=torch.float64)
            design[:, :3] = torch.eye(3, dtype=torch.float64)
            residual = torch.zeros(n, 9, dtype=torch.float64)
            residual[:, :3] = torch.randn(n, 3, dtype=torch.float64) * 0.2
            observation = dict(
                design=design,
                residual=residual,
                noise_logdet=torch.zeros(n, dtype=torch.float64),
                count=torch.full((n,), 3),
            )
            mean, lower, lp = reference_posterior(observation)
            fields = dict(
                context=torch.randn(n, 4),
                reference_mean=torch.zeros(n, 3, dtype=torch.float64),
                reference_lower=torch.eye(3, dtype=torch.float64).expand(n, -1, -1),
                posterior_mean=mean,
                posterior_lower=lower,
                reference_lp=lp,
                xp_available_count=torch.full((n,), 110),
                reference_mapping=torch.zeros(n, 9),
                physical_design=design,
                native_value=residual,
                native_mask=(design.abs().sum(-1) > 0),
                native_noise_variance=torch.ones(n, 9),
                source_index=torch.tensor([0, 0, 0, -2, -2, -2, -2, -2, -2]).expand(
                    n, -1
                ),
                **observation,
            )
            for key, value in fields.items():
                group.create_dataset(key, data=value.numpy())
    config = dict(
        density_reference_sha256="a" * 64,
        parent=dict(library_revision="b" * 40),
        seed=4,
        particles=32,
        evaluation_particles=32,
        head_lr=1e-5,
        updates=2,
        batch_size=16,
        micro_batch=8,
        snapshots=[1, 2],
        draws=32,
        corner_draws=32,
        gate_particles=64,
        save_all_snapshots=True,
    )
    module = runner_module()
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        states = module.train(
            config,
            tmp_path,
            dict(
                cache_path=str(path),
                cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            "c" * 40,
            device="cpu",
        )
        from masked_stellar_autoencoder.training.multisurvey_flow import (
            GaussianResidualHead,
            NSFResidualHead,
        )

        data, _ = module.tensor_data(path, roles=("tune",))
        control = GaussianResidualHead(4).double()
        control.load_state_dict(
            torch.load(tmp_path / "gaussian-selected.pth", weights_only=True)["head"]
        )
        control.eval().requires_grad_(False)
        module.attach_defensive_proposal(data["tune"], control, torch.device("cpu"))
        saved = torch.load(tmp_path / "nsf-selected.pth", weights_only=True)
        flow = NSFResidualHead(4, **saved["head_config"]).double()
        flow.load_state_dict(saved["head"])
        flow.eval().requires_grad_(False)
        gate = module.integration_gate(
            flow, "nsf", data["tune"], torch.device("cpu"), particles=64, seed=91
        )
        assert gate["pass_gate"], gate
        score = module.evaluate(
            flow,
            "nsf",
            data["tune"],
            torch.device("cpu"),
            particles=128,
            seed=92,
            include_scores=True,
        )
        assert len(score["per_star_nll"]) == 16
        assert score["nll"] == np.mean(score["per_star_nll"])
    finally:
        torch.set_num_threads(previous)
    assert set(states) == {"gaussian", "mixture", "copula", "nsf"}
    assert all(value["status"] == "complete" for value in states.values()), states
    for kind in states:
        assert (tmp_path / f"{kind}-step-0001.pth").is_file()
        assert (tmp_path / f"{kind}-step-0002.pth").is_file()
        with np.load(
            tmp_path / (kind + "-tune.npz"), allow_pickle=False
        ) as predictions:
            assert predictions["joint_draws"].shape == (10, 32, 3)
            assert predictions["source_id"].dtype == np.int64
            assert np.isfinite(predictions["per_star_nll"]).all()
            assert np.isfinite(predictions["observed_crps"]).all()
            assert (
                (predictions["observed_pit"] >= 0) & (predictions["observed_pit"] <= 1)
            ).all()


def test_new_experiment_keeps_source_pins_and_uses_project_gpu_lock():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "flexible_policy", scripts / "multisurvey_flexible_launch_policy.py"
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    plan = yaml.safe_load(
        Path("configs/multisurvey-flexible-launch-20261105.yaml").read_text()
    )["jobs"][0]
    env = policy.bind_flexible_policy(
        plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + plan["run_id"]}
    )
    assert env["MSA_XP_EXPERIMENTS"] == "0"
    current = yaml.safe_load(
        Path("configs/multisurvey-flexible-20261105.yaml").read_text()
    )["parent"]
    prior = yaml.safe_load(
        Path("configs/multisurvey-distribution-20261104.yaml").read_text()
    )["experiments"][0]["parent"]
    assert all(
        current[k] == v
        for k, v in prior.items()
        if k not in {"run_id", "seed", "library_revision"}
    )


def test_refinement_reuses_original_cache_and_changes_only_integration_budget():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "flexible_policy", scripts / "multisurvey_flexible_launch_policy.py"
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    plan = yaml.safe_load(
        Path("configs/multisurvey-flexible-refinement-launch-20261105.yaml").read_text()
    )["jobs"][0]
    env = policy.bind_flexible_refinement_policy(
        plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + plan["run_id"]}
    )
    config = yaml.safe_load(Path(env["MSA_MULTISURVEY_POLICY"]).read_text())
    original = yaml.safe_load(policy.CONFIG.read_text())
    assert all(
        config[k] == original[k]
        for k in original
        if k not in {"run_id", "particles", "evaluation_particles"}
    )
    assert (
        config["gate_particles"]
        >= config["evaluation_particles"]
        > original["evaluation_particles"]
    )
    assert env["MSA_XP_EXPERIMENTS"] == "0"


def test_defensive_rescore_is_evaluation_only_on_pinned_completed_training():
    import pytest

    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "flexible_policy", scripts / "multisurvey_flexible_launch_policy.py"
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    plan = yaml.safe_load(
        Path("configs/multisurvey-flexible-rescore-launch-20261105.yaml").read_text()
    )["jobs"][0]
    env = {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + plan["run_id"]}
    result = policy.bind_flexible_rescore_policy(plan, dict(env))
    assert (
        result["MSA_XP_EXPERIMENTS"] == "0"
        and result["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    )
    plan["resources"]["gpu"] = 2
    with pytest.raises(ValueError, match="rescoring contract"):
        policy.bind_flexible_rescore_policy(plan, dict(env))


def test_calibration_plan_keeps_preselected_heads_and_performs_no_training():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "flexible_policy", scripts / "multisurvey_flexible_launch_policy.py"
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    plan = yaml.safe_load(
        Path(
            "configs/multisurvey-flexible-calibration-launch-20261105.yaml"
        ).read_text()
    )["jobs"][0]
    env = policy.bind_flexible_calibration_policy(
        plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + plan["run_id"]}
    )
    config = yaml.safe_load(Path(env["MSA_MULTISURVEY_POLICY"]).read_text())
    assert config["primary_head"] == "mixture"
    assert set(config["heads"]) == {"gaussian", "mixture", "copula", "nsf"}
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0" and config["cal_stars"] == 2048
