"""Pure helper contracts only: never execute the actual CUDA preflight locally."""

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "remediation_preflight", ROOT / "scripts/run_remediation_joint_preflight.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config():
    return yaml.safe_load(
        (ROOT / "configs/remediation-joint-preflight-20261026.yaml").read_text()
    )


def test_bound_config_and_all_masks():
    cfg = config()
    runner.validate_config(cfg)
    mask = runner.all_masks()
    assert mask.shape == (128, 6) and mask.dtype == torch.bool
    assert torch.unique(mask, dim=0).shape[0] == 64
    with pytest.raises(ValueError):
        runner.validate_config(dict(cfg, updates=10))


def test_correlated_synthetic_data_and_whole_vector_covariance_gate():
    generator = torch.Generator().manual_seed(1)
    x, y, mean = runner.synthetic_batch(30000, generator)
    lower, slope = runner.truth_parameters()
    torch.testing.assert_close(mean, x @ slope.T)
    truth = lower @ lower.T
    checks = runner.covariance_checks(y - mean, truth, config()["gates"])
    assert checks["passed"] and all(checks["cross_sign_matches"])
    diagonal = torch.randn((30000, 6), generator=generator, dtype=torch.float64)
    assert not runner.covariance_checks(diagonal, truth, config()["gates"])["passed"]


def test_sparse_gate_requires_ess_and_replica_refinement():
    mask = runner.all_masks()
    diagnostic = dict(
        exact=False,
        relative_ess=torch.ones(128),
        independent_replica_log_prob_gap=torch.zeros(128),
        refinement_log_prob_gap=torch.zeros(128),
    )
    assert runner.sparse_checks(diagnostic, mask, config()["gates"])["passed"]
    diagnostic["relative_ess"][1] = 0.001
    assert not runner.sparse_checks(diagnostic, mask, config()["gates"])["passed"]
    diagnostic["relative_ess"].fill_(1)
    diagnostic["refinement_log_prob_gap"].fill_(0.11)
    assert not runner.sparse_checks(diagnostic, mask, config()["gates"])["passed"]
    assert runner.sparse_checks({"exact": True}, mask, config()["gates"])["passed"]


def test_exact_complete_nsf_path_matches_integrated_objective_and_gradient():
    from masked_stellar_autoencoder.training.remediation_joint6 import JointSixHead

    torch.manual_seed(9)
    head = JointSixHead("nsf", latent_dim=4, width=8, particles=8).double()
    x, y, _ = runner.synthetic_batch(3, torch.Generator().manual_seed(2))
    full = torch.ones_like(y, dtype=torch.bool)
    integrated = head.criterion(x, y, torch.zeros_like(y), full)
    exact = runner.complete_objective(head, x, y)
    torch.testing.assert_close(exact, integrated, rtol=1e-10, atol=1e-10)
    pars = tuple(head.parameters())
    a = torch.autograd.grad(integrated, pars, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(exact, pars, allow_unused=True)
    for left, right in zip(a, b, strict=True):
        if left is not None:
            torch.testing.assert_close(left, right, rtol=5e-5, atol=5e-5)
