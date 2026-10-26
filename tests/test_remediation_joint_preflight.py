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
