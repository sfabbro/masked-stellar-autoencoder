"""Synthetic only; no native checkpoints or stellar models."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchregress.models.copula_spline")

from masked_stellar_autoencoder.training.remediation_copula6 import CopulaSplineSixHead

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_remediation_pilot as pilot


def test_six_target_sparse_marginal_does_not_integrate_missing_coordinates():
    head = CopulaSplineSixHead(3, 8, 64, 17).double()
    context = torch.zeros(3, 3, dtype=torch.double)
    values = torch.tensor(
        [[0.1] * 6, [0.0, 0.0, 0.0, 0.2, 0.0, 0.0], [0.0] * 6], dtype=torch.double
    )
    mask = torch.tensor([[1] * 6, [0, 0, 0, 1, 0, 0], [0] * 6], dtype=torch.bool)
    diag = head.diagnostics(context, values, torch.zeros_like(values), mask)
    torch.testing.assert_close(
        diag["relative_ess"], torch.ones(3, dtype=torch.double), atol=1e-12, rtol=0
    )
    assert diag["independent_replica_log_prob_gap"].max() < 1e-12
    assert diag["refinement_log_prob_gap"].max() < 1e-12
    assert torch.isfinite(
        head.criterion(context, values, torch.zeros_like(values), mask)
    )


def test_six_target_draws_are_joint_not_scalar_reassembly():
    head = CopulaSplineSixHead(3, 8, 64, 17).double()
    with torch.no_grad():
        head.flow.net[-1].bias[6 + 1] = 0.8
    draw = head.sample(
        torch.zeros(1, 3, dtype=torch.double), 10000, torch.Generator().manual_seed(7)
    )
    assert draw.shape == (10000, 1, 6)
    assert torch.corrcoef(draw[:, 0].T)[0, 1] > 0.5


def test_device_preflight_is_copy_only_and_restores_rng():
    torch.manual_seed(12)
    head = CopulaSplineSixHead(3, 8, 64, 17).double()
    context = torch.randn(4, 3, dtype=torch.double)
    target = torch.randn(4, 6, dtype=torch.double)
    sigma = torch.full_like(target, 0.1)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    state = {k: v.clone() for k, v in head.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = pilot.copula_device_preflight(head, context, target, sigma, mask)
    assert result["passed"] and result["copy_only"]
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, state[key])
