"""Synthetic analytic/API checks only; no local native-data model replay."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/xp/monitor"))
import independent_lowdim_nsf_kernel as kernel

from masked_stellar_autoencoder.training.multilabel_experiments import MultiLabelHead


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_lowdim_kernel_matches_public_oracle_and_analytic_jacobian(dimensions):
    torch.manual_seed(29)
    head = MultiLabelHead(
        "flow", latent_dim=4, target_dim=dimensions, width=16, flow_type="nsf"
    ).double()
    with torch.no_grad():
        head.output.weight.normal_(std=0.1)
        head.output.bias.normal_(std=0.1)
    state = head.state_dict()
    x = torch.randn(4, 4, dtype=torch.double)
    y = torch.randn(4, dimensions, dtype=torch.double)
    context = head(x)
    actual = kernel.density(y, context, state)
    torch.testing.assert_close(
        actual, head.distribution(x).log_prob(y), atol=1e-11, rtol=1e-11
    )
    z, ld = kernel.transform(y, context, state)
    torch.testing.assert_close(
        kernel.inverse(z, context, state), y, atol=1e-10, rtol=1e-10
    )
    for row in range(4):
        jac = torch.autograd.functional.jacobian(
            lambda value: kernel.transform(value[None], context[row : row + 1], state)[
                0
            ][0],
            y[row],
        )
        torch.testing.assert_close(
            torch.linalg.slogdet(jac)[1], ld[row], atol=1e-10, rtol=1e-10
        )


@pytest.mark.parametrize("kind", ["gaussian", "mdn"])
def test_independent_scalar_head_matches_public_analytic_oracle(kind):
    import numpy as np
    from verify_alpha_frozen_heads import comparison, head_arrays, order_threshold

    head = MultiLabelHead(kind, latent_dim=4, target_dim=1, width=16).double()
    torch.manual_seed(71)
    with torch.no_grad():
        head.output.weight.normal_(std=0.15)
        head.output.bias.normal_(std=0.15)
    x = torch.randn(5, 4, dtype=torch.double)
    y = torch.randn(5, 1, dtype=torch.double)
    mean, variance, log_prob = head_arrays(x, y, head.state_dict(), kind)
    expected_mean, expected_variance = head.moments(x)
    torch.testing.assert_close(mean, expected_mean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(variance, expected_variance, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(log_prob, head.log_prob(x, y), atol=1e-12, rtol=1e-12)
    assert order_threshold(np.arange(9), 0.1) == 8
    with pytest.raises(ValueError, match="Unbounded"):
        order_threshold(np.arange(4), 0.05)
    assert not comparison(np.array([0.0]), np.array([np.nan]))["passed"]
