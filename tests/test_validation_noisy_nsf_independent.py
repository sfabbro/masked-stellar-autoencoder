"""Finite-QMC label-error convolution invariants; no production density helpers."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/xp/monitor"))
from verify_validation_noisy_nsf_fp64 import manual_label_convolution, measurement_nodes


def normal_log_density(x):
    return -0.5 * (x.square() + math.log(2 * math.pi)).sum(-1)


def test_qmc_exact_antithetic_zero_error_limit_and_invalid_errors():
    nodes = measurement_nodes().double()
    assert nodes.shape == (32, 3)
    torch.testing.assert_close(nodes[:16], -nodes[16:], rtol=0, atol=0)
    y = torch.tensor([[0.1, -0.3, 0.7], [2.0, 3.0, -4.0]], dtype=torch.float64)
    got = manual_label_convolution(y, torch.zeros_like(y), nodes, normal_log_density)
    torch.testing.assert_close(got, normal_log_density(y), atol=1e-14, rtol=1e-14)
    with pytest.raises(ValueError, match="nonnegative"):
        manual_label_convolution(y, -torch.ones_like(y), nodes, normal_log_density)
    with pytest.raises(ValueError, match="contract"):
        measurement_nodes(64)


def test_finite_qmc_matches_numpy_and_native_density_jacobian():
    nodes = measurement_nodes().double()
    y = torch.tensor([[0.2, -0.4, 1.1], [-1.0, 0.7, 2.0]], dtype=torch.float64)
    sigma = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.3, 0.2]], dtype=torch.float64)
    actual = manual_label_convolution(y, sigma, nodes, normal_log_density)
    x = y.numpy()[None] + sigma.numpy()[None] * nodes.numpy()[:, None]
    lp = -0.5 * (x * x + np.log(2 * np.pi)).sum(-1)
    largest = lp.max(0)
    expected = largest + np.log(np.exp(lp - largest).mean(0))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-14, atol=1e-14)
    scale = torch.tensor([900.0, 0.8, 0.4], dtype=torch.float64)
    native_y, native_sigma = y * scale, sigma * scale
    jac = scale.log().sum()
    native = manual_label_convolution(
        native_y, native_sigma, nodes, lambda q: normal_log_density(q / scale) - jac
    )
    torch.testing.assert_close(native, actual - jac, rtol=1e-14, atol=1e-14)
