"""Analytical tests for the independent validation-wave replay; no production flows."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/xp/monitor"))
import independent_nsf_kernel as kernel
from verify_validation_nsf_fp64 import strict_compare


def test_manual_spline_identity_roundtrip_logjacobian_and_tails():
    x = torch.tensor(
        [-10.0, -7.1, -0.31, 0.22, 6.7, 10.0], dtype=torch.float64, requires_grad=True
    )
    rw = torch.zeros(len(x), 16, dtype=torch.float64)
    rd = torch.zeros(len(x), 15, dtype=torch.float64)
    y, ld = kernel.spline(x, rw, rw, rd)
    xr, ild = kernel.spline(y, rw, rw, rd, inverse=True)
    torch.testing.assert_close(y, x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(xr, x, atol=1e-12, rtol=1e-12)
    derivative = torch.autograd.grad(y.sum(), x)[0]
    torch.testing.assert_close(derivative.log(), ld, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ld + ild, torch.zeros_like(ld), atol=1e-12, rtol=1e-12)


def test_strict_gate_unchanged_boundary_shape_and_nonfinite():
    a = np.zeros(2)
    assert strict_compare(a, a + 4e-5)["passed"]
    assert not strict_compare(a, a + 6e-5)["passed"]
    assert not strict_compare(a, np.array([np.nan, 0]))["passed"]
    with pytest.raises(ValueError, match="shapes"):
        strict_compare(a, np.zeros(1))
