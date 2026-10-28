"""Analytic invariants for the independent evaluation auditor."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).parents[1]
    / "experiments/xp/monitor/verify_validation_evaluations.py"
)
spec = importlib.util.spec_from_file_location("independent_evaluation", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_finite_sample_rank_is_not_interpolated_and_rejects_short_calibration():
    assert audit.rank_quantile(np.arange(9), 0.2) == 7
    with pytest.raises(ValueError, match="insufficient"):
        audit.rank_quantile(np.arange(2), 0.1)


def test_bootstrap_preserves_pairing_and_bias_sign():
    a = np.arange(12.0) - 3
    np.testing.assert_array_equal(audit.bootstrap(a, a, 5), np.zeros((2, 3)))
    assert audit.stat(np.array([-2.0, 1.0]))[1] == -0.5


def test_zero_family_gamma_is_deterministic_and_label_scale_equivariant():
    residual = np.linspace(0.1, 2, 60)
    sigma = np.ones(60)
    assert audit.gamma_selection(residual, sigma, np.zeros(60), 0.1) == 0
    family = np.linspace(0.1, 0.2, 60)
    assert audit.gamma_selection(residual, sigma, family, 0.1) == audit.gamma_selection(
        10 * residual, 10 * sigma, 10 * family, 0.1
    )
