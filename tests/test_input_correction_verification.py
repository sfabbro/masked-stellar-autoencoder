"""Independent correction equations agree with the pinned experiment adapters."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/xp/monitor"))
from verify_input_corrections import correction_reference, independent_extrapolate

from masked_stellar_autoencoder.training.input_measurement import (
    apply_per_star_rc,
    apply_shared_rc,
    fit_univariate_rc,
    rational_extrapolate,
)


@pytest.mark.parametrize("method", ["rc_shared", "rc_per_star"])
def test_independent_rc(method):
    rng = np.random.default_rng(42)
    fit = rng.normal(size=(200, 3)).astype("f4")
    sigma = np.array([0.2, 0.7, 2.0])
    values = rng.normal(size=(20, 3)).astype("f4")
    own = rng.uniform(0.1, 2, size=values.shape)
    models = fit_univariate_rc(fit, sigma)
    actual = (
        apply_shared_rc(models, values)
        if method == "rc_shared"
        else apply_per_star_rc(models, values, own)
    )
    np.testing.assert_allclose(
        correction_reference(fit, sigma, method, values, own),
        actual,
        rtol=1e-6,
        atol=1e-7,
    )


def test_independent_affine_extrapolation():
    levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    c = np.linspace(1.05, 8.0, 400)[123]
    y = np.column_stack(
        [
            np.full(5, -0.2),
            2 + 3 / (c + np.array(levels)),
            -7 + 0.4 / (c + np.array(levels)),
        ]
    )
    expected = independent_extrapolate(levels, y)
    np.testing.assert_allclose(
        expected, rational_extrapolate(levels, y), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        expected, [-0.2, 2 + 3 / (c - 1), -7 + 0.4 / (c - 1)], atol=1e-8
    )
