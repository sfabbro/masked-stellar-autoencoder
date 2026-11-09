import numpy as np
import pytest

from masked_stellar_autoencoder.training.marginal_calibration import (
    central_bounds,
    central_mass,
)


def test_rank_correction_and_unattainable_coverage():
    scores = np.linspace(0.1, 0.9, 9)
    assert central_mass(scores, 0.9) == pytest.approx(0.8)
    assert np.isposinf(central_mass(scores, 0.95))
    with pytest.raises(ValueError):
        central_mass([np.nan], 0.9)


def test_bounds_use_cal_rank_without_changing_density():
    grid = np.linspace(0, 1, 101)
    quantiles = np.column_stack([grid, 2 * grid + 1])
    before = quantiles.copy()
    lo, hi = central_bounds(quantiles, grid, 0.8)
    np.testing.assert_allclose(lo, [0.1, 1.2])
    np.testing.assert_allclose(hi, [0.9, 2.8])
    np.testing.assert_array_equal(before, quantiles)
    lo, hi = central_bounds(quantiles, grid, np.inf)
    assert np.isneginf(lo).all() and np.isposinf(hi).all()
    lo, hi = central_bounds(quantiles, grid, 1.0)
    assert np.isneginf(lo).all() and np.isposinf(hi).all()
