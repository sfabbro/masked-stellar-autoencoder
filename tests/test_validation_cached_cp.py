import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "cached_cp", Path("experiments/xp/monitor/verify_validation_cached_cp.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_order_statistic_and_native_affine_equivariance():
    target = np.arange(30.0).reshape(10, 3)
    mean = np.zeros_like(target)
    q = module.threshold(mean, target, 0.2)
    np.testing.assert_array_equal(q, target[8])
    center, scale = np.array([5000.0, 3.0, -0.2]), np.array([800.0, 0.7, 0.3])
    np.testing.assert_allclose(
        module.threshold(center + scale * mean, center + scale * target, 0.2), q * scale
    )


def test_short_and_nonfinite_calibration_rejected():
    with pytest.raises(ValueError, match="insufficient"):
        module.threshold(np.zeros((2, 3)), np.ones((2, 3)), 0.1)
    with pytest.raises(ValueError, match="finite"):
        module.threshold(np.zeros((10, 3)), np.full((10, 3), np.nan), 0.1)
