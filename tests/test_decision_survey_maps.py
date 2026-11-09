import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "decision_maps", Path(__file__).parents[1] / "scripts/run_decision_survey_maps.py"
)
maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps)


def test_new_map_recovers_offset_and_preserves_full_covariance():
    rng = np.random.default_rng(9)
    mean = rng.normal(size=(200, 1, 3))
    cov = np.broadcast_to(np.eye(3) * 0.01, (200, 1, 3, 3)).copy()
    value = mean[:, 0, 1] + 0.3
    fit = maps.fit_map(
        (np.ones((200, 1)), mean, cov, value, np.full(200, 0.01)), 1, "offset"
    )
    assert abs(fit["parameters"][0] - 0.3) < 1e-3
    covariance = np.array([[[[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]]]])
    _, variance, a, _ = maps.project(
        np.zeros((1, 1, 3)), covariance, np.array([0.0, 2.0, 0.0, 0.2]), 0, "affine"
    )
    np.testing.assert_array_equal(a, [1.0, 2.0, 0.0])
    np.testing.assert_allclose(variance, 44.0)
