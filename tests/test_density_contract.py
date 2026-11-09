import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from masked_stellar_autoencoder.training.density_contract import (
    TARGETS,
    physical_log_density,
    standardized_coordinates,
    target_axes,
)


def test_declared_six_target_axes_reject_invalid_and_duplicate_quantities():
    assert target_axes(TARGETS) == tuple(range(6))
    with pytest.raises(ValueError):
        target_axes(("age", "age"))
    with pytest.raises(ValueError):
        target_axes(("unknown",))


def test_positive_age_density_normalizes_in_physical_units():
    center = np.zeros(6)
    center[4] = np.log(5)
    scale = np.ones(6)
    scale[4] = 0.4

    def density(age):
        y = np.array([5000, 2.5, -0.5, 0.1, age, 2.0])
        z, jac = standardized_coordinates(y, center, scale)
        return np.exp(physical_log_density(norm.logpdf(z[4]), jac, ("age",)))

    integral, _ = quad(density, 0, np.inf, epsabs=1e-8)
    np.testing.assert_allclose(integral, 1, atol=1e-7)
    with pytest.raises(ValueError):
        standardized_coordinates(np.array([5000, 2.5, 0, 0, -1, 2]), center, scale)


def test_coordinate_jacobians_compose_to_joint_density():
    y = np.array([5000, 2.5, -0.5, 0.1, 5.0, 2.0])
    _, jac = standardized_coordinates(y, np.zeros(6), np.ones(6))
    joint = physical_log_density(-3.0, jac, ("age", "distance"))
    given = physical_log_density(-1.0, jac, ("distance",))
    conditional = physical_log_density(-2.0, jac, ("age",))
    np.testing.assert_allclose(joint - given, conditional)
    all_axes = target_axes(TARGETS)
    assert all_axes == tuple(range(6))
