"""Abundance conversion preserves partial measurements and covariance."""

import numpy as np
import pytest

from masked_stellar_autoencoder.training.multisurvey_abundances import (
    iron_from_global_and_relative,
)


def test_iron_conversion_propagates_positive_and_negative_covariance():
    cov = np.array([[[0.04, 0.01], [0.01, 0.09]], [[0.04, -0.01], [-0.01, 0.09]]])
    result = iron_from_global_and_relative(
        [-0.5, -1], [0.1, 0.2], reference_offset_dex=0.05, covariance=cov
    )
    np.testing.assert_allclose(result["value"], [-0.35, -0.75])
    np.testing.assert_allclose(result["variance"], [0.15, 0.11])
    assert result["observed"].tolist() == [True, True]


def test_missing_iron_or_unknown_covariance_is_never_imputed():
    cov = np.array([[[0.04, np.nan], [np.nan, 0.09]], [[0.04, 0], [0, 0.09]]])
    result = iron_from_global_and_relative(
        [-0.5, -1], [0.1, np.nan], reference_offset_dex=0, covariance=cov
    )
    assert result["observed"].tolist() == [True, False]
    assert np.isnan(result["variance"]).all() and np.isnan(result["value"][1])
    missing = iron_from_global_and_relative([-0.5], [0.1], reference_offset_dex=0)
    assert np.isnan(missing["variance"][0])


def test_invalid_covariance_and_reference_offset_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        iron_from_global_and_relative(
            [0], [0], reference_offset_dex=0, covariance=[[[1, 0.2], [0.1, 1]]]
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        iron_from_global_and_relative(
            [0], [0], reference_offset_dex=0, covariance=[[[1, 2], [2, 1]]]
        )
    with pytest.raises(ValueError, match="reference"):
        iron_from_global_and_relative([0], [0], reference_offset_dex=float("nan"))
    with pytest.raises(ValueError, match="aligned"):
        iron_from_global_and_relative(
            [0], [0], reference_offset_dex=0, covariance=np.eye(2)
        )


def test_scalar_covariance_and_boolean_reference_offset():
    result = iron_from_global_and_relative(
        -0.5, 0.1, reference_offset_dex=0.0, covariance=np.eye(2)
    )
    assert np.isclose(result["value"], -0.4)
    assert np.isclose(result["variance"], 2.0)
    with pytest.raises(ValueError, match="reference"):
        iron_from_global_and_relative(-0.5, 0.1, reference_offset_dex=True)
