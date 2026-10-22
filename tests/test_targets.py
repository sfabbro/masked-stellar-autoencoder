"""Frozen target registry: name-driven transforms and per-row Jacobians."""

import math

import numpy as np
import pytest

from masked_stellar_autoencoder.training.targets import (
    ASYMMETRIC_QUANTILES,
    IDENTITY,
    INTERFACE_ONLY,
    LOG_GYR,
    LOG_KPC,
    OBSERVABLE_KINDS,
    RAW_SIGNED_PARALLAX,
    REGISTRY,
    SYMMETRIC,
    TARGET_NAMES,
    TARGET_SPECS,
    TARGET_TRANSFORMS,
    TARGET_UNITS,
    TRAINABLE,
    density_offsets,
    log_abs_det_jacobian,
    model_coordinates,
    native_from_coordinates,
    native_from_standardized,
    native_log_prob,
    target_index,
    target_spec,
    validate_targets,
)


def test_registry_declares_the_six_targets_in_frozen_order():
    assert TARGET_NAMES == ("teff", "logg", "fe_h", "distance", "age", "alpha")
    assert TARGET_UNITS == ("K", "dex", "dex", "kpc", "Gyr", "dex")
    assert TARGET_TRANSFORMS == (
        "identity",
        "identity",
        "identity",
        "log_kpc",
        "log_gyr",
        "identity",
    )
    assert len(TARGET_SPECS) == len(REGISTRY) == 6
    assert target_spec("distance").positive and target_spec("distance").error_kind == (
        RAW_SIGNED_PARALLAX
    )
    assert target_spec("age").error_kind == ASYMMETRIC_QUANTILES
    assert target_spec("alpha").status == INTERFACE_ONLY
    assert target_spec("alpha").observable_kind == "proxy"
    assert target_spec("teff").status == TRAINABLE
    assert {spec.observable_kind for spec in TARGET_SPECS} <= set(OBSERVABLE_KINDS)


def test_unregistered_duplicate_or_reordered_targets_are_rejected():
    with pytest.raises(ValueError, match="not in the frozen registry"):
        target_spec("parallax")
    with pytest.raises(ValueError, match="duplicate"):
        validate_targets(("teff", "teff"))
    with pytest.raises(ValueError, match="registry order"):
        validate_targets(("age", "distance"))
    with pytest.raises(ValueError):
        validate_targets(())
    assert validate_targets(("fe_h", "distance", "age")) == (
        "fe_h",
        "distance",
        "age",
    )
    assert target_index("age") == 4


def test_identity_transform_has_unit_jacobian_and_round_trips():
    values = np.array([[4200.0, 4.1, -1.2], [6100.0, 2.0, 0.3]])
    names = ("teff", "logg", "fe_h")
    coordinates = model_coordinates(names, values)
    np.testing.assert_array_equal(coordinates, values)
    np.testing.assert_array_equal(native_from_coordinates(names, coordinates), values)
    np.testing.assert_array_equal(
        log_abs_det_jacobian(names, values), np.zeros_like(values)
    )
    scale = np.array([100.0, 0.2, 0.1])
    np.testing.assert_allclose(
        density_offsets(names, values, scale),
        np.broadcast_to(np.log(scale), values.shape),
        rtol=0,
        atol=0,
    )


def _expected_lognormal_log_density(native, center, scale):
    """Independent NumPy log density of LogNormal(log-mean=center, log-sd=scale)."""
    standardized = (np.log(native) - center) / scale
    return -0.5 * (standardized**2 + math.log(2 * math.pi)) - np.log(scale) - np.log(
        native
    )


@pytest.mark.parametrize(
    "name,scale",
    [("distance", 0.37), ("age", 0.61)],
)
def test_log_transform_jacobian_gives_the_exact_lognormal_density(name, scale):
    native = np.array([0.4, 1.0, 3.5, 12.0]) ** (1 if name == "age" else 1)
    center = 1.3
    standardized = (np.log(native) - center) / scale
    log_prob_standardized = -0.5 * (standardized**2 + math.log(2 * math.pi))
    actual = native_log_prob(
        (name,), log_prob_standardized, native, np.array([scale])
    )
    np.testing.assert_allclose(
        actual, _expected_lognormal_log_density(native, center, scale), rtol=1e-13, atol=1e-13
    )
    offsets = density_offsets((name,), native, np.array([scale]))
    np.testing.assert_allclose(
        offsets[:, 0], np.log(scale) + np.log(native), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        log_abs_det_jacobian((name,), native)[:, 0], -np.log(native), rtol=0, atol=0
    )


def test_a_single_scalar_jacobian_is_wrong_for_log_transforms():
    """The legacy scalar Jacobian is off by exactly +log d / +log age per row."""
    values = np.array([[0.5], [4.0], [20.0]])
    scale = np.array([0.3])
    per_row = density_offsets(("distance",), values, scale).sum(1)
    scalar = float(np.log(scale).sum())
    np.testing.assert_allclose(
        per_row - scalar, np.log(values[:, 0]), rtol=1e-14, atol=1e-14
    )
    assert np.abs(per_row - scalar).max() > 1.0


def test_non_positive_distance_or_age_raises_instead_of_producing_infinity():
    for name in ("distance", "age"):
        with pytest.raises(ValueError, match="positive"):
            target_spec(name).to_coordinate(np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="positive"):
            density_offsets((name,), np.array([-1.0]), np.array([1.0]))
    # Identity targets declare positivity metadata but never reject a simulated
    # score: only the log transforms need a strictly positive native argument.
    np.testing.assert_array_equal(
        target_spec("teff").to_coordinate(np.array([-5000.0])), np.array([-5000.0])
    )
    assert target_spec("teff").positive and not target_spec("teff").log_transformed


def test_inverse_mapping_is_name_driven_not_positional():
    names = ("teff", "distance", "age")
    standardized = np.array([[0.5, 1.0, -1.0], [-0.2, 0.3, 0.4]])
    center = np.array([5000.0, 1.0, 2.0])
    scale = np.array([500.0, 0.5, 0.7])
    native = native_from_standardized(names, standardized, center, scale)
    coordinate = standardized * scale + center
    np.testing.assert_allclose(native[:, 0], coordinate[:, 0], rtol=1e-14)
    np.testing.assert_allclose(native[:, 1], np.exp(coordinate[:, 1]), rtol=1e-14)
    np.testing.assert_allclose(native[:, 2], np.exp(coordinate[:, 2]), rtol=1e-14)
    # The same model coordinates read under a different declared order would be
    # rejected outright, so no positional inversion is possible.
    with pytest.raises(ValueError, match="registry order"):
        native_from_standardized(("age", "teff"), standardized[:, :2], center[:2], scale[:2])
    with pytest.raises(ValueError, match="(row|values)"):
        native_from_standardized(names, standardized[:, :2], center, scale)


def test_density_offsets_ignore_unobserved_coordinates():
    names = ("teff", "distance", "age")
    native = np.array([[5000.0, 1.0, 2.0], [4500.0, 4.0, 8.0]])
    scale = np.array([500.0, 0.5, 0.7])
    observed = np.array([[True, False, True], [False, False, False]])
    offsets = density_offsets(names, native, scale, observed=observed)
    np.testing.assert_allclose(offsets[0, 0], np.log(500.0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(offsets[0, 1], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(
        offsets[0, 2], np.log(0.7) + np.log(2.0), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_array_equal(offsets[1], np.zeros(3))
    terms = log_abs_det_jacobian(names, native, observed=observed)
    np.testing.assert_array_equal(terms[1], np.zeros(3))


def test_non_three_dimensional_target_vectors_are_supported():
    names = ("teff", "logg", "fe_h", "distance", "age")
    native = np.array([[5000.0, 4.0, -1.0, 2.0, 3.0]])
    scale = np.array([500.0, 0.2, 0.1, 0.4, 0.6])
    offsets = density_offsets(names, native, scale)
    assert offsets.shape == (1, 5)
    expected = np.log(scale) + np.array([0.0, 0.0, 0.0, np.log(2.0), np.log(3.0)])
    np.testing.assert_allclose(offsets[0], expected, rtol=1e-14, atol=1e-14)


def test_spec_transforms_are_their_own_inverse_round_trip():
    for spec in TARGET_SPECS:
        native = np.array([1.5, 3.0, 7.5])
        np.testing.assert_allclose(
            spec.to_native(spec.to_coordinate(native)), native, rtol=1e-14
        )
    assert REGISTRY["distance"].transform == LOG_KPC
    assert REGISTRY["age"].transform == LOG_GYR
    assert REGISTRY["teff"].transform == IDENTITY
    assert REGISTRY["distance"].error_kind == RAW_SIGNED_PARALLAX
    assert REGISTRY["_".join(["fe", "h"])].error_kind == SYMMETRIC
