import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stellar_sky_candidates import nearest_two, unambiguous_candidates, unit_vectors


def test_spherical_neighbors_independent_of_ra_wrap_and_poles():
    ra = np.array([359.9999, 0.0006, 123, 310, 180])
    dec = np.array([0, 0, 89.9999, -45, 0])
    qa = np.array([0, 20, 310.0001])
    qd = np.array([0, 89.9999, -45])
    rows, sep = nearest_two(ra, dec, qa, qd)
    brute = (
        SkyCoord(qa * u.deg, qd * u.deg)[:, None]
        .separation(SkyCoord(ra * u.deg, dec * u.deg)[None, :])
        .arcsec
    )
    expected = np.argsort(brute, axis=1)[:, :2]
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_allclose(
        sep, np.take_along_axis(brute, expected, axis=1), atol=1e-7, rtol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(unit_vectors(ra, dec), axis=1), 1, atol=1e-15
    )


def test_second_neighbor_and_all_many_to_one_candidates_rejected():
    rows = np.array([[1, 2], [1, 3], [4, 5], [6, 7], [8, 9]])
    sep = np.array([[0.1, 3], [0.2, 4], [0.3, 0.5], [2, 4], [0.1, 4]])
    assert unambiguous_candidates(rows, sep).tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    with pytest.raises(ValueError):
        unit_vectors([360], [0])
    with pytest.raises(ValueError):
        unit_vectors([1], [91])
    with pytest.raises(ValueError):
        nearest_two([1], [0], [1], [0])
    r, s = nearest_two([0, 180], [0, 0], [], [])
    assert r.shape == s.shape == (0, 2)
