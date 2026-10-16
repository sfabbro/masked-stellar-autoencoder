import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from hipparcos_reference import ordinary_reference_mask, propagate_position


def test_negative_parallax_retained_and_special_gaia_matches_excluded():
    n = 6
    yes = np.ones(n)
    m = ordinary_reference_mask(
        np.full(n, 5),
        yes,
        yes,
        yes,
        [-2, 0, 2, 2, np.nan, 1],
        yes,
        yes,
        [8, 8, 16, 72, 8, 8],
    )
    assert m.tolist() == [True, True, False, False, False, True]


def test_epoch_motion_matches_independent_astropy_at_wrap_poles_and_high_pm():
    ra = np.array([359.99999, 0, 127, 80]) * u.deg
    dec = np.array([0, 89.99, -89.99, 40]) * u.deg
    pmra = np.array([10000, 100, -500, 20]) * u.mas / u.yr
    pmdec = np.array([100, 200, 300, -40]) * u.mas / u.yr
    c = SkyCoord(
        ra=ra,
        dec=dec,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        obstime=Time(1991.25, format="jyear"),
    )
    expected = c.apply_space_motion(Time(2016, format="jyear"))
    got = propagate_position(
        ra.to_value(u.rad),
        dec.to_value(u.rad),
        pmra.to_value(u.mas / u.yr),
        pmdec.to_value(u.mas / u.yr),
        24.75,
    )
    own = SkyCoord(
        x=got[:, 0], y=got[:, 1], z=got[:, 2], representation_type="cartesian"
    )
    assert expected.separation(own).arcsec.max() < 1e-4
    zero = propagate_position(
        ra.to_value(u.rad), dec.to_value(u.rad), np.zeros(4), np.zeros(4), 24.75
    )
    original = SkyCoord(
        x=zero[:, 0], y=zero[:, 1], z=zero[:, 2], representation_type="cartesian"
    )
    assert c.separation(original).arcsec.max() < 1e-7
