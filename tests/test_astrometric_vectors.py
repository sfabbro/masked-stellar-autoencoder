import numpy as np
import pandas as pd

from masked_stellar_autoencoder.training.astrometric_vectors import directions


def test_motion_accepts_series_preserves_units_and_cos_declination():
    r = directions(
        pd.Series([0.0]), pd.Series([60.0]), pd.Series([1000.0]), pd.Series([0.0]), 1.0
    )
    longitude = np.rad2deg(np.arctan2(r[:, 1], r[:, 0]))
    np.testing.assert_allclose(longitude, [1000 / 3600000 / 0.5], atol=1e-9, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1.0)


def test_epoch_zero_and_ra_wrap_do_not_create_singular_vectors():
    a = directions([0.0, 360.0, 359.999], [90.0, 90.0, 0.0])
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a[0], a[1], atol=1e-14)
    b = directions([359.999], [0.0], [1000.0], [0.0], 16.0)
    longitude = np.mod(np.rad2deg(np.arctan2(b[:, 1], b[:, 0])), 360.0)
    np.testing.assert_allclose(
        longitude, [(359.999 + 16000 / 3600000) % 360], atol=1e-9
    )
