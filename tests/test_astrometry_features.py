import numpy as np

from masked_stellar_autoencoder.training.astrometry_features import (
    apply_parallax_input_policy,
    parallax_label_asinh,
    parallax_label_error_asinh,
    parallax_label_error_log10,
    parallax_label_log10,
)


def test_snr_clipped_policy():
    train = np.array([[1.0, 0.01], [2.0, 0.02]], dtype=np.float32)
    valid = train.copy()
    test = train.copy()
    et = np.array([[0.1, 0.001], [0.2, 0.002]], dtype=np.float32)
    ev = et.copy()
    es = et.copy()
    apply_parallax_input_policy(
        train, valid, test, et, ev, es, 0, "snr_clipped", snr_cap=50.0
    )
    assert np.allclose(train[:, 0], [10.0, 10.0])
    assert np.allclose(et[:, 0], 1.0)


def test_log10_labels_mask_nonpositive():
    pi = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    y, m = parallax_label_log10(pi, floor_mas=1e-4)
    assert m[0] and not m[1] and not m[2]
    assert np.isfinite(y[0])
    assert np.isnan(y[1]) and np.isnan(y[2])


def test_log10_label_error_positive():
    pi = np.array([10.0])
    e = np.array([1.0])
    s = parallax_label_error_log10(pi, e, floor_mas=1e-4)
    assert s[0] > 0 and np.isfinite(s[0])


def test_asinh_labels():
    pi = np.array([1.0, -1.0, 0.0, np.nan, np.inf, -np.inf], dtype=np.float64)
    y, m = parallax_label_asinh(pi, scale_mas=2.0)
    assert np.all(m == [True, True, True, False, False, False])
    assert np.allclose(y[:3], np.arcsinh(pi[:3] / 2.0))
    assert np.all(np.isnan(y[3:]))


def test_asinh_label_error():
    pi = np.array([10.0, -10.0, 0.0, np.nan, np.inf])
    e = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    s = parallax_label_error_asinh(pi, e, scale_mas=2.0)

    # 1 / (scale * sqrt((pi/scale)^2 + 1)) * e
    # for pi=0, scale=2 -> 1 / (2 * sqrt(1)) * 1 = 0.5
    assert np.isclose(s[2], 0.5)

    # Error should be positive for finite values
    assert np.all(s[:3] > 0)
    assert np.all(np.isfinite(s[:3]))

    # Non-finite inputs should result in NaN error
    assert np.all(np.isnan(s[3:]))
