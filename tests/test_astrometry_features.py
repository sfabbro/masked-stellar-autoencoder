import os
import sys

import numpy as np

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "training"))

from astrometry_features import (
    apply_parallax_input_policy,
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


def test_asinh_label_error():
    # Mathematical test:
    # d/dpi arcsinh(pi/scale) = 1 / (scale * sqrt((pi/scale)^2 + 1))

    # 1. Test at pi = 0
    pi = np.array([0.0])
    e = np.array([1.0])
    s = parallax_label_error_asinh(pi, e, scale_mas=1.0)
    # denominator should be 1.0 * sqrt(0 + 1) = 1.0
    # result: 1.0 / 1.0 = 1.0
    assert np.isclose(s[0], 1.0)

    # 2. Test at pi = 1.0, scale = 1.0
    pi = np.array([1.0])
    e = np.array([2.0])
    s = parallax_label_error_asinh(pi, e, scale_mas=1.0)
    # denominator should be 1.0 * sqrt(1 + 1) = sqrt(2)
    # result: 2.0 / sqrt(2) = sqrt(2)
    assert np.isclose(s[0], np.sqrt(2.0))

    # 3. Test at pi = 2.0, scale = 0.5
    pi = np.array([2.0])
    e = np.array([0.5])
    s = parallax_label_error_asinh(pi, e, scale_mas=0.5)
    # denominator should be 0.5 * sqrt((2/0.5)^2 + 1) = 0.5 * sqrt(16 + 1) = 0.5 * sqrt(17)
    # result: 0.5 / (0.5 * sqrt(17)) = 1 / sqrt(17)
    assert np.isclose(s[0], 1.0 / np.sqrt(17.0))


def test_asinh_label_error_nonfinite():
    pi = np.array([1.0, np.nan, np.inf, 1.0], dtype=np.float64)
    e = np.array([np.nan, 1.0, 1.0, np.inf], dtype=np.float64)
    s = parallax_label_error_asinh(pi, e, scale_mas=1.0)

    # According to the function's logic:
    # m = np.isfinite(pi) & np.isfinite(e)
    # So non-finite elements should be NaN because out is initialized with np.nan
    # and m is False for these indices
    assert np.all(np.isnan(s))
