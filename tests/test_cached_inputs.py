import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.cached_inputs import cache_natural_inputs


def test_cached_profile_is_exact_readonly_preserves_signed_parallax_and_source(
    tmp_path,
):
    path = tmp_path / "input.h5"
    original = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype="f4")
    with h5py.File(path, "w") as f:
        f["input"] = original
        f["feature_names"] = np.array(["G", "PARALLAX"], dtype="S16")
        f["p"] = np.array([0.5, -9999, -0.3], dtype="f4")
        f["seen"] = [True, False, True]
    with h5py.File(path) as f:
        cached = cache_natural_inputs(
            f, dict(feature_index=1, override_dataset="p", availability_dataset="seen")
        )
        np.testing.assert_array_equal(
            cached, np.array([[1, 0.5], [2, -9999], [3, -0.3]], dtype="f4")
        )
        np.testing.assert_array_equal(f["input"][:], original)
        assert not cached.flags.writeable
        with pytest.raises(ValueError):
            cached[0, 0] = 0
