"""Read-only natural-input RAM cache with the admitted parallax override."""

import numpy as np


def cache_natural_inputs(data, profile):
    values = data["input"][:].astype("f4", copy=False)
    j = profile["feature_index"]
    names = data["feature_names"][:].astype("U").tolist()
    if values.ndim != 2 or len(names) != values.shape[1] or names[j] != "PARALLAX":
        raise ValueError("Input layout or parallax feature identity differs")
    parallax = data[profile["override_dataset"]][:]
    available = data[profile["availability_dataset"]][:].astype(bool)
    if (
        parallax.shape != (len(values),)
        or available.shape != parallax.shape
        or not np.isfinite(parallax[available]).all()
        or not np.all(parallax[~available] == -9999)
    ):
        raise ValueError("Parallax availability and admitted values differ")
    values[:, j] = parallax
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite admitted input")
    values.setflags(write=False)
    return values
