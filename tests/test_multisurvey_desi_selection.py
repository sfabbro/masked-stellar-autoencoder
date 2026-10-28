"""Synthetic repeat resolution and base-preserving overlay checks."""

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.multisurvey_desi_selection import (
    create_overlay,
    select_one_per_star,
)
from masked_stellar_autoencoder.training.multisurvey_overlay import (
    file_sha256,
    write_observation_sidecar,
)


def test_selection_uses_snr_then_source_row_not_label_values():
    indices = select_one_per_star(
        np.array([11, 22, 11, 22, 22]),
        np.array([8, 9, 2, 4, 7]),
        np.array([20.0, 30.0, 20.0, 15.0, 30.0]),
    )
    np.testing.assert_array_equal(indices, [2, 4])
    with pytest.raises(ValueError, match="S/N"):
        select_one_per_star(np.array([1]), np.array([2]), np.array([np.nan]))
    with pytest.raises(ValueError, match="unique"):
        select_one_per_star(np.array([1, 2]), np.array([4, 4]), np.array([1.0, 2.0]))


def test_overlay_retains_base_roles_and_counts_new_logg(tmp_path):
    base = tmp_path / "base.h5"
    with h5py.File(base, "w") as h:
        h["source_id"] = np.array([11, 22, 33], dtype="i8")
        h["input"] = np.arange(6, dtype="f4").reshape(3, 2)
        h["role"] = np.array([b"fit", b"fit", b"devtest"])
        h["primary_input_eligible"] = np.ones(3, bool)
        h["observed"] = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], bool)
        h["quality"] = np.ones((3, 3), bool)
        h.create_group("native_observations")
    meta = {
        "release": "synthetic",
        "pipeline": "rv",
        "dependency_id": "one-spectrum",
        "targets": {
            n: {"quantity": n, "unit": u, "field": n}
            for n, u in (("teff", "K"), ("logg", "dex(cm/s2)"), ("fe_h", "dex"))
        },
    }
    rows = {
        "source_id": np.array([11, 22, 22, 99], dtype="i8"),
        "source_row": np.array([0, 1, 2, 3], dtype="i8"),
        "snr_r": np.array([10.0, 11.0, 20.0, 12.0]),
        "value": np.array([[5000.0, 2.0, -1.0]] * 4),
        "error": np.full((4, 3), 0.1),
        "observed": np.ones((4, 3), bool),
    }
    raw = write_observation_sidecar(
        tmp_path / "raw.h5",
        {"desi_dr1_rvs_raw": rows},
        {"desi_dr1_rvs_raw": meta},
        provenance={"origin": "synthetic"},
    )
    result = create_overlay(
        base, file_sha256(base), {}, raw["path"], raw["sha256"], tmp_path / "output"
    )
    assert (
        result["selected"]["selected_rows"] == 3
        and result["selected"]["discarded_from_view"] == 1
    )
    assert result["counts"]["roles"]["fit"]["desi_logg_not_previously_observed"] == 1
    assert result["counts"]["roles"]["fit"]["overlap_desi_and_previous_logg"] == 1
    assert result["counts"]["selected_desi_not_in_base"] == 1
    with h5py.File(tmp_path / "output/overlay.h5", "r") as h:
        np.testing.assert_array_equal(h["role"][:], [b"fit", b"fit", b"devtest"])
        np.testing.assert_array_equal(
            h["input"][:], np.arange(6, dtype="f4").reshape(3, 2)
        )
        np.testing.assert_array_equal(
            h["native_observations/desi_dr1_rvs_raw/source_row"][:], [0, 2, 3]
        )
