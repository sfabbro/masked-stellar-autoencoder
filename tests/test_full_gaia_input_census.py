import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


spec = importlib.util.spec_from_file_location(
    "input_census", Path(__file__).parents[1] / "scripts/run_full_gaia_input_census.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_census_counts_unique_stars_preserves_roles_and_admits_no_xp(tmp_path):
    gaia = tmp_path / "full_gaia_gaia_stage_20261108_v2"
    photo = tmp_path / "full_gaia_photometry_stage_20261108_v1"
    (gaia / "labelled-gaia").mkdir(parents=True)
    (photo / "raw-observations").mkdir(parents=True)
    ids = np.arange(100000000000000001, 100000000000000006, dtype="i8")
    for folder in [gaia, photo]:
        np.save(folder / "label-candidate-source-id.npy", ids)
    pq.write_table(pa.table({
        "source_id": ids, "candidate_row": [0, 1, 2, 3, 4],
        "phot_g_mean_mag": [17., None, 18., None, 19.],
        "parallax": [-0.2, 0.5, 1.0, 0.3, 0.4], "parallax_error": [0.1, None, 0.1, 0.2, 0.1],
        "has_xp_continuous": [False, True, False, False, False],
    }), gaia / "labelled-gaia/000.parquet")
    pq.write_table(pa.table({
        "source_id": [int(ids[1]), int(ids[1])], "candidate_row": [1, 1],
        "bp_1": [1., 2.], "bpe_1": [0.1, 0.2],
    }), photo / "raw-observations/000.parquet")
    prepared = tmp_path / "expanded_six_source_20261107_v5"
    prepared.mkdir()
    with h5py.File(prepared / "prepared.h5", "w") as f:
        f["source_id"] = ids[[0, 1, 2, 4]]
        f["role"] = np.array([b"tune", b"fit", b"external", b"excluded"])
    result = m.census(tmp_path, {
        "feature_names": ["G", "bp_1", "PARALLAX"],
        "error_names": [None, "bpe_1", "parallax_error"],
    }, lambda *args: None)
    assert result["candidate_stars"] == 5
    assert result["channels"]["bp_1"]["visible_unique_stars"] == 1
    assert result["channels"]["bp_1"]["known_positive_error_unique_stars"] == 1
    eligible = result["input_eligible_before_native_quality"]
    assert eligible["tune"]["no_xp"] == 1
    assert eligible["fit"]["xp"] == 1
    assert eligible["external"]["no_xp"] == 1
    assert eligible["excluded"]["no_xp"] == 1
    assert sum(eligible["unassigned"].values()) == 0  # Parallax alone is insufficient.
    assert result["training_admitted"] is False


def test_census_rejects_float_ids_and_wrong_candidate_rows():
    ids = np.array([100000000000000001], dtype="i8")
    with pytest.raises(ValueError, match="Exact integer"):
        m.positions(pa.record_batch({"source_id": ids.astype(float), "candidate_row": [0]}), ids)
    with pytest.raises(ValueError, match="does not identify"):
        m.positions(pa.record_batch({"source_id": [int(ids[0]) + 1], "candidate_row": [0]}), ids)


def test_bit_union_keeps_channels_above_64_and_repeated_rows():
    bits = np.zeros((2, 3), dtype="u8")
    masks = np.zeros((3, 140), dtype=bool)
    masks[0, [0, 63, 64, 139]] = True
    masks[1, 65] = True
    m.add_bits(bits, np.array([0, 0, 1]), masks)
    assert bits[0, 0] == (np.uint64(1) | (np.uint64(1) << np.uint64(63)))
    assert bits[0, 1] == 3
    assert bits[0, 2] == 1 << 11
    assert not bits[1].any()
