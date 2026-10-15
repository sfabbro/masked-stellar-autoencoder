import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

spec = importlib.util.spec_from_file_location(
    "views", Path(__file__).parents[1] / "scripts/stellar_dataset_views.py"
)
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


def test_index_keeps_all_fields_and_separates_label_holdouts(tmp_path):
    benchmark = tmp_path / "benchmark.h5"
    registry = tmp_path / "labels.h5"
    ids = np.array([20, 10, 30], dtype="<i8")
    digest = hashlib.sha256(ids.tobytes()).hexdigest()
    with h5py.File(benchmark, "w") as f:
        f.attrs["provenance"] = json.dumps({"source_id_sha256": digest})
        dtype = [("source_id", "<i8"), ("original_field", "<f8")]
        f.create_dataset(
            "train/records", data=np.array([(20, 4.0), (10, 5.0)], dtype=dtype)
        )
        f.create_dataset("validation/records", data=np.array([(30, 6.0)], dtype=dtype))
        f.create_dataset("reference_scaler/scale", data=[1.0])
    with h5py.File(registry, "w") as f:
        f.create_dataset(
            "selected/labels",
            data=np.array(
                [(10, 0), (30, 2)], dtype=[("source_id", "<i8"), ("split", "u1")]
            ),
        )
    output, scratch = tmp_path / "out", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    result = views.dataset_view(output, scratch, registry, benchmark, digest, (2, 1))
    assert result["counts"]["train"]["sft_fit_eligible"] == 1
    assert result["counts"]["validation"]["sft_fit_eligible"] == 0
    with h5py.File(output / "dataset-view.h5") as f:
        assert f["train/selected_label_row"][:].tolist() == [-1, 0]
        assert f["train/benchmark/records"]["original_field"].tolist() == [4.0, 5.0]
        assert f["validation/label_holdout_split"][:].tolist() == [2]


def test_posteriors_keep_asymmetry_units_and_reject_wrong_units(tmp_path):
    columns = [fits.Column(name="GaiaEDR3", format="K", array=[123])]
    for prefix, unit in [("Age", "Gyr"), ("Dist", "kpc")]:
        for suffix, value in zip(["05", "16", "50", "84", "95"], [1, 2, 3, 6, 9]):
            columns.append(
                fits.Column(name=prefix + suffix, format="D", unit=unit, array=[value])
            )
    columns.append(fits.Column(name="OutputFlag", format="3A", array=["000"]))
    source = tmp_path / "starhorse.fits"
    fits.BinTableHDU.from_columns(columns).writeto(source)
    output, scratch = tmp_path / "out", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    result = views.posterior_tables(output, scratch, [source])
    assert not result["independent_age_distance_truth"]
    with h5py.File(output / "posterior-labels.h5") as f:
        assert f["starhorse/Age_percentiles"][:].tolist() == [[1, 2, 3, 6, 9]]
        assert f["starhorse/Dist_percentiles"].attrs["unit"] == "kpc"
        assert f["starhorse/flags/OutputFlag"][:].tolist() == [b"000"]
    with fits.open(source, mode="update") as f:
        f[1].header["TUNIT2"] = "yr"
    other = tmp_path / "bad"
    other.mkdir()
    with pytest.raises(ValueError, match="Unexpected unit"):
        views.posterior_tables(output, other, [source])
