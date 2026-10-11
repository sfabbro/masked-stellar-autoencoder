import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "xp_builder",
    Path(__file__).resolve().parents[1] / "scripts/benchmarks/build_xp_benchmark.py",
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_proportional_counts_are_exact():
    assert builder.allocations([3, 7, 10], 7).tolist() == [1, 2, 4]


def test_split_preserving_covariance_roundtrip(tmp_path, monkeypatch):
    source = tmp_path / "source.h5"
    dtype = [("source_id", "i8"), ("G", "f4"), ("bp_1", "f4"), ("bpe_1", "f4")]
    with h5py.File(source, "w") as f:
        f["training"] = np.array(
            [(10, 15.0, 2.0, 0.1), (11, 16.0, 3.0, 0.2)], dtype=dtype
        )
        f["validation"] = np.array([(12, 17.0, 4.0, 0.3)], dtype=dtype)
    rows = []
    for source_id in (10, 11, 12):
        row = {"source_id": source_id}
        for b in ("bp", "rp"):
            row[b + "_coefficient_correlations"] = "[" + ",".join(["0.25"] * 1485) + "]"
            for k, v in [
                ("basis_function_id", 1),
                ("n_parameters", 55),
                ("n_relevant_bases", 20),
                ("standard_deviation", 1.2),
            ]:
                row[b + "_" + k] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        tmp_path / "XpContinuousMeanSpectrum_test.csv.gz", index=False
    )
    output = tmp_path / "benchmark.h5"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--source",
            str(source),
            "--csv-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--validation-keys",
            "validation",
            "--train-rows",
            "2",
            "--validation-rows",
            "1",
            "--workers",
            "1",
        ],
    )
    builder.main()
    with h5py.File(output) as f:
        assert f["train/records"]["source_id"].tolist() == [10, 11]
        assert f["validation/records"]["source_id"].tolist() == [12]
        np.testing.assert_array_equal(
            f["train/records"]["bpe_1"], np.array([0.1, 0.2], dtype="f4")
        )
        assert f["train/xp_correlations"].shape == (2, 2, 1485)
        assert np.all(f["train/xp_correlations"][:] == 0.25)
        assert f["validation/xp_covariance_available"][:].all()
    assert (
        json.loads(Path(str(output) + ".manifest.json").read_text())["status"]
        == "complete"
    )
