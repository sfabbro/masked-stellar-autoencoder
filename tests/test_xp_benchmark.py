import hashlib
import importlib.util
import json
import subprocess
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
            "2",
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )
    subprocess.run([sys.executable, str(builder.__file__), *sys.argv[1:]], check=True)
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
    # Reuse must retain exact records/provenance; no new random sample is drawn.
    reused = tmp_path / "reused.h5"
    argv = list(sys.argv[1:])
    argv[argv.index("--output") + 1] = str(reused)
    argv += [
        "--reuse-sample",
        str(output),
        "--expected-source-id-sha256",
        hashlib.sha256(np.array([10, 11, 12], dtype="i8").tobytes()).hexdigest(),
    ]
    subprocess.run([sys.executable, str(builder.__file__), *argv], check=True)
    with h5py.File(output) as original, h5py.File(reused) as copy:
        for split in ("train", "validation"):
            for field in ("records", "source_row", "source_shard", "xp_correlations"):
                np.testing.assert_array_equal(
                    original[split][field][:], copy[split][field][:]
                )


def test_binary_reader_joins_unsorted_ids_and_preserves_band_metadata(tmp_path):
    path = tmp_path / "raw.hdf5"
    with h5py.File(path, "w") as f:
        f["source_id"] = [99, 90, 101]
        for band, value in (("bp", 0.125), ("rp", 0.25)):
            f.create_dataset(
                band + "_coefficient_correlations",
                (3,),
                dtype=np.dtype(("f4", (1485,))),
            )[:] = np.full((3, 1485), value, dtype="f4")
            for key, values in (
                ("basis_function_id", [56, 57, 58]),
                ("n_parameters", [55, 55, 55]),
                ("n_relevant_bases", [20, 21, 22]),
                ("standard_deviation", [1.125, 1.25, 1.5]),
            ):
                f[band + "_" + key] = values
    _, positions, corr, meta = builder.read_hdf5_correlations(path, np.array([90, 99]))
    assert positions.tolist() == [1, 0]
    assert np.all(corr[:, 0] == 0.125) and np.all(corr[:, 1] == 0.25)
    np.testing.assert_array_equal(meta[:, 0, 0], [56, 57])
    _, positions, corr, meta = builder.read_hdf5_correlations(path, np.array([100]))
    assert corr.shape == (0, 2, 1485) and meta.shape == (0, 2, 4)


def test_selected_line_reader_preserves_metadata_and_empty_selection(tmp_path):
    path = tmp_path / "XpContinuousMeanSpectrum_test.csv.gz"
    record = {"source_id": 99}
    for band in ("bp", "rp"):
        record[band + "_coefficient_correlations"] = (
            "[" + ",".join(["0.125"] * 1485) + "]"
        )
        for key, value in (
            ("basis_function_id", 56),
            ("n_parameters", 55),
            ("n_relevant_bases", 20),
            ("standard_deviation", 1.125),
        ):
            record[band + "_" + key] = value
    pd.DataFrame([record]).to_csv(path, index=False)
    _, positions, correlation, metadata = builder.read_correlations(
        path, np.array([90, 99])
    )
    assert positions.tolist() == [1]
    assert np.all(correlation == np.float32(0.125))
    np.testing.assert_array_equal(metadata, [[[56, 55, 20, 1.125]] * 2])
    _, positions, correlation, metadata = builder.read_correlations(
        path, np.array([90])
    )
    assert len(positions) == 0
    assert correlation.shape == (0, 2, 1485)
    assert metadata.shape == (0, 2, 4)
