import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table


def test_combiner_writes_training_compatible_keys_and_rebuilds(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "data" / "combine-partial-tables.py"
    Table(
        {
            "source_id": np.array([1, 2], dtype=np.int64),
            "G": np.array([12.0, 13.0]),
        }
    ).write(tmp_path / "partialtable-11.fits")

    for _ in range(2):
        subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        with h5py.File(tmp_path / "pretrain_dataset_incomplete.h5", "r") as datafile:
            np.testing.assert_array_equal(
                datafile["sslset11_part0"]["source_id"], np.array([1, 2])
            )


def test_combiner_accepts_canfar_input_and_output_paths(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "partials"
    output = tmp_path / "training" / "pretrain.h5"
    input_dir.mkdir()
    Table({"source_id": np.array([7], dtype=np.int64)}).write(
        input_dir / "partialtable-4.fits"
    )
    env = os.environ.copy()
    env["MSA_PREPROCESS_DIR"] = str(input_dir)
    env["MSA_PRETRAIN_HDF5_OUT"] = str(output)

    subprocess.run(
        [sys.executable, str(repo / "data" / "combine-partial-tables.py")],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output, "r") as datafile:
        np.testing.assert_array_equal(datafile["sslset4_part0"]["source_id"], [7])
