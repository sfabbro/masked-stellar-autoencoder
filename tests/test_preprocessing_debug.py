import os
import runpy
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
PREPROCESSOR = REPO / "data" / "pretraining-partial-table-maker.py"


def load_preprocessor(monkeypatch, tmp_path):
    monkeypatch.setenv("DUSTMAPS_DATA_DIR", str(tmp_path))
    from dustmaps.config import config

    monkeypatch.setattr(
        type(config),
        "__setitem__",
        lambda self, key, value: self._options.__setitem__(key, value),
    )
    return runpy.run_path(str(PREPROCESSOR))


def test_source_index_reader_accepts_generator_compound_hdf5(monkeypatch, tmp_path):
    module = load_preprocessor(monkeypatch, tmp_path)
    dataset = np.array(
        [(101, 1), (102, 0), (103, 1)],
        dtype=[("source_id", "i8"), ("has_xp_coeffs", "i1")],
    )

    assert module["_source_ids_with_xp"](dataset) == [101, 103]


def test_gaia_flux_errors_propagate_to_magnitude_errors(monkeypatch, tmp_path):
    module = load_preprocessor(monkeypatch, tmp_path)
    factor = 2.5 / np.log(10)
    result = module["flux_error_to_mag"](np.array([10.0, 0.0]), np.array([2.0, 1.0]))

    assert result[0] == pytest.approx(factor * 0.2)
    assert np.isnan(result[1])


def test_source_reader_emits_gaia_magnitude_and_astrometry_errors(
    monkeypatch, tmp_path
):
    module = load_preprocessor(monkeypatch, tmp_path)
    source_dir = tmp_path / "gaia"
    source_dir.mkdir()
    process_source_file = module["process_source_file"]
    module_globals = process_source_file.__globals__
    module_globals["GAIA_SOURCE_DIR"] = source_dir
    source_file = source_dir / "GaiaSource_000000-000001.hdf5"
    fields = {
        "source_id": np.array([10, 11], dtype=np.int64),
        "phot_g_mean_mag": np.array([12.0, 13.0]),
        "phot_bp_mean_mag": np.array([12.5, 13.5]),
        "phot_rp_mean_mag": np.array([11.5, 12.5]),
        "phot_g_mean_flux": np.array([10.0, 20.0]),
        "phot_bp_mean_flux": np.array([5.0, 10.0]),
        "phot_rp_mean_flux": np.array([8.0, 16.0]),
        "phot_g_mean_flux_error": np.array([1.0, 2.0]),
        "phot_bp_mean_flux_error": np.array([0.5, 1.0]),
        "phot_rp_mean_flux_error": np.array([0.8, 1.6]),
        "parallax": np.array([1.0, 2.0]),
        "parallax_error": np.array([0.1, 0.2]),
        "pmra": np.array([3.0, 4.0]),
        "pmdec": np.array([5.0, 6.0]),
        "pmra_error": np.array([0.3, 0.4]),
        "pmdec_error": np.array([0.5, 0.6]),
        "ra": np.array([15.0, 16.0]),
        "dec": np.array([20.0, 21.0]),
    }
    with h5py.File(source_file, "w") as h5:
        for name, values in fields.items():
            h5.create_dataset(name, data=values)

    class FakeSFDQuery:
        def __call__(self, coords):
            return np.array([0.05, 0.06])

    module_globals["SFDQuery"] = FakeSFDQuery
    row = process_source_file(([10], "GaiaSource_000000-000001")).iloc[0]

    for feature in ("G", "BP", "RP"):
        assert row[f"e_{feature}"] == pytest.approx(
            module["flux_error_to_mag"](
                np.array([fields[f"phot_{feature.lower()}_mean_flux"][0]]),
                np.array([fields[f"phot_{feature.lower()}_mean_flux_error"][0]]),
            )[0]
        )
    assert row["e_parallax"] == pytest.approx(0.1)
    assert row["e_pmra"] == pytest.approx(0.3)
    assert row["e_pmdec"] == pytest.approx(0.5)


def test_xp_errors_receive_the_same_flux_scaling_as_coefficients(monkeypatch, tmp_path):
    module = load_preprocessor(monkeypatch, tmp_path)
    columns = module["_xp_scale_labels"]()
    frame = pd.DataFrame([{"G": 10.0, **{column: 2.0 for column in columns}}])

    scaled = module["_scale_xp_measurements"](frame)
    expected = 2.0 / (10 ** ((8.5 - 10.0) / 2.5))
    assert scaled.loc[0, "bp_1"] == pytest.approx(expected)
    assert scaled.loc[0, "bpe_1"] == pytest.approx(expected)
    assert scaled.loc[0, "rp_55"] == pytest.approx(expected)
    assert scaled.loc[0, "rpe_55"] == pytest.approx(expected)


def test_crossmatch_worker_failure_is_propagated(monkeypatch, tmp_path):
    module = load_preprocessor(monkeypatch, tmp_path)

    class FailedExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args):
            future = Future()
            future.set_exception(RuntimeError("broken crossmatch chunk"))
            return future

    module_globals = module["parallel_crossmatch_np"].__globals__
    monkeypatch.setitem(module_globals, "ProcessPoolExecutor", FailedExecutor)
    monkeypatch.setitem(
        module_globals,
        "read_hdf_chunked_np",
        lambda *args, **kwargs: iter([np.zeros(1)]),
    )

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, count):
            pass

    monkeypatch.setitem(module_globals, "tqdm", lambda *args, **kwargs: Progress())

    with pytest.raises(RuntimeError, match="broken crossmatch chunk"):
        module["parallel_crossmatch_np"](
            "unused.h5",
            "catalog",
            np.array([], dtype=np.int64),
            workers=1,
            save_path=str(tmp_path / "cache"),
        )


def test_crossmatch_bounds_in_flight_chunks(monkeypatch, tmp_path):
    module = load_preprocessor(monkeypatch, tmp_path)
    executor_state = {"active": 0, "peak": 0}

    class CompletedExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args):
            future = Future()
            future.set_result(None)
            executor_state["active"] += 1
            executor_state["peak"] = max(
                executor_state["peak"], executor_state["active"]
            )
            return future

    def complete_one(futures, return_when):
        completed = next(iter(futures))
        executor_state["active"] -= 1
        return {completed}, futures - {completed}

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, count):
            pass

    module_globals = module["parallel_crossmatch_np"].__globals__
    monkeypatch.setitem(module_globals, "ProcessPoolExecutor", CompletedExecutor)
    monkeypatch.setitem(module_globals, "wait", complete_one)
    monkeypatch.setitem(
        module_globals,
        "read_hdf_chunked_np",
        lambda *args, **kwargs: iter([np.zeros(1) for _ in range(11)]),
    )
    monkeypatch.setitem(
        module_globals,
        "tqdm",
        lambda *args, **kwargs: args[0] if args else Progress(),
    )

    module["parallel_crossmatch_np"](
        "unused.h5",
        "catalog",
        np.array([], dtype=np.int64),
        workers=2,
        save_path=str(tmp_path / "cache"),
    )

    assert executor_state["peak"] == 4


def test_source_index_builder_honors_canfar_input_and_output_paths(tmp_path):
    source_dir = tmp_path / "gaia" / "GaiaSource"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "GaiaSource_000000-000001.h5"
    with h5py.File(source_file, "w") as h5:
        h5.create_dataset("source_id", data=np.array([10, 11], dtype=np.int64))
        h5.create_dataset("has_xp_continuous", data=np.array([1, 0], dtype=np.uint8))

    output_path = tmp_path / "catalogues" / "andrae2023" / "source_ids.h5"
    env = os.environ.copy()
    env["MSA_GAIA_SOURCE_DIR"] = str(source_dir)
    env["MSA_SOURCE_IDS_FILE"] = str(output_path)
    subprocess.run(
        [sys.executable, str(REPO / "data" / "source_ids_x_file_names.py")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as h5:
        fields = h5["GaiaSource_000000-000001"].dtype.names
        assert fields == ("source_id", "has_xp_coeffs")
        np.testing.assert_array_equal(
            h5["GaiaSource_000000-000001"]["source_id"], [10, 11]
        )
