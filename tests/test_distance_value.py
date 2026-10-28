import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_distance_value import fit_distance_head, positive_distance, score_parallax
from run_pristine_value import replay


def test_distance_units_and_negative_measurements_are_retained():
    score = score_parallax(
        np.array([2.0, 4.0]), np.array([0.5, -0.1]), np.array([0.1, 0.2])
    )
    assert (
        score["count"] == 2 and score["reference_nonpositive_parallaxes_retained"] == 1
    )
    assert score["mae_mas"] == pytest.approx(0.175)
    assert score["bias_mas"] == pytest.approx(0.175)
    assert not score["formal_sigma_is_total_calibrated_accuracy"]
    for bad in [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0]]:
        with pytest.raises(ValueError):
            score_parallax(bad, [1, 2], [0.1, 0.2])


def test_head_scaling_uses_only_train_log_distance_and_reload():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 5))
    distance = np.exp(x[:, 0] / 3)
    fit = np.arange(40)
    test = np.arange(40, 80)
    pred, state = fit_distance_head(x, distance, fit, test)
    changed = distance.copy()
    changed[test] = 1e6
    pred2, state2 = fit_distance_head(x, changed, fit, test)
    np.testing.assert_array_equal(pred, pred2)
    np.testing.assert_array_equal(state["center"], x[fit].mean(0))
    assert state["ycenter"] == np.log(distance[fit]).mean()
    np.testing.assert_array_equal(pred, positive_distance(replay(x[test], state2)))


def test_raw_gaia_baseline_is_scored_without_inversion_or_sign_cut():
    from run_distance_value import parallax_metrics

    result = parallax_metrics(
        np.array([-0.2, 0.5]), np.array([-0.1, 0.4]), np.array([0.1, 0.1])
    )
    assert result["count"] == 2
    assert result["mae_mas"] == pytest.approx(0.1)
    assert result["bias_mas"] == pytest.approx(0)


def test_actual_nested_benchmark_records_and_verified_index(tmp_path):
    import json

    import h5py
    from run_distance_value import EXPECTED_IDS, benchmark_ids

    benchmark, index = tmp_path / "benchmark.h5", tmp_path / "index.h5"
    ids = np.array([2**60 + 1, 2**60 + 3], dtype="i8")
    with h5py.File(benchmark, "w") as f, h5py.File(index, "w") as view:
        f.attrs["provenance"] = json.dumps({"source_id_sha256": EXPECTED_IDS})
        for split in ["train", "validation"]:
            rows = np.zeros(2, dtype=[("source_id", "<i8"), ("bp_1", "<f4")])
            rows["source_id"] = ids
            f.create_group(split).create_dataset("records", data=rows)
            view.create_group(split).create_dataset("source_id", data=ids)
    for actual in benchmark_ids(benchmark, index):
        np.testing.assert_array_equal(actual, ids)
    with h5py.File(benchmark, "r+") as f:
        f.attrs["provenance"] = json.dumps({"source_id_sha256": "wrong"})
    with pytest.raises(ValueError, match="provenance"):
        benchmark_ids(benchmark, index)

    with h5py.File(benchmark, "r+") as f:
        f.attrs["provenance"] = json.dumps({"source_id_sha256": EXPECTED_IDS})
    with h5py.File(index, "r+") as f:
        del f["train/source_id"]
        f["train"].create_dataset("source_id", data=ids[:1])
    with pytest.raises(ValueError, match="count differs"):
        benchmark_ids(benchmark, index)
