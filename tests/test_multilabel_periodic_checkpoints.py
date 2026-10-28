"""Full periodic state and tuning-only checkpoint choice are recoverable."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_multilabel_screen import (
    interval_crossed,
    save_periodic_checkpoint,
    select_by_reserved_tuning,
)


def test_checkpoint_boundaries_include_short_epoch_batches():
    assert not interval_crossed(8_945_664, 8_978_432, 9_000_000)
    assert interval_crossed(8_978_432, 9_000_000, 9_000_000)
    assert not interval_crossed(9_000_000, 9_032_768, 9_000_000)
    assert interval_crossed(17_978_432, 18_000_000, 9_000_000)
    assert not interval_crossed(0, 256, 0)
    with pytest.raises(ValueError, match="positive integers"):
        interval_crossed(0, 10, -1)


def test_periodic_full_state_roundtrip_both_adams_and_rng(tmp_path):
    model, head = torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    head_optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss = head(model(torch.ones(4, 3))).square().mean()
    loss.backward()
    optimizer.step()
    head_optimizer.step()
    state = {
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "model_optimizer_state_dict": optimizer.state_dict(),
        "head_optimizer_state_dict": head_optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "supervised_rng_state": np.random.default_rng(42).bit_generator.state,
        "mask_rng_state": torch.Generator().manual_seed(4).get_state(),
        "completed_presentations": 9_000_000,
        "completed_updates": 275,
        "policy": {"selection_policy": "reserved_tune_mean_observed_nll"},
    }
    receipt = save_periodic_checkpoint(tmp_path, state)
    assert receipt["all_saved_state_values_reload_equal"]
    assert len(receipt["sha256"]) == 64
    assert receipt == json.loads((tmp_path / "latest-checkpoint.json").read_text())
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        save_periodic_checkpoint(tmp_path, state)


def test_tuning_selection_retains_first_tie_and_does_not_read_test_labels(tmp_path):
    data = {
        "source_id": np.array([11, 22, 33], dtype="i8"),
        "value": np.zeros((3, 3)),
        "error": np.ones((3, 3)),
        "label_center": np.zeros(3),
        "label_scale": np.ones(3),
    }
    rows = np.array([0, 1])
    result = {
        "mean": np.zeros((2, 3)),
        "observed_log_prob_native": np.array([-3.0, -6.0]),
    }
    for seen, density in [(9, [-3.0, -6.0]), (18, [-3.0, -6.0]), (27, [-1.0, -2.0])]:
        result["observed_log_prob_native"] = np.array(density)
        select_by_reserved_tuning(
            tmp_path, {"presentations": seen, "file": f"{seen}.pth"}, data, rows, result
        )
        selected = json.loads((tmp_path / "selected-checkpoint.json").read_text())
        assert selected["presentations"] == (9 if seen <= 18 else 27)
        assert not selected["calibration_or_test_used_for_selection"]
        assert selected["tuning_rows"] == 2
    assert len(list(tmp_path.glob("tuning-*.npz"))) == 3


def test_ttt_saves_exact_selected_batch_baselines_for_cp(tmp_path, monkeypatch):
    import run_multilabel_screen as runner

    model = torch.nn.Linear(3, 3)
    data = {
        "input": np.arange(24, dtype="f4").reshape(8, 3),
        "visible": np.ones((8, 3), dtype=bool),
        "source_id": np.arange(101, 109, dtype="i8"),
        "value": np.zeros((8, 3), dtype="f4"),
    }
    monkeypatch.setattr(runner, "xp_indices", lambda features: np.arange(3))
    monkeypatch.setattr(
        runner,
        "refine_one",
        lambda model, head, x, visible, xp, source_id, steps: (x, {}),
    )

    def predicted(model, head, data, rows, device):
        return {"mean": np.nextafter(data["input"][rows], np.float32(np.inf))}

    monkeypatch.setattr(runner, "predictions", predicted)
    received = {}

    def calibrated(output, data, test, cal, means_test, means_cal, prefix=""):
        received[prefix] = (means_test.copy(), means_cal.copy())
        return {}

    monkeypatch.setattr(runner, "calibrate_and_save", calibrated)
    runner.postprocess(
        tmp_path,
        model,
        model,
        data,
        np.arange(4),
        np.arange(4, 8),
        torch.device("cpu"),
        {"feature_columns": [], "ttt_steps": 3},
        True,
    )
    with np.load(tmp_path / "ttt-matched-clean-predictions.npz") as saved:
        expected_test, expected_cal = received["ttt-matched-clean-"]
        np.testing.assert_array_equal(
            saved["test_standardized_prediction"], expected_test
        )
        np.testing.assert_array_equal(
            saved["calibration_standardized_prediction"], expected_cal
        )
        np.testing.assert_array_equal(
            saved["test_source_id"], data["source_id"][saved["test_rows"]]
        )
