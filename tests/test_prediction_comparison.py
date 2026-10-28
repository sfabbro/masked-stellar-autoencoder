"""Budget and real-input source-ID policy checks for the prediction pilots."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_prediction_comparison import replay_batches  # noqa: E402


def test_exact_replay_budget_cycles_training_rows_without_validation():
    train = np.arange(21).reshape(7, 3)
    batches = list(replay_batches(train, 10, 4))
    np.testing.assert_array_equal(
        np.concatenate([b[0] for b in batches]), [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    )
    np.testing.assert_array_equal(
        np.concatenate([b[1] for b in batches]), train[[0, 1, 2, 3, 4, 5, 6, 0, 1, 2]]
    )
    assert [len(b[0]) for b in batches] == [4, 3, 3]


@pytest.mark.parametrize("length,budget,batch", [(0, 10, 4), (3, 0, 4), (3, 10, 0)])
def test_invalid_replay_budget_rejected(length, budget, batch):
    with pytest.raises(ValueError, match="positive"):
        list(replay_batches(np.zeros((length, 3)), budget, batch))


def test_native_inputs_executes_input_policy_and_source_association(
    tmp_path, monkeypatch
):
    import h5py
    import run_prediction_comparison as runner

    records = np.array(
        [(101, -2.0), (102, 0.0), (103, 1.0)],
        dtype=[("source_id", "i8"), ("value", "f8")],
    )
    cohort = tmp_path / "cohort.h5"
    with h5py.File(cohort, "w") as f:
        f.create_dataset("raw_layers/native/measurements", data=records)
    raw = np.array([[10.0, 17.0, 0.1], [11.0, 18.0, 0.2], [12.0, 19.0, 0.3]])
    data = dict(
        native_fit=np.array([0, 1]),
        tests={"native": dict(used_row=np.array([2]), measurements=records[2:])},
        fit_truth=records["value"],
        source_id=records["source_id"],
        raw_values=raw,
        center=np.zeros(3),
        scale=np.ones(3),
        feature_columns=["G", "ra", "EBV"],
    )
    monkeypatch.setattr(runner, "prepare", lambda _: data)
    monkeypatch.setattr(runner, "identity_split", lambda ids: np.zeros(len(ids), int))
    _, fit, _, safe, visible, _, _ = runner.native_inputs(cohort, limit=2)
    np.testing.assert_array_equal(fit, [0, 1])
    np.testing.assert_array_equal(safe[:, 1], -9999.0)
    np.testing.assert_array_equal(visible[:, 1], False)
    np.testing.assert_allclose(safe[:, [0, 2]], raw[:, [0, 2]])
