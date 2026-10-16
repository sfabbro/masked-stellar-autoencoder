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
