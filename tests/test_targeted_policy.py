import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from targeted_policy import dropout_offset, experiment_seed, resolve_policy


def test_old_defaults_and_new_seed_are_explicit():
    suite = {"checkpoint": "old", "checkpoint_sha256": "hash", "encoder_lr": 1e-6}
    assert resolve_policy(suite, {}) == suite
    assert experiment_seed(suite, {}) == 42 and dropout_offset(42) == 0
    assert experiment_seed(suite, {"experiment_seed": 43}) == 43
    assert dropout_offset(43) == 10_000_000
    with pytest.raises(ValueError):
        resolve_policy(suite, {"policy_overrides": {"checkpoint": "new"}})
    with pytest.raises(ValueError):
        resolve_policy(suite, {"policy_overrides": {"reconstruction_presentations": 1}})
    assert (
        resolve_policy(suite, {"policy_overrides": {"encoder_lr": 0}})["encoder_lr"]
        == 0
    )
