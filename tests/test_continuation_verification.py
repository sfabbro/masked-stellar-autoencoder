import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "continuation_verification",
    Path(__file__).resolve().parents[1]
    / "experiments/xp/monitor/verify_prediction_continuation.py",
)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_streams_respect_pass_boundaries_and_no_warmup():
    row, label, state, updates = v.streams(
        fit_count=7,
        rows_per_pass=5,
        passes=3,
        groups=2,
        batch=4,
        effective_batch=4,
        seed=1042,
    )
    assert updates == 6
    assert (
        row
        == hashlib.sha256(np.tile(np.arange(5, dtype="i8"), 3).tobytes()).hexdigest()
    )
    rng = np.random.default_rng(1042)
    draws = [rng.integers(7, size=4).tobytes() for _ in range(12)]
    assert label == hashlib.sha256(b"".join(draws)).hexdigest()
    assert state == rng.bit_generator.state


def test_optimizer_checks_inherited_step_deltas_not_absolute_steps():
    def opt(step):
        return dict(
            param_groups=[dict(lr=0.001, params=[0])],
            state={
                0: dict(
                    step=torch.tensor(step),
                    exp_avg=torch.zeros(2),
                    exp_avg_sq=torch.zeros(2),
                )
            },
        )

    assert v.optimizer_audit(opt(806), opt(9056), 8250, 0.001)["step_increment"] == 8250
    with pytest.raises(AssertionError):
        v.optimizer_audit(opt(806), opt(8556), 8250, 0.001)
    with pytest.raises(AssertionError):
        v.exact(np.array([1.0], dtype="f4"), np.array([1.0], dtype="f8"))
