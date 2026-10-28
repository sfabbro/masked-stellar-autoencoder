"""Winner/Adam identity and full checkpoint/stream preservation."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_stellar_improvement import bind_prediction_policy
from prediction_continuation import load_segment, save_epoch


def test_restore_winner_adam_and_atomic_epoch_checkpoint(tmp_path):
    torch.manual_seed(2)
    model = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    loss = head(model(torch.ones(3, 2))).square().mean()
    loss.backward()
    optimizer.step()
    head_optimizer.step()
    rng = np.random.default_rng(1042)
    mask = torch.Generator().manual_seed(1042)
    path = tmp_path / "checkpoint.pth"
    receipt = save_epoch(
        path,
        model,
        head,
        optimizer,
        head_optimizer,
        arm=dict(head="residual", loss="huber"),
        feature_columns=["a", "b"],
        label_center=0.0,
        label_scale=1.0,
        presentations=9000000,
        completed_updates=581,
        label_rng_state=rng.bit_generator.state,
        mask_rng_state=mask.get_state(),
    )
    new_model = torch.nn.Linear(2, 2)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-5)
    state = load_segment(
        path,
        receipt["sha256"],
        new_model,
        new_opt,
        features=["a", "b"],
        center=0.0,
        scale=1.0,
    )
    for actual, expected in zip(
        new_model.parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    old_values = list(optimizer.state.values())
    new_values = list(new_opt.state.values())
    for actual, expected in zip(new_values, old_values, strict=True):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert (
        state["completed_updates"] == 581
        and state["label_rng_state"] == rng.bit_generator.state
    )
    assert not path.with_suffix(".partial").exists()
    with pytest.raises(ValueError, match="digest"):
        load_segment(
            path,
            "0" * 64,
            new_model,
            new_opt,
            features=["a", "b"],
            center=0.0,
            scale=1.0,
        )


def test_long_policy_binds_actual_parent_and_declared_replay_budget():
    result = bind_prediction_policy(
        dict(task="prediction-continuation", run_id="residual_huber_long_30e_v1"), {}
    )
    assert result["MSA_EXPERIMENT_PRESENTATIONS"] == "270000000"
    assert result["MSA_PRETRAIN_RESUME"].endswith("_72.pth")
