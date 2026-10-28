"""Grid-major replay must preserve likelihood context/target association."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/xp/monitor"))
from contrastive_flow_reference import grid_mle

from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    _CONTRASTIVE_FLOW_SPEC,
    build_contrastive_flow,
)


def test_grid_major_replay_matches_full_flow_scores():
    pytest.importorskip("zuko")
    torch.manual_seed(42)
    head = build_contrastive_flow(_CONTRASTIVE_FLOW_SPEC).eval()
    latent = torch.randn(4, 256)
    grid = torch.linspace(-2, 2, 9)
    actual, scores = grid_mle(head, latent, grid)
    expected = torch.stack(
        [
            head.flow_loss.distribution(g.reshape(1, 1).expand(4, 1)).log_prob(
                head.summary(latent)
            )
            for g in grid
        ]
    )
    torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual, head.standardized_mle(latent, grid))
