import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "decision_six", Path(__file__).parents[1] / "scripts/run_decision_six_pilot.py"
)
wave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wave)


def test_chunking_retains_joint_vectors_and_star_axis():
    import torch

    class CorrelatedHead:
        def sample(self, context, count, generator):
            base = torch.randn((count, len(context), 1), generator=generator)
            return base + torch.arange(6)[None, None, :] + context[None, :, :]

    context = torch.tensor([[0.0], [100.0]])
    result = wave.bounded_whole_draws(
        CorrelatedHead(), context, 257, torch.Generator().manual_seed(12), 128
    )
    assert result.shape == (257, 2, 6)
    np.testing.assert_allclose(result[..., 5] - result[..., 0], 5.0, atol=1e-5)
    assert result[:, 1, 0].mean() > 95
    with pytest.raises(ValueError):
        wave.bounded_whole_draws(CorrelatedHead(), context, 0, None)
