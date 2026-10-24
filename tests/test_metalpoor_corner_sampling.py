"""Synthetic corner-sampler invariants; never local native model inference."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sample_metalpoor_corners import (
    draw_standardized,
    histogram_mass_levels,
    native_and_plot_samples,
)

from masked_stellar_autoencoder.training.final_six_recipes import SparseSixHead


@pytest.mark.parametrize("kind", ["gaussian", "factorized_nsf"])
def test_independent_heads_use_independent_base_coordinates(kind):
    torch.manual_seed(23)
    head = SparseSixHead(kind, latent_dim=4).double().eval()
    with torch.no_grad():
        draws = draw_standardized(
            head,
            torch.zeros(1, 4, dtype=torch.double),
            2048,
            generator=torch.Generator().manual_seed(65),
        )
    assert draws.shape == (2048, 6) and torch.isfinite(draws).all()
    corr = np.corrcoef(draws.numpy().T)
    assert np.max(np.abs(corr - np.eye(6))) < 0.1


def test_mdn_draws_share_a_component_across_targets():
    head = SparseSixHead("mdn", latent_dim=4).double().eval()
    with torch.no_grad():
        head.head.output.weight.zero_()
        head.head.output.bias.zero_()
        means = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.double)[
            :, None
        ].expand(4, 6)
        head.head.output.bias[4:28] = means.flatten()
        head.head.output.bias[28:] = -4
        draws = draw_standardized(
            head,
            torch.zeros(1, 4, dtype=torch.double),
            2048,
            generator=torch.Generator().manual_seed(61),
        )
    assert np.corrcoef(draws.numpy().T)[0, 5] > 0.99


def test_native_transform_preserves_signed_alpha_and_positive_age_distance():
    value = torch.tensor([[1.0, 2.0, -3.0, -0.5, 0.0, 1.0]], dtype=torch.double)
    native, plotted = native_and_plot_samples(value, torch.zeros(6), torch.ones(6))
    assert native[0, 3] == -0.5 and native[0, 4] == 1
    torch.testing.assert_close(native[:, 4:].log(), plotted[:, 4:])
    levels, mass = histogram_mass_levels(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert all(0 < level <= 4 for level in levels)
    assert mass[0] >= 0.68 and mass[1] >= 0.95
