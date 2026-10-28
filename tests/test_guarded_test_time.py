"""Synthetic invariants for decoder-frozen, held-out guarded TTT."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from masked_stellar_autoencoder.training.guarded_test_time import (
    adapt_guarded,
    split_columns,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        self.decoder = nn.Linear(8, 8)
        self.reconstruction_layer = nn.Linear(8, 8)


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 6)

    def moments(self, x):
        mean = self.net(x)
        return mean, torch.ones_like(mean)


def test_heldout_and_update_masks_disjoint_stateless_observed_only():
    visible = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    first, update = split_columns(visible, list(range(8)), 17, 0)
    for step in range(5):
        validation, update = split_columns(visible, list(range(8)), 17, step)
        np.testing.assert_array_equal(first, validation)
        assert not np.intersect1d(validation, update).size
        assert 2 not in validation and 2 not in update
    with pytest.raises(ValueError, match="four"):
        split_columns(visible, [0, 1, 2], 17, 0)


@pytest.mark.parametrize(
    "scope,optimizer",
    [("encoder_only", "sgd"), ("layernorm", "sgd"), ("layernorm", "adam")],
)
def test_guarded_ttt_restores_all_state_and_supports_six_outputs(scope, optimizer):
    torch.manual_seed(85)
    model, head = TinyModel(), TinyHead()
    x = torch.arange(8).reshape(1, 8).float() / 8
    visible = torch.ones_like(x, dtype=torch.bool)
    state = copy.deepcopy(model.state_dict())
    head_state = copy.deepcopy(head.state_dict())
    rng = torch.get_rng_state().clone()
    mean, variance, report = adapt_guarded(
        model,
        head,
        x,
        visible,
        list(range(8)),
        17,
        baseline_mean=torch.zeros(1, 6),
        baseline_variance=torch.ones(1, 6),
        steps=2,
        scope=scope,
        optimizer_name=optimizer,
    )
    assert mean.shape == variance.shape == (1, 6)
    assert report["selected_heldout_mse"] <= report["initial_heldout_mse"]
    assert report["selected_step"] in {0, 1, 2}
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    for key, value in head.state_dict().items():
        assert torch.equal(value, head_state[key])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


def test_clean_step_zero_fallback_reuses_exact_saved_predictions():
    torch.manual_seed(93)
    model, head = TinyModel(), TinyHead()
    baseline = torch.full((1, 6), 123.456)
    mean, _, report = adapt_guarded(
        model,
        head,
        torch.ones(1, 8),
        torch.ones(1, 8, dtype=torch.bool),
        list(range(8)),
        17,
        baseline_mean=baseline,
        baseline_variance=torch.ones_like(baseline),
        maximum_relative_latent_drift=1e-30,
        steps=1,
    )
    assert report["selected_step"] == 0
    assert torch.equal(mean, baseline)


def test_heldout_target_values_do_not_enter_adaptation_gradient():
    torch.manual_seed(91)
    model, head = TinyModel(), TinyHead()
    x = torch.arange(8).reshape(1, 8).float() / 8
    visible = torch.ones_like(x, dtype=torch.bool)
    validation, _ = split_columns(visible, list(range(8)), 17, 0)
    altered = x.clone()
    altered[:, validation] += 99
    gradients = []
    hook = next(model.encoder.parameters()).register_hook(
        lambda grad: gradients.append(grad.detach().clone())
    )
    for values in [x, altered]:
        adapt_guarded(
            model,
            head,
            values,
            visible,
            list(range(8)),
            17,
            baseline_mean=torch.zeros(1, 6),
            baseline_variance=torch.ones(1, 6),
            steps=1,
        )
    hook.remove()
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)


def test_saved_moment_density_matches_independent_torch_gaussian_and_unit_shift():
    from masked_stellar_autoencoder.training.guarded_test_time import (
        gaussian_log_prob_from_moments,
    )

    y = np.array([[0.5, -0.2], [0.1, 0.8]])
    mean = np.array([[0.0, 0.1], [0.2, 0.4]])
    variance = np.array([[1.0, 0.2], [0.3, 0.8]])
    expected = (
        torch.distributions.Normal(torch.tensor(mean), torch.tensor(variance).sqrt())
        .log_prob(torch.tensor(y))
        .sum(-1)
        .numpy()
    )
    np.testing.assert_allclose(
        gaussian_log_prob_from_moments(y, mean, variance), expected, atol=1e-15
    )
    scale = np.array([3.0, 0.2])
    np.testing.assert_allclose(
        gaussian_log_prob_from_moments(y * scale, mean * scale, variance * scale**2),
        expected - np.log(scale).sum(),
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="positive"):
        gaussian_log_prob_from_moments(y, mean, variance * 0)
