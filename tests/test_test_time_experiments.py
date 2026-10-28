import inspect

import torch
from torch import nn

from masked_stellar_autoencoder.training.test_time_experiments import (
    adapt_one,
    augmented_view,
    average_predictions,
    masked_object,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 5), nn.LayerNorm(5), nn.Tanh())
        self.decoder = nn.Linear(5, 4)
        self.reconstruction_layer = nn.Linear(4, 4)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(5, 1)

    def forward(self, z):
        return self.output(z)[:, :, None]


def fixture():
    torch.manual_seed(7)
    return Model().eval(), Head().eval(), torch.tensor([[0.3, -0.2, 0.8, -9999.0]])


def test_stateless_views_preserve_missing_and_have_exact_zero_limit():
    _, _, x = fixture()
    visible = x != -9999
    zero, hidden = augmented_view(x, visible, torch.ones(1, 4), [19], [0, 1, 2, 3])
    assert torch.equal(zero, x) and not hidden.any()
    a, _ = augmented_view(
        x,
        visible,
        torch.ones(1, 4),
        [19],
        [0, 1, 2, 3],
        noise_scale=1,
        mask_probability=0.3,
    )
    b, _ = augmented_view(
        x,
        visible,
        torch.ones(1, 4),
        [19],
        [0, 1, 2, 3],
        noise_scale=1,
        mask_probability=0.3,
    )
    assert torch.equal(a, b) and a[0, 3] == -9999 and torch.equal(x, zero)


def test_adaptation_restores_all_states_and_zero_steps_are_identity():
    model, head, x = fixture()
    visible = x != -9999
    states = {k: v.clone() for k, v in model.state_dict().items()}
    hs = {k: v.clone() for k, v in head.state_dict().items()}
    baseline = head(model.encoder(x))[:, :, 0].mean(1)
    for mode in ["latent_recon", "layernorm_recon"]:
        identity, _ = adapt_one(
            model, head, x, visible, [0, 1, 2], 19, mode=mode, steps=0
        )
        assert torch.equal(identity, baseline)
        result, _ = adapt_one(
            model,
            head,
            x,
            visible,
            [0, 1, 2],
            19,
            mode=mode,
            steps=3,
            lr=0.01,
            consistency=0.1,
        )
        assert torch.isfinite(result).all()
        for k, v in model.state_dict().items():
            assert torch.equal(states[k], v)
        for k, v in head.state_dict().items():
            assert torch.equal(hs[k], v)
        assert all(
            p.grad is None for p in list(model.parameters()) + list(head.parameters())
        )
    assert "labels" not in inspect.signature(adapt_one).parameters


def test_object_order_and_model_global_rng_are_unchanged():
    model, head, x = fixture()
    y = x.clone()
    y[:, :3] += 0.1
    rng = torch.get_rng_state().clone()
    a, _ = adapt_one(
        model,
        head,
        x,
        x != -9999,
        [0, 1, 2],
        19,
        mode="layernorm_recon",
        steps=2,
        lr=0.01,
    )
    adapt_one(
        model,
        head,
        y,
        y != -9999,
        [0, 1, 2],
        20,
        mode="layernorm_recon",
        steps=2,
        lr=0.01,
    )
    b, _ = adapt_one(
        model,
        head,
        x,
        x != -9999,
        [0, 1, 2],
        19,
        mode="layernorm_recon",
        steps=2,
        lr=0.01,
    )
    assert torch.equal(a, b) and torch.equal(rng, torch.get_rng_state())


def test_averaging_zero_view_and_permutation_identity():
    model, head, x = fixture()
    values = torch.cat([x, x + 0.05])
    values[:, 3] = -9999
    visible = values != -9999
    sigma = torch.ones(2, 3) * 0.1
    baseline = head(model.encoder(values))[:, :, 0].mean(1)
    assert torch.equal(
        average_predictions(model, head, values, visible, sigma, [19, 20], [0, 1, 2]),
        baseline,
    )
    a = average_predictions(
        model,
        head,
        values,
        visible,
        sigma,
        [19, 20],
        [0, 1, 2],
        noise_scale=1,
        mask_probability=0.2,
    )
    b = average_predictions(
        model,
        head,
        values.flip(0),
        visible.flip(0),
        sigma.flip(0),
        [20, 19],
        [0, 1, 2],
        noise_scale=1,
        mask_probability=0.2,
    )
    torch.testing.assert_close(a, b.flip(0), rtol=0, atol=0)


def test_masked_reconstruction_targets_are_never_visible_to_encoder():
    _, _, values = fixture()
    visible = values != -9999
    hidden, score = masked_object(
        values, visible, [0, 1, 2], 19, seed=1042, probability=0.2
    )
    assert score.any() and (~score & visible).any()
    assert torch.all(hidden[score] == -9999)
    assert torch.equal(hidden[~score], values[~score])


def test_missing_layernorm_rejects_without_changing_parameter_flags():
    import pytest

    model, head, values = fixture()
    model.encoder = nn.Sequential(nn.Linear(4, 5), nn.Tanh())
    before = [
        p.requires_grad for p in list(model.parameters()) + list(head.parameters())
    ]
    with pytest.raises(ValueError, match="no LayerNorm"):
        adapt_one(
            model, head, values, values != -9999, [0, 1, 2], 19, mode="layernorm_recon"
        )
    assert before == [
        p.requires_grad for p in list(model.parameters()) + list(head.parameters())
    ]
