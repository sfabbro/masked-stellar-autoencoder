"""Analytic optimizer, source-only mask and exact per-object reset checks."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from masked_stellar_autoencoder.training.mae_test_time import (
    adapt_masked_mse,
    masked_mse,
    object_adaptation_scope,
    stateless_columns,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.decoder = nn.Linear(4, 4)
        self.reconstruction_layer = nn.Linear(4, 4)
        self.register_buffer("counter", torch.tensor(0.0))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 3)

    def moments(self, x):
        y = self.net(x)
        return y, torch.ones_like(y)


def test_mask_prefix_observation_and_object_order_independent():
    visible = torch.tensor([[True, False, True, True]])
    a = [stateless_columns(visible, [0, 1, 2, 3], 123, i, 0.75) for i in range(20)]
    for i, v in enumerate(a):
        assert 1 not in v and len(v) == 2
        np.testing.assert_array_equal(
            v, stateless_columns(visible, [0, 1, 2, 3], 123, i, 0.75)
        )
    assert len(stateless_columns(visible, [0, 1, 2, 3], 123, 0, 0.2)) == 1


@pytest.mark.parametrize("scope", ["encoder_decoder", "layernorm"])
def test_repeatable_adaptation_exact_reset_and_frozen_head(scope):
    torch.manual_seed(23)
    model = Model()
    head = Head()
    x = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    visible = torch.ones_like(x, dtype=torch.bool)
    before = copy.deepcopy(model.state_dict())
    hb = copy.deepcopy(head.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    first = adapt_masked_mse(
        model, head, x, visible, [0, 1, 2, 3], 42, steps=3, scope=scope, lr=1e-7
    )
    second = adapt_masked_mse(
        model, head, x, visible, [0, 1, 2, 3], 42, steps=3, scope=scope, lr=1e-7
    )
    assert torch.equal(first[0], second[0])
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    for k, v in head.state_dict().items():
        assert torch.equal(v, hb[k])
    assert flags == [p.requires_grad for p in model.parameters()]
    assert torch.equal(rng, torch.get_rng_state())


def test_exception_restores_state_and_sgd_momentum_matches_scalar_reference():
    model = Model()
    head = Head()
    before = copy.deepcopy(model.state_dict())
    with (
        pytest.raises(RuntimeError),
        object_adaptation_scope(model, head, "encoder_decoder") as params,
    ):
        with torch.no_grad():
            params[0].add_(1)
            model.counter.add_(1)
        raise RuntimeError("intentional")
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    p = nn.Parameter(torch.tensor([[0.7]], dtype=torch.float64))
    optimizer = torch.optim.SGD([p], lr=0.01, momentum=0.9)
    value = 0.7
    velocity = 0.0
    for _ in range(20):
        optimizer.zero_grad()
        loss = masked_mse(p, torch.tensor([[0.2]], dtype=torch.float64), [0])
        loss.backward()
        optimizer.step()
        velocity = 0.9 * velocity + 2 * (value - 0.2)
        value -= 0.01 * velocity
    assert p.item() == pytest.approx(value, abs=1e-15)
