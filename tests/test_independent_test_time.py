"""Small invariants for the independent TTA audit, with no production replay."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location(
    "tta_audit",
    Path(__file__).parents[1] / "experiments/xp/monitor/verify_test_time_screen.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_independent_views_are_per_source_and_preserve_withheld_entries():
    x = np.array([[1, 2, -9999, 9], [3, 4, -9999, 8]], dtype="f4")
    visible = x != -9999
    sigma = np.ones((2, 3), dtype="f4")
    actual, hidden = AUDIT.view(
        x, visible, sigma, [17, 23], [0, 1, 2], 1042, 3, 1.0, 0.2
    )
    permuted, _ = AUDIT.view(
        x[::-1].copy(),
        visible[::-1].copy(),
        sigma,
        [23, 17],
        [0, 1, 2],
        1042,
        3,
        1.0,
        0.2,
    )
    np.testing.assert_array_equal(actual, permuted[::-1])
    np.testing.assert_array_equal(actual[:, 2:], x[:, 2:])
    np.testing.assert_array_equal(actual[hidden], -9999)
    zero, _ = AUDIT.view(x, visible, sigma, [17, 23], [0, 1, 2], 1042, 0, 0.0, 0.0)
    np.testing.assert_array_equal(zero, x)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.reconstruction_layer = nn.Identity()


class Head(nn.Module):
    def forward(self, z):
        return z[:, :1, None]


def test_latent_first_adam_step_matches_analytic_absolute_gradient():
    # Holdout is nonempty; identity decoder has a negative gradient on exactly
    # held-out coordinates. First Adam update equals lr*g/(abs(g)+eps).
    model, head = LinearModel(), Head()
    x = np.array([[0.3, 0.7]], dtype="f4")
    policy = dict(
        seed=1042,
        steps=1,
        adapt_mask_probability=0.2,
        anchor=0.1,
        consistency=0.1,
        layernorm_lr=0.0001,
    )
    corrupted, hidden = AUDIT.view(
        x,
        np.ones_like(x, dtype=bool),
        np.zeros_like(x),
        [19],
        [0, 1],
        1042,
        0,
        0.0,
        0.2,
    )
    if not hidden.any():
        k = int(AUDIT.random_stream(19, 1042, 0, "fallback").integers(2))
        hidden[0, k] = True
        corrupted[0, k] = -9999
    if hidden.all():
        hidden[0, 1] = False
        corrupted[0, 1] = x[0, 1]
    expected = corrupted.copy()
    gradient = -1 / int(hidden.sum())
    expected[hidden] -= 0.01 * gradient / (abs(gradient) + 1e-8)
    actual, diagnostic = AUDIT.adaptation(
        model,
        head,
        x,
        np.ones_like(x, dtype=bool),
        19,
        np.array([0, 1]),
        policy,
        "latent_recon",
    )
    np.testing.assert_allclose(actual, expected[0, 0], rtol=0, atol=1e-6)
    assert diagnostic["steps"] == 1 and diagnostic["scored_features"] == 1


def test_layernorm_updates_restore_all_state_and_repeat_independently():
    torch.manual_seed(17)
    model = LinearModel()
    model.encoder = nn.Sequential(nn.Linear(2, 3), nn.LayerNorm(3))
    model.decoder = nn.Linear(3, 2)
    model.eval()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    policy = dict(
        seed=1042,
        steps=3,
        adapt_mask_probability=0.2,
        anchor=0.1,
        consistency=0.1,
        layernorm_lr=0.0001,
    )
    x = np.array([[0.3, 0.7]], dtype="f4")
    first, _ = AUDIT.adaptation(
        model,
        Head(),
        x,
        np.ones_like(x, dtype=bool),
        19,
        np.array([0, 1]),
        policy,
        "layernorm_recon_consistency",
    )
    second, _ = AUDIT.adaptation(
        model,
        Head(),
        x,
        np.ones_like(x, dtype=bool),
        19,
        np.array([0, 1]),
        policy,
        "layernorm_recon_consistency",
    )
    assert first == second
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, initial[key])
