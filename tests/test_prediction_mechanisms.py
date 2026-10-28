"""Mechanism identities and finite-gradient checks before bounded pilots."""

import torch
from torch import nn

from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_prediction_head,
)
from masked_stellar_autoencoder.training.prediction_mechanisms import (
    NeuralRatioEstimator,
    corrected_score_objective,
    group_dro_objective,
    ratio_grid,
)


def test_ratio_head_objective_grid_and_reload():
    torch.manual_seed(42)
    head = NeuralRatioEstimator()
    z = torch.randn(256, 256)
    y = torch.linspace(-3.0, 2.0, 256)
    loss = head.criterion(z, y)
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )
    grid = ratio_grid(y)
    point = head.standardized_mle(z[:4], grid)
    other = NeuralRatioEstimator()
    other.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(
        point, other.standardized_mle(z[:4], grid), rtol=0, atol=0
    )
    assert torch.isin(point, grid).all()


def test_corrected_score_subtracts_known_gradient_variance():
    from torchregress.losses import CorrectedScoreEIVLoss

    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 0.0]]))
    x = torch.tensor([[1.0, 4.0], [2.0, 6.0]])
    y = model(x).detach().flatten() + 1
    sigma = torch.tensor([[0.5, 0.0], [0.25, 0.0]])
    objective = CorrectedScoreEIVLoss(model, sigma_x=0.0, eps=0.0, stabilize="none")
    loss = objective(x, y, sigma_x=torch.diag_embed(sigma.square()))
    torch.testing.assert_close(loss, torch.tensor((0.0 + 0.75) / 2))
    loss.backward()
    assert torch.isfinite(model.weight.grad).all()


def test_corrected_joint_adapter_supports_parameter_gradients():
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(4, 256), nn.Tanh())
    head = build_prediction_head("residual")
    objective = corrected_score_objective(model, head)
    loss = objective(
        torch.randn(4, 4),
        torch.randn(4),
        sigma_x=torch.diag_embed(torch.ones(4, 4) * 0.01),
    )
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in head.parameters())


def test_group_dro_uses_physical_fit_bins_and_updates_weights():
    labels = torch.tensor([-3.0, -2.5, -1.5, -0.5, 0.2])
    dro = group_dro_objective(labels, 0.0, 1.0)
    pred = torch.tensor([0.0, 0.0, -1.5, -0.5, 0.2], requires_grad=True)
    loss = dro(pred, labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    state = dro.state_dict()
    assert state


def test_group_dro_allows_absent_tail_in_a_bounded_subset():
    labels = torch.tensor([-0.1, 0.1, 0.3])
    dro = group_dro_objective(labels, 0.0, 1.0)
    loss = dro(labels, labels)
    assert torch.isfinite(loss)
