import torch

from masked_stellar_autoencoder.training.nsf_confirmation import (
    confirmation_score,
    marginal_crps,
)


def test_crps_matches_univariate_energy_and_ignores_missing_targets():
    x = torch.tensor([[[-2.0, 99.0]], [[0.0, 98.0]], [[2.0, 97.0]]], requires_grad=True)
    y = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[True, False]])
    score = marginal_crps(x, y, mask)
    torch.testing.assert_close(score, torch.tensor(1 / 3))
    score.backward()
    assert torch.isfinite(x.grad).all() and (x.grad[:, :, 1] == 0).all()


def test_paired_objectives_use_identical_noise_and_energy_control():
    x = torch.randn(32, 3, 2, requires_grad=True)
    y = torch.zeros(3, 2)
    mask = torch.ones_like(y, dtype=torch.bool)
    sigma = torch.full_like(y, 0.1)
    known = mask.clone()
    a, ca = confirmation_score(
        x,
        y,
        mask,
        sigma,
        known,
        [0.1, 0.1],
        torch.Generator().manual_seed(3),
        energy_weight=1.0,
        crps_weight=0.0,
    )
    b, cb = confirmation_score(
        x,
        y,
        mask,
        sigma,
        known,
        [0.1, 0.1],
        torch.Generator().manual_seed(3),
        energy_weight=1.0,
        crps_weight=1.0,
    )
    torch.testing.assert_close(ca["masked_energy_score"], cb["masked_energy_score"])
    torch.testing.assert_close(b - a, cb["marginal_crps"])
    b.backward()
    assert torch.isfinite(x.grad).all()


def test_correlated_noise_is_gated_to_native_rows_and_keeps_gradients_finite():
    x = torch.randn(32, 3, 6, requires_grad=True)
    y = torch.zeros(3, 6)
    mask = torch.ones_like(y, dtype=torch.bool)
    sigma = torch.full_like(y, 0.1)
    base, _ = confirmation_score(
        x,
        y,
        mask,
        sigma,
        mask,
        [0.1] * 6,
        torch.Generator().manual_seed(4),
        energy_weight=1.0,
        crps_weight=1.0,
    )
    off, _ = confirmation_score(
        x,
        y,
        mask,
        sigma,
        mask,
        [0.1] * 6,
        torch.Generator().manual_seed(4),
        energy_weight=1.0,
        crps_weight=1.0,
        native_rows=[False] * 3,
        correlation=0.3,
        native_student_t=True,
    )
    torch.testing.assert_close(base, off)
    on, _ = confirmation_score(
        x,
        y,
        mask,
        sigma,
        mask,
        [0.1] * 6,
        torch.Generator().manual_seed(4),
        energy_weight=1.0,
        crps_weight=1.0,
        native_rows=[True, False, True],
        correlation=-0.3,
        native_student_t=True,
    )
    assert torch.isfinite(on)
    on.backward()
    assert torch.isfinite(x.grad).all()
