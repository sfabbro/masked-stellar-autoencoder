import numpy as np
import torch

from masked_stellar_autoencoder.training.xp_experiments import (
    configure_optimizer,
    experiment_objective,
    noise_reliability,
    spectral_relative_error,
)


def test_optimizer_reset_preserves_group_settings_and_isolates_lr_restart():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5, weight_decay=0.02)
    parameter.square().sum().backward()
    optimizer.step()
    momentum = optimizer.state[parameter]["exp_avg"].clone()
    configure_optimizer(optimizer, {})
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], momentum)
    policy = configure_optimizer(optimizer, {"reset_optimizer": True})
    assert not optimizer.state and optimizer.param_groups[0]["lr"] == 1e-5
    assert policy["reset_moments"]
    configure_optimizer(optimizer, {"reset_optimizer": True, "lr_factor": 2})
    assert not optimizer.state and optimizer.param_groups[0]["lr"] == 2e-5
    assert optimizer.param_groups[0]["weight_decay"] == 0.02


def test_spectral_loss_matches_independent_projection_and_masks_visible_errors():
    target = torch.tensor([[0.0, 1.0, 2.0], [0.0, float("nan"), 2.0]])
    prediction = torch.tensor([[0.0, 2.0, 999.0], [0.0, 1.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[False, True, False], [False, False, True]])
    scale = torch.tensor([1.0, 2.0, 3.0])
    center = torch.tensor([0.0, 10.0, -3.0])
    basis = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0], [0.0, -1.0, 2.0]])
    wavelength = torch.tensor([400.0, 500.0, 600.0])
    fraction, count = spectral_relative_error(
        prediction, target, mask, [1, 2], scale, center, basis, wavelength
    )
    raw_target = np.array([12.0, 3.0]) @ basis.numpy()[1:]
    residual = np.array([2.0, 0.0]) @ basis.numpy()[1:]
    expected = np.trapezoid(abs(residual), wavelength.numpy()) / np.trapezoid(
        abs(raw_target), wavelength.numpy()
    )
    np.testing.assert_allclose(fraction.detach().numpy(), [expected, 0.0], rtol=1e-6)
    assert count.tolist() == [1, 0]
    tiny_fraction, tiny_count = spectral_relative_error(
        prediction, target, mask, [1, 2], scale, center, basis * 1e-16, wavelength
    )
    torch.testing.assert_close(fraction, tiny_fraction)
    torch.testing.assert_close(count, tiny_count)
    options = {"loss": "spectral_mae", "spectral_weight": 0.1}
    spectral = dict(scale=scale, center=center, basis=basis, wavelength=wavelength)
    full = experiment_objective(
        prediction, target, mask, [1, 2], options, mask.sum(), spectral=spectral
    )
    parts = sum(
        experiment_objective(
            prediction[i : i + 1],
            target[i : i + 1],
            mask[i : i + 1],
            [1, 2],
            options,
            mask.sum(),
            spectral=spectral,
        )
        for i in range(2)
    )
    torch.testing.assert_close(full, parts)
    full.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad[0, 2] == 0


def test_corrected_signal_weights_change_loss_and_keep_microbatch_denominator():
    weights = noise_reliability(
        np.array([[-1.0, -1.0], [1.0, 1.0]]), np.array([[0.1, 1.0], [0.1, 1.0]])
    )
    np.testing.assert_allclose(weights, [0.99, 0.05])
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(target, dtype=torch.bool)
    options = {"loss": "signal_weighted_mae"}
    reliability = torch.tensor(weights)
    full = experiment_objective(
        prediction, target, mask, [0, 1], options, mask.sum(), reliability=reliability
    )
    parts = sum(
        experiment_objective(
            prediction[i : i + 1],
            target[i : i + 1],
            mask[i : i + 1],
            [0, 1],
            options,
            mask.sum(),
            reliability=reliability,
        )
        for i in range(2)
    )
    torch.testing.assert_close(full, parts)
    full.backward()
    torch.testing.assert_close(prediction.grad[0], reliability / mask.sum())
