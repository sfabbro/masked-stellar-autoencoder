import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.xp_representation import (
    continuum_operator,
    multiscale_spectral_relative_error,
    paired_latent_regularizer,
)


def test_continuum_preserves_constant_on_irregular_grid():
    wavelength = torch.tensor([400.0, 410.0, 430.0, 460.0], dtype=torch.float64)
    operator = continuum_operator(wavelength, 60)
    torch.testing.assert_close(
        operator @ torch.ones(4, dtype=torch.float64),
        torch.ones(4, dtype=torch.float64),
    )
    # Independently evaluate the window at 410 nm with trapezoid weights.
    np.testing.assert_allclose(operator[1].numpy(), [5 / 45, 15 / 45, 25 / 45, 0])


def test_multiscale_independent_projection_unit_invariance_and_hidden_gradients():
    wave = torch.tensor([400.0, 410.0, 430.0, 460.0])
    basis = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]])
    target = torch.tensor([[2.0, 0.1], [0.0, 0.0], [float("nan"), 1.0]])
    pred = torch.tensor([[3.0, 999.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, True], [False, True]])
    scale, center = torch.ones(2), torch.zeros(2)
    loss, counts = multiscale_spectral_relative_error(
        pred, target, mask, [0, 1], scale, center, basis, wave, width_nm=40
    )
    flux = target[0].numpy() @ basis.numpy()
    residual = np.ones(4)
    smoothing = continuum_operator(wave, 40).numpy()
    continuum = residual @ smoothing.T
    expected = (
        np.trapezoid(abs(continuum), wave.numpy())
        + np.trapezoid(abs(residual - continuum), wave.numpy())
    ) / np.trapezoid(abs(flux), wave.numpy())
    np.testing.assert_allclose(loss.detach().numpy(), [expected, 0, 0], rtol=1e-6)
    assert counts.tolist() == [1, 0, 0]
    tiny, _ = multiscale_spectral_relative_error(
        pred, target, mask, [0, 1], scale, center, basis * 1e-16, wave, width_nm=40
    )
    torch.testing.assert_close(tiny, loss)
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[0, 1] == 0


def test_invalid_spectral_grid_and_hidden_prediction_fail():
    with pytest.raises(ValueError):
        continuum_operator(torch.tensor([2.0, 1.0]))
    with pytest.raises(FloatingPointError):
        multiscale_spectral_relative_error(
            torch.tensor([[float("nan")]]),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            [0],
            torch.ones(1),
            torch.zeros(1),
            torch.ones(1, 2),
            torch.tensor([400.0, 500.0]),
        )


def test_latent_invariance_reference_scaling_and_numpy_covariance():
    first = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], requires_grad=True)
    second = first + torch.tensor([[0.1, -0.2]])
    scale = torch.tensor([2.0, 3.0])
    result = paired_latent_regularizer(first, second, scale)
    a, b = (
        first.detach().numpy() / scale.numpy(),
        second.detach().numpy() / scale.numpy(),
    )
    expected = ((a - b) ** 2).mean()
    for values in (a, b):
        expected += (
            0.1 / 2 * np.maximum(0.5 - np.sqrt(values.var(0, ddof=1) + 1e-8), 0).mean()
        )
        expected += (
            0.01
            / 2
            * np.mean(np.cov(values, rowvar=False)[~np.eye(2, dtype=bool)] ** 2)
        )
    np.testing.assert_allclose(result.detach().numpy(), expected, rtol=1e-5)
    torch.testing.assert_close(
        result, paired_latent_regularizer(first * 2, second * 2, scale * 2)
    )
    result.backward()
    assert torch.isfinite(first.grad).all()


def test_latent_collapse_penalty_and_invalid_references():
    constant = torch.ones(4, 3, requires_grad=True)
    loss = paired_latent_regularizer(constant, constant, torch.ones(3))
    assert loss > 0
    loss.backward()
    assert torch.isfinite(constant.grad).all()
    with pytest.raises(ValueError):
        paired_latent_regularizer(constant, constant, torch.zeros(3))


@pytest.mark.parametrize(
    "objective", ["multiscale_spectral_mae", "full_decoder_spectral_mae"]
)
def test_objective_global_denominator_and_visible_coefficient_behavior(objective):
    from masked_stellar_autoencoder.training.xp_experiments import experiment_objective

    target = torch.tensor([[2.0, 0.1, 0.0], [3.0, 0.2, 0.0]])
    prediction = torch.tensor(
        [[2.2, 1.3, 999.0], [3.4, 1.5, 999.0]], requires_grad=True
    )
    mask = torch.tensor([[True, False, False], [True, False, False]])
    wave = torch.tensor([400.0, 410.0, 430.0, 460.0])
    spectral = dict(
        scale=torch.ones(3),
        center=torch.zeros(3),
        basis=torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        wavelength=wave,
        smoothing=continuum_operator(wave, 40),
    )
    options = dict(loss=objective, spectral_weight=0.1)
    full = experiment_objective(
        prediction, target, mask, [0, 1], options, mask.sum(), spectral=spectral
    )
    parts = sum(
        experiment_objective(
            prediction[i : i + 1],
            target[i : i + 1],
            mask[i : i + 1],
            [0, 1],
            options,
            mask.sum(),
            spectral=spectral,
        )
        for i in range(2)
    )
    torch.testing.assert_close(full, parts)
    plain = experiment_objective(
        prediction,
        target,
        mask,
        [0, 1],
        dict(loss=objective, spectral_weight=0),
        mask.sum(),
        spectral=spectral,
    )
    torch.testing.assert_close(plain, torch.tensor(0.3))
    full.backward()
    assert (prediction.grad[:, 2] == 0).all()
    if objective == "multiscale_spectral_mae":
        assert (prediction.grad[:, 1] == 0).all()
    else:
        assert (prediction.grad[:, 1] != 0).all()
