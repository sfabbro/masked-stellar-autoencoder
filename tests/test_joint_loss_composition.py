"""Numerical gates for proposed joint loss composition; no runner is launched."""

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.parallax_likelihood import signed_parallax_nll
from masked_stellar_autoencoder.training.xp_experiments import experiment_objective


def fixture():
    # Unequal missing-label counts and a wholly unlabelled first microbatch.
    target = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 4.0, 2.0],
            [4.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
        ],
        dtype=torch.float64,
    )
    score = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [False, True, True],
            [True, False, True],
            [False, True, False],
        ],
    )
    observed = torch.tensor([float("nan"), -0.2, 0.0, 0.3, 0.7], dtype=torch.float64)
    sigma = torch.tensor([0.1, 0.1, 0.2, 0.0, 0.4], dtype=torch.float64)
    spectral = {
        "scale": torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64),
        "center": torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64),
        "basis": torch.tensor(
            [[1.0, 0.5, 0.2], [0.3, 0.7, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        ),
        "wavelength": torch.tensor([400.0, 550.0, 800.0], dtype=torch.float64),
    }
    return target, score, observed, sigma, spectral


@pytest.mark.parametrize("spectral_weight", [0.0, 0.1])
def test_ragged_microbatches_preserve_joint_loss_and_shared_parameter_gradient(
    spectral_weight,
):
    target, score, observed, sigma, spectral = fixture()
    valid_count = int((torch.isfinite(observed) & (sigma > 0)).sum())
    global_count = score.sum()
    arm = {"loss": "spectral_mae", "spectral_weight": spectral_weight}

    def calculate(slices):
        theta = torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
        pieces = []
        for sl in slices:
            # Shared theta represents an encoder parameter feeding both branches.
            prediction = target[sl] * 0.8 + theta
            mu = torch.ones(len(target[sl]), dtype=torch.float64) * (11.0 + theta)
            nll, count = signed_parallax_nll(mu, observed[sl], sigma[sl])
            replay = experiment_objective(
                prediction,
                target[sl],
                score[sl],
                [0, 1],
                arm,
                global_count,
                spectral=spectral,
            )
            # Local MEANS must be weighted by actual label count/global count.
            pieces.append(nll * (count / valid_count) + 0.01 * replay)
        total = sum(pieces)
        gradient = torch.autograd.grad(total, theta)[0]
        return total.detach(), gradient.detach()

    whole = calculate([slice(None)])
    split = calculate([slice(0, 1), slice(1, 3), slice(3, 5)])
    for actual, expected in zip(split, whole, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_spectral_term_independent_numpy_reference_and_flux_unit_invariance():
    target, score, _, _, spectral = fixture()
    prediction = target * 0.8 + 0.13
    arm = {"loss": "spectral_mae", "spectral_weight": 0.1}
    actual = experiment_objective(
        prediction,
        target,
        score,
        [0, 1],
        arm,
        score.sum(),
        spectral=spectral,
    )
    y, p, m = target.numpy(), prediction.numpy(), score.numpy()
    scale, center, basis, wave = (
        spectral[k].numpy() for k in ("scale", "center", "basis", "wavelength")
    )
    total = np.abs(p - y)[m].sum()
    for row in range(len(y)):
        # Independent per-star integration; visible XP residuals are exactly zero.
        flux = (y[row, :2] * scale[:2] + center[:2]) @ basis[:2]
        delta = ((p[row, :2] - y[row, :2]) * m[row, :2] * scale[:2]) @ basis[:2]
        fraction = np.trapezoid(np.abs(delta), wave) / np.trapezoid(np.abs(flux), wave)
        total += 0.1 * fraction * m[row, :2].sum()
    np.testing.assert_allclose(actual.item(), total / m.sum(), rtol=1e-14)
    converted = dict(spectral, basis=spectral["basis"] * 1e-18)
    changed = experiment_objective(
        prediction,
        target,
        score,
        [0, 1],
        arm,
        score.sum(),
        spectral=converted,
    )
    torch.testing.assert_close(actual, changed, rtol=1e-14, atol=1e-14)
