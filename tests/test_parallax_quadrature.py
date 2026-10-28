"""Independent units, signed observations and nonlinear integration checks."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

pytest.importorskip("torchregress.losses.forward_quadrature")

from masked_stellar_autoencoder.training.parallax_quadrature import (
    SignedParallaxQuadratureNLLLoss,
)


def args(mean=10.0, std=0.2, observed=-0.1, error=0.5):
    return tuple(
        torch.tensor([x], dtype=torch.double) for x in (mean, std, observed, error)
    )


def test_mu10_is_one_mas_and_explicit_zero_point_sign():
    a = args(std=0, observed=0.98, error=0.1)
    loss, count = SignedParallaxQuadratureNLLLoss(zero_point_mas=-0.02)(
        *a, parallax_used_as_model_input=False
    )
    assert count == 1
    assert loss.item() == pytest.approx(
        math.log(0.1) + 0.5 * math.log(2 * math.pi), abs=1e-12
    )
    other, _ = SignedParallaxQuadratureNLLLoss(zero_point_mas=0.02)(
        *a, parallax_used_as_model_input=False
    )
    assert other > loss


def test_signed_negative_parallax_and_floor_against_adaptive_integration():
    mean, std, observed, error, zero_point, floor = 10.2, 0.3, -0.2, 0.7, -0.025, 0.15
    effective = math.hypot(error, floor)
    integral, _ = quad(
        lambda z: (
            np.exp(-0.5 * z * z)
            / np.sqrt(2 * np.pi)
            * np.exp(
                -0.5
                * (
                    (observed - (10 ** (2 - 0.2 * (mean + std * z)) + zero_point))
                    / effective
                )
                ** 2
            )
            / (effective * np.sqrt(2 * np.pi))
        ),
        -12,
        12,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    result, count = SignedParallaxQuadratureNLLLoss(
        96, zero_point_mas=zero_point, systematic_floor_mas=floor
    )(*args(mean, std, observed, error), parallax_used_as_model_input=False)
    assert count == 1 and result.item() == pytest.approx(-math.log(integral), abs=1e-12)


def test_missing_errors_not_invented_even_with_floor_and_nan_latents():
    mean = torch.tensor(
        [10.0, float("nan"), float("nan"), float("nan")], requires_grad=True
    )
    std = torch.tensor([0.2, float("nan"), float("nan"), float("nan")])
    y = torch.tensor([-0.1, -0.3, 0.5, 0.7])
    error = torch.tensor([0.5, float("nan"), 0.0, -0.2])
    loss, count = SignedParallaxQuadratureNLLLoss(systematic_floor_mas=1.0)(
        mean, std, y, error, parallax_used_as_model_input=False
    )
    assert count == 1 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(mean.grad).all() and not mean.grad[1:].any()


def test_empty_labels_and_eligibility_and_zero_weight_counts():
    mean = torch.tensor([float("nan")], requires_grad=True)
    zero, count = SignedParallaxQuadratureNLLLoss()(
        mean,
        mean,
        torch.tensor([-0.2]),
        torch.tensor([0.1]),
        parallax_used_as_model_input=False,
        eligible=torch.tensor([False]),
    )
    assert count == 0 and zero == 0
    zero.backward()
    assert mean.grad.item() == 0
    zero, count = SignedParallaxQuadratureNLLLoss()(
        *args(),
        parallax_used_as_model_input=False,
        weights=torch.zeros(1, dtype=torch.double),
    )
    assert count == 0 and zero == 0


def test_latent_gradients_and_autocast():
    a = list(args())
    a[0].requires_grad_()
    a[1].requires_grad_()
    fn = SignedParallaxQuadratureNLLLoss()
    expected, _ = fn(*a, parallax_used_as_model_input=False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = fn(*a, parallax_used_as_model_input=False)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    for tensor in a[:2]:
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_double_use_requires_explicit_boolean_and_is_rejected():
    fn = SignedParallaxQuadratureNLLLoss()
    with pytest.raises(TypeError):
        fn(*args())
    with pytest.raises(ValueError, match="both model input"):
        fn(*args(), parallax_used_as_model_input=True)
    with pytest.raises(TypeError, match="boolean"):
        fn(*args(), parallax_used_as_model_input=None)


@pytest.mark.parametrize(
    "kwargs", [{"systematic_floor_mas": -1}, {"zero_point_mas": float("nan")}]
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SignedParallaxQuadratureNLLLoss(**kwargs)
