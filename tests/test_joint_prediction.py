"""Independent numerical gates for the first joint prediction components."""

import math

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.joint_prediction import (
    ResidualJointHead,
    masked_gaussian_measurement_nll,
    signed_log_distance_nll,
)


def test_every_partial_target_pattern_matches_numpy_native_noise_convolution():
    rng = np.random.default_rng(12)
    mean = rng.normal(size=(8, 3))
    scale = np.broadcast_to(
        np.array([[0.7, 0, 0], [0.2, 0.9, 0], [-0.1, 0.15, 0.5]]), (8, 3, 3)
    ).copy()
    target = rng.normal(size=(8, 3))
    error = np.broadcast_to([0.15, 0.3, 0.1], (8, 3)).copy()
    mask = np.array([[bool(row & (1 << col)) for col in range(3)] for row in range(8)])
    target[~mask] = np.nan
    expected = 0.0
    for row in range(1, 8):
        dims = np.flatnonzero(mask[row])
        covariance = (scale[row] @ scale[row].T)[np.ix_(dims, dims)] + np.diag(
            error[row, dims] ** 2
        )
        residual = target[row, dims] - mean[row, dims]
        expected += 0.5 * (
            residual @ np.linalg.solve(covariance, residual)
            + np.linalg.slogdet(covariance)[1]
            + len(dims) * np.log(2 * np.pi)
        )
    actual = masked_gaussian_measurement_nll(
        *(torch.tensor(x) for x in (mean, scale, target, error))
    )
    assert actual.observation_count == int(mask.sum()) == 12
    assert actual.row_count == 7
    np.testing.assert_allclose(actual.numerator.item(), expected, rtol=1e-13)


def test_native_correlated_measurement_covariance_and_missing_fields():
    mean = torch.zeros((2, 3), dtype=torch.float64)
    scale = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    target = torch.tensor(
        [[1.0, 2.0, float("nan")], [float("nan"), float("nan"), float("nan")]],
        dtype=torch.float64,
    )
    errors = torch.tensor(
        [[0.2, 0.4, float("nan")], [float("nan")] * 3], dtype=torch.float64
    )
    noise = torch.full((2, 3, 3), float("nan"), dtype=torch.float64)
    noise[0, :2, :2] = torch.tensor([[0.04, 0.04], [0.04, 0.16]], dtype=torch.float64)
    actual = masked_gaussian_measurement_nll(
        mean, scale, target, errors, measurement_covariance=noise
    )
    c = np.eye(2) + noise[0, :2, :2].numpy()
    y = np.array([1.0, 2.0])
    expected = 0.5 * (
        y @ np.linalg.solve(c, y) + np.linalg.slogdet(c)[1] + 2 * np.log(2 * np.pi)
    )
    assert actual.observation_count == 2
    assert actual.numerator.item() == pytest.approx(expected)
    for bad, message in [
        ([[0.04, 0.5], [0.5, 0.16]], "positive semidefinite"),
        ([[0.04, 0.04], [0.03, 0.16]], "symmetric"),
        ([[0.08, 0.04], [0.04, 0.16]], "diagonal"),
    ]:
        noise[0, :2, :2] = torch.tensor(bad, dtype=torch.float64)
        with pytest.raises(ValueError, match=message):
            masked_gaussian_measurement_nll(
                mean, scale, target, errors, measurement_covariance=noise
            )


def test_ragged_losses_and_gradients_equal_whole_optimizer_batch():
    target = torch.tensor(
        [
            [float("nan")] * 3,
            [1.0, float("nan"), 2.0],
            [2.0, 1.0, 3.0],
            [float("nan"), -0.1, 0.5],
        ],
        dtype=torch.float64,
    )
    error = torch.ones_like(target) * 0.2
    masks = torch.isfinite(target)
    pi = torch.tensor([float("nan"), -0.1, 0.0, 2.0], dtype=torch.float64)
    pe = torch.ones_like(pi) * 0.1

    def calculate(slices):
        theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        loss = theta * 0
        for sl in slices:
            n = len(target[sl])
            mean = theta.expand(n, 3)
            scale = torch.eye(3, dtype=torch.float64).expand(n, 3, 3) * theta.exp()
            joint = masked_gaussian_measurement_nll(mean, scale, target[sl], error[sl])
            distance = signed_log_distance_nll(theta.expand(n), pi[sl], pe[sl])
            loss = loss + joint.numerator / int(masks.sum()) + distance.numerator / 3
        return loss.detach(), torch.autograd.grad(loss, theta)[0]

    for actual, expected in zip(
        calculate([slice(None)]),
        calculate([slice(0, 1), slice(1, 3), slice(3, 4)]),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_empty_labels_have_zero_gradient_even_with_nan_measurements():
    mean = torch.ones(3, 3, requires_grad=True)
    scale = torch.eye(3).expand(3, 3, 3).clone().requires_grad_()
    target = torch.full_like(mean, float("nan"))
    errors = torch.full_like(mean, float("nan"))
    loss = masked_gaussian_measurement_nll(mean, scale, target, errors)
    loss.mean.backward()
    assert loss.observation_count == loss.row_count == 0
    assert torch.count_nonzero(mean.grad) == torch.count_nonzero(scale.grad) == 0
    distance = torch.ones(3, requires_grad=True)
    loss = signed_log_distance_nll(distance, target[:, 0], errors[:, 0])
    loss.mean.backward()
    assert loss.observation_count == 0 and torch.count_nonzero(distance.grad) == 0


def test_empty_loss_does_not_overflow_by_summing_unused_finite_predictions():
    mean = torch.full((10, 3), 1e38, requires_grad=True)
    scale = torch.eye(3).expand(10, 3, 3)
    missing = torch.full_like(mean, float("nan"))
    loss = masked_gaussian_measurement_nll(mean, scale, missing, missing)
    assert loss.numerator.item() == 0
    loss.mean.backward()
    assert torch.count_nonzero(mean.grad) == 0


def test_head_large_offdiagonal_has_no_unused_exponential_nan_gradient():
    head = ResidualJointHead(input_dim=2, width=4)
    with torch.no_grad():
        # Packed tril order: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2).
        head.output.bias[4] = 1000.0
    output = head(torch.zeros(1, 2))
    assert output["scale_tril"][0, 1, 0] == 1000
    output["scale_tril"].sum().backward()
    assert torch.isfinite(head.output.bias.grad).all()


def test_signed_parallax_numpy_units_per_star_zero_point_and_analytic_gradient():
    s = torch.tensor(
        [0.0, math.log(2), math.log(10)], dtype=torch.float64, requires_grad=True
    )
    pi = torch.tensor([-1.0, 0.0, 0.4], dtype=torch.float64)
    error = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float64)
    zero = torch.tensor([-0.02, 0.01, 0.03], dtype=torch.float64)
    actual = signed_log_distance_nll(s, pi, error, zero_point_mas=zero)
    q = np.array([1.0, 0.5, 0.1])
    residual = pi.numpy() - zero.numpy() - q
    expected = np.sum(
        0.5 * (residual / error.numpy()) ** 2
        + np.log(error.numpy())
        + 0.5 * np.log(2 * np.pi)
    )
    assert actual.observation_count == 3
    assert actual.numerator.item() == pytest.approx(expected)
    actual.numerator.backward()
    np.testing.assert_allclose(
        s.grad.numpy(), residual * q / error.numpy() ** 2, rtol=1e-13
    )


def test_gaussian_gradcheck_and_invalid_covariance_no_silent_repair():
    target = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float64)
    error = torch.ones_like(target) * 0.2
    mean = torch.zeros_like(target, requires_grad=True)
    raw = torch.tensor(
        [[0.1, 0.2, 0.3, -0.1, 0.1, 0.2]], dtype=torch.float64, requires_grad=True
    )
    index = torch.tril_indices(3, 3)

    def loss(m, r):
        packed = torch.where((index[0] == index[1])[None], r.exp(), r)
        scale = packed.new_zeros((1, 3, 3))
        scale[:, index[0], index[1]] = packed
        return masked_gaussian_measurement_nll(m, scale, target, error).numerator

    assert torch.autograd.gradcheck(loss, (mean, raw), atol=1e-5)
    scale = torch.eye(3).expand(1, 3, 3).clone().double()
    scale[:, 0, 1] = 0.1
    with pytest.raises(ValueError, match="lower triangular"):
        masked_gaussian_measurement_nll(mean, scale, target, error)
    with pytest.raises(ValueError, match="exact target shape"):
        masked_gaussian_measurement_nll(
            mean,
            torch.eye(3).expand(1, 3, 3).double(),
            target,
            error,
            eligible=torch.ones(1, dtype=torch.bool),
        )


def test_head_initialization_reload_and_optimizer_resume_exact():
    torch.manual_seed(19)
    head = ResidualJointHead(
        input_dim=8, width=16, initial_log_distance=math.log(2)
    ).double()
    x = torch.randn(7, 8, dtype=torch.float64)
    first = head(x)
    torch.testing.assert_close(
        first["mean"], torch.zeros(7, 3, dtype=torch.float64), rtol=0, atol=0
    )
    torch.testing.assert_close(
        first["scale_tril"], torch.eye(3).expand(7, 3, 3).double(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        first["log_distance"],
        torch.full((7,), math.log(2), dtype=torch.float32).double(),
        rtol=0,
        atol=0,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    y = torch.randn(7, 3, dtype=torch.float64)

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        p = model(x)
        loss = masked_gaussian_measurement_nll(
            p["mean"], p["scale_tril"], y, torch.ones_like(y) * 0.1
        ).mean
        loss = (
            loss
            + signed_log_distance_nll(
                p["log_distance"],
                torch.ones(7, dtype=torch.float64),
                torch.ones(7, dtype=torch.float64) * 0.1,
            ).mean
        )
        loss.backward()
        opt.step()

    step(head, optimizer)
    import copy

    checkpoint = copy.deepcopy(
        {"head": head.state_dict(), "optimizer": optimizer.state_dict()}
    )
    restored = ResidualJointHead(input_dim=8, width=16).double()
    restored.load_state_dict(checkpoint["head"], strict=True)
    other = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    other.load_state_dict(checkpoint["optimizer"])
    for name, prediction in head(x).items():
        torch.testing.assert_close(prediction, restored(x)[name], rtol=0, atol=0)
    step(head, optimizer)
    step(restored, other)
    for value, expected in zip(head.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("s", [1000.0, -1000.0, float("nan")])
def test_distance_numerical_extremes_fail_instead_of_clipping(s):
    with pytest.raises(ValueError, match="model parallax"):
        signed_log_distance_nll(
            torch.tensor([s]), torch.tensor([-0.1]), torch.tensor([0.2])
        )
