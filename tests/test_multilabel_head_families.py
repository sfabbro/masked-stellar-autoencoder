"""Synthetic correctness tests for the extended prediction-head families.

Every numeric assertion is against an independently computed NumPy reference, not
against the implementation under test. All heads are deterministic: the output
layer weight is zeroed and the bias is set to the parameters under test, so the
head output does not depend on the latent or on training state.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from masked_stellar_autoencoder.training.multilabel_experiments import (
    MultiLabelHead,
    _standard_normal_expectiles,
    _standard_normal_quantiles,
    masked_gaussian_row_log_prob,
    masked_mixture_row_log_prob,
)

LOG_2PI = math.log(2 * math.pi)


def _pin(head, bias):
    head = head.double()
    values = torch.as_tensor(bias, dtype=head.output.bias.dtype)
    if values.shape != head.output.bias.shape:
        raise ValueError("bias must match the head output layout")
    with torch.no_grad():
        head.output.weight.zero_()
        head.output.bias.copy_(values)
    return head.eval()


def _numpy_gaussian_log_prob(mean, covariance, target):
    residual = np.asarray(target, dtype="f8") - np.asarray(mean, dtype="f8")
    covariance = np.asarray(covariance, dtype="f8")
    _, logdet = np.linalg.slogdet(covariance)
    solved = np.linalg.solve(covariance, residual.T).T
    return -0.5 * ((solved * residual).sum(-1) + logdet + residual.shape[-1] * LOG_2PI)


def _pack_tril(scale_tril):
    """Pack a lower-triangular factor in ``torch.tril_indices`` order.

    The head output packs the lower triangle row-major with the diagonal
    exponentiated, so the diagonal entries appear at packed positions 0, 2, 5...
    """
    scale_tril = np.asarray(scale_tril, dtype="f8")
    rows, columns = np.tril_indices(scale_tril.shape[0])
    packed = scale_tril[rows, columns].copy()
    diagonal = rows == columns
    packed[diagonal] = np.log(packed[diagonal])
    return packed


def _runner():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "head_families_runner", scripts / "run_multilabel_screen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_covariance_nll_matches_an_independent_numpy_gaussian():
    mean = np.array([0.4, -1.2, 0.7])
    scale_tril = np.array([[0.9, 0.0, 0.0], [0.35, 0.6, 0.0], [-0.2, 0.15, 0.45]])
    covariance = scale_tril @ scale_tril.T
    target = np.array([[0.1, -0.9, 0.2], [-0.6, -1.5, 1.1], [1.3, 0.4, 0.0]])
    head = _pin(
        MultiLabelHead("mvgaussian", latent_dim=2, width=8),
        np.r_[mean, _pack_tril(scale_tril)],
    )
    latent = torch.zeros(3, 2, dtype=torch.float64)
    y = torch.as_tensor(target, dtype=torch.float64)
    reference = _numpy_gaussian_log_prob(mean, covariance, target)
    np.testing.assert_allclose(
        head.log_prob(latent, y).detach().numpy(), reference, rtol=1e-12, atol=1e-12
    )
    # Intrinsic training loss is the per-coordinate negative log likelihood.
    sigma = torch.full_like(y, 0.1)
    loss = head.criterion(latent, y, sigma)
    np.testing.assert_allclose(
        float(loss.detach()), float(-reference.mean() / 3), rtol=1e-12
    )


def test_noise_aware_full_covariance_adds_the_measurement_covariance():
    mean = np.array([0.2, 0.5, -0.1])
    scale_tril = np.array([[0.7, 0.0, 0.0], [0.1, 0.5, 0.0], [-0.05, 0.2, 0.4]])
    target = np.array([[0.0, 0.4, 0.1], [1.0, -0.2, 0.6]])
    errors = np.array([[0.2, 0.3, 0.1], [0.15, 0.05, 0.25]])
    head = _pin(
        MultiLabelHead("mvgaussian", latent_dim=2, width=8, noise_aware=True),
        np.r_[mean, _pack_tril(scale_tril)],
    )
    latent = torch.zeros(2, 2, dtype=torch.float64)
    y = torch.as_tensor(target, dtype=torch.float64)
    sigma = torch.as_tensor(errors, dtype=torch.float64)
    intrinsic = _numpy_gaussian_log_prob(mean, scale_tril @ scale_tril.T, target)
    observed = np.array(
        [
            _numpy_gaussian_log_prob(
                mean, scale_tril @ scale_tril.T + np.diag(errors[row] ** 2), target[row]
            )
            for row in range(len(target))
        ]
    )
    np.testing.assert_allclose(
        head.log_prob(latent, y).detach().numpy(), intrinsic, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        head.log_prob(latent, y, sigma).detach().numpy(),
        observed,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.allclose(observed, intrinsic)


def test_masked_marginal_equals_the_lower_dimensional_gaussian():
    mean = np.array([0.3, -0.4, 0.8])
    scale_tril = np.array([[1.0, 0.0, 0.0], [0.4, 0.7, 0.0], [-0.1, 0.25, 0.5]])
    covariance = scale_tril @ scale_tril.T
    target = np.array([[0.2, -0.1, 0.9], [1.1, 0.3, -0.2]])
    keep = [0, 2]
    y = torch.zeros(2, 3, dtype=torch.float64)
    y[:, keep] = torch.as_tensor(target[:, keep], dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [True, False, True]])
    actual = masked_gaussian_row_log_prob(
        torch.as_tensor(mean, dtype=torch.float64)[None, :].expand(2, -1),
        torch.as_tensor(scale_tril, dtype=torch.float64)[None, :, :].expand(2, -1, -1),
        y,
        mask=mask,
    )
    reference = _numpy_gaussian_log_prob(
        mean[keep], covariance[np.ix_(keep, keep)], target[:, keep]
    )
    np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-12, atol=1e-12)
    # The partition identity: the fully observed pattern reproduces the joint
    # density, and a row with no observed coordinate contributes zero.
    full = masked_gaussian_row_log_prob(
        torch.as_tensor(mean, dtype=torch.float64)[None, :],
        torch.as_tensor(scale_tril, dtype=torch.float64)[None, :, :],
        torch.as_tensor(target[:1], dtype=torch.float64),
    )
    joint = _numpy_gaussian_log_prob(mean, covariance, target[:1])
    np.testing.assert_allclose(full.numpy(), joint, rtol=1e-12, atol=1e-12)
    empty = masked_gaussian_row_log_prob(
        torch.as_tensor(mean, dtype=torch.float64)[None, :],
        torch.as_tensor(scale_tril, dtype=torch.float64)[None, :, :],
        torch.full((1, 3), float("nan"), dtype=torch.float64),
    )
    np.testing.assert_array_equal(empty.numpy(), [0.0])
    # A non-positive-definite observed subvector raises; it is never jittered.
    bad = torch.zeros(1, 2, 2, dtype=torch.float64)
    bad[0, 1, 0] = bad[0, 1, 1] = 1.0
    with pytest.raises(ValueError, match="positive definite|strictly positive"):
        masked_gaussian_row_log_prob(
            torch.zeros(1, 2, dtype=torch.float64),
            bad,
            torch.zeros(1, 2, dtype=torch.float64),
        )


def test_mixture_masked_marginal_matches_independent_numpy_mixture():
    weights = np.array([0.3, 0.7])
    means = np.array([[[0.0, 0.5, -0.4], [0.8, -0.2, 0.1]]])
    stds = np.array([[[0.5, 0.4, 0.3], [0.2, 0.6, 0.7]]])
    target = np.array([[0.25, 0.1, 0.0]])
    keep = [1, 2]

    def reference_component(row_mean, row_std):
        per_coordinate = -0.5 * (
            (target[0, keep] - row_mean[keep]) ** 2 / row_std[keep] ** 2
            + np.log(row_std[keep] ** 2)
            + LOG_2PI
        )
        return per_coordinate.sum()

    components = np.array(
        [reference_component(means[0, k], stds[0, k]) for k in range(2)]
    )
    expected = np.logaddexp(
        np.log(weights[0]) + components[0], np.log(weights[1]) + components[1]
    )
    y = torch.full((1, 3), float("nan"), dtype=torch.float64)
    y[0, keep] = torch.as_tensor(target[0, keep], dtype=torch.float64)
    actual = masked_mixture_row_log_prob(
        torch.as_tensor(np.log(weights), dtype=torch.float64)[None, :],
        torch.as_tensor(means, dtype=torch.float64),
        torch.as_tensor(np.log(stds), dtype=torch.float64),
        y,
        mask=torch.tensor([[False, True, True]]),
    )
    np.testing.assert_allclose(actual.numpy(), [expected], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("noise_aware", [False, True])
def test_single_component_mdn_equals_the_diagonal_gaussian(noise_aware):
    mean = np.array([0.3, -0.7, 1.1])
    variance = np.array([0.4, 0.09, 1.6])
    log_stds = 0.5 * np.log(variance)
    target = np.array([[0.1, -0.9, 0.8], [0.6, 0.0, 1.9]])
    errors = np.full((2, 3), 0.2)
    gaussian = _pin(
        MultiLabelHead("gaussian", latent_dim=2, width=8, noise_aware=noise_aware),
        np.r_[mean, np.log(variance)],
    )
    mdn = _pin(
        MultiLabelHead(
            "mdn",
            latent_dim=2,
            width=8,
            mdn_components=1,
            **{"noise_aware": noise_aware},
        ),
        np.r_[0.0, mean, log_stds],
    )
    latent = torch.zeros(2, 2, dtype=torch.float64)
    y = torch.as_tensor(target, dtype=torch.float64)
    sigma = torch.as_tensor(errors, dtype=torch.float64)
    np.testing.assert_allclose(
        mdn.log_prob(latent, y).detach().numpy(),
        gaussian.log_prob(latent, y).detach().numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        mdn.log_prob(latent, y, sigma).detach().numpy(),
        gaussian.log_prob(latent, y, sigma).detach().numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        float(mdn.criterion(latent, y, sigma).detach()),
        float(gaussian.criterion(latent, y, sigma).detach()),
        rtol=1e-10,
    )
    mdn_mean, mdn_variance = mdn.moments(latent)
    np.testing.assert_allclose(
        mdn_mean.detach().numpy(),
        np.broadcast_to(mean, (2, 3)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mdn_variance.detach().numpy(),
        np.broadcast_to(variance, (2, 3)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_mdn_component_permutation_invariance():
    head = MultiLabelHead("mdn", latent_dim=2, width=8, mdn_components=2).double()
    torch.manual_seed(3)
    with torch.no_grad():
        head.output.weight.normal_(0, 0.3)
        head.output.bias.zero_()
    latent = torch.randn(4, 2, dtype=torch.float64)
    y = torch.randn(4, 3, dtype=torch.float64)
    sigma = torch.full((4, 3), 0.1, dtype=torch.float64)
    before = head.criterion(latent, y, sigma)
    state = {k: v.clone() for k, v in head.state_dict().items()}
    swapped = MultiLabelHead("mdn", latent_dim=2, width=8, mdn_components=2).double()
    swapped.load_state_dict(state)
    with torch.no_grad():
        logits = swapped.output.bias[:2].flip(0)
        means = swapped.output.bias[2:8].view(2, 3).flip(0)
        stds = swapped.output.bias[8:].view(2, 3).flip(0)
        swapped.output.bias[:2] = logits
        swapped.output.bias[2:8] = means.reshape(-1)
        swapped.output.bias[8:] = stds.reshape(-1)
    np.testing.assert_allclose(
        float(swapped.criterion(latent, y, sigma).detach()),
        float(before.detach()),
        rtol=1e-6,
    )


def test_quantile_head_recovers_gaussian_quantiles_without_crossing():
    torch.manual_seed(7)
    head = MultiLabelHead(
        "quantile", target_dim=1, latent_dim=1, width=8, quantiles=(0.16, 0.5, 0.84)
    )
    samples = torch.randn(512, 1)
    latent = torch.zeros(512, 1)
    sigma = torch.ones_like(samples)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = head.criterion(latent, samples, sigma)
        loss.backward()
        optimizer.step()
    levels = head._level_values(head(latent[:1])).detach().numpy()[0, :, 0]
    assert levels[0] < levels[1] < levels[2], "quantile levels crossed"
    np.testing.assert_allclose(levels, [-0.9945, 0.0, 0.9945], atol=0.12)
    assert not head.has_density
    assert np.isnan(head.log_prob(latent[:1], samples[:1]).detach().numpy()).all()
    mean, variance = head.moments(latent[:1])
    np.testing.assert_allclose(mean.detach().numpy(), [[0.0]], atol=0.12)
    assert float(variance.detach()) > 0


def test_quantile_crossover_penalty_orders_crossed_levels():
    from torchregress.losses.quantile import MultiQuantileLoss, QuantileCrossoverLoss

    levels = [0.16, 0.5, 0.84]
    ordered = torch.tensor([[[-1.0], [0.0], [1.0]]])
    crossed = torch.tensor([[[1.0], [0.0], [-1.0]]])
    target = torch.zeros(1, 1)
    base = MultiQuantileLoss(levels)
    penalized = QuantileCrossoverLoss(levels, crossover_penalty=10.0)
    np.testing.assert_allclose(
        float(penalized(ordered, target)), float(base(ordered, target)), rtol=1e-6
    )
    assert float(penalized(crossed, target)) > float(base(crossed, target))


def test_expectile_location_scale_recovery_matches_gaussian_constants():
    levels = (0.1, 0.5, 0.9)
    constants = _standard_normal_expectiles(levels)
    assert constants[1].abs() < 1e-10, "the 0.5 expectile of a Gaussian is the mean"
    assert bool(torch.all(constants[1:] > constants[:-1]))
    # For Z ~ N(0, 1), E_tau[Z] = c_tau, so a Gaussian has exact expectile values.
    mu, sigma = 2.5, 1.7
    predicted = (mu + sigma * constants).detach().numpy()
    head = _pin(
        MultiLabelHead(
            "expectile", target_dim=1, latent_dim=1, width=8, expectiles=levels
        ),
        predicted,
    )
    mean, variance = head.moments(torch.zeros(1, 1, dtype=torch.float64))
    np.testing.assert_allclose(mean.detach().numpy(), [[mu]], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        variance.detach().numpy(), [[sigma**2]], rtol=1e-6, atol=1e-6
    )
    assert not head.has_density
    assert head.density_kind is None


def test_gaussian_quantile_constants_are_the_inverse_normal_cdf():
    z = _standard_normal_quantiles((0.05, 0.16, 0.5, 0.84, 0.95)).detach().numpy()
    np.testing.assert_allclose(
        z,
        [-1.644853626951, -0.994457883210, 0.0, 0.994457883210, 1.644853626951],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("dims", [2, 4, 5])
def test_non_three_dimensional_denominators(dims):
    torch.manual_seed(11)
    latent = torch.randn(6, 3)
    y = torch.randn(6, dims, dtype=torch.float64)
    sigma = torch.full((6, dims), 0.1, dtype=torch.float64)
    for kind, extra in [
        ("gaussian", {}),
        ("mvgaussian", {}),
        ("lowrank", {"lowrank_rank": 1}),
        ("mdn", {"mdn_components": 2}),
        ("quantile", {"quantiles": (0.16, 0.5, 0.84)}),
        ("expectile", {"expectiles": (0.1, 0.5, 0.9)}),
    ]:
        head = MultiLabelHead(kind, target_dim=dims, latent_dim=3, width=8, **extra)
        with torch.no_grad():
            head.output.weight.normal_(0, 0.1)
            head.output.bias.zero_()
        loss = float(head.criterion(latent, y, sigma).detach())
        assert np.isfinite(loss)
        assert head.loss_denominator == "per_coordinate"
        mean, variance = head.moments(latent)
        assert mean.shape == (6, dims) and variance.shape == (6, dims)
        if head.has_density:
            per_coordinate = (
                -head.log_prob(latent, y).detach().mean().detach().numpy() / dims
            )
            if kind in {"mvgaussian", "lowrank"} or kind == "gaussian":
                np.testing.assert_allclose(loss, per_coordinate, rtol=1e-5, atol=1e-6)
        with pytest.raises(ValueError, match="Unsupported"):
            MultiLabelHead("not_a_head", target_dim=dims)


def test_flow_normalization_uses_the_declared_target_dimension():
    torch.manual_seed(13)
    head = MultiLabelHead("flow", target_dim=2, latent_dim=3, width=8, flow_type="nsf")
    head.point_nodes = head.point_nodes[:8]
    latent = torch.randn(5, 3)
    y = torch.randn(5, 2)
    sigma = torch.full_like(y, 0.1)
    loss = head.criterion(latent, y, sigma)
    reference = -head.log_prob(latent, y).detach().mean().detach().numpy() / 2
    np.testing.assert_allclose(float(loss.detach()), reference, rtol=1e-6)


def test_huber_head_declares_no_density_and_keeps_nan_log_prob():
    head = MultiLabelHead("huber", latent_dim=3, width=8)
    assert not head.has_density and head.density_kind is None
    latent = torch.randn(2, 3)
    assert np.isnan(head.log_prob(latent, torch.randn(2, 3)).detach().numpy()).all()


@pytest.mark.parametrize(
    "kind,extra",
    [
        ("mvgaussian", {}),
        ("lowrank", {"lowrank_rank": 2}),
        ("mdn", {"mdn_components": 4}),
        ("quantile", {}),
        ("expectile", {}),
    ],
)
def test_float64_scoring_is_batch_stable_for_every_family(kind, extra):
    torch.manual_seed(19)
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(4, 4), nn.SiLU())
    head = MultiLabelHead(kind, target_dim=3, latent_dim=4, width=16, **extra)
    with torch.no_grad():
        head.output.weight.normal_(0, 0.2)
    data = dict(
        input=np.random.default_rng(21).normal(size=(128, 4)).astype("f4"),
        value=np.random.default_rng(22).normal(size=(128, 3)),
        error=np.full((128, 3), 0.05),
        label_center=np.array([0.1, 0.3, -0.4]),
        label_scale=np.array([0.9, 1.7, 1.2]),
    )
    module = _runner()
    batched = module.predictions(model, head, data, np.arange(128), torch.device("cpu"))
    singles = module.predictions(
        model, head, data, np.arange(128), torch.device("cpu"), batch=1
    )
    for key in batched:
        if batched[key].dtype.kind != "f":
            np.testing.assert_array_equal(batched[key], singles[key])
            continue
        finite = np.isfinite(batched[key]) & np.isfinite(singles[key])
        np.testing.assert_allclose(
            batched[key][finite], singles[key][finite], rtol=1e-10, atol=1e-10
        )
        np.testing.assert_array_equal(np.isnan(batched[key]), np.isnan(singles[key]))


def test_noise_aware_arms_save_distinct_intrinsic_and_observed_densities():
    torch.manual_seed(23)
    model = nn.Module()
    model.encoder = nn.Identity()
    head = MultiLabelHead("mvgaussian", latent_dim=3, width=16, noise_aware=True)
    with torch.no_grad():
        head.output.weight.normal_(0, 0.2)
    data = dict(
        input=np.zeros((6, 3), dtype="f4"),
        value=np.random.default_rng(24).normal(size=(6, 3)),
        error=np.full((6, 3), 0.25),
        label_center=np.zeros(3),
        label_scale=np.ones(3),
    )
    result = _runner().predictions(model, head, data, np.arange(6), torch.device("cpu"))
    assert str(result["density_kind"]) == "observed_label"
    assert str(result["loss_denominator"]) == "per_coordinate"
    assert bool(result["observed_log_prob_available"]) is True
    intrinsic = result["intrinsic_log_prob_standardized"]
    observed = result["observed_log_prob_standardized"]
    assert np.isfinite(intrinsic).all() and np.isfinite(observed).all()
    assert not np.allclose(observed, intrinsic)
    np.testing.assert_allclose(
        result["observed_variance"], result["variance"] + data["error"] ** 2
    )
    # The saved variance stays intrinsic (no silent + sigma^2) and the Jacobian
    # terms are per row and per coordinate, not one scalar.
    assert result["jacobian"].shape == (6, 3)
    np.testing.assert_allclose(result["jacobian"], np.zeros((6, 3)), atol=0)
