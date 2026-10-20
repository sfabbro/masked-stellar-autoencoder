"""Numerical and isolation checks for the bounded native three-label screen."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from masked_stellar_autoencoder.training.multilabel_experiments import (
    MultiLabelHead,
    complete_roles,
    conformal_intervals,
    hash_selected,
    label_scaling,
    qmc_normal_nodes,
    refine_one,
)


def test_complete_roles_reject_overlap_and_keep_target_masks():
    observed = np.ones((8, 3), bool)
    quality = observed.copy()
    quality[0, 1] = False
    roles = {
        k: np.arange(8) // 2 == i
        for i, k in enumerate(["fit", "tune", "calibration", "test"])
    }
    actual = complete_roles(observed, quality, roles)
    np.testing.assert_array_equal(actual["fit"], [1])
    roles["test"][1] = True
    with pytest.raises(ValueError, match="overlap"):
        complete_roles(observed, quality, roles)


def test_scaling_uses_fit_only():
    values = np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0], [1000.0, 2000.0, 3000.0]])
    center, scale = label_scaling(values, [0, 1])
    np.testing.assert_array_equal(center, [2, 4, 6])
    np.testing.assert_array_equal(scale, [1, 2, 3])


def test_three_dimensional_gaussian_measurement_likelihood_matches_numpy():
    torch.manual_seed(10)
    head = MultiLabelHead("gaussian", latent_dim=4, width=16, noise_aware=True)
    latent = torch.randn(7, 4)
    target = torch.randn(7, 3)
    sigma = torch.rand(7, 3) * 0.3 + 0.01
    # Zero initialization gives N(0,I); measurement variance adds exactly.
    expected = 0.5 * (
        target.numpy() ** 2 / (1 + sigma.numpy() ** 2)
        + np.log(1 + sigma.numpy() ** 2)
        + math.log(2 * math.pi)
    )
    np.testing.assert_allclose(
        float(head.criterion(latent, target, sigma).detach()),
        expected.mean(),
        rtol=1e-6,
    )
    lp = head.log_prob(latent, target, sigma)
    np.testing.assert_allclose(lp.detach().numpy(), -expected.sum(1), rtol=1e-6)
    head.criterion(latent, target, sigma).backward()
    assert (
        head.output.weight.grad is not None
        and torch.isfinite(head.output.weight.grad).all()
    )


def test_qmc_nodes_are_antithetic_and_stable():
    first = qmc_normal_nodes(32)
    assert torch.equal(first, qmc_normal_nodes(32))
    assert torch.equal(first[:16], -first[16:])
    assert torch.isfinite(first).all()
    with pytest.raises(ValueError):
        qmc_normal_nodes(9)


@pytest.mark.parametrize("flow_type", ["nsf", "maf"])
def test_actual_three_dimensional_flow_nll_and_gradients(flow_type):
    torch.manual_seed(17)
    head = MultiLabelHead("flow", latent_dim=4, width=16, flow_type=flow_type)
    head.point_nodes = qmc_normal_nodes(8)
    latent = torch.randn(5, 4, requires_grad=True)
    target = torch.randn(5, 3)
    sigma = torch.full_like(target, 0.1)
    loss = head.criterion(latent, target, sigma)
    np.testing.assert_allclose(
        loss.detach().numpy(),
        -head.distribution(latent).log_prob(target).detach().mean().numpy() / 3,
        rtol=1e-6,
    )
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    mean, var = head.moments(latent.detach())
    assert mean.shape == var.shape == (5, 3)
    assert torch.isfinite(mean).all() and torch.isfinite(var).all() and (var > 0).all()


def test_cp_thresholds_match_independent_order_statistics():
    error = torch.arange(1.0, 101.0)[:, None] * torch.tensor([[1.0, 2.0, 3.0]])
    result = conformal_intervals(
        torch.zeros_like(error), error, torch.zeros(4, 3), alpha=0.1
    )
    marginal = math.ceil(101 * 0.9)
    joint = math.ceil(101 * (1 - 0.1 / 3))
    np.testing.assert_array_equal(
        result["marginal_threshold"].numpy(), error[marginal - 1].numpy()
    )
    np.testing.assert_array_equal(
        result["simultaneous_threshold"].numpy(), error[joint - 1].numpy()
    )
    assert (result["simultaneous_upper"] >= result["marginal_upper"]).all()


def test_multilabel_reconstruction_refinement_restores_all_model_state():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(5, 4), nn.LayerNorm(4), nn.Tanh())
            self.decoder = nn.Linear(4, 5)
            self.reconstruction_layer = nn.Identity()

    torch.manual_seed(4)
    model = Model().eval()
    head = MultiLabelHead("gaussian", latent_dim=4, width=16).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    head_state = {k: v.clone() for k, v in head.state_dict().items()}
    x = torch.tensor([[0.2, 0.3, 0.8, -0.1, -9999.0]])
    result, diagnostic = refine_one(model, head, x, x != -9999, [0, 1, 2, 3], 19)
    assert result.shape == (1, 3) and diagnostic["steps"] == 3
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    for key, value in head.state_dict().items():
        assert torch.equal(value, head_state[key])


def test_policy_has_independent_methods_and_matched_training_budgets():
    policy = yaml.safe_load(
        (
            Path(__file__).parents[1] / "configs/multilabel-screen-20261020.yaml"
        ).read_text()
    )
    assert len(policy["arms"]) == len({arm["run_id"] for arm in policy["arms"]}) == 14
    assert policy["reconstruction_presentations"] == 10_000_000
    forward = [arm for arm in policy["arms"] if arm["kind"] == "forward"]
    assert len(forward) == 1 and forward[0]["reconstruction_presentations"] == 0
    assert len(policy["ensemble_policy"]["members"]) == 3
    assert policy["ensemble_policy"]["total_reconstruction_presentations"] == 30_000_000
    assert all(
        arm["kind"] in {"huber", "gaussian", "flow", "forward"}
        for arm in policy["arms"]
    )
    ids = np.array([25, 1, 8, 42])
    selected = hash_selected(ids, 2)
    np.testing.assert_array_equal(ids[selected], ids[::-1][hash_selected(ids[::-1], 2)])


def test_plain_observed_density_does_not_double_convolve_label_errors():
    """Only intrinsic-density noise-aware arms convolve native errors at scoring."""
    import importlib.util
    import sys

    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "multilabel_runner_test", scripts / "run_multilabel_screen.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    model = nn.Module()
    model.encoder = nn.Identity()
    data = dict(
        input=np.zeros((2, 4), dtype="f4"),
        value=np.zeros((2, 3)),
        error=np.full((2, 3), 2.0),
        label_center=np.zeros(3),
        label_scale=np.ones(3),
    )
    for noise_aware in (False, True):
        head = MultiLabelHead(
            "gaussian", latent_dim=4, width=16, noise_aware=noise_aware
        ).eval()
        result = runner.predictions(
            model, head, data, np.arange(2), torch.device("cpu")
        )
        if noise_aware:
            assert np.all(
                result["observed_log_prob_standardized"]
                < result["intrinsic_log_prob_standardized"]
            )
        else:
            np.testing.assert_array_equal(
                result["observed_log_prob_standardized"],
                result["intrinsic_log_prob_standardized"],
            )


def test_forward_errors_ignore_unrelated_missing_xp_but_reject_used_missing():
    from masked_stellar_autoencoder.training.multilabel_experiments import (
        native_xp_error_subset,
    )

    errors = np.array([[2.0, 4.0, 0.0], [np.nan, np.nan, 0.0], [3.0, 6.0, 0.0]])
    g = np.array([8.5, np.nan, 8.5])
    rows, sigma = native_xp_error_subset(
        errors, np.array([2.0, 2.0, 1.0]), np.array([0, 1]), g, [2, 0, 2]
    )
    np.testing.assert_array_equal(rows, [0, 2])
    np.testing.assert_array_equal(sigma, [[1.0, 2.0], [1.5, 3.0]])
    import pytest

    with pytest.raises(ValueError, match="finite"):
        native_xp_error_subset(errors, np.ones(3), np.array([0, 1]), g, [1])


def test_nsf_chunked_moments_preserve_fixed_samples_and_gradients():
    torch.manual_seed(314)
    head = MultiLabelHead("flow", latent_dim=4, width=16, flow_type="nsf")
    head.point_nodes = qmc_normal_nodes(32)
    with torch.no_grad():
        head.output.weight.normal_(std=0.02)
    latent = torch.randn(3, 4, requires_grad=True)
    dist = head.distribution(latent)
    sample = dist.transform.inv(head.point_nodes[:, None, :].expand(-1, 3, -1))
    mean = sample.double().mean(0).to(latent.dtype)
    variance = (sample - mean).square().double().mean(0).to(latent.dtype)
    expected_gradient = torch.autograd.grad((mean + variance).sum(), latent)[0]
    actual_mean, actual_variance = head.moments(latent)
    actual_gradient = torch.autograd.grad(
        (actual_mean + actual_variance).sum(), latent
    )[0]
    torch.testing.assert_close(actual_mean, mean, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(actual_variance, variance, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-5, atol=5e-5)


def test_flow_moments_bound_inverse_node_allocation(monkeypatch):
    from types import SimpleNamespace

    head = MultiLabelHead("flow", latent_dim=4, width=16, flow_type="nsf")
    calls = []

    def inverse(nodes):
        calls.append(nodes.shape)
        assert len(nodes) <= 16
        return nodes * 2 + 3

    monkeypatch.setattr(
        head,
        "distribution",
        lambda latent: SimpleNamespace(transform=SimpleNamespace(inv=inverse)),
    )
    mean, variance = head.moments(torch.zeros(128, 4))
    assert len(calls) == 16 and sum(shape[0] for shape in calls) == 256
    expected = head.point_nodes.double() * 2 + 3
    torch.testing.assert_close(
        mean.double(), expected.mean(0).expand(128, -1), rtol=5e-5, atol=5e-5
    )
    torch.testing.assert_close(
        variance.double(),
        ((expected - expected.mean(0)) ** 2).mean(0).expand(128, -1),
        rtol=5e-5,
        atol=5e-5,
    )
