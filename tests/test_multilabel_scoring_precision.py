"""Evaluation precision regressions; training weights and modes stay untouched."""

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
    qmc_normal_nodes,
)


def runner():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "multilabel_precision_runner", scripts / "run_multilabel_screen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UnitGaussian(nn.Module):
    kind = "gaussian"
    noise_aware = False

    def moments(self, latent):
        mean = latent[:, :1].expand(-1, 3)
        return mean, torch.ones_like(mean)

    def log_prob(self, latent, target):
        mean, _ = self.moments(latent)
        return (-0.5 * ((target - mean).square() + math.log(2 * math.pi))).sum(-1)


def test_prediction_clone_preserves_small_signal_modes_parameters_and_rng():
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(2, 1, bias=False), nn.Dropout(0.3))
    with torch.no_grad():
        model.encoder[0].weight.fill_(1)
    model.train()
    head = UnitGaussian().train()
    data = dict(
        input=np.array([[1e8, 1], [1e8, 3]], dtype="f4"),
        value=np.repeat(np.array([[1e8 + 1], [1e8 + 3]]), 3, axis=1),
        error=np.ones((2, 3)),
        label_center=np.zeros(3),
        label_scale=np.ones(3),
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = runner().predictions(model, head, data, np.arange(2), torch.device("cpu"))
    np.testing.assert_array_equal(result["mean"], data["value"])
    np.testing.assert_allclose(
        result["intrinsic_log_prob_standardized"],
        -1.5 * math.log(2 * math.pi),
        rtol=0,
        atol=1e-14,
    )
    assert result["mean"].dtype == np.float64
    assert model.training and model.encoder.training and model.encoder[1].training
    assert head.training and model.encoder[0].weight.dtype == torch.float32
    assert model.encoder[0].weight.requires_grad
    assert model.encoder[0].weight.grad is None
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_standardization_and_native_density_conversion_happen_in_float64():
    model = nn.Module()
    model.encoder = nn.Identity()
    head = MultiLabelHead("gaussian", latent_dim=4, width=16)
    data = dict(
        input=np.zeros((2, 4), dtype="f4"),
        value=np.full((2, 3), 1e8 + 8, dtype="f4"),
        error=np.full((2, 3), 1, dtype="f4"),
        label_center=np.full(3, 1e8, dtype="f4"),
        label_scale=np.full(3, 3, dtype="f4"),
    )
    result = runner().predictions(model, head, data, np.arange(2), torch.device("cpu"))
    expected = -1.5 * ((8 / 3) ** 2 + math.log(2 * math.pi))
    np.testing.assert_allclose(
        result["intrinsic_log_prob_standardized"], expected, rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(
        result["intrinsic_log_prob_native"],
        expected - 3 * math.log(3),
        rtol=0,
        atol=1e-13,
    )


@pytest.mark.parametrize("flow_type", ["nsf", "cinn", "bpf"])
def test_nonidentity_flow_scoring_is_batch_stable_without_changing_saved_nodes(
    flow_type,
):
    torch.manual_seed(214)
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(4, 4), nn.SiLU())
    head = MultiLabelHead("flow", latent_dim=4, width=16, flow_type=flow_type)
    head.point_nodes = qmc_normal_nodes(8)
    with torch.no_grad():
        head.output.weight.normal_(0, 0.1)
    nodes = head.point_nodes.clone()
    data = dict(
        input=np.random.default_rng(14).normal(size=(9, 4)).astype("f4"),
        value=np.random.default_rng(15).normal(size=(9, 3)),
        error=np.full((9, 3), 0.05),
        label_center=np.array([0.1, 0.3, -0.4]),
        label_scale=np.array([0.9, 1.7, 1.2]),
    )
    module = runner()
    batched = module.predictions(model, head, data, np.arange(9), torch.device("cpu"))
    singles = module.predictions(
        model, head, data, np.arange(9), torch.device("cpu"), batch=1
    )
    for key in batched:
        if batched[key].dtype.kind != "f":
            # Declared metadata (density kind, denominator, names, masks) is
            # carried alongside the float64 scoring arrays and is compared
            # separately; it is not part of the numerical payload.
            np.testing.assert_array_equal(batched[key], singles[key])
            continue
        assert batched[key].dtype == np.float64
        np.testing.assert_allclose(batched[key], singles[key], rtol=1e-10, atol=1e-10)
    assert torch.equal(head.point_nodes, nodes)
    assert head.point_nodes.dtype == torch.float32


def _log_target_data(names, native, center, scale):
    return dict(
        input=np.zeros(native.shape, dtype="f4"),
        value=np.asarray(native, dtype="f8"),
        error=np.full(native.shape, 0.1, dtype="f8"),
        label_center=np.asarray(center, dtype="f8"),
        label_scale=np.asarray(scale, dtype="f8"),
        target_names=names,
        observed=np.ones(native.shape, dtype=bool),
        quality=np.ones(native.shape, dtype=bool),
    )


def test_native_density_uses_per_row_per_coordinate_jacobian_for_log_targets():
    """log-distance/log-age densities are corrected row by row, not by one scalar."""
    model = nn.Module()
    model.encoder = nn.Identity()
    head = MultiLabelHead("gaussian", target_dim=2, latent_dim=2, width=8).eval()
    names = ("distance", "age")
    native = np.array([[0.5, 1.0], [4.0, 3.0], [20.0, 30.0]])
    center = np.array([1.0, 2.0])
    scale = np.array([0.4, 0.6])
    standardized = (np.log(native) - center) / scale
    standard_log_prob = -0.5 * (
        standardized**2 + math.log(2 * math.pi)
    ).sum(1)
    data = _log_target_data(names, native, center, scale)
    result = runner().predictions(
        model, head, data, np.arange(3), torch.device("cpu")
    )
    np.testing.assert_allclose(
        result["intrinsic_log_prob_standardized"], standard_log_prob, rtol=1e-13
    )
    expected_native = standard_log_prob - np.log(scale).sum() - np.log(native).sum(1)
    np.testing.assert_allclose(
        result["intrinsic_log_prob_native"], expected_native, rtol=1e-13
    )
    expected_jacobian = np.log(scale)[None, :] + np.log(native)
    np.testing.assert_allclose(
        result["jacobian"], expected_jacobian, rtol=1e-13, atol=1e-13
    )
    assert result["jacobian"].shape == (3, 2)
    assert str(result["density_kind"]) == "intrinsic"
    assert str(result["loss_denominator"]) == "per_coordinate"
    assert bool(result["observed_log_prob_available"]) is False
    # The legacy single scalar is wrong by exactly +log d / +log age per row,
    # i.e. the correct native density is lower by the row sum of log(native).
    legacy = standard_log_prob - float(np.log(scale).sum())
    np.testing.assert_allclose(
        legacy - expected_native, np.log(native).sum(1), rtol=1e-13
    )


def test_runner_rejects_a_target_order_outside_the_frozen_registry():
    model = nn.Module()
    model.encoder = nn.Identity()
    head = MultiLabelHead("gaussian", target_dim=2, latent_dim=2, width=8).eval()
    native = np.ones((2, 2))
    data = _log_target_data(
        ("age", "distance"), native, np.zeros(2), np.ones(2)
    )
    import pytest

    with pytest.raises(ValueError, match="registry order"):
        runner().predictions(model, head, data, np.arange(2), torch.device("cpu"))


def test_huber_arm_reports_no_density_instead_of_a_nan_hidden_behind_metadata():
    model = nn.Module()
    model.encoder = nn.Identity()
    head = MultiLabelHead("huber", latent_dim=3, width=8).eval()
    data = dict(
        input=np.zeros((2, 3), dtype="f4"),
        value=np.zeros((2, 3)),
        error=np.ones((2, 3)),
        label_center=np.zeros(3),
        label_scale=np.ones(3),
    )
    result = runner().predictions(model, head, data, np.arange(2), torch.device("cpu"))
    assert str(result["density_kind"]) == "none"
    assert np.isnan(result["intrinsic_log_prob_native"]).all()
    assert np.isnan(result["observed_log_prob_native"]).all()
    assert np.isnan(result["variance"]).all()
