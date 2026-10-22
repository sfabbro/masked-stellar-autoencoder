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
        assert batched[key].dtype == np.float64
        np.testing.assert_allclose(batched[key], singles[key], rtol=1e-10, atol=1e-10)
    assert torch.equal(head.point_nodes, nodes)
    assert head.point_nodes.dtype == torch.float32
