"""Weighted star likelihoods are explicit and remain split-bound."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("torchregress.losses.multisource")
from torchregress.losses import MarginalizedMultiSourceGaussianNLLLoss

from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
from masked_stellar_autoencoder.training.multisurvey_model import (
    CanonicalAtmosphereModel,
    CanonicalGaussianHead,
    multisurvey_objective,
)
from masked_stellar_autoencoder.training.multisurvey_training import ConsensusTrainer


def load_fixture(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_objective():
    fixture = load_fixture("test_multisurvey_model")
    center, scale, relative, _ = fixture.spline_setup()

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 5)

    torch.manual_seed(17)
    model = CanonicalAtmosphereModel(
        Toy().double(), CanonicalGaussianHead(5, center, scale, width=8).double()
    )
    batch = fixture.batch()
    batch["input"] = torch.randn(2, 4, dtype=torch.float64)
    batch["role"] = ["fit", "fit"]
    kwargs = dict(
        expected_registry_sha256="a" * 64,
        expected_source_registry=("a", "b", "anchor"),
        expected_training_source_binding="b" * 64,
        assume_diagonal_catalog_covariance=True,
    )
    return model, batch, relative, kwargs


def test_weighted_objective_matches_explicit_per_star_mean_and_gradients():
    model, batch, relative, kwargs = synthetic_objective()
    weights = torch.tensor([9.0, 1.0], dtype=torch.float64)
    output = multisurvey_objective(
        model, batch, relative, torch.ones(3, 3), star_weights=weights, **kwargs
    )
    individual = MarginalizedMultiSourceGaussianNLLLoss(reduction="none")(
        output["mean"], output["scale_tril"], output["packed_observations"]
    )
    expected = (9.0 * individual[0] + individual[1]) / 10.0
    torch.testing.assert_close(output["supervised"], expected)
    gradient = torch.autograd.grad(
        output["supervised"], model.mae.encoder.weight, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected, model.mae.encoder.weight)[0]
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([1, 1]),
        torch.tensor([1.0, 1.0], requires_grad=True),
    ],
)
def test_invalid_weights_are_rejected(weights):
    model, batch, relative, kwargs = synthetic_objective()
    with pytest.raises(ValueError, match="weights"):
        multisurvey_objective(
            model, batch, relative, torch.ones(3, 3), star_weights=weights, **kwargs
        )


def test_weighted_batch_cannot_silently_drop_an_unlabelled_star():
    model, batch, relative, kwargs = synthetic_objective()
    batch["observed"][1] = False
    batch["error_known"][1] = False
    with pytest.raises(ValueError, match="at least one label"):
        multisurvey_objective(
            model,
            batch,
            relative,
            torch.ones(3, 3),
            star_weights=torch.ones(2),
            **kwargs,
        )


def test_trainer_requires_bound_sampling_contract_and_weights(tmp_path):
    fixture = load_fixture("test_multisurvey_training")
    original, dataset = fixture.setup_trainer(tmp_path)
    batch = collate_multisurvey([dataset[0], dataset[1]])
    mask = torch.tensor([[True, False], [True, False]])
    with pytest.raises(ValueError, match="pinned sampling contract"):
        original.step(batch, reconstruction_mask=mask, star_weights=torch.ones(2))
    trainer = ConsensusTrainer(
        original.model,
        original.relative_head,
        original.optimizer,
        original.gate,
        original.source_floor,
        parent_sha256="a" * 64,
        normalization_sha256="b" * 64,
        library_revision="c" * 40,
        msa_revision="d" * 40,
        assume_diagonal_catalog_covariance=True,
        reconstruction_weight=0.03,
        reconstruction_width=2,
        sampling_contract_sha256="e" * 64,
    )
    assert trainer.contract["sampling_contract_sha256"] == "e" * 64
    with pytest.raises(ValueError, match="pinned sampling contract"):
        trainer.step(batch, reconstruction_mask=mask)
    metrics = trainer.step(
        batch, reconstruction_mask=mask, star_weights=torch.tensor([1.5, 0.5])
    )
    assert trainer.steps == 1 and torch.isfinite(torch.tensor(metrics["total"]))
    trainer.sampling_contract_sha256 = "f" * 64
    with pytest.raises(ValueError, match="sampling contract changed"):
        trainer.step(batch, reconstruction_mask=mask, star_weights=torch.ones(2))
    dataset.close()
