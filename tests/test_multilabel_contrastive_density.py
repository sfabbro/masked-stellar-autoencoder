"""Independent InfoNCE equation and source-policy checks for matched MSA flows."""

import pytest
import torch

from masked_stellar_autoencoder.training.multilabel_experiments import MultiLabelHead


@pytest.mark.parametrize("objective", ["contrastive", "nll_contrastive"])
def test_joint_context_objective_matches_independent_equation_and_gradients(objective):
    torch.manual_seed(17)
    head = MultiLabelHead(
        "flow",
        latent_dim=4,
        width=8,
        flow_type="maf",
        flow_objective=objective,
        contrastive_weight=0.1,
    )
    torch.nn.init.normal_(head.output.weight, std=0.2)
    latent = torch.randn(5, 4, requires_grad=True)
    target = torch.randn(5, 3)
    sigma = torch.ones_like(target)
    ids = torch.arange(5)
    actual = head.criterion(latent, target, sigma, source_ids=ids)
    context = head(latent)
    positive = head.loss_fn.log_prob(context, target)
    negative = head.loss_fn.log_prob(context.roll(1, 0), target)
    contrastive = torch.nn.functional.softplus(negative - positive).mean()
    expected = (
        contrastive
        if objective == "contrastive"
        else -positive.mean() + 0.1 * contrastive
    ) / 3
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, latent, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, latent)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
    with pytest.raises(ValueError, match="duplicate"):
        head.criterion(
            latent, target, sigma, source_ids=torch.zeros(5, dtype=torch.long)
        )


def test_contrastive_rejects_silent_noise_convolution_composition():
    with pytest.raises(ValueError, match="clean joint flow"):
        MultiLabelHead("flow", noise_aware=True, flow_objective="contrastive")


def test_small_fit_smoke_has_no_false_negative_duplicates():
    import numpy as np

    from masked_stellar_autoencoder.training.multilabel_experiments import (
        supervised_fit_rows,
    )

    fit = np.arange(64) + 1000
    rows = supervised_fit_rows(np.random.default_rng(42), fit, 256)
    np.testing.assert_array_equal(np.sort(rows), fit)
    assert np.all(rows != np.roll(rows, 1))
