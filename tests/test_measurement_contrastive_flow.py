"""Explicit finite-noise InfoNCE composition and both-context gradients."""

import math

import pytest
import torch

from masked_stellar_autoencoder.training.multilabel_experiments import MultiLabelHead


@pytest.mark.parametrize("objective", ["contrastive", "nll_contrastive"])
def test_measurement_contexts_match_independent_node_loop_and_gradients(objective):
    torch.manual_seed(41)
    head = MultiLabelHead(
        "flow",
        latent_dim=4,
        width=8,
        flow_type="maf",
        noise_aware=True,
        mc_samples=8,
        flow_objective=objective,
        measurement_contrastive=True,
    ).double()
    torch.nn.init.normal_(head.output.weight, std=0.15)
    latent = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(6, 3, dtype=torch.float64)
    sigma = torch.full_like(y, 0.25)
    actual = head.criterion(latent, y, sigma, source_ids=torch.arange(6))
    positive_dist = head.distribution(latent)
    negative_dist = head.distribution(latent.roll(1, 0))
    positive = torch.logsumexp(
        torch.stack(
            [positive_dist.log_prob(y + sigma * z) for z in head.measurement_nodes]
        ),
        0,
    ) - math.log(8)
    negative = torch.logsumexp(
        torch.stack(
            [negative_dist.log_prob(y + sigma * z) for z in head.measurement_nodes]
        ),
        0,
    ) - math.log(8)
    rank = torch.logaddexp(positive, negative) - positive
    expected = (
        rank.mean()
        if objective == "contrastive"
        else -positive.mean() + 0.1 * rank.mean()
    ) / 3
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, latent, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, latent)[0]
    torch.testing.assert_close(ga, ge)
    assert torch.isfinite(ga).all() and bool(ga.abs().sum() > 0)
    with pytest.raises(ValueError, match="duplicate"):
        head.criterion(latent, y, sigma, source_ids=torch.zeros(6, dtype=torch.int64))


def test_zero_error_limit_matches_clean_library_contrastive_objective():
    torch.manual_seed(19)
    clean = MultiLabelHead(
        "flow", latent_dim=4, width=8, flow_type="maf", flow_objective="nll_contrastive"
    ).double()
    noisy = MultiLabelHead(
        "flow",
        latent_dim=4,
        width=8,
        flow_type="maf",
        noise_aware=True,
        mc_samples=8,
        flow_objective="nll_contrastive",
        measurement_contrastive=True,
    ).double()
    for attr in ["input", "block", "output"]:
        getattr(noisy, attr).load_state_dict(getattr(clean, attr).state_dict())
    noisy.loss_fn.base_loss.load_state_dict(clean.loss_fn.state_dict())
    x = torch.randn(5, 4, dtype=torch.float64)
    y = torch.randn(5, 3, dtype=torch.float64)
    ids = torch.arange(5)
    s = torch.ones_like(y) * 1e-14
    torch.testing.assert_close(
        noisy.criterion(x, y, s, source_ids=ids),
        clean.criterion(x, y, s, source_ids=ids),
        rtol=1e-10,
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="Explicit measurement"):
        MultiLabelHead("gaussian", measurement_contrastive=True)
