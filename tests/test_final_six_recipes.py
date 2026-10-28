"""Synthetic likelihood, target-unit, source-mask and regularizer invariants."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_final_six_recipes import cp_threshold, draw_balanced_rows, selection_score

from masked_stellar_autoencoder.training.final_six_recipes import (
    AdaptedTriLoss,
    SparseSixHead,
    normalization,
    prediction_order_loss,
    standardized_arrays,
)


def test_sparse_mdn_keeps_zero_error_coordinates_and_exact_marginal():
    torch.manual_seed(58)
    head = SparseSixHead("mdn", latent_dim=4).double()
    z = torch.randn(3, 4, dtype=torch.double)
    y = torch.randn(3, 6, dtype=torch.double)
    sigma = torch.zeros_like(y)
    sigma[:, 0] = 0.2
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, [0, 4, 5]] = True
    weights, means, ls = head.head._mixture_parameters(head.head(z))
    var = (2 * ls).exp() + sigma[:, None].square()
    comp = (
        torch.distributions.Normal(means, var.sqrt())
        .log_prob(y[:, None])
        .masked_fill(~mask[:, None], 0)
        .sum(-1)
    )
    expected = torch.logsumexp(weights + comp, -1)
    torch.testing.assert_close(head.row_log_prob(z, y, sigma, mask), expected)
    changed = y.clone()
    changed[:, 4] += 5
    assert not torch.allclose(
        head.row_log_prob(z, y, sigma, mask), head.row_log_prob(z, changed, sigma, mask)
    )
    changed[:, 1:4] = 900  # All these dimensions are unobserved.
    torch.testing.assert_close(
        head.row_log_prob(z, changed, sigma, mask),
        head.row_log_prob(z, changed.masked_fill(~mask, 0), sigma, mask),
    )


def test_gaussian_masked_observed_density_and_log_native_moments():
    torch.manual_seed(59)
    head = SparseSixHead("gaussian", latent_dim=4).double()
    z = torch.randn(2, 4, dtype=torch.double)
    y = torch.randn(2, 6, dtype=torch.double)
    sigma = torch.ones_like(y) * 0.3
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, [0, 5]] = True
    mean, lv = head.head(z).chunk(2, -1)
    expected = (
        torch.distributions.Normal(mean, (lv.exp() + sigma.square()).sqrt())
        .log_prob(y)
        .masked_fill(~mask, 0)
        .sum(-1)
    )
    torch.testing.assert_close(head.row_log_prob(z, y, sigma, mask), expected)
    native_mean, native_variance = head.native_moments(
        z, torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)
    )
    lognormal = torch.distributions.LogNormal(mean[:, 4:], lv[:, 4:].exp().sqrt())
    torch.testing.assert_close(native_mean[:, 4:], lognormal.mean)
    torch.testing.assert_close(native_variance[:, 4:], lognormal.variance)


def test_sparse_scaling_native_jacobian_and_posterior_error_semantics():
    value = (
        np.tile([5000.0, 2.5, -0.5, 0.1, 5.0, 2.0], (20, 1))
        + np.arange(20)[:, None] * 0.01
    )
    observed = np.ones_like(value, dtype=bool)
    observed[0, 3] = False
    value[0, 3] = np.nan
    center, scale = normalization(value, observed, np.ones(20, dtype=bool))
    errors = np.full_like(value, np.nan)
    known = np.zeros_like(observed)
    known[:, 0] = True
    errors[:, 0] = 50
    y, sigma, mask, jac = standardized_arrays(
        value, observed, errors, known, center, scale
    )
    np.testing.assert_allclose(jac[:, 4], np.log(scale[4]) + np.log(value[:, 4]))
    assert np.isfinite(y).all() and y[0, 3] == 0 and not mask[0, 3]
    assert (sigma[:, 4:] == 0).all()
    known[:, 4] = True
    errors[:, 4] = 1
    with pytest.raises(ValueError, match="posterior"):
        standardized_arrays(value, observed, errors, known, center, scale)


def test_fit_balancing_never_draws_heldout_and_cp_order_statistic():
    rng = np.random.default_rng(12)
    fit = np.arange(20)
    mask = np.ones((30, 6), bool)
    rows = draw_balanced_rows(rng, fit, mask, 8)
    assert len(np.unique(rows)) == len(rows) and np.isin(rows, fit).all()
    assert cp_threshold(np.arange(9), 0.1) == 8
    with pytest.raises(ValueError, match="Insufficient"):
        cp_threshold(np.arange(4), 0.05)


def test_ranking_respects_uncertain_pairs_and_source_scales():
    mean = torch.randn(4, 6, requires_grad=True)
    target = torch.zeros_like(mean)
    mask = torch.ones_like(mean, dtype=torch.bool)
    sigma = torch.ones_like(mean)
    source = torch.zeros_like(mean, dtype=torch.long)
    loss = prediction_order_loss(mean, target, mask, sigma, source)
    assert loss == 0
    loss.backward()
    assert torch.equal(mean.grad, torch.zeros_like(mean))
    mean.grad = None
    target[:, 0] = torch.arange(4.0)
    source[:, 0] = torch.arange(
        4
    )  # No within-source pairs: don't enforce cross-source orders.
    assert prediction_order_loss(mean, target, mask, sigma * 0, source) == 0


def test_binned_cdf_is_paper_midpoint_surrogate_and_two_view_gradients():
    edge = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    logits = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    value = torch.tensor([0.5])
    ce, cdf = AdaptedTriLoss.binned_terms(logits, value, edge)
    indicator = torch.tensor([[0.0, 0.0, 1.0]])
    expected = (
        (logits.softmax(-1).cumsum(-1) - indicator).square() * edge.diff()
    ).sum()
    torch.testing.assert_close(cdf, expected)
    aux = AdaptedTriLoss(4, [edge] * 6)
    first = torch.randn(4, 4, requires_grad=True)
    second = torch.randn(4, 4, requires_grad=True)
    labels = torch.randn(4, 6)
    mask = torch.ones_like(labels, dtype=torch.bool)
    source = torch.zeros_like(labels, dtype=torch.long)
    loss = aux(first, second, labels, mask, source)
    loss.backward()
    assert (
        torch.isfinite(loss)
        and first.grad.abs().sum() > 0
        and second.grad.abs().sum() > 0
    )


def test_selection_uses_all_targets_and_rejects_missing_tune_labels():
    result = dict(
        observed=np.ones((20, 6), bool), observed_log_prob_native=np.full((20, 6), -2.0)
    )
    assert selection_score(result) == (2.0, [2.0] * 6)
    result["observed"][:, 5] = False
    with pytest.raises(ValueError, match="lacks"):
        selection_score(result)


def test_small_calibration_makes_simultaneous_set_explicitly_unbounded():
    scores = np.arange(22, dtype=float)
    assert np.isfinite(cp_threshold(scores, 0.05))
    assert cp_threshold(scores, 0.05 / 6, allow_unbounded=True) == float("inf")


def test_factorized_nsf_uses_only_observed_marginals_and_both_context_gradients():
    torch.manual_seed(61)
    head = SparseSixHead("factorized_nsf", latent_dim=4, mixed_weight=0.1).double()
    with torch.no_grad():
        for scalar in head.heads:
            scalar.output.weight.normal_(std=0.1)
            scalar.output.bias.normal_(std=0.1)
    latent = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    y = torch.randn(3, 6, dtype=torch.double)
    sigma = torch.zeros_like(y)
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, [0, 4]] = True
    expected = sum(head.heads[j].log_prob(latent, y[:, j : j + 1]) for j in [0, 4])
    torch.testing.assert_close(
        head.row_log_prob(latent, y, sigma, mask), expected, rtol=1e-10, atol=1e-10
    )
    changed = y.clone()
    changed[:, 1:4] += 100
    torch.testing.assert_close(
        head.row_log_prob(latent, changed, sigma, mask),
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    loss = head.criterion(latent, y, sigma, mask, torch.arange(3))
    loss.backward()
    assert (
        torch.isfinite(loss)
        and torch.isfinite(latent.grad).all()
        and latent.grad.abs().sum() > 0
    )


def test_auxiliary_views_do_not_advance_reconstruction_generator():
    from masked_stellar_autoencoder.training.final_six_recipes import (
        independent_masked_views,
    )

    x = torch.arange(80).reshape(10, 8).float()
    visible = torch.ones_like(x, dtype=torch.bool)
    reconstruction = torch.Generator().manual_seed(82)
    before = reconstruction.get_state().clone()
    augmentation = torch.Generator().manual_seed(1783)
    first, second = independent_masked_views(
        x, visible, [2, 3, 4, 5, 6, 7], augmentation, fraction=0.5
    )
    assert torch.equal(reconstruction.get_state(), before)
    assert not torch.equal(first, second)
    torch.testing.assert_close(first[:, :2], x[:, :2])


def test_native_mixture_variance_uses_stable_centered_formula():
    head = SparseSixHead("mdn", latent_dim=4).double()
    with torch.no_grad():
        head.head.output.weight.zero_()
        head.head.output.bias.zero_()
        head.head.output.bias[4 + 4 * 6 :] = -20
    center = torch.tensor([5000.0, 3.0, -0.5, 0.1, 1.5, 0.7], dtype=torch.double)
    mean, variance = head.native_moments(
        torch.zeros(2, 4, dtype=torch.double), center, torch.ones(6, dtype=torch.double)
    )
    assert torch.isfinite(mean).all() and (variance > 0).all()
    torch.testing.assert_close(
        variance[:, :4],
        torch.full((2, 4), float(np.exp(-40)), dtype=torch.double),
        atol=1e-28,
        rtol=1e-9,
    )
