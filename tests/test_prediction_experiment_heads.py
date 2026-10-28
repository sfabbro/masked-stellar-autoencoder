"""Independent objective identities and head-state checks for bounded pilots."""

import copy
import math

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_contrastive_flow,
    build_prediction_head,
    contrastive_flow_spec,
    contrastive_grid,
    fit_lds_huber,
    point_prediction,
    prediction_loss,
    rank_contrast_loss,
)


@pytest.mark.parametrize(
    "kind,members", [("mlp", 1), ("residual", 1), ("batchensemble", 4)]
)
def test_common_initial_predictions_member_shapes_and_full_state_reload(kind, members):
    torch.manual_seed(42)
    head = build_prediction_head(kind, input_dim=7, width=16).double()
    inputs = torch.arange(35, dtype=torch.float64).reshape(5, 7) / 35
    output = head(inputs)
    assert output.shape == (5, members, 2)
    assert torch.equal(output, torch.zeros_like(output))
    loss = prediction_loss(output, torch.arange(5, dtype=torch.float64), "huber")
    loss.backward()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    optimizer.step()
    assert torch.isfinite(head(inputs)).all()
    restored = build_prediction_head(kind, input_dim=7, width=16).double()
    restored.load_state_dict(copy.deepcopy(head.state_dict()))
    assert torch.equal(head(inputs), restored(inputs))
    assert point_prediction(output).shape == (5,)


def test_quoted_label_variance_matches_independent_convolution_and_rejects_bad_errors():
    output = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
    truth = torch.tensor([0.0], dtype=torch.float64)
    quoted = torch.tensor([3.0], dtype=torch.float64)
    plain = prediction_loss(output, truth, "gaussian_nll")
    convolved = prediction_loss(
        output, truth, "gaussian_nll_label_error", label_variance=quoted
    )
    # y equals the mean, so only the log-determinant term changes.
    reference = 0.5 * math.log(2 * math.pi * 4.0)
    np.testing.assert_allclose(convolved.item(), reference, rtol=1e-14)
    assert convolved > plain
    with pytest.raises(ValueError, match="strictly positive"):
        prediction_loss(
            output,
            truth,
            "gaussian_nll_label_error",
            label_variance=torch.tensor([0.0]),
        )


def test_lds_huber_upweights_the_isolated_label():
    spec = {
        "kernel": "gaussian",
        "kernel_width_bins": 2,
        "n_bins": 100,
        "reweight_factor": 1.0,
        "base_loss": "huber",
    }
    isolated = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0], dtype=torch.float64)
    fitted = fit_lds_huber(isolated, spec)
    weights = fitted._weights_per_bin
    assert float(weights[-1]) > float(weights[0])
    output = torch.zeros(5, 1, 2, dtype=torch.float64, requires_grad=True)
    weighted = prediction_loss(output, isolated, "lds_huber", lds=fitted)
    weighted.backward()
    assert torch.isfinite(weighted)
    assert torch.isfinite(output.grad).all()
    assert output.grad[4, 0, 0] != 0
    with pytest.raises(ValueError, match="LDS spec"):
        fit_lds_huber(isolated, {**spec, "n_bins": 10})


def test_gaussian_crps_against_independent_normal_integral_and_gradcheck():
    mean = np.array([-1.2, 0.3, 2.1])
    sd = np.array([0.4, 1.1, 0.8])
    truth = np.array([-0.6, -0.9, 2.1])
    # Independent CDF integration of CRPS: integral (F(z)-1[z>=y])**2 dz.
    reference = []
    for mu, sigma, y in zip(mean, sd, truth, strict=True):
        # Integrate either side separately to keep the target discontinuity exact.
        grid_left = np.linspace(mu - 12 * sigma, y, 20001)
        grid_right = np.linspace(y, mu + 12 * sigma, 20001)
        cdf_left = np.array(
            [0.5 * (1 + math.erf((x - mu) / sigma / math.sqrt(2))) for x in grid_left]
        )
        cdf_right = np.array(
            [0.5 * (1 + math.erf((x - mu) / sigma / math.sqrt(2))) for x in grid_right]
        )
        reference.append(
            np.trapezoid(cdf_left**2, grid_left)
            + np.trapezoid((cdf_right - 1) ** 2, grid_right)
        )
    output = torch.tensor(
        np.stack((mean, np.log(sd**2)), axis=-1)[:, None], requires_grad=True
    )
    y_tensor = torch.tensor(truth)
    actual = prediction_loss(output, y_tensor, "gaussian_crps")
    np.testing.assert_allclose(actual.item(), np.mean(reference), rtol=1e-7)
    assert torch.autograd.gradcheck(
        lambda value: prediction_loss(value, y_tensor, "gaussian_crps"), (output,)
    )


def test_gaussian_nll_matches_independent_formula_and_member_losses():
    output = torch.tensor(
        [[[0.0, 0.0], [2.0, math.log(4.0)]]], dtype=torch.float64, requires_grad=True
    )
    truth = torch.tensor([1.0], dtype=torch.float64)
    reference = np.mean(
        [0.5 * (math.log(2 * math.pi) + 1.0), 0.5 * (math.log(8 * math.pi) + 0.25)]
    )
    actual = prediction_loss(output, truth, "gaussian_nll")
    np.testing.assert_allclose(actual.item(), reference, rtol=1e-14)
    # Scoring the averaged output instead would be a different objective.
    averaged = prediction_loss(output.mean(dim=1, keepdim=True), truth, "gaussian_nll")
    assert not torch.isclose(actual, averaged)
    actual.backward()
    assert torch.isfinite(output.grad).all()


def test_huber_does_not_fit_log_variance_and_empty_loss_is_differentiable():
    output = torch.tensor([[[0.5, 123.0]]], requires_grad=True)
    loss = prediction_loss(output, torch.tensor([0.0]), "huber")
    assert loss.item() == 0.125
    loss.backward()
    assert output.grad[0, 0, 1] == 0
    empty = torch.empty(0, 1, 2, requires_grad=True)
    loss = prediction_loss(empty, torch.empty(0), "gaussian_crps")
    assert loss.item() == 0
    loss.backward()
    assert empty.grad.shape == empty.shape


@pytest.mark.parametrize("kind", ["huber", "gaussian_nll", "gaussian_crps"])
def test_bad_shapes_and_missing_labels_rejected_before_score(kind):
    output = torch.zeros(2, 1, 2)
    with pytest.raises(ValueError, match="target must be"):
        prediction_loss(output, torch.zeros(2, 1), kind)
    with pytest.raises(ValueError, match="finite eligible"):
        prediction_loss(output, torch.tensor([0.0, float("nan")]), kind)
    with pytest.raises(ValueError, match="shape"):
        prediction_loss(torch.zeros(2, 2), torch.zeros(2), kind)


@pytest.mark.parametrize("bad_logvar", [-1000.0, 1000.0])
def test_bad_variance_rejected_without_clipping(bad_logvar):
    output = torch.tensor([[[0.0, bad_logvar]]], dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly positive"):
        prediction_loss(output, torch.zeros(1), "gaussian_crps")


def test_rank_contrast_translation_and_positive_scale_invariant_features():
    features = torch.tensor(
        [[1.0, 0.2], [0.4, 0.8], [-0.2, 1.0]], dtype=torch.float64, requires_grad=True
    )
    targets = torch.tensor([-0.5, 0.0, 0.4], dtype=torch.float64)
    actual = rank_contrast_loss(features, targets)
    # Row normalization removes a common positive feature scale; label translation
    # leaves pairwise label distances identical.
    expected = rank_contrast_loss(features * 17.0, targets + 2.0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.backward()
    assert torch.isfinite(features.grad).all()
    one = torch.ones(1, 2, requires_grad=True)
    empty_pair_loss = rank_contrast_loss(one, torch.zeros(1))
    assert empty_pair_loss.item() == 0
    empty_pair_loss.backward()


_FLOW_SPEC = {
    "n_features": 8,
    "context_dim": 1,
    "flow_type": "nsf",
    "n_transforms": 3,
    "hidden_features": [64, 64],
    "temperature": 1.0,
    "margin": 0.0,
    "negative_shifts": [1, 2, 4, 8, 16, 32, 64, 128],
    "grid_points": 201,
    "grid_margin_standardized": 0.5,
}


def test_contrastive_flow_spec_rejects_a_changed_transform_count():
    changed = dict(_FLOW_SPEC)
    changed["n_transforms"] = 4
    with pytest.raises(ValueError, match="contrastive flow spec"):
        contrastive_flow_spec(changed)
    assert contrastive_flow_spec(_FLOW_SPEC) == _FLOW_SPEC


def test_contrastive_flow_rolls_are_deterministic_and_the_batch_must_exceed_them():
    pytest.importorskip("zuko")
    head = build_contrastive_flow(_FLOW_SPEC)
    too_small = torch.zeros(128, 256)
    with pytest.raises(ValueError, match="negative shift"):
        head.criterion(too_small, torch.zeros(128))
    label = torch.linspace(-1.0, 1.0, 129)
    loss = head.criterion(torch.zeros(129, 256), label)
    loss.backward()
    assert torch.isfinite(loss)
    grid = contrastive_grid(label, _FLOW_SPEC)
    estimate = head.standardized_mle(torch.zeros(4, 256), grid)
    assert estimate.shape == (4,)
    assert ((estimate[:, None] - grid).abs() < 1e-6).any(dim=1).all()


def test_one_dimensional_contrastive_flow_mle_tracks_the_context():
    pytest.importorskip("zuko")
    from torchregress.losses import create_contrastive_flow_loss

    torch.manual_seed(0)
    flow = create_contrastive_flow_loss(
        n_features=1,
        context_dim=1,
        flow_type="nsf",
        n_transforms=2,
        hidden_features=[32, 32],
        temperature=1.0,
        margin=0.0,
    )
    context = torch.linspace(-1.5, 1.5, 128)
    data = context[:, None]
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
    shifts = (1, 2, 4, 8, 16, 32)
    for _ in range(80):
        negatives = torch.stack([torch.roll(context, shift) for shift in shifts], 1)
        loss = flow(context[:, None], data, negative_context=negatives.unsqueeze(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    grid = torch.linspace(-2.0, 2.0, 161)
    chosen = []
    for start in range(0, len(data), 32):
        block = data[start : start + 32]
        query = grid.view(1, -1, 1).expand(len(block), -1, 1).reshape(-1, 1)
        target = block.unsqueeze(1).expand(-1, grid.numel(), -1).reshape(-1, 1)
        with torch.enable_grad():
            score = flow.log_prob(query.detach(), target.detach()).reshape(
                len(block), -1
            )
        chosen.append(grid[score.argmax(dim=1)])
    estimate = torch.cat(chosen)
    error = (estimate - context).abs().mean()
    assert error < estimate.abs().mean()
    assert error < 0.35
