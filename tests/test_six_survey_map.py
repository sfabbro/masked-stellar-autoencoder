import pytest
import torch

from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap


def test_all_six_reference_coordinates_are_identity_with_positive_derivative():
    support = torch.ones((3, 6), dtype=torch.bool)
    model = SixSurveyMap(support, [0] * 6, kind="spline")
    with torch.no_grad():
        model.bias.normal_()
        model.raw_slope.normal_()
    x = (
        torch.linspace(-8, 8, 33)[:, None, None]
        .expand(-1, 3, 6)
        .clone()
        .requires_grad_()
    )
    result = model(x, torch.arange(3))
    torch.testing.assert_close(result[:, 0], x[:, 0])
    gradient = torch.autograd.grad(result.sum(), x)[0]
    assert (gradient > 0).all()


def test_arithmetic_gauge_is_identity_everywhere_including_spline_tails():
    model = SixSurveyMap(
        torch.ones((3, 6), dtype=torch.bool), [-1] * 6, zero_sum=range(6), kind="spline"
    )
    with torch.no_grad():
        model.bias.normal_()
        model.raw_slope.normal_()
    x = torch.linspace(-8, 8, 31)[:, None, None].expand(-1, 3, 6)
    torch.testing.assert_close(
        model(x, torch.arange(3)).mean(1), x[:, 0], atol=2e-6, rtol=2e-6
    )
    model(x, torch.arange(3)).square().mean().backward()
    assert torch.isfinite(model.raw_slope.grad).all()


def test_unsupported_target_is_rejected_instead_of_projected():
    support = torch.ones((2, 6), dtype=torch.bool)
    support[1, 4:] = False
    model = SixSurveyMap(support, [0] * 6)
    with pytest.raises(ValueError, match="unsupported"):
        model(torch.zeros(4, 2, 6), torch.arange(2), torch.ones(2, 6, dtype=torch.bool))
