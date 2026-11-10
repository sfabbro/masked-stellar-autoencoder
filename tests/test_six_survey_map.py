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


def test_mixed_source_families_are_exact_special_cases_of_the_spline():
    support = torch.ones((5, 6), dtype=torch.bool)
    kinds = ["affine", "identity", "offset", "affine", "spline"]
    model = SixSurveyMap(support, [0] * 6, kind="affine", source_kinds=kinds)
    assert model.bins == 8 and model.export_kind == "spline"
    with torch.no_grad():
        model.bias.normal_()
        model.raw_slope.normal_()
    x = torch.linspace(-7, 7, 29)[:, None, None].expand(-1, 5, 6)
    y = model(x, torch.arange(5))
    torch.testing.assert_close(y[:, 1], x[:, 1])  # identity row
    offset = y[:, 2] - x[:, 2]
    torch.testing.assert_close(offset, offset[:1].expand_as(offset))  # pure shift
    slope = (y[1:, 3] - y[:-1, 3]) / (x[1:, 3] - x[:-1, 3])
    torch.testing.assert_close(slope, slope[:1].expand_as(slope))  # one slope
    _, slopes = model.coefficients()
    assert not torch.allclose(slopes[4], slopes[4, :, :1].expand_as(slopes[4]))


def test_mixed_families_load_legacy_checkpoints_unchanged():
    support = torch.ones((3, 6), dtype=torch.bool)
    legacy = SixSurveyMap(support, [0] * 6, kind="affine")
    with torch.no_grad():
        legacy.bias.normal_()
        legacy.raw_slope.normal_()
    state = legacy.state_dict()
    assert "slope_mode" not in state
    same = SixSurveyMap(support, [0] * 6, kind="affine")
    same.load_state_dict(state, strict=True)
    x = torch.randn(4, 3, 6)
    torch.testing.assert_close(same(x, torch.arange(3)), legacy(x, torch.arange(3)))


def test_identity_rows_cannot_join_a_zero_sum_target():
    with pytest.raises(ValueError, match="zero-sum"):
        SixSurveyMap(
            torch.ones((2, 6), dtype=torch.bool),
            [-1] * 6,
            zero_sum=[0],
            source_kinds=["identity", "affine"],
        )
