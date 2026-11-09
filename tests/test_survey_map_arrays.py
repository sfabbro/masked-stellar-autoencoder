import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap
from masked_stellar_autoencoder.training.survey_map_arrays import map_axis


@pytest.mark.parametrize("kind", ["offset", "affine", "spline"])
def test_exported_map_matches_exact_torch_in_native_units_including_log_teachers(kind):
    model = SixSurveyMap(torch.ones((2, 6), dtype=torch.bool), [0] * 6, kind=kind)
    with torch.no_grad():
        model.bias.normal_()
        model.raw_slope.normal_()
    coordinate = torch.linspace(-8, 8, 31)[:, None, None].expand(-1, 2, 6)
    expected = model(coordinate, torch.arange(2)).detach().numpy()
    b, a = model.coefficients()
    center = np.array([5000, 3, -0.5, 0, 1, 0])
    scale = np.array([1000, 1, 1, 1, 1, 1])
    coefficients = dict(
        center=center,
        scale=scale,
        bias=b.detach().numpy(),
        slopes=a.detach().numpy(),
        kind=kind,
    )
    for j in range(6):
        native = coordinate[:, 1, j].numpy() * scale[j] + center[j]
        native = np.exp(native) if j >= 4 else native
        result = map_axis(native, j, 1, coefficients)
        truth = expected[:, 1, j] * scale[j] + center[j]
        truth = np.exp(truth) if j >= 4 else truth
        np.testing.assert_allclose(
            result, truth, rtol=2e-6, atol=2e-3 if j == 0 else 2e-6
        )
