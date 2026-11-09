import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.consensus_survey_map import ConsensusSurveyMap
from masked_stellar_autoencoder.training.full_gaia_inputs import (
    AUXILIARY,
    PHOTOMETRY,
    XP,
    auxiliary_context,
    full_gaia_policy,
    input_eligible,
)


def test_no_xp_and_additional_bands_are_admitted_without_dropping_photometry():
    names = PHOTOMETRY + XP + AUXILIARY
    policy = full_gaia_policy(names)
    a = np.full((4, len(names)), -9999.0)
    a[0, names.index("G")] = 17
    a[1, names.index("G_SDSS")] = 18
    a[2, names.index("bp_1")] = 1
    np.testing.assert_array_equal(input_eligible(a, policy), [True, True, True, False])
    assert sum(policy.encoder) == 134
    assert all(policy.encoder[names.index(n)] for n in PHOTOMETRY)
    assert not any(policy.encoder[names.index(n)] for n in AUXILIARY)


def test_auxiliary_context_preserves_negative_parallax_missing_errors_and_sky_wrap():
    names = PHOTOMETRY + XP + AUXILIARY
    a = np.full((2, len(names)), np.nan)
    e = a.copy()
    for r, ra in enumerate([0, 360]):
        a[r, names.index("RA")] = ra
        a[r, names.index("DEC")] = 0
        a[r, names.index("PARALLAX")] = -0.2
        a[r, names.index("EBV")] = 0.3
    e[0, names.index("PARALLAX")] = 0.1
    x, m, labels = auxiliary_context(a, e, names)
    assert x[0, labels.index("PARALLAX")] == pytest.approx(-0.2)
    assert m[0, labels.index("sigma_PARALLAX")]
    assert not m[1, labels.index("sigma_PARALLAX")]
    np.testing.assert_allclose(x[0, -3:], x[1, -3:], atol=1e-7)
    assert "EBV" in labels and "pmra" in labels


@pytest.mark.parametrize("kind", ["affine", "spline"])
def test_weighted_common_gauge_has_no_preferred_survey_and_mean_identity(kind):
    support = torch.ones((3, 6), dtype=torch.bool)
    weights = torch.tensor([[1.0] * 6, [2.0] * 6, [4.0] * 6])
    model = ConsensusSurveyMap(support, weights, kind=kind)
    with torch.no_grad():
        model.bias.normal_()
        model.raw_slope.normal_()
    theta = torch.linspace(-7, 7, 41)[:, None, None].expand(-1, 3, 6)
    mapped = model(theta, torch.arange(3))
    torch.testing.assert_close(
        (mapped * model.consensus_weights[None]).sum(1),
        theta[:, 0],
        atol=3e-6,
        rtol=3e-6,
    )
    perm = torch.tensor([2, 0, 1])
    other = ConsensusSurveyMap(support[perm], weights[perm], kind=kind)
    with torch.no_grad():
        other.bias.copy_(model.bias[perm])
        other.raw_slope.copy_(model.raw_slope[perm])
    torch.testing.assert_close(other(theta, torch.arange(3)), mapped[:, perm])
    assert model.free.all()


def test_consensus_does_not_invent_unsupported_survey_targets():
    support = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1]], dtype=torch.bool
    )
    model = ConsensusSurveyMap(support, support.float())
    with torch.no_grad():
        model.bias.normal_()
        model.raw_slope.normal_()
    b, a = model.coefficients()
    assert (b[~support] == 0).all()
    assert (a[~support] == 1).all()
    with pytest.raises(ValueError):
        ConsensusSurveyMap(support, torch.ones_like(support, dtype=torch.float32))
