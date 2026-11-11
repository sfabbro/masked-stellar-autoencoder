import math

import pytest
import torch

from masked_stellar_autoencoder.training.quantity_kinds import QuantityKinds


def _draws(fe_h, alpha_fe, center, scale):
    x = torch.zeros(1, len(fe_h), 6)
    x[0, :, 2] = (torch.tensor(fe_h) - center[2]) / scale[2]
    x[0, :, 3] = (torch.tensor(alpha_fe) - center[3]) / scale[3]
    return x


def test_salaris_conversion_in_dex_per_source():
    center = torch.tensor([5000.0, 3.0, -0.5, 0.1, 0.0, 0.0])
    scale = torch.tensor([800.0, 1.0, 0.5, 0.15, 1.0, 1.0])
    q = QuantityKinds(
        ["fe_h", "m_h", "fe_h"], ["alpha_fe", "alpha_m", "mg_fe"], center, scale
    )
    x = _draws([-2.0, -2.0, -2.0], [0.4, 0.4, 0.4], center, scale)
    y = q(x, torch.tensor([0, 1, 2]))
    f = math.log10(0.638 * 10**0.4 + 0.362)
    to_dex = lambda v, j: float(v) * float(scale[j]) + float(center[j])  # noqa: E731
    assert to_dex(y[0, 0, 2], 2) == pytest.approx(-2.0)  # fe_h unchanged
    assert to_dex(y[0, 1, 2], 2) == pytest.approx(-2.0 + f, abs=1e-5)  # [M/H]
    assert to_dex(y[0, 1, 3], 3) == pytest.approx(0.4 - f, abs=1e-5)  # [alpha/M]
    assert to_dex(y[0, 2, 3], 3) == pytest.approx(0.4)  # mg_fe: own map only
    torch.testing.assert_close(y[..., [0, 1, 4, 5]], x[..., [0, 1, 4, 5]])


def test_solar_alpha_gives_mh_equal_feh_and_gradients_flow():
    center = torch.zeros(6)
    scale = torch.ones(6)
    q = QuantityKinds(["m_h"], ["alpha_m"], center, scale)
    x = _draws([-1.0], [0.0], center, scale).requires_grad_()
    y = q(x, torch.tensor([0]))
    assert float(y[0, 0, 2]) == pytest.approx(-1.0, abs=1e-6)
    assert float(y[0, 0, 3]) == pytest.approx(0.0, abs=1e-6)
    y[..., 2].sum().backward()
    assert x.grad[0, 0, 3] > 0  # [M/H] rises with [alpha/Fe]


def test_all_default_kinds_are_a_no_op_and_unknown_kinds_fail():
    q = QuantityKinds(
        ["fe_h"] * 2, ["alpha_fe", "mg_fe"], torch.zeros(6), torch.ones(6)
    )
    x = torch.randn(3, 2, 6)
    assert q(x, torch.tensor([0, 1])) is x
    with pytest.raises(ValueError):
        QuantityKinds(["z"], ["alpha_fe"], torch.zeros(6), torch.ones(6))
