"""Independent invariants for SLS region semantics and adapter checkpointing."""

import math

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.sls_experiments import (
    SLSHead,
    finite_sample_threshold,
    region_log_volume,
)


def test_sls_conformal_finite_rank_matches_library_and_test_labels_irrelevant():
    from torchregress.losses import SLSConformal

    torch.manual_seed(17)
    head = SLSHead(latent_dim=4).double()
    x, y = torch.randn(64, 4).double(), torch.randn(64, 3).double()
    g, q, _ = head.region_values(x, y)
    for alpha in (0.1, 0.05):
        expected = finite_sample_threshold((g / q).detach().numpy(), alpha)
        cp = SLSConformal(head.sls, alpha=alpha)
        cp.calibrate(x, y)
        assert float(cp.q_hat) == pytest.approx(expected, abs=1e-12)
    assert np.isinf(finite_sample_threshold(np.arange(3), 0.05))


def test_precision_volume_sign_and_native_jacobian():
    # G(y)=4||y||² <=9 is a radius1.5 ball; native scale product=30.
    logv = region_log_volume(
        np.array([9.0]), np.array([3 * math.log(2)]), 1.0, np.array([2.0, 3.0, 5.0])
    )
    assert np.exp(logv)[0] == pytest.approx(4 / 3 * math.pi * 1.5**3 * 30)


@pytest.mark.parametrize("mode", ["full", "low_rank"])
def test_sls_center_zero_frontier_schedule_reload_and_gradients(mode):
    torch.manual_seed(19)
    head = SLSHead(latent_dim=4, mode=mode).double()
    x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    center, variance = head.moments(x)
    g, _, _ = head.region_values(x, center)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-25)
    assert torch.isnan(variance).all() and not head.has_density
    loss = head.criterion(x, torch.randn(8, 3, dtype=torch.float64))
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    clone = SLSHead(latent_dim=4, mode=mode).double()
    clone.load_state_dict(head.state_dict(), strict=True)
    assert clone.criterion_calls.item() == 1
    clone.eval()
    clone.criterion(x.detach(), torch.zeros(8, 3, dtype=torch.float64))
    assert clone.criterion_calls.item() == 1


def test_sls_rejects_unimplemented_measurement_objectives():
    with pytest.raises(ValueError, match="measurement-error"):
        SLSHead(noise_aware=True)
    with pytest.raises(ValueError, match="measurement-error"):
        SLSHead(measurement_contrastive=True)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_conformal_alpha_is_a_probability(alpha):
    with pytest.raises(ValueError, match="alpha"):
        finite_sample_threshold(np.arange(10), alpha)
