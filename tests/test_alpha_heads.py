"""Alpha interface contract: disabled by default, gauged, no missing-label training.

There are no alpha labels in the current cohort, so these tests check the
*contract* only: the head cannot be used by accident, the affine link is
identified by the declared gauge, and unlabelled rows are dropped rather than
imputed.
"""

import math

import pytest
import torch

from masked_stellar_autoencoder.training import alpha_heads as ah

DOUBLE = torch.float64


def test_head_is_disabled_by_default():
    head = ah.SourceTaggedAlphaHead(input_dim=8, width=16)
    assert not head.enabled
    with pytest.raises(RuntimeError):
        head(torch.zeros(2, 8), torch.zeros(2, dtype=torch.long))
    enabled = ah.SourceTaggedAlphaHead(input_dim=8, width=16, enabled=True)
    assert enabled.enabled


def test_source_tagged_output_gathers_the_row_source():
    head = ah.SourceTaggedAlphaHead(input_dim=8, width=16, enabled=True).double()
    with torch.no_grad():
        head.net[-1].bias[:3] = torch.tensor([0.0, 1.0, 2.0], dtype=DOUBLE)
        head.net[-1].bias[3:] = torch.tensor([0.0, 1.0, 2.0], dtype=DOUBLE)
        head.offsets.zero_()
        head.slopes.fill_(1.0)
    features = torch.zeros(3, 8, dtype=DOUBLE)
    outputs = head(features, torch.tensor([2, 0, 1]))
    # The latent alpha is shared by every source (one physical quantity); the
    # per-source layer only supplies the scale. With a zero-init last weight the
    # latent mean is exactly bias[0] = 0, so every row has the same mean ...
    assert outputs["mean"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    # ... while log sigma is gathered from the row's own source tag.
    assert outputs["sigma"].tolist() == pytest.approx(
        [math.exp(2.0), math.exp(0.0), math.exp(1.0)], rel=1e-14
    )
    with pytest.raises(ValueError):
        head(features, torch.tensor([0, 5, 1]))
    with pytest.raises(ValueError):
        head(features, torch.tensor([0, -1, 1]))
    with pytest.raises(ValueError):
        head(torch.zeros(3, 7, dtype=DOUBLE), torch.tensor([0, 0, 0]))


def test_anchor_gauge_pins_the_reference_source():
    gauge = ah.AlphaGauge(mode="anchor", reference_source="GALAH")
    offsets = torch.tensor([0.3, -0.2, 0.7], dtype=DOUBLE)
    slopes = torch.tensor([0.9, 0.4, 1.3], dtype=DOUBLE)
    fixed_offsets, fixed_slopes = ah.apply_gauge(
        offsets, slopes, gauge, source_names=("APOGEE", "GALAH", "DESI")
    )
    assert fixed_offsets[1].item() == 0.0
    assert fixed_slopes[1].item() == 1.0
    assert fixed_offsets[0].item() == pytest.approx(0.5)
    assert fixed_slopes[0].item() == pytest.approx(1.5)
    with pytest.raises(ValueError):
        ah.apply_gauge(
            offsets, slopes, ah.AlphaGauge(mode="anchor"), source_names=("A", "B", "C")
        )
    with pytest.raises(ValueError):
        ah.apply_gauge(
            offsets,
            slopes,
            ah.AlphaGauge(mode="anchor", reference_source="APOGEE"),
            source_names=("A", "B"),
        )


def test_sum_zero_gauge_identifies_shift_and_scale():
    gauge = ah.AlphaGauge(mode="sum_zero", counts=(100.0, 50.0, 25.0))
    offsets = torch.tensor([0.3, -0.2, 0.7], dtype=DOUBLE)
    slopes = torch.tensor([0.9, 0.4, 1.3], dtype=DOUBLE)
    fixed_offsets, fixed_slopes = ah.apply_gauge(
        offsets, slopes, gauge, source_names=("APOGEE", "GALAH", "DESI")
    )
    weights = torch.tensor(gauge.counts, dtype=DOUBLE)
    assert float((fixed_offsets * weights).sum()) == pytest.approx(0.0, abs=1e-14)
    assert float((fixed_slopes * weights).sum() / weights.sum()) == pytest.approx(
        1.0, abs=1e-14
    )
    with pytest.raises(ValueError):
        ah.apply_gauge(
            offsets,
            slopes,
            ah.AlphaGauge(mode="sum_zero"),
            source_names=("A", "B", "C"),
        )
    with pytest.raises(ValueError):
        ah.AlphaGauge(mode="sum_zero", counts=(0.0, 0.0, 0.0))


def test_effective_sigma_uses_the_fitted_kappa_without_a_floor():
    sigma = ah.effective_alpha_sigma(
        torch.tensor([0.1], dtype=DOUBLE),
        torch.tensor([0.05], dtype=DOUBLE),
        torch.tensor([7.3], dtype=DOUBLE),
    )
    assert sigma.item() == pytest.approx((0.01 + (7.3 * 0.05) ** 2) ** 0.5, rel=1e-14)
    with pytest.raises(ValueError):
        ah.effective_alpha_sigma(
            torch.tensor([0.0], dtype=DOUBLE),
            torch.tensor([0.05], dtype=DOUBLE),
            torch.tensor([1.0], dtype=DOUBLE),
        )
    with pytest.raises(ValueError):
        ah.effective_alpha_sigma(
            torch.tensor([0.1], dtype=DOUBLE),
            torch.tensor([0.05], dtype=DOUBLE),
            torch.tensor([0.0], dtype=DOUBLE),
        )


def test_missing_labels_are_dropped_and_untagged_rows_raise():
    mean = torch.zeros(3, dtype=DOUBLE, requires_grad=True)
    log_sigma = torch.zeros(3, dtype=DOUBLE, requires_grad=True)
    labels = torch.tensor([0.1, float("nan"), 0.2], dtype=DOUBLE)
    sources = torch.tensor([0, 1, 2], dtype=torch.long)
    total, count = ah.source_tagged_alpha_nll(mean, log_sigma, labels, sources)
    assert count == 2
    assert torch.isfinite(total)
    with pytest.raises(ValueError):
        ah.source_tagged_alpha_nll(
            mean[:1], log_sigma[:1], labels[:1], torch.tensor([-1], dtype=torch.long)
        )
    empty, count = ah.source_tagged_alpha_nll(
        mean, log_sigma, torch.full((3,), float("nan"), dtype=DOUBLE), sources
    )
    assert count == 0
    assert float(empty.detach()) == 0.0
    (grad,) = torch.autograd.grad(empty, mean)
    assert torch.isfinite(grad).all()


def test_claim_requires_all_five_gates():
    assert ah.alpha_claim_allowed({}) is False
    assert ah.alpha_claim_allowed({ah.ALPHA_GATES[0]: True}) is False
    assert ah.alpha_claim_allowed({name: True for name in ah.ALPHA_GATES}) is True
    with pytest.raises(ValueError):
        ah.alpha_claim_allowed({"native_alpha": True})
