"""Context conditioner contract: fit-only normalization, mask bits, fusion.

The context vector is 14 continuous features + 8 mask/quality bits = 22, fused
after the 256-d latent to 384. These tests pin the two properties that make the
context admissible at all: the normalization statistics come from fit-role rows
only, and a missing feature is only ever encoded as "median value + mask bit 0",
never as a silent fill.
"""

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training import context_heads as ch

N_FEATURES = len(ch.CONTINUOUS_CONTEXT)


def _synthetic(n=200, seed=0):
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(n, N_FEATURES))
    values[:, ch.CONTINUOUS_CONTEXT.index("ebv_sfd")] = generator.uniform(0.0, 3.0, n)
    observed = np.ones((n, N_FEATURES), dtype=bool)
    roles = np.array(["fit"] * (n // 2) + ["test"] * (n - n // 2))
    return values, observed, roles


def test_frozen_layout_is_14_plus_8():
    assert len(ch.CONTINUOUS_CONTEXT) == 14
    assert len(ch.MASK_BITS) == 8
    assert ch.CONTEXT_DIMENSIONS == 22
    assert len(set(ch.CONTINUOUS_CONTEXT)) == 14
    assert len(set(ch.MASK_BITS)) == 8
    assert ch.MASK_BITS[5:] == (
        "flag_ebv_unphysical",
        "flag_parallax_negative",
        "flag_zero_point_applied",
    )


def test_fit_only_statistics_ignore_non_fit_rows():
    values, observed, roles = _synthetic()
    fit = ch.fit_mask_from_roles(roles)
    assert fit.sum() == 100
    normalizer = ch.ContextNormalizer.fit(values, fit)
    perturbed = values.copy()
    perturbed[~fit] *= 1000.0
    other = ch.ContextNormalizer.fit(perturbed, fit)
    assert np.array_equal(normalizer.center, other.center)
    assert np.array_equal(normalizer.scale, other.scale)
    assert normalizer.n_fit == other.n_fit
    # ... and the center really is the fit median, the scale the fit IQR.
    column = values[fit, 0]
    assert normalizer.center[0] == pytest.approx(np.median(column), abs=1e-15)
    assert normalizer.scale[0] == pytest.approx(
        np.percentile(column, 75) - np.percentile(column, 25), abs=1e-15
    )


def test_median_maps_to_zero_and_the_iqr_to_one():
    values, observed, roles = _synthetic()
    fit = ch.fit_mask_from_roles(roles)
    normalizer = ch.ContextNormalizer.fit(values, fit)
    standardized = normalizer.standardize(values, observed)
    assert standardized.shape == values.shape
    for index in (0, 5, N_FEATURES - 1):
        column = values[fit, index]
        q25, median, q75 = np.percentile(column, [25.0, 50.0, 75.0])
        probe = np.empty((3, N_FEATURES))
        probe[:] = normalizer.center
        probe[:, index] = [q25, median, q75]
        mapped = normalizer.standardize(probe, np.ones((3, N_FEATURES), dtype=bool))
        # The median maps to 0 and the raw IQR maps to a span of exactly 1;
        # the quartiles are NOT symmetric about the median in a finite sample.
        assert mapped[1, index] == pytest.approx(0.0, abs=1e-12)
        assert mapped[2, index] - mapped[0, index] == pytest.approx(1.0, abs=1e-12)


def test_missing_values_are_zero_plus_a_clear_mask_bit():
    values, observed, roles = _synthetic()
    fit = ch.fit_mask_from_roles(roles)
    normalizer = ch.ContextNormalizer.fit(values, fit)
    missing = observed.copy()
    missing[0, ch.CONTINUOUS_CONTEXT.index("pmdec_masyr")] = False
    values = values.copy()
    values[0, ch.CONTINUOUS_CONTEXT.index("pmdec_masyr")] = np.nan
    context = normalizer.transform(values, missing, zero_point_applied=False)
    column = ch.CONTINUOUS_CONTEXT.index("pmdec_masyr")
    assert context[0, column] == 0.0
    assert context[0, N_FEATURES + ch.MASK_BITS.index("mask_pm")] == 0.0
    assert context[1, N_FEATURES + ch.MASK_BITS.index("mask_pm")] == 1.0
    assert np.isfinite(context).all()
    assert context.shape == (values.shape[0], 22)


def test_quality_flags_follow_the_audited_thresholds():
    values, observed, roles = _synthetic()
    normalizer = ch.ContextNormalizer.fit(values, ch.fit_mask_from_roles(roles))
    values = values.copy()
    plx = ch.CONTINUOUS_CONTEXT.index("parallax_mas")
    ebv = ch.CONTINUOUS_CONTEXT.index("ebv_sfd")
    values[:, plx] = np.abs(values[:, plx]) + 0.1
    values[0, plx] = -0.4
    values[1, ebv] = ch.EBV_UNPHYSICAL_THRESHOLD + 0.5
    applied = np.zeros(values.shape[0], dtype=bool)
    applied[2] = True
    context = normalizer.transform(values, observed, zero_point_applied=applied)
    bits = {
        name: context[:, N_FEATURES + index] for index, name in enumerate(ch.MASK_BITS)
    }
    assert bits["flag_parallax_negative"][0] == 1.0
    assert bits["flag_parallax_negative"][1] == 0.0
    assert bits["flag_ebv_unphysical"][1] == 1.0
    assert bits["flag_ebv_unphysical"][2] == 0.0
    assert bits["flag_zero_point_applied"][2] == 1.0
    assert bits["flag_zero_point_applied"][3] == 0.0
    # An unobserved EBV cannot be "unphysical" even if its stored value is huge.
    hidden = observed.copy()
    hidden[4, ebv] = False
    values[4, ebv] = 500.0
    context = normalizer.transform(values, hidden, zero_point_applied=False)
    assert context[4, N_FEATURES + ch.MASK_BITS.index("flag_ebv_unphysical")] == 0.0
    assert context[4, N_FEATURES + ch.MASK_BITS.index("mask_ebv")] == 0.0


def test_observed_non_finite_values_raise():
    values, observed, roles = _synthetic()
    normalizer = ch.ContextNormalizer.fit(values, ch.fit_mask_from_roles(roles))
    values = values.copy()
    values[3, 2] = np.nan
    with pytest.raises(ValueError):
        normalizer.transform(values, observed, zero_point_applied=False)


def test_degenerate_and_malformed_inputs_are_rejected():
    values, observed, roles = _synthetic()
    fit = ch.fit_mask_from_roles(roles)
    constant = values.copy()
    constant[:, 4] = 1.0
    with pytest.raises(ValueError):
        ch.ContextNormalizer.fit(constant, fit)
    with pytest.raises(ValueError):
        ch.ContextNormalizer.fit(values[:, : N_FEATURES - 1], fit)
    with pytest.raises(ValueError):
        ch.ContextNormalizer.fit(values, np.zeros(3, dtype=bool))
    with pytest.raises(ValueError):
        ch.ContextNormalizer.fit(values, fit.astype(int))
    # A fit subset with fewer than two observed rows per feature must raise.
    lonely = np.zeros(values.shape[0], dtype=bool)
    lonely[0] = True
    with pytest.raises(ValueError):
        ch.ContextNormalizer.fit(values, lonely)


def test_roles_helper_rejects_unknown_labels():
    roles = np.array(["fit", "tune", "calibration", "test", "test"])
    mask = ch.fit_mask_from_roles(roles)
    assert mask.tolist() == [True, False, False, False, False]
    assert ch.fit_mask_from_roles(roles, "test").sum() == 2
    with pytest.raises(ValueError):
        ch.fit_mask_from_roles(np.array(["fit", "Fit"]))
    with pytest.raises(ValueError):
        ch.fit_mask_from_roles(np.array(["tune", "test"]))
    with pytest.raises(ValueError):
        ch.fit_mask_from_roles(roles, "validation")


def test_conditioner_shapes_parameters_and_batch_invariance():
    model = ch.ContextConditioner()
    assert model.parameter_count == 13952
    context = torch.randn(128, 22)
    summary = model(context)
    assert summary.shape == (128, 128)
    latent = torch.randn(128, 256)
    fused = model.fuse(latent, context)
    assert fused.shape == (128, 384)
    assert torch.equal(fused[:, :256], latent)
    # float32 BLAS may reorder a matmul across batch shapes; exact equality is
    # required on the float64 scoring path, not here.
    assert torch.allclose(
        model.fuse(latent[:1], context[:1]), fused[:1], rtol=0, atol=1e-6
    )
    # Gradients reach both the latent and the context path.
    latent = latent.clone().requires_grad_(True)
    context = context.clone().requires_grad_(True)
    model.fuse(latent, context).square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert context.grad.abs().sum() > 0


def test_conditioner_fail_closed():
    model = ch.ContextConditioner()
    with pytest.raises(ValueError):
        model(torch.randn(4, 21))
    with pytest.raises(ValueError):
        model(torch.randn(4, 22, 1))
    with pytest.raises(ValueError):
        model(torch.full((4, 22), float("nan")))
    with pytest.raises(TypeError):
        model(torch.zeros(4, 22, dtype=torch.int64))
    with pytest.raises(ValueError):
        model.fuse(torch.randn(4, 384), torch.randn(4, 22))
    with pytest.raises(ValueError):
        model.fuse(torch.randn(4, 256), torch.randn(5, 22))
    with pytest.raises(ValueError):
        model.fuse(torch.full((4, 256), float("inf")), torch.randn(4, 22))
    with pytest.raises(ValueError):
        ch.ContextConditioner(context_dim=0)


def test_extreme_but_finite_context_stays_finite():
    model = ch.ContextConditioner()
    context = torch.zeros(3, 22)
    context[0] = 1e6
    context[1] = -1e6
    fused = model.fuse(torch.zeros(3, 256), context)
    assert torch.isfinite(fused).all()
