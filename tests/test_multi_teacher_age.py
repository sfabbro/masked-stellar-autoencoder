"""Contract tests for the multi-teacher age correction machinery.

Every number here is synthetic, so the truth is known: the teacher links,
their inflations and a deliberate teacher bias must be recovered, the
teachers must enter through the likelihood instead of being averaged, the
cluster term must recover a shared age and be blind to order/duplication, the
meta-correction must recover a monotone offset without seeing test rows, and
the variance decomposition must add exactly with no clipping.

The quadrature is checked against the exact Gaussian convolution it
approximates (``gaussian_convolution_nll``), including the regimes where each
of the two fixed rules fails - which is why ``rule="auto"`` exists.
"""

import math

import pytest
import torch

from masked_stellar_autoencoder.training import multi_teacher_age as mt

DOUBLE = torch.float64
FEATURES = mt.DEFAULT_TEACHER_FEATURE_NAMES
N_FEATURES = len(FEATURES)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _features(n_rows: int) -> torch.Tensor:
    return torch.zeros(n_rows, N_FEATURES, dtype=DOUBLE)


def _affine_calibration(
    name: str,
    offset: float,
    slope: float,
    sigma: float,
    kappa: float,
) -> mt.TeacherCalibration:
    """Affine teacher link pinned to known truth (for recovery tests)."""
    calibration = mt.TeacherCalibration(
        name, mode="affine", initial_intrinsic_sigma=sigma, initial_kappa=kappa
    ).double()
    with torch.no_grad():
        calibration.offset.fill_(offset)
        calibration.log_slope.fill_(math.log(slope))
        calibration.log_intrinsic_sigma.fill_(math.log(sigma))
        calibration.log_kappa.fill_(math.log(kappa))
    return calibration


def _label(name: str, tau: float, sigma_quoted: float, quality: float = 1.0):
    return mt.TeacherAgeLabel(
        name, f"{name}_scale", tau, sigma_quoted=sigma_quoted, quality=quality
    )


#: Three teachers on one latent scale. ``willett`` is deliberately biased by
#: ``exp(0.85) = 2.34`` in age and has the largest quoted errors, so a method
#: that merges teachers cannot fit this table.
TEACHER_SPEC = {
    "starhorse": dict(
        offset=0.00,
        slope=1.00,
        sigma=0.12,
        kappa=1.4,
        quoted=(0.06, 0.24),
        present=1.00,
    ),
    "apokasc3": dict(
        offset=-0.12,
        slope=1.08,
        sigma=0.08,
        kappa=1.7,
        quoted=(0.03, 0.20),
        present=0.60,
    ),
    "willett": dict(
        offset=0.85,
        slope=0.92,
        sigma=0.22,
        kappa=2.4,
        quoted=(0.02, 0.30),
        present=0.50,
    ),
}


def _synthetic_teacher_table(n_rows: int = 2000, seed: int = 11):
    """Generate ``(tau_true, features, labels)`` with known teacher systematics."""
    torch.manual_seed(seed)
    tau_true = 0.7 + 0.7 * torch.randn(n_rows, dtype=DOUBLE)
    rows = [dict() for _ in range(n_rows)]
    for name, spec in TEACHER_SPEC.items():
        low, high = spec["quoted"]
        quoted = low + (high - low) * torch.rand(n_rows, dtype=DOUBLE)
        noise = (spec["sigma"] ** 2 + (spec["kappa"] * quoted) ** 2).sqrt()
        observed = (
            spec["offset"]
            + spec["slope"] * tau_true
            + noise * torch.randn(n_rows, dtype=DOUBLE)
        )
        present = torch.rand(n_rows, dtype=DOUBLE) < spec["present"]
        for row in range(n_rows):
            if bool(present[row]):
                rows[row][name] = _label(name, float(observed[row]), float(quoted[row]))
    labels = mt.TeacherLabelBatch.from_rows(rows, teacher_names=tuple(TEACHER_SPEC))
    return tau_true, _features(n_rows), labels


def _fit_lbfgs(parameters, closure) -> None:
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=0.5,
        max_iter=600,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )
    optimizer.step(closure)


# --------------------------------------------------------------------------- #
# Teacher labels
# --------------------------------------------------------------------------- #
def test_teacher_label_derives_and_validates_uncertainty():
    label = mt.TeacherAgeLabel.from_percentiles_gyr(
        "starhorse", "starhorse_v2", age_50=4.0, age_16=2.0, age_84=8.0
    )
    assert label.sigma_quoted == pytest.approx(0.5 * math.log(4.0), abs=1e-15)
    assert label.sigma_quoted_source == "derived_16_84"
    assert label.age_gyr == pytest.approx(4.0, rel=1e-15)
    assert set(label.tau_percentiles) == {"0.16", "0.50", "0.84"}

    # A 5/95 pair is converted with the Gaussian-equivalent 1.6449, not 1.96.
    wide = mt.TeacherAgeLabel.from_percentiles_gyr(
        "willett", "willett_age_Dnu", age_50=3.0, age_5=1.0, age_95=9.0
    )
    assert wide.sigma_quoted == pytest.approx(
        0.5 * math.log(9.0) / 1.6448536269514715, rel=1e-12
    )
    assert wide.sigma_quoted_source == "derived_5_95_gaussian"

    # A linear age error is propagated as sigma_tau = sigma_gyr / age.
    linear = mt.TeacherAgeLabel.from_gyr(
        "apokasc3", "apokasc3", 4.0, sigma_quoted_gyr=0.8
    )
    assert linear.sigma_quoted == pytest.approx(0.2, rel=1e-15)
    with pytest.raises(ValueError):
        mt.TeacherAgeLabel.from_gyr(
            "apokasc3", "apokasc3", 4.0, 0.1, sigma_quoted_gyr=0.8
        )

    # No uncertainty at all is rejected: an error is never invented.
    with pytest.raises(ValueError):
        mt.TeacherAgeLabel("apokasc3", "apokasc3", 1.0)
    # Crossed percentiles are rejected, not reordered.
    with pytest.raises(ValueError):
        mt.TeacherAgeLabel("t", "s", 1.0, sigma_quoted=0.1, tau16=2.0, tau84=1.0)
    with pytest.raises(ValueError):
        mt.TeacherAgeLabel("t", "s", 1.0, sigma_quoted=0.0)
    with pytest.raises(ValueError):
        mt.TeacherAgeLabel("t", "s", 1.0, sigma_quoted=0.1, quality=1.5)
    with pytest.raises(ValueError):
        mt.TeacherAgeLabel("", "s", 1.0, sigma_quoted=0.1)
    with pytest.raises(ValueError):
        mt.TeacherAgeLabel.from_percentiles_gyr(
            "t", "s", age_50=-1.0, age_16=0.5, age_84=2.0
        )


def test_teacher_label_batch_is_sparse_and_scale_checked():
    rows = [
        {"a": _label("a", 0.5, 0.1), "b": _label("b", 0.9, 0.2)},
        {"a": _label("a", 0.7, 0.1)},
    ]
    batch = mt.TeacherLabelBatch.from_rows(rows, teacher_names=("a", "b"))
    assert batch.n_rows == 2 and batch.n_teachers == 2
    assert torch.equal(batch.observed, torch.isfinite(batch.tau))
    assert batch.observed.tolist() == [[True, True], [True, False]]
    assert torch.equal(
        batch.quality, torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=DOUBLE)
    )
    assert batch.age_scales == ("a_scale", "b_scale")
    inventory = batch.describe()
    assert [entry["n_labels"] for entry in inventory["teachers"]] == [2, 1]
    assert inventory["partial_label_policy"] == "per_teacher_mask; no imputation"
    # Reindexing the teacher axis keeps the labels aligned.
    swapped = batch.select_teachers(("b", "a"))
    assert torch.equal(swapped.tau[1], batch.tau[0])
    assert swapped.index_of("b") == 0
    with pytest.raises(ValueError):
        batch.index_of("c")
    # One teacher may not mix age scales, and keys must match the label.
    with pytest.raises(ValueError):
        mt.TeacherLabelBatch.from_rows(
            [
                {"a": _label("a", 0.5, 0.1)},
                {"a": mt.TeacherAgeLabel("a", "other", 0.6, sigma_quoted=0.1)},
            ],
            teacher_names=("a",),
        )
    with pytest.raises(ValueError):
        mt.TeacherLabelBatch.from_rows(
            [{"a": _label("b", 0.5, 0.1)}], teacher_names=("a",)
        )
    with pytest.raises(ValueError):
        mt.TeacherLabelBatch.from_rows(
            [{"c": _label("c", 0.5, 0.1)}], teacher_names=("a",)
        )


# --------------------------------------------------------------------------- #
# Teacher calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["affine", "mlp", "stage_affine"])
def test_teacher_calibration_transform_inverse_and_monotonicity(mode):
    kwargs = {"n_stages": 3} if mode == "stage_affine" else {}
    calibration = mt.TeacherCalibration(
        "t", mode=mode, initial_offset=0.2, initial_slope=1.3, **kwargs
    ).double()
    features = torch.randn(7, N_FEATURES, dtype=DOUBLE)
    features[:, FEATURES.index("stage")] = torch.randint(0, 3, (7,)).to(DOUBLE)
    tau = torch.linspace(-2.0, 3.0, 7, dtype=DOUBLE)
    mapping = calibration(features)
    assert mapping.slope.shape == tau.shape
    assert bool((mapping.slope > 0).all())
    mapped = calibration.transform(tau, features, mapping=mapping)
    assert torch.allclose(
        calibration.inverse(mapped, features, mapping=mapping), tau, atol=1e-12
    )
    assert torch.allclose(mapped, mapping.offset + mapping.slope * tau, atol=1e-14)
    # The link is strictly monotone in tau wherever the slope is positive.
    fine = torch.linspace(-3.0, 3.0, 41, dtype=DOUBLE)
    # A one-row mapping broadcasts over a tau grid (diagnostic curve).
    one_row = calibration.transform(fine, features[:1])
    assert one_row.shape == fine.shape
    assert bool((one_row[1:] > one_row[:-1]).all())
    with pytest.raises(ValueError):
        calibration.transform(fine, features[:2])
    with pytest.raises(ValueError):
        calibration.transform(torch.zeros(6, dtype=DOUBLE), features)
    with pytest.raises(ValueError):
        calibration(torch.zeros(7, N_FEATURES + 1, dtype=DOUBLE))


def test_teacher_calibration_identity_gauge_and_stage_bins():
    reference = mt.TeacherCalibration(
        "ref", mode="affine", initial_offset=0.4, initial_slope=1.2
    ).double()
    reference.freeze_identity()
    features = torch.randn(5, N_FEATURES, dtype=DOUBLE)
    tau = torch.linspace(-1.0, 2.0, 5, dtype=DOUBLE)
    assert reference.is_identity
    # The reference teacher *is* the scale: a + b*tau == tau bitwise.
    assert torch.equal(reference.transform(tau, features), tau)
    assert torch.equal(reference.inverse(tau, features), tau)
    assert bool((reference.offset == 0).all()) and bool(
        (reference.log_slope == 0).all()
    )
    with pytest.raises(ValueError):
        mt.TeacherCalibration("x", mode="mlp").double().freeze_identity()

    binned = mt.TeacherCalibration(
        "t", mode="stage_affine", n_stages=3, initial_intrinsic_sigma=0.2
    ).double()
    grid = torch.zeros(3, N_FEATURES, dtype=DOUBLE)
    grid[:, FEATURES.index("stage")] = torch.tensor([0.0, 1.0, 2.0], dtype=DOUBLE)
    with torch.no_grad():
        binned.offset.copy_(torch.tensor([-0.3, 0.0, 0.4], dtype=DOUBLE))
    offsets = binned(grid).offset
    assert offsets.tolist() == pytest.approx([-0.3, 0.0, 0.4])
    bad = grid.clone()
    bad[0, FEATURES.index("stage")] = 3.0
    with pytest.raises(ValueError):
        binned(bad)
    bad[0, FEATURES.index("stage")] = 0.5
    with pytest.raises(ValueError):
        binned(bad)


def test_teacher_effective_sigma_and_kappa_are_explicit():
    calibration = _affine_calibration("t", 0.0, 1.1, 0.2, 2.5)
    features = _features(3)
    mapping = calibration(features)
    quoted = torch.tensor([0.0, 0.1, 0.4], dtype=DOUBLE)
    expected = (0.2**2 + (2.5 * quoted) ** 2).sqrt()
    assert torch.allclose(mapping.effective_sigma(quoted), expected, rtol=1e-14)
    # kappa is a free positive inflation, reported rather than clamped.
    assert calibration.inflation_at_least_one
    assert calibration.diagnostics(features)["kappa"] == pytest.approx(2.5, rel=1e-12)
    with torch.no_grad():
        calibration.log_kappa.fill_(math.log(0.6))
    assert not calibration.inflation_at_least_one
    assert calibration.diagnostics(features)["kappa_at_least_one"] is False
    with pytest.raises(ValueError):
        mapping.effective_sigma(torch.tensor([-0.1, 0.1, 0.2], dtype=DOUBLE))


# --------------------------------------------------------------------------- #
# Latent integration
# --------------------------------------------------------------------------- #
def test_quadrature_matches_the_exact_gaussian_convolution():
    calibration = _affine_calibration("t", 0.3, 1.4, 0.2, 2.0)
    features = _features(6)
    mu = torch.tensor([0.0, 1.0, -2.0, 0.5, 2.0, -1.0], dtype=DOUBLE)
    log_sigma = torch.log(torch.tensor([0.01, 0.2, 0.8, 0.05, 0.5, 0.3], dtype=DOUBLE))
    quoted = torch.tensor([0.05, 0.1, 0.2, 0.02, 0.3, 0.01], dtype=DOUBLE)
    observed = (
        1.4 * mu + 0.3 + torch.tensor([0.0, 0.1, -0.2, 0.05, 0.3, -0.1], dtype=DOUBLE)
    )
    mapping = calibration(features)
    exact = mt.gaussian_convolution_nll(observed, quoted, mu, log_sigma, mapping)

    auto, diagnostics = mt.teacher_marginal_nll(
        observed, quoted, mu, log_sigma, mapping, rule="auto", order=32
    )
    # Measured agreement: 2.2e-16 in float64 at order 32.
    assert torch.allclose(auto, exact, atol=1e-12)
    assert diagnostics.latent_rule_rows + diagnostics.residual_rule_rows == 6
    assert diagnostics.min_latent_resolution <= 1.0
    assert diagnostics.min_residual_resolution <= 1.0

    # Each single rule fails on the rows outside its domain; auto does not.
    latent_only, _ = mt.teacher_marginal_nll(
        observed, quoted, mu, log_sigma, mapping, rule="gh_latent", order=32
    )
    residual_only, _ = mt.teacher_marginal_nll(
        observed, quoted, mu, log_sigma, mapping, rule="gh_residual", order=32
    )
    assert float((latent_only - exact).abs().max().detach()) > 1e-8
    assert float((residual_only - exact).abs().max().detach()) > 0.1
    # Geometric convergence of the auto rule (measured 1e-2, 7e-5, 2e-9).
    coarse, _ = mt.teacher_marginal_nll(
        observed, quoted, mu, log_sigma, mapping, rule="auto", order=8
    )
    assert float((coarse - exact).abs().max().detach()) < 1e-3
    assert float((auto - exact).abs().max().detach()) < float(
        (coarse - exact).abs().max().detach()
    )
    # The fixed QMC node set is the same Gaussian expectation, coarser.
    qmc, _ = mt.teacher_marginal_nll(
        observed, quoted, mu, log_sigma, mapping, nodes="qmc", qmc_count=256
    )
    assert float((qmc - exact).abs().max().detach()) < 1e-2
    with pytest.raises(ValueError):
        mt.teacher_marginal_nll(
            observed, quoted, mu, log_sigma, mapping, rule="not-a-rule"
        )


def test_quadrature_rule_selection_reports_resolution():
    """Rows where one rule is exact and the other stalls are routed correctly."""
    calibration = _affine_calibration("t", 0.0, 1.4, 0.01, 1.0)
    features = _features(2)
    mu = torch.zeros(2, dtype=DOUBLE)
    # Row 0: broad teacher error -> latent-space rule. Row 1: precise -> residual.
    log_sigma = torch.log(torch.tensor([0.01, 1.0], dtype=DOUBLE))
    quoted = torch.tensor([0.5, 0.01], dtype=DOUBLE)
    observed = torch.tensor([0.0, 0.0], dtype=DOUBLE)
    mapping = calibration(features)
    ratio = mapping.mapped_scale(log_sigma.exp()) / mapping.effective_sigma(quoted)
    ratio = ratio.detach()
    assert float(ratio[0]) < 0.1 and float(ratio[1]) > 10.0

    exact = mt.gaussian_convolution_nll(observed, quoted, mu, log_sigma, mapping)
    nll, diagnostics = mt.teacher_marginal_nll(
        observed, quoted, mu, log_sigma, mapping, rule="auto", order=32
    )
    assert diagnostics.latent_rule_rows == 1
    assert diagnostics.residual_rule_rows == 1
    assert diagnostics.min_latent_resolution < 0.1
    assert diagnostics.min_residual_resolution < 0.1
    assert torch.allclose(nll, exact, atol=1e-12)
    # Explicit rare branch: gh_residual needs a positive latent scale.
    with pytest.raises(ValueError):
        mt.teacher_marginal_nll(
            observed,
            quoted,
            mu,
            torch.full((2,), -800.0, dtype=DOUBLE),
            mapping,
            rule="gh_residual",
        )


# --------------------------------------------------------------------------- #
# Joint recovery of teacher systematics
# --------------------------------------------------------------------------- #
def test_recover_teacher_offsets_scales_and_kappa_with_a_biased_teacher():
    tau_true, features, labels = _synthetic_teacher_table()
    calibrations = {
        name: mt.TeacherCalibration(
            name, mode="affine", initial_intrinsic_sigma=0.1, initial_kappa=1.0
        ).double()
        for name in TEACHER_SPEC
    }
    # The reference teacher defines the scale: a = 0, b = 1 by definition.
    calibrations["starhorse"].freeze_identity()
    log_sigma = torch.full((labels.n_rows,), math.log(1e-3), dtype=DOUBLE)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = mt.multi_teacher_age_loss(
            tau_true, log_sigma, features, labels, calibrations
        ).mean
        loss.backward()
        return loss

    optimizer = torch.optim.LBFGS(
        [p for c in calibrations.values() for p in c.parameters() if p.requires_grad],
        lr=0.5,
        max_iter=600,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )
    optimizer.step(closure)
    fitted = mt.multi_teacher_age_loss(
        tau_true, log_sigma, features, labels, calibrations
    )
    assert fitted.observation_count == 3 and fitted.label_count > 3000

    for name, spec in TEACHER_SPEC.items():
        mapping = calibrations[name](features)
        offset = float(mapping.offset[0].detach())
        slope = float(mapping.slope[0].detach())
        sigma = float(mapping.intrinsic_sigma[0].detach())
        kappa = float(mapping.kappa[0].detach())
        if name == "starhorse":
            assert offset == 0.0 and slope == 1.0  # frozen gauge, bitwise
        else:
            assert offset == pytest.approx(spec["offset"], abs=0.05)
        assert slope == pytest.approx(spec["slope"], abs=0.04)
        assert sigma == pytest.approx(spec["sigma"], abs=0.06)
        assert kappa == pytest.approx(spec["kappa"], abs=0.3)

    # The deliberately biased teacher is recovered as biased, not merged away.
    willett = calibrations["willett"](features)
    assert float(willett.offset[0].detach()) > 0.75  # exp(0.85) = 2.34 in age
    assert float(willett.offset[0].detach()) - float(
        calibrations["apokasc3"](features).offset[0].detach()
    ) == pytest.approx(0.97, abs=0.06)
    # ... and learning it is what makes the fit work: the untrained links lose.
    untrained = {
        name: mt.TeacherCalibration(
            name, mode="affine", initial_intrinsic_sigma=0.1, initial_kappa=1.0
        ).double()
        for name in TEACHER_SPEC
    }
    untrained["starhorse"].freeze_identity()
    baseline = mt.multi_teacher_age_loss(
        tau_true, log_sigma, features, labels, untrained
    ).mean
    assert float(fitted.mean.detach()) < 0.5 * float(baseline.detach())


# --------------------------------------------------------------------------- #
# The teachers enter through the likelihood
# --------------------------------------------------------------------------- #
def test_teachers_combine_through_the_likelihood_not_by_averaging():
    precise = _affine_calibration("precise", 0.0, 1.0, 0.1, 1.0)
    noisy = _affine_calibration("noisy", 0.5, 1.0, 0.2, 2.0)
    calibrations = {"precise": precise, "noisy": noisy}
    quoted_precise, quoted_noisy = 0.05, 0.5
    observed_precise, observed_noisy = 1.0, 2.0
    rows = [
        {
            "precise": _label("precise", observed_precise, quoted_precise),
            "noisy": _label("noisy", observed_noisy, quoted_noisy),
        }
    ]
    labels = mt.TeacherLabelBatch.from_rows(rows, teacher_names=("precise", "noisy"))
    features = _features(1)
    log_sigma = torch.full((1,), math.log(1e-6), dtype=DOUBLE)

    mu = torch.zeros(1, dtype=DOUBLE, requires_grad=True)
    result = mt.multi_teacher_age_loss(mu, log_sigma, features, labels, calibrations)
    # A star with two teachers has *two* log-likelihood terms, one per teacher.
    assert result.observation_count == 2 and result.label_count == 2
    assert torch.allclose(
        result.numerator,
        result.teacher("precise").mean + result.teacher("noisy").mean,
        atol=1e-14,
    )
    # Changing one teacher's value leaves the other teacher's term untouched.
    rows_shifted = [dict(rows[0])]
    rows_shifted[0]["noisy"] = _label("noisy", observed_noisy + 0.5, quoted_noisy)
    shifted = mt.multi_teacher_age_loss(
        mu,
        log_sigma,
        features,
        mt.TeacherLabelBatch.from_rows(
            rows_shifted, teacher_names=("precise", "noisy")
        ),
        calibrations,
    )
    assert torch.equal(
        shifted.teacher("precise").numerator, result.teacher("precise").numerator
    )
    assert not torch.equal(
        shifted.teacher("noisy").numerator, result.teacher("noisy").numerator
    )

    # The latent optimum is the precision-weighted combination, never the mean
    # of the two teacher values (1.25 in latent units here).
    optimum = torch.optim.LBFGS(
        [mu], lr=0.5, max_iter=400, tolerance_grad=1e-14, tolerance_change=1e-16
    )

    def closure() -> torch.Tensor:
        optimum.zero_grad()
        value = mt.multi_teacher_age_loss(
            mu, log_sigma, features, labels, calibrations
        ).mean
        value.backward()
        return value

    optimum.step(closure)
    variance_precise = 0.1**2 + quoted_precise**2
    variance_noisy = 0.2**2 + (2.0 * quoted_noisy) ** 2
    expected = (
        observed_precise / variance_precise + (observed_noisy - 0.5) / variance_noisy
    ) / (1.0 / variance_precise + 1.0 / variance_noisy)
    arithmetic = 0.5 * ((observed_precise - 0.0) + (observed_noisy - 0.5))
    learned = float(mu[0].detach())
    assert learned == pytest.approx(expected, abs=5e-3)
    assert abs(learned - arithmetic) > 0.15
    # The precise teacher dominates, as a likelihood must.
    assert abs(learned - observed_precise) < 0.05


def test_partial_labels_and_teacher_specific_denominators():
    calibration = mt.TeacherCalibration("shared", mode="affine").double()
    calibrations = {"a": calibration, "b": calibration}
    features = _features(4)
    mu = torch.zeros(4, dtype=DOUBLE)
    log_sigma = torch.zeros(4, dtype=DOUBLE)
    rows = [
        {"a": _label("a", 0.3, 0.1), "b": _label("b", 0.4, 0.1)},
        {"a": _label("a", 0.9, 0.1)},
        {"a": _label("a", 1.4, 0.1)},
    ]
    labels = mt.TeacherLabelBatch.from_rows(rows, teacher_names=("a", "b"))
    result = mt.multi_teacher_age_loss(
        mu[:3], log_sigma[:3], features[:3], labels, calibrations
    )
    assert result.teacher("a").count == 3 and result.teacher("a").weight_sum == 3.0
    assert result.teacher("b").count == 1 and result.teacher("b").weight_sum == 1.0
    assert result.label_count == 4 and result.row_count == 3
    # Teacher "a" has its own denominator: a star that only carries "b" cannot
    # move it, and "b"'s denominator counts only its own labels.
    mean_a = result.teacher("a").mean
    extended = mt.multi_teacher_age_loss(
        mu,
        log_sigma,
        features,
        mt.TeacherLabelBatch.from_rows(
            [*rows, {"b": _label("b", -2.0, 0.1)}], teacher_names=("a", "b")
        ),
        calibrations,
    )
    assert torch.equal(extended.teacher("a").mean, mean_a)
    assert extended.teacher("b").count == 2 and extended.teacher("b").weight_sum == 2.0

    # Declared label quality is a weight in that teacher's own denominator.
    weighted = mt.multi_teacher_age_loss(
        mu[:3],
        log_sigma[:3],
        features[:3],
        mt.TeacherLabelBatch.from_rows(
            [
                {"a": _label("a", 0.3, 0.1, quality=0.5), "b": _label("b", 0.4, 0.1)},
                {"a": _label("a", 0.9, 0.1, quality=0.0)},
                {"a": _label("a", 1.4, 0.1)},
            ],
            teacher_names=("a", "b"),
        ),
        calibrations,
    )
    assert weighted.teacher("a").count == 3
    assert weighted.teacher("a").quality_zero == 1
    assert weighted.teacher("a").weight_sum == pytest.approx(1.5)
    # An all-absent teacher is inactive and contributes a differentiable zero.
    absent = mt.TeacherLabelBatch.from_rows(
        [{"a": _label("a", 0.3, 0.1)}], teacher_names=("a", "b")
    )
    only_a = mt.multi_teacher_age_loss(
        mu[:1], log_sigma[:1], features[:1], absent, calibrations
    )
    assert only_a.teacher("b").count == 0 and not only_a.teacher("b").active
    assert float(only_a.teacher("b").mean.detach()) == 0.0
    # An empty table is a differentiable zero, not a NaN.
    empty = mt.multi_teacher_age_loss(
        mu[:2],
        log_sigma[:2],
        features[:2],
        mt.TeacherLabelBatch.from_rows([{}, {}], teacher_names=("a",)),
        {"a": calibration},
    )
    assert empty.observation_count == 0 and empty.label_count == 0
    assert float(empty.mean.detach()) == 0.0
    trainable = mu.detach().clone().requires_grad_(True)
    empty_grad = mt.multi_teacher_age_loss(
        trainable[:2],
        trainable[:2],
        features[:2],
        mt.TeacherLabelBatch.from_rows([{}, {}], teacher_names=("a",)),
        {"a": calibration},
    )
    (grad,) = torch.autograd.grad(empty_grad.mean, trainable)
    assert torch.equal(grad, torch.zeros_like(grad))


# --------------------------------------------------------------------------- #
# Cluster coevality
# --------------------------------------------------------------------------- #
def test_cluster_coeval_recovers_shared_age_and_is_order_invariant():
    torch.manual_seed(5)
    ages_true = torch.tensor([-0.5, 1.0, 2.0], dtype=DOUBLE)
    counts = [7, 12, 5]
    cluster_ids = torch.cat(
        [
            torch.full((count,), index, dtype=torch.int64)
            for index, count in enumerate(counts)
        ]
    )
    tau = torch.cat(
        [
            ages_true[index] + 0.15 * torch.randn(count, dtype=DOUBLE)
            for index, count in enumerate(counts)
        ]
    )
    result = mt.cluster_coeval_loss(tau, cluster_ids)
    assert result.profiled and result.observation_count == 3 and result.row_count == 24
    assert result.cluster_counts == (7, 12, 5)
    for index, age in enumerate(ages_true.tolist()):
        assert float(result.cluster_age[index]) == pytest.approx(age, abs=0.1)
    # The profiled age is the argmin of the loss over that cluster's age.
    grid = torch.linspace(-1.5, 0.5, 4001, dtype=DOUBLE)
    losses = []
    other = result.cluster_age[1:].detach().clone()
    for value in grid:
        supplied = torch.stack([value, other[0], other[1]])
        losses.append(
            float(
                mt.cluster_coeval_loss(tau, cluster_ids, cluster_age=supplied).numerator
            )
        )
    best = float(grid[int(torch.tensor(losses).argmin())])
    assert best == pytest.approx(float(result.cluster_age[0]), abs=1e-3)
    # Order and duplication invariance: the objective is a ratio of sums.
    permutation = torch.randperm(tau.numel())
    permuted = mt.cluster_coeval_loss(tau[permutation], cluster_ids[permutation])
    assert float(permuted.numerator) == pytest.approx(
        float(result.numerator), rel=1e-14
    )
    duplicated = mt.cluster_coeval_loss(
        torch.cat([tau, tau]), torch.cat([cluster_ids, cluster_ids])
    )
    assert float(duplicated.numerator) == pytest.approx(
        float(result.numerator), rel=1e-12
    )
    assert torch.allclose(duplicated.cluster_age, result.cluster_age, atol=1e-12)
    # Missing (negative) ids are masked out, not treated as a cluster.
    with_missing = mt.cluster_coeval_loss(
        torch.cat([tau, torch.tensor([9.0], dtype=DOUBLE)]),
        torch.cat([cluster_ids, torch.tensor([-1], dtype=torch.int64)]),
    )
    assert with_missing.row_count == 24
    assert float(with_missing.numerator) == pytest.approx(
        float(result.numerator), rel=1e-14
    )
    with pytest.raises(TypeError):
        mt.cluster_coeval_loss(tau, cluster_ids.to(DOUBLE))
    unassigned = torch.zeros(3, dtype=DOUBLE, requires_grad=True)
    empty = mt.cluster_coeval_loss(unassigned, torch.full((3,), -1, dtype=torch.int64))
    assert empty.observation_count == 0 and empty.row_count == 0
    assert float(empty.mean.detach()) == 0.0
    (grad,) = torch.autograd.grad(empty.mean, unassigned)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_cluster_coeval_weights_and_learnable_ages():
    cluster_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
    tau = torch.tensor([0.0, 0.2, -0.4, 1.0, 1.4], dtype=DOUBLE)
    base = mt.cluster_coeval_loss(tau, cluster_ids)
    # A zero selection weight drops the member instead of biasing the mean.
    weights = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0], dtype=DOUBLE)
    dropped = mt.cluster_coeval_loss(tau, cluster_ids, selection_weights=weights)
    assert dropped.row_count == 4
    assert torch.allclose(
        dropped.cluster_age,
        mt.cluster_coeval_loss(
            tau[:2].new_tensor([0.0, 0.2, 1.0, 1.4]), torch.tensor([0, 0, 1, 1])
        ).cluster_age,
    )
    assert float(base.numerator) != float(dropped.numerator)
    # Declared variances reweight the members (cluster random effect included).
    with_variance = mt.cluster_coeval_loss(
        tau,
        cluster_ids,
        sigma_cluster=torch.tensor([0.1, 0.3], dtype=DOUBLE),
        tau_sigma=torch.full((5,), 0.2, dtype=DOUBLE),
    )
    assert with_variance.observation_count == 2
    assert float(with_variance.numerator) != float(base.numerator)
    # A learnable shared age is a parameter: gradients flow through it.
    learnable = mt.LearnableClusterAges.from_members(tau, cluster_ids)
    assert torch.allclose(
        learnable.ages.detach(),
        torch.tensor([-0.0666666667, 1.2], dtype=DOUBLE),
        atol=1e-6,
    )
    learned = mt.cluster_coeval_loss(tau, cluster_ids, cluster_age=learnable.ages)
    assert not learned.profiled
    learned.mean.backward()
    assert learnable.ages.grad is not None
    assert torch.isfinite(learnable.ages.grad).all()
    assert torch.allclose(
        learnable.per_member(cluster_ids), learnable.ages[cluster_ids]
    )
    missing_member = learnable.per_member(torch.tensor([-1], dtype=torch.int64))
    assert float(missing_member[0].detach()) == 0.0
    with pytest.raises(ValueError):
        mt.cluster_coeval_loss(tau, cluster_ids, reduction="median")


# --------------------------------------------------------------------------- #
# Meta-correction
# --------------------------------------------------------------------------- #
def test_meta_correction_is_the_identity_before_fitting_and_monotone():
    torch.manual_seed(2)
    features = torch.randn(32, 3, dtype=DOUBLE)
    tau = torch.linspace(-2.0, 3.0, 32, dtype=DOUBLE)
    correction = mt.MetaCorrection(3, target_scales=("cluster", "dr4_flame")).double()
    assert torch.equal(correction.apply(tau, features, "cluster"), tau)
    assert torch.equal(correction.apply(tau, features, "dr4_flame"), tau)
    diagnostics = correction.diagnostics(tau, features, "cluster")
    assert diagnostics["monotone"] and diagnostics["identity_max_abs"] == 0.0
    assert diagnostics["slope_min"] == pytest.approx(1.0) and diagnostics[
        "slope_max"
    ] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        correction.apply(tau, features, "unknown_scale")
    with pytest.raises(ValueError):
        correction.apply(tau, features[:, :2], "cluster")
    with pytest.raises(ValueError):
        correction.apply(tau, features[:4], "cluster")


def test_meta_correction_recovers_a_synthetic_monotone_offset():
    torch.manual_seed(4)
    features = torch.randn(64, 3, dtype=DOUBLE)
    base = torch.linspace(-1.5, 2.5, 64, dtype=DOUBLE)
    # A known monotone (affine) age-scale offset of +0.4 in ln and a 1.25 slope.
    target = 1.25 * base + 0.4
    correction = mt.MetaCorrection(
        3, target_scales=("cluster", "dr4_flame"), mode="affine"
    ).double()
    fit = correction.fit(
        base, features, target, target_scale="cluster", steps=1500, lr=0.05
    )
    assert fit.improved and fit.anchor_rows_used == 64 and fit.anchor_rows_ignored == 0
    assert fit.residual_rms < 1e-6
    assert fit.slope_min > 0 and fit.slope_max > 0
    assert torch.allclose(
        correction.apply(base, features, "cluster"), target, atol=1e-6
    )
    assert 1.0 < fit.slope_min < fit.slope_max
    # Only the named scale is adapted: the other target block stays identity.
    assert torch.equal(correction.apply(base, features, "dr4_flame"), base)
    # A smooth monotone systematic the affine family cannot represent exactly is
    # still reduced, and the correction stays monotone everywhere.
    curved = base + 0.5 * torch.tanh(base)
    smooth = mt.MetaCorrection(3, target_scales=("cluster",), mode="mlp").double()
    torch.manual_seed(0)
    smooth_fit = smooth.fit(
        base, features, curved, target_scale="cluster", steps=800, lr=0.03, loss="huber"
    )
    assert smooth_fit.final_loss < 0.25 * smooth_fit.initial_loss
    # Declared per-anchor weights change the fit without changing its contract.
    row_weights = torch.linspace(0.5, 1.5, base.numel(), dtype=DOUBLE)
    weighted = mt.MetaCorrection(3, target_scales=("cluster",), mode="affine").double()
    weighted_fit = weighted.fit(
        base,
        features,
        target,
        target_scale="cluster",
        steps=1200,
        lr=0.05,
        weights=row_weights,
    )
    assert weighted_fit.improved and weighted_fit.residual_rms < 1e-3
    grid = torch.linspace(-3.0, 3.0, 121, dtype=DOUBLE)
    one = smooth(grid, features[:1].expand(grid.numel(), -1), "cluster")
    assert bool((one.slope > 0).all())
    assert bool((one.tau_corrected[1:] > one.tau_corrected[:-1]).all())


def test_meta_correction_fit_ignores_non_anchor_rows():
    torch.manual_seed(6)
    features = torch.randn(64, 3, dtype=DOUBLE)
    base = torch.linspace(-1.5, 2.5, 64, dtype=DOUBLE)
    anchor_target = 1.2 * base + 0.3
    # Test rows live in the same tensors but carry an absurd offset.
    mixed_base = torch.cat([base, base + 0.1])
    mixed_features = torch.cat([features, features + 0.2])
    mixed_target = torch.cat([anchor_target, anchor_target - 5.0])
    anchors = torch.cat(
        [torch.ones(64, dtype=torch.bool), torch.zeros(64, dtype=torch.bool)]
    )
    masked = mt.MetaCorrection(3, target_scales=("cluster",), mode="affine").double()
    reference = mt.MetaCorrection(3, target_scales=("cluster",), mode="affine").double()
    reference.load_state_dict(masked.state_dict())
    with_mask = masked.fit(
        mixed_base,
        mixed_features,
        mixed_target,
        target_scale="cluster",
        anchor_mask=anchors,
    )
    without_test_rows = reference.fit(
        base, features, anchor_target, target_scale="cluster"
    )
    assert with_mask.anchor_rows_used == 64 and with_mask.anchor_rows_ignored == 64
    assert with_mask.final_loss == without_test_rows.final_loss
    assert torch.equal(
        masked.apply(base, features, "cluster"),
        reference.apply(base, features, "cluster"),
    )
    # The fitted correction is the anchors', not the test rows' absurd offset.
    assert (
        abs(
            float(
                (masked.apply(base, features, "cluster") - anchor_target)
                .mean()
                .detach()
            )
        )
        < 0.1
    )
    with pytest.raises(ValueError):
        masked.fit(
            base,
            features,
            anchor_target,
            target_scale="cluster",
            anchor_mask=torch.zeros(64, dtype=torch.bool),
        )


# --------------------------------------------------------------------------- #
# Systematic-variance decomposition
# --------------------------------------------------------------------------- #
def test_variance_decomposition_adds_exactly_and_rejects_negatives():
    model = torch.tensor([0.04, 0.09], dtype=DOUBLE)
    within = torch.tensor([0.01, 0.02], dtype=DOUBLE)
    between = torch.tensor([0.05, 0.0], dtype=DOUBLE)
    scale = torch.tensor([0.02, 0.03], dtype=DOUBLE)
    decomposition = mt.systematic_variance_decomposition(model, within, between, scale)
    assert decomposition.labels == (
        "Var_model",
        "Var_within_teacher",
        "Var_between_teacher",
        "Var_scale",
    )
    assert torch.equal(decomposition.total, model + within + between + scale)
    assert torch.equal(decomposition.model, model)
    assert torch.equal(decomposition.between_teacher, between)
    assert torch.allclose(decomposition.total_sigma.square(), decomposition.total)
    fractions = decomposition.fractions
    assert torch.allclose(sum(fractions), torch.ones(2, dtype=DOUBLE), atol=1e-15)
    assert all(bool((fraction >= 0).all()) for fraction in fractions)
    # as_report is the scalar (already reduced) view used by age.csv.
    with pytest.raises(ValueError):
        decomposition.as_report()
    scalar = mt.systematic_variance_decomposition(0.04, 0.01, 0.05, 0.02)
    report = scalar.as_report()
    assert [row["component"] for row in report] == [
        "Var_model",
        "Var_within_teacher",
        "Var_between_teacher",
        "Var_scale",
        "total",
    ]
    assert report[-1]["variance"] == pytest.approx(0.12)
    assert report[-1]["fraction"] == 1.0
    assert sum(row["fraction"] for row in report[:-1]) == pytest.approx(1.0)
    per_star = decomposition.as_dict()
    assert set(per_star) >= {"total", "total_sigma", "frac_Var_scale"}
    assert torch.equal(per_star["total"], decomposition.total)
    # Broadcasting is exact and order-preserving.
    broadcast = mt.systematic_variance_decomposition(
        0.1, 0.2, torch.zeros(3, dtype=DOUBLE), 0.3
    )
    assert broadcast.total.tolist() == pytest.approx([0.6, 0.6, 0.6])
    # No clipping: an impossible component raises instead of being zeroed.
    with pytest.raises(ValueError):
        mt.systematic_variance_decomposition(0.1, -0.01, 0.2, 0.0)
    with pytest.raises(ValueError):
        mt.systematic_variance_decomposition(0.1, float("nan"), 0.2, 0.0)
    with pytest.raises(ValueError):
        mt.systematic_variance_decomposition(0.1, float("inf"), 0.2, 0.0)
    with pytest.raises(ValueError):
        mt.systematic_variance_decomposition(0.1, 0.2, 0.3, 0.4, labels=("a", "b"))
    # A zero total has undefined fractions (NaN), never a silent guard.
    zero = mt.systematic_variance_decomposition(0.0, 0.0, 0.0, 0.0)
    assert float(zero.total) == 0.0 and float(zero.total_sigma) == 0.0
    assert all(bool(torch.isnan(fraction)) for fraction in zero.fractions)
    assert zero.as_report()[-1]["fraction"] is None


# --------------------------------------------------------------------------- #
# Head wiring
# --------------------------------------------------------------------------- #
def test_head_wires_latent_and_calibrations_with_a_named_scale():
    torch.manual_seed(1)
    head = mt.MultiTeacherAgeHead(
        ("starhorse", "apokasc3"),
        input_dim=8,
        width=16,
        reference_teacher="apokasc3",
        calibration_mode="affine",
    ).double()
    assert head.reference_teacher == "apokasc3"
    assert head.calibrations["apokasc3"].is_identity
    rows = [
        {
            "starhorse": _label("starhorse", 0.5, 0.1),
            "apokasc3": _label("apokasc3", 0.6, 0.08),
        },
        {"apokasc3": _label("apokasc3", 1.1, 0.08)},
    ]
    labels = mt.TeacherLabelBatch.from_rows(rows, teacher_names=head.teacher_names)
    latent_features = torch.randn(2, 8, dtype=DOUBLE)
    physical = _features(2)
    result = head.loss(latent_features, physical, labels)
    assert result.observation_count == 2 and result.label_count == 3
    assert result.reference_teacher == "apokasc3"
    assert result.age_scale == "apokasc3_scale"
    result.mean.backward()
    # Only free teacher parameters get gradients; the gauge stays frozen.
    assert head.calibrations["apokasc3"].offset.grad is None
    assert head.calibrations["apokasc3"].log_slope.grad is None
    assert head.calibrations["apokasc3"].log_kappa.grad is not None
    assert head.calibrations["starhorse"].offset.grad is not None
    assert head.latent.trunk.output.weight.grad is not None
    # 4 free (starhorse) + 2 free (frozen gauge keeps only its noise terms).
    assert len(head.calibration_parameters()) == 6
    assert len(head.latent_parameters()) == 10
    manifest = head.diagnostics(physical, labels)
    assert manifest["age_scale"] == "apokasc3_scale"
    assert manifest["gauge"] == "reference_identity(a=0, b=1)"
    assert manifest["evidence_status"] == mt.MULTI_TEACHER_EVIDENCE_STATUS
    assert len(manifest["teachers"]) == 2
    # Unknown teachers and wrong feature widths are rejected, not ignored.
    with pytest.raises(ValueError):
        head.loss(
            latent_features,
            physical,
            mt.TeacherLabelBatch.from_rows(
                [{"willett": _label("willett", 0.5, 0.1)}], teacher_names=("willett",)
            ),
        )
    with pytest.raises(ValueError):
        head.loss(latent_features, torch.zeros(2, 3, dtype=DOUBLE), labels)
    with pytest.raises(ValueError):
        mt.MultiTeacherAgeHead(("a", "b"), reference_teacher="c")
    with pytest.raises(ValueError):
        mt.MultiTeacherAgeHead(())


def test_diagnostics_are_manifest_ready_and_no_truth_is_claimed():
    assert "not physical truth" in mt.MULTI_TEACHER_EVIDENCE_STATUS
    assert "observed_label" in mt.MULTI_TEACHER_EVIDENCE_STATUS
    assert "named age scale" in mt.MULTI_TEACHER_EVIDENCE_STATUS
    assert mt.SYSTEMATIC_VARIANCE_LABELS == (
        "Var_model",
        "Var_within_teacher",
        "Var_between_teacher",
        "Var_scale",
    )
    assert mt.AGE_COORDINATE == "tau = ln(age / Gyr)"
    _, features, labels = _synthetic_teacher_table(n_rows=32, seed=3)
    calibration = mt.TeacherCalibration("starhorse", mode="mlp").double()
    mu = torch.zeros(32, dtype=DOUBLE)
    log_sigma = torch.zeros(32, dtype=DOUBLE)
    result = mt.multi_teacher_age_loss(
        mu,
        log_sigma,
        features,
        labels.select_teachers(("starhorse",)),
        {"starhorse": calibration},
    )
    diagnostics = result.diagnostics()
    assert diagnostics["loss_denominator"] == "per_teacher_observation_mean"
    assert diagnostics["partial_label_policy"] == "per_teacher_mask; no imputation"
    assert diagnostics["coordinate"] == mt.AGE_COORDINATE
    assert diagnostics["n_active_teachers"] == 1
    assert diagnostics["teachers"][0]["teacher"] == "starhorse"
    assert diagnostics["quadrature"]["latent_rule_rows"] >= 0


# --------------------------------------------------------------------------- #
# Remaining declared paths: weights, stage bins on the head, batch plumbing
# --------------------------------------------------------------------------- #
def test_teacher_weights_zero_a_teacher_out_of_the_denominator():
    calibration = mt.TeacherCalibration("a", mode="affine").double()
    other = mt.TeacherCalibration("b", mode="affine").double()
    labels = mt.TeacherLabelBatch.from_rows(
        [{"a": _label("a", 0.4, 0.1), "b": _label("b", -1.0, 0.1)}],
        teacher_names=("a", "b"),
    )
    features = _features(1)
    mu = torch.zeros(1, dtype=DOUBLE)
    log_sigma = torch.zeros(1, dtype=DOUBLE)
    equal = mt.multi_teacher_age_loss(
        mu, log_sigma, features, labels, {"a": calibration, "b": other}
    )
    silenced = mt.multi_teacher_age_loss(
        mu,
        log_sigma,
        features,
        labels,
        {"a": calibration, "b": other},
        teacher_weights=(1.0, 0.0),
    )
    assert equal.observation_count == 2 and silenced.observation_count == 1
    assert torch.equal(silenced.numerator, equal.teacher("a").mean)
    with pytest.raises(ValueError):
        mt.multi_teacher_age_loss(
            mu,
            log_sigma,
            features,
            labels,
            {"a": calibration, "b": other},
            teacher_weights=(1.0, -1.0),
        )
    with pytest.raises(ValueError):
        mt.multi_teacher_age_loss(
            mu,
            log_sigma,
            features,
            labels,
            {"a": calibration, "b": other},
            teacher_weights=(1.0,),
        )
    # Batch plumbing: an all-absent teacher keeps an empty scale, and casting
    # the batch is value-preserving.
    sparse = mt.TeacherLabelBatch.from_rows(
        [{"a": _label("a", 0.4, 0.1)}], teacher_names=("a", "b")
    )
    assert sparse.age_scales == ("a_scale", "")
    cast = sparse.to(dtype=torch.float32)
    assert cast.tau.dtype == torch.float32
    assert torch.allclose(cast.tau, sparse.tau.float(), equal_nan=True)
    assert torch.equal(cast.observed, sparse.observed)


def test_head_supports_stage_binned_affine_links():
    head = mt.MultiTeacherAgeHead(
        ("starhorse", "apokasc3"),
        input_dim=4,
        width=8,
        reference_teacher="starhorse",
        calibration_mode="stage_affine",
        n_stages=2,
    ).double()
    features = torch.zeros(3, N_FEATURES, dtype=DOUBLE)
    features[:, FEATURES.index("stage")] = torch.tensor([0.0, 1.0, 1.0], dtype=DOUBLE)
    labels = mt.TeacherLabelBatch.from_rows(
        [
            {
                "starhorse": _label("starhorse", 0.2, 0.1),
                "apokasc3": _label("apokasc3", 0.25, 0.1),
            },
            {"apokasc3": _label("apokasc3", 0.9, 0.1)},
            {"apokasc3": _label("apokasc3", 1.1, 0.1)},
        ],
        teacher_names=head.teacher_names,
    )
    result = head.loss(torch.randn(3, 4, dtype=DOUBLE), features, labels)
    assert result.observation_count == 2 and result.label_count == 4
    assert torch.isfinite(result.mean)
    bad = features.clone()
    bad[0, FEATURES.index("stage")] = 2.0
    with pytest.raises(ValueError):
        head.loss(torch.randn(3, 4, dtype=DOUBLE), bad, labels)
