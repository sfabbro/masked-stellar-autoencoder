"""Contract C6: metric definitions, schema round-trip, strict I/O and bootstrap.

Every test here is deterministic (fixed seeds) and asserts either an analytic
value (Gaussian NLL/CRPS, interval score, region volume) or a statistical
property with an explicit tolerance (PIT uniformity, grouped-bootstrap width,
conformal coverage lives in ``tests/test_conformal_ranks.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from masked_stellar_autoencoder.reporting import calibration, io, metrics, schema


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------
def test_point_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.5, 1.5, 3.0, 4.5, 4.0])
    result = metrics.point_metrics(y, pred, outlier_k=3.0)
    residual = pred - y
    assert result["n"] == 5
    assert result["n_eff"] == 5.0
    assert result["mae"] == pytest.approx(np.mean(np.abs(residual)))
    assert result["rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert result["bias"] == pytest.approx(np.mean(residual))
    assert result["nmad"] == pytest.approx(
        1.4826 * np.median(np.abs(residual - np.median(residual)))
    )
    assert result["r2"] == pytest.approx(
        1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)
    )
    assert result["atten_slope"] == pytest.approx(
        np.cov(y, pred, ddof=0)[0, 1] / np.var(pred)
    )
    assert result["tail_count"] == 1  # ceil(0.1 * 5) = 1
    assert result["tail_mae"] == pytest.approx(np.max(np.abs(residual)))
    sd = np.std(y, ddof=1)
    assert result["outlier_threshold"] == pytest.approx(3.0 * sd)
    assert result["outlier_frac"] == pytest.approx(np.mean(np.abs(residual) > 3.0 * sd))
    assert result["null_reason"] == ""
    assert result["descriptive_only"] is True


def test_point_metrics_perfect_and_empty():
    y = np.linspace(-2.0, 2.0, 25)
    perfect = metrics.point_metrics(y, y.copy())
    assert perfect["mae"] == 0.0
    assert perfect["rmse"] == 0.0
    assert perfect["bias"] == 0.0
    assert perfect["nmad"] == 0.0
    assert perfect["r2"] == 1.0
    assert perfect["atten_slope"] == pytest.approx(1.0)
    assert perfect["descriptive_only"] is False

    empty = metrics.point_metrics([], [])
    assert empty["n"] == 0
    assert empty["mae"] is None
    assert empty["null_reason"] == schema.NullReason.SPLIT_EMPTY.value

    # Constant target: R^2 / correlation / slope are undefined, not zero.
    constant = metrics.point_metrics(np.ones(30), np.ones(30) + 0.1)
    assert constant["mae"] == pytest.approx(0.1)
    assert constant["r2"] is None
    assert constant["null_reason"] == schema.NullReason.NOT_APPLICABLE.value


# ---------------------------------------------------------------------------
# proper scores
# ---------------------------------------------------------------------------
def test_gaussian_nll_and_crps_have_known_values():
    sigma = 2.0
    exact_nll = 0.5 * math.log(2.0 * math.pi * sigma**2) + 0.5 * (3.0 / sigma) ** 2
    result = metrics.nll_gaussian(np.array([3.0]), 0.0, sigma, density_kind="intrinsic")
    assert result["nll"] == pytest.approx(exact_nll)
    assert result["density_kind"] == "intrinsic"

    # CRPS(N(0,1), y=0) = 1/sqrt(pi) * (2*phi(0) - 1) ... = 2 phi(0) - 1/sqrt(pi)
    exact_crps = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    crps = metrics.crps_gaussian(
        np.array([0.0]), 0.0, 1.0, density_kind="observed_label"
    )
    assert crps["crps"] == pytest.approx(exact_crps)
    assert crps["density_kind"] == "observed_label"


def test_density_kind_is_mandatory_everywhere():
    with pytest.raises(ValueError, match="density_kind"):
        metrics.nll_from_log_prob(np.zeros(5), density_kind="observed")
    with pytest.raises(ValueError, match="density_kind"):
        metrics.crps_from_samples(np.zeros((2, 4)), np.zeros(2), density_kind="truth")
    with pytest.raises(ValueError, match="loss_denominator"):
        metrics.nll_from_log_prob(
            np.zeros(5), density_kind="intrinsic", loss_denominator="per_star"
        )


def test_nll_from_log_prob_value_and_failure_modes():
    log_prob = np.array([-1.0, -2.0, -3.0, -4.0])
    result = metrics.nll_from_log_prob(log_prob, density_kind="intrinsic")
    assert result["nll"] == pytest.approx(2.5)
    assert result["n"] == 4
    assert result["loss_denominator"] == "per_row"
    assert result["null_reason"] == ""

    # A non-finite density is a failed arm, not a row to silently drop.
    with pytest.raises(ValueError, match="non-finite"):
        metrics.nll_from_log_prob(np.array([-1.0, np.nan]), density_kind="intrinsic")
    empty = metrics.nll_from_log_prob([], density_kind="intrinsic")
    assert empty["nll"] is None
    assert empty["null_reason"] == schema.NullReason.SPLIT_EMPTY.value


def test_crps_from_samples_matches_analytic_gaussian():
    rng = np.random.default_rng(101)
    y = rng.normal(size=40)
    samples = rng.normal(size=(40, 20000))  # N(0,1) predictive
    sample_crps = metrics.crps_from_samples(samples, y, density_kind="observed_label")
    analytic = metrics.crps_gaussian(y, 0.0, 1.0, density_kind="observed_label")
    assert sample_crps["crps"] == pytest.approx(analytic["crps"], abs=0.02)
    assert sample_crps["n_samples"] == 20000
    assert sample_crps["density_kind"] == "observed_label"

    # Identical samples equal to y give exactly zero CRPS.
    identical = np.repeat(y[:, None], 8, axis=1)
    assert metrics.crps_from_samples(identical, y, density_kind="intrinsic")[
        "crps"
    ] == pytest.approx(0.0, abs=1e-12)
    # One sample cannot define CRPS; never silently return a proxy.
    with pytest.raises(ValueError, match="at least 2 samples"):
        metrics.crps_from_samples(
            np.ones((3, 1)), np.zeros(3), density_kind="intrinsic"
        )


def test_pit_and_rank_histogram_detect_miscalibration():
    rng = np.random.default_rng(103)
    y = rng.normal(size=300)
    n_samples = 200
    calibrated = rng.normal(size=(300, n_samples))
    pit_cal = metrics.pit_from_samples(calibrated, y)
    diagnostics = metrics.pit_diagnostics(pit_cal)
    assert 0.0 < diagnostics["pit_ks"] < 0.15
    assert diagnostics["pit_ks_pvalue"] > 0.05
    assert diagnostics["pit_mean"] == pytest.approx(0.5, abs=0.05)
    assert diagnostics["null_reason"] == ""

    # Wrong mean and far too narrow: the PIT piles up on {0, 1}.
    miscalibrated = 0.1 * rng.normal(size=(300, n_samples))
    pit_bad = metrics.pit_from_samples(miscalibrated, y)
    bad = metrics.pit_diagnostics(pit_bad)
    assert bad["pit_ks"] > 0.3
    assert bad["pit_ks_pvalue"] < 1e-6

    # Duplicate ranks are broken reproducibly.
    tie_samples = np.zeros((4, 10))
    ties = metrics.pit_from_samples(tie_samples, np.zeros(4), randomized=True, seed=9)
    ties_again = metrics.pit_from_samples(
        tie_samples, np.zeros(4), randomized=True, seed=9
    )
    np.testing.assert_array_equal(ties, ties_again)
    np.testing.assert_allclose(
        metrics.pit_from_samples(tie_samples, np.zeros(4)), np.full(4, 0.5)
    )

    histogram = metrics.rank_histogram(calibrated, y)
    assert len(histogram["counts"]) == n_samples + 1
    assert sum(histogram["counts"]) == 300
    assert histogram["chi2_pvalue"] > 0.01
    bad_hist = metrics.rank_histogram(miscalibrated, y)
    assert bad_hist["chi2"] > histogram["chi2"]

    empty = metrics.pit_diagnostics([])
    assert empty["pit_ks"] is None
    assert empty["null_reason"] == schema.NullReason.SPLIT_EMPTY.value


# ---------------------------------------------------------------------------
# energy / variogram
# ---------------------------------------------------------------------------
def test_energy_score_identical_samples_is_zero():
    y = np.array([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    samples = np.repeat(y[:, None, :], 16, axis=1)
    assert metrics.energy_score(samples, y)["energy_score"] == 0.0

    rng = np.random.default_rng(107)
    dispersed = rng.normal(size=(2, 16, 3)) + y[:, None, :]
    assert metrics.energy_score(dispersed, y)["energy_score"] > 0.0
    with pytest.raises(ValueError, match="beta"):
        metrics.energy_score(samples, y, beta=2.0)
    with pytest.raises(ValueError, match="dimension"):
        metrics.energy_score(np.zeros((2, 4, 2)), y)


def test_variogram_score_identical_samples_is_zero():
    y = np.array([[0.0, 1.0, 2.0]])
    samples = np.repeat(y[:, None, :], 16, axis=1)
    # The sample mean of S identical pair distances can round by one ulp.
    assert metrics.variogram_score(samples, y)["variogram_score"] == pytest.approx(
        0.0, abs=1e-12
    )

    rng = np.random.default_rng(109)
    dispersed = rng.normal(size=(1, 32, 3)) + y[:, None, :]
    assert metrics.variogram_score(dispersed, y)["variogram_score"] > 0.0
    result = metrics.variogram_score(dispersed, y, weights=np.ones((3, 3)))
    assert result["n_pairs"] == 3
    with pytest.raises(ValueError, match="2 coordinates"):
        metrics.variogram_score(np.zeros((1, 4, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="p must be"):
        metrics.variogram_score(dispersed, y, p=1.5)


# ---------------------------------------------------------------------------
# intervals and coverage
# ---------------------------------------------------------------------------
def test_interval_score_closed_form():
    y = np.array([0.0, 2.0])
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    result = metrics.interval_score(y, lo, hi, alpha=0.1)
    # Row 1: width 2, no penalty.  Row 2: width 2 + (2/0.1)*(2-1) = 22.
    assert result["interval_score"] == pytest.approx((2.0 + 22.0) / 2.0)
    assert result["width"] == pytest.approx(2.0)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["n_invalid"] == 0

    # Inverted bounds are excluded and counted, never silently averaged in.
    inverted = metrics.interval_score(
        np.array([0.0, 0.0]), np.array([2.0, -1.0]), np.array([1.0, 1.0]), alpha=0.1
    )
    assert inverted["n"] == 1
    assert inverted["n_invalid"] == 1
    with pytest.raises(ValueError, match="alpha"):
        metrics.interval_score(y, lo, hi, alpha=1.0)


def test_marginal_and_mondrian_coverage():
    y = np.array([0.0, 0.0, 0.0, 10.0])
    lo = np.array([-1.0, -1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0, 1.0])
    marginal = metrics.coverage(y, lo, hi)
    assert marginal["coverage"] == pytest.approx(0.75)
    assert marginal["n"] == 4
    assert marginal["n_inside"] == 3

    rows = metrics.mondrian_coverage(y, lo, hi, np.array(["a", "a", "a", "b"]))
    by_stratum = {row["stratum"]: row for row in rows}
    assert by_stratum["a"]["coverage"] == 1.0
    assert by_stratum["b"]["coverage"] == 0.0
    assert by_stratum["a"]["descriptive_only"] is True  # n=3 < 20
    with pytest.raises(ValueError, match="strata"):
        metrics.mondrian_coverage(y, lo, hi, np.array(["a"]))


def test_coverage_calibration_slope_tracks_width():
    width = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    covered = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    result = metrics.coverage_calibration_slope(covered, width)
    assert result["calib_slope"] > 0.0
    constant = metrics.coverage_calibration_slope(covered, np.full(6, 2.0))
    assert constant["calib_slope"] is None
    assert constant["null_reason"] == schema.NullReason.NOT_APPLICABLE.value


# ---------------------------------------------------------------------------
# grouped bootstrap
# ---------------------------------------------------------------------------
def _duplicated_design(seed: int = 5):
    rng = np.random.default_rng(seed)
    n_sources, copies = 100, 10
    sources = np.repeat(np.arange(n_sources), copies)
    cluster_effect = rng.normal(size=n_sources)
    delta = cluster_effect[sources]
    return delta, sources


def test_grouped_bootstrap_is_wider_than_row_bootstrap_on_duplicated_ids():
    delta, sources = _duplicated_design()
    grouped = metrics.grouped_paired_bootstrap(
        delta, groups=sources, n_boot=2000, seed=5
    )
    naive = metrics.grouped_paired_bootstrap(
        delta, groups=np.arange(delta.size), n_boot=2000, seed=5
    )
    assert grouped["bootstrap_kind"] == "star"
    assert grouped["n_groups"] == [100]
    assert naive["n_groups"] == [1000]
    assert grouped["point"] == pytest.approx(naive["point"])
    naive_width = naive["ci_hi"] - naive["ci_lo"]
    grouped_width = grouped["ci_hi"] - grouped["ci_lo"]
    # With 10 identical copies per source, independent-row resampling ignores
    # the cluster signal: sqrt(10) ~ 3.2x too narrow in theory.
    assert grouped_width > 2.0 * naive_width


def test_two_way_bootstrap_contract_and_permutation_invariance():
    delta, sources = _duplicated_design(seed=8)
    rng = np.random.default_rng(11)
    blocks = rng.integers(0, 12, size=delta.size)
    first = metrics.grouped_paired_bootstrap(
        delta, groups=(sources, blocks), n_boot=200, seed=3
    )
    assert first["bootstrap_kind"] == "two_way"
    assert len(first["n_groups"]) == 2
    assert first["n_groups"][0] == 100

    order = rng.permutation(delta.size)
    second = metrics.grouped_paired_bootstrap(
        delta[order], groups=(sources[order], blocks[order]), n_boot=200, seed=3
    )
    assert second["ci_lo"] == pytest.approx(first["ci_lo"], abs=1e-12)
    assert second["ci_hi"] == pytest.approx(first["ci_hi"], abs=1e-12)

    with pytest.raises(ValueError, match="needs two grouping"):
        metrics.grouped_paired_bootstrap(
            delta, groups=sources, bootstrap_kind="two_way"
        )
    with pytest.raises(ValueError, match="exactly one grouping"):
        metrics.grouped_paired_bootstrap(
            delta, groups=(sources, blocks), bootstrap_kind="star"
        )
    with pytest.raises(ValueError, match="align with delta"):
        metrics.grouped_paired_bootstrap(delta, groups=sources[:-1])


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------
def test_multiplicity_helpers():
    pvalues = [0.01, 0.02, 0.2, 0.5]
    bonferroni = metrics.bonferroni(pvalues, alpha=0.05, n_tests=16)
    assert bonferroni["threshold"] == pytest.approx(0.05 / 16)
    assert bonferroni["adjusted"] == pytest.approx([0.16, 0.32, 1.0, 1.0])
    assert bonferroni["reject"] == [False, False, False, False]
    assert metrics.bonferroni([0.001], alpha=0.05)["reject"] == [True]

    bh = metrics.benjamini_hochberg(pvalues, q=0.05)
    assert bh["adjusted"] == pytest.approx([0.04, 0.04, 0.26666666666666666, 0.5])
    assert bh["reject"] == [True, True, False, False]

    assert metrics.multiplicity_note("bonferroni", 16) == "bonferroni_k=16"
    assert metrics.multiplicity_note("bh_fdr", 16) == "bh_fdr_q=0.05"
    assert metrics.multiplicity_note("none", 16) == "none"


# ---------------------------------------------------------------------------
# schema + strict I/O
# ---------------------------------------------------------------------------
def _metrics_row(**overrides):
    values = dict(
        run_id="run-1",
        model_id="mvgaussian",
        label="teff",
        unit="K",
        stratum="bulk",
        n=2046,
        n_eff=2046.0,
        mae=68.7,
        rmse=90.0,
        bias=-1.2,
        nmad=64.0,
        r2=0.98,
        atten_slope=0.99,
        outlier_frac=0.004,
        tail_mae=180.0,
        evidence_status="producer_only",
        bootstrap_kind="star",
        bootstrap_reps=1000,
        multiplicity="bonferroni_k=16",
        null_reason="not_computed",
        notes="unit test row",
    )
    values.update(overrides)
    return io.build_row("metrics", values)


def test_schema_round_trip_with_null_reason(tmp_path):
    row = _metrics_row()
    assert row["null_reason"] == schema.NullReason.NOT_COMPUTED.value
    assert row["crps"] is None
    assert row["verified"] is False
    path = io.write_csv(tmp_path / "metrics.csv", "metrics", [row])
    text = path.read_text()
    assert "nan" not in text.lower()
    assert "inf" not in text.replace("infer", "").lower()
    back = io.read_csv(path, "metrics")
    assert back == [row]

    # Every table's required columns exist and include the null contract.
    for table, columns in schema.ROW_SCHEMAS.items():
        assert "null_reason" in columns
        assert "evidence_status" in columns
    assert set(schema.ROW_SCHEMAS) == {"metrics", "joint", "distance", "age"}


def test_null_reason_is_required_and_never_stale():
    row = _metrics_row()
    without_reason = dict(row)
    without_reason["null_reason"] = None
    with pytest.raises(io.NullReasonError, match="require a null_reason"):
        io.validate_row("metrics", without_reason)

    stale = dict(
        _metrics_row(),
        crps=1.0,
        nll=1.0,
        nll_space="intrinsic",
        pit_ks=0.5,
        pit_cvm=0.1,
        pit_local_w1=0.01,
        cov_68=0.7,
        cov_90=0.9,
        cov_95=0.95,
        cov_nominal=0.9,
        width_90=1.0,
        interval_score_90=1.0,
        calib_slope=0.1,
        ci_lo=-1.0,
        ci_hi=1.0,
        null_reason="not_computed",
    )
    with pytest.raises(io.NullReasonError, match="no metric cell is null"):
        io.validate_row("metrics", stale)

    with pytest.raises(io.ContractError, match="not in the enum"):
        io.validate_row("metrics", _metrics_row(null_reason="because"))


def test_bare_nan_and_infinity_are_never_written(tmp_path):
    row = _metrics_row()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(io.NonFiniteValueError, match="non-finite"):
            io.write_csv(tmp_path / "bad.csv", "metrics", [dict(row, mae=bad)])
    with pytest.raises(io.NonFiniteValueError, match="non-finite"):
        io.write_json(tmp_path / "bad.json", {"value": float("nan")})

    # build_row sanitises a non-finite metric *only* into a documented null.
    sanitised = io.build_row(
        "metrics", dict(row, mae=float("nan"), null_reason="not_computed")
    )
    assert sanitised["mae"] is None
    with pytest.raises(io.NullReasonError):
        io.build_row("metrics", dict(row, mae=float("nan"), null_reason=None))


def test_cross_field_contract_rules():
    row = _metrics_row()
    with pytest.raises(io.ContractError, match="required columns"):
        io.validate_row(
            "metrics", {key: value for key, value in row.items() if key != "run_id"}
        )
    with pytest.raises(io.ContractError, match="unknown columns"):
        io.validate_row("metrics", dict(row, surprise=1.0))
    with pytest.raises(io.ContractError, match="nll_space"):
        io.validate_row("metrics", dict(row, nll=1.0, nll_space=None))
    with pytest.raises(io.ContractError, match="failure_id"):
        io.validate_row("metrics", dict(row, evidence_status="strict_replay_failed"))
    with pytest.raises(io.ContractError, match="contradicts"):
        io.validate_row("metrics", dict(row, verified=True))
    with pytest.raises(io.ContractError, match="valid_range"):
        io.validate_row("metrics", dict(row, cov_90=1.5))
    # A failed row may carry values, as long as its provenance is explicit.
    failed = io.build_row(
        "metrics",
        dict(
            row,
            evidence_status="strict_replay_failed",
            failure_id="F12",
            null_reason="failed_verdict",
        ),
    )
    assert failed["failure_id"] == "F12"
    assert failed["verified"] is False


def test_schema_document_is_versioned(tmp_path):
    path = io.write_schema_json(
        tmp_path / "schema.json", provenance={"source_sha": "a" * 64}
    )
    document = io.read_json(path)
    assert document["schema_version"] == schema.SCHEMA_VERSION
    assert document["contract"] == "C6"
    assert set(document["tables"]) == {"metrics", "joint", "distance", "age"}
    assert "failed_verdict" in document["null_reason_enum"]
    assert document["provenance"]["source_sha"] == "a" * 64
    for table in ("metrics", "joint", "distance", "age"):
        for field in document["tables"][table]["columns"]:
            if field["name"] == "null_reason":
                assert set(field["enum"]) == set(schema.NULL_REASONS)


# ---------------------------------------------------------------------------
# certificate and whole-directory contract
# ---------------------------------------------------------------------------
def _certificate():
    return schema.build_certificate(
        **{
            "run_id": "run-1",
            "source_sha": "a" * 64,
            "library_commit": "0e77eee",
            "checkpoint_sha256": "b" * 64,
            "cohort_sha256": "c" * 64,
            "scaler_sha256": "d" * 64,
            "eval": {
                "dtype": "float64",
                "device": "cpu",
                "batch_size": 128,
                "torch": "2.9.0",
            },
            "density_kind": "observed_label",
            "loss_denominator": "per_row",
            "partial_label_policy": "gaussian_exact_marginal",
            "n_observed_per_label": {"teff": 2046, "logg": 2046, "fe_h": 2046},
            "conformal": {
                "method": "split_rank",
                "alpha": 0.1,
                "n_cal": 2046,
                "order_stat_index": 1843,
                "finite_sample_exact": True,
                "joint_method": "rank_score_mahalanobis",
                "strata": ["feh_lt_minus2", "dwarf", "giant"],
            },
            "density_integration": {
                "method": "gh",
                "nodes": 32,
                "seed": 0,
                "buffer_sha256": None,
                "convergence": {"gh16": 0.1, "gh32": 0.01, "rel_change": 0.001},
            },
            "n": {"fit": 19299, "tune": 2118, "cal": 2046, "test": 97695},
        }
    )


def test_certificate_requires_exact_conformal_provenance():
    certificate = _certificate()
    schema.validate_certificate(certificate)
    broken = dict(certificate)
    broken["conformal"] = dict(certificate["conformal"], finite_sample_exact=False)
    with pytest.raises(ValueError, match="finite_sample_exact"):
        schema.validate_certificate(broken)
    missing = dict(certificate)
    missing.pop("jacobian")
    with pytest.raises(ValueError, match="missing required keys"):
        schema.validate_certificate(missing)
    with pytest.raises(ValueError, match="unknown keys"):
        schema.build_certificate(bogus=1)


def test_failed_verdicts_are_never_erased(tmp_path):
    certificate = _certificate()
    schema.record_verdict(
        certificate,
        "density",
        "fail",
        evidence_status="verified",
        detail="float32 replay divergence",
        failure_id="F2",
    )
    schema.record_verdict(certificate, "density", "pass", evidence_status="verified")
    assert [
        entry["status"] for entry in certificate["verdicts"]["density"]["history"]
    ] == ["fail"]
    assert certificate["verdicts"]["density"]["current"]["status"] == "pass"
    assert len(certificate["failed_verdicts"]) == 1
    schema.validate_certificate(certificate)

    path = io.write_certificate(tmp_path / "prediction-certificate.json", certificate)
    io.update_certificate_verdicts(path, {"cp": {"status": "fail", "failure_id": "F4"}})
    reread = io.read_certificate(path)
    assert {entry["failure_id"] for entry in reread["failed_verdicts"]} == {"F2", "F4"}

    with pytest.raises(ValueError, match="failure_id"):
        schema.record_verdict(certificate, "point", "fail")


def test_contract_directory_emits_all_four_tables(tmp_path):
    metrics_row = _metrics_row()
    joint_row = io.build_row(
        "joint",
        dict(
            run_id="run-1",
            model_id="mvgaussian",
            label_set="teff+logg+fe_h",
            stratum="bulk",
            n=2046,
            joint_cov_90=0.91,
            joint_cov_90_bonferroni=0.92,
            joint_cov_90_rank_score=0.90,
            region_volume_native=1.5,
            levelset_method="analytic_ellipsoid",
            energy_score=0.8,
            variogram_score=0.4,
            replay_status="producer_only",
            evidence_status="producer_only",
            null_reason="not_computed",
        ),
    )
    distance_row = io.build_row(
        "distance",
        dict(
            run_id="run-1",
            kind="mas",
            stratum="bulk",
            n=1000,
            bias_mas=0.01,
            mae_mas=0.5,
            p68_mas=0.3,
            cov90_mas=0.9,
            frac_ci_empty=0.0,
            frac_ci_upper_unbounded=0.0,
            ci_kind="predictive",
            ref_type="HIP",
            evidence_status="producer_only",
            null_reason="not_computed",
        ),
    )
    age_row = io.build_row(
        "age",
        dict(
            run_id="run-1",
            stratum="bulk",
            n=500,
            bias_Gyr=0.1,
            mae_Gyr=1.2,
            bias_logGyr=0.01,
            mae_logGyr=0.1,
            cov68=0.68,
            cov90=0.9,
            width90_log=0.4,
            gt10Gyr_n=120,
            gt10Gyr_bias_log=0.02,
            evidence_status="producer_only",
            null_reason="not_computed",
        ),
    )
    written = io.write_contract_directory(
        tmp_path / "run-1",
        metrics=[metrics_row],
        joint=[joint_row],
        distance=[distance_row],
        age=[age_row],
        certificate=_certificate(),
        schema_provenance={"source_sha": "a" * 64},
    )
    assert set(written) == {
        "metrics",
        "joint",
        "distance",
        "age",
        "certificate",
        "schema",
    }
    assert io.read_csv(written["metrics"], "metrics")[0] == metrics_row
    assert io.read_csv(written["joint"], "joint")[0] == joint_row
    assert io.read_csv(written["distance"], "distance")[0] == distance_row
    assert io.read_csv(written["age"], "age")[0] == age_row
    assert io.read_certificate(written["certificate"])["run_id"] == "run-1"


def test_region_helpers_are_exposed_from_calibration():
    # Guard the integration point used by the metrics/joint writers.
    assert calibration.finite_sample_rank(2046, 0.1) == 1843
    assert calibration.ellipsoid_region_volume(np.eye(3), 1.0) > 0.0
