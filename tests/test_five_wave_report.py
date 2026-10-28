"""Synthetic saved-array contracts; no native model or catalogue is executed."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "five_report", Path("scripts/five_wave_report.py")
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture():
    target = np.array(
        [
            [4500.0, 2.0, -2.8, 0.3, 10.0, 2.0],
            [5500.0, 4.0, -0.2, 0.1, 5.0, 1.0],
            [4700.0, 2.5, -2.2, 0.2, 8.0, 3.0],
            [6000.0, 4.2, 0.1, 0.0, 2.0, 0.5],
        ]
    )
    point = target + np.array([20.0, 0.02, 0.05, 0.01, 0.2, 0.1])
    return dict(
        source_id=np.array([10, 20, 30, 40], dtype="i8"),
        target_native=target,
        point_native=point,
        observed=np.ones((4, 6), bool),
        label_source=np.full((4, 6), "synthetic"),
        xp_status=np.array(
            ["complete_xp", "complete_xp", "official_no_xp", "official_no_xp"]
        ),
        evaluation_mode=np.full(4, "natural"),
        quantile_levels=np.array([0.05, 0.5, 0.95]),
        quantiles_native=np.stack([point - 0.2, point, point + 0.2], 1),
    )


def test_quantile_baseline_has_no_invented_pit_or_joint_distribution(tmp_path):
    data = fixture()
    data["intrinsic_pit"] = np.full((4, 6), 0.5)
    with pytest.raises(ValueError, match="unique predictive CDF"):
        report.validate_predictions(data, "quantile3")
    with pytest.raises(ValueError, match="no normalized joint law"):
        report.corner_figures({}, tmp_path, "synthetic", "quantile3", {})


def test_source_availability_and_masked_stress_population_are_distinct():
    data = fixture()
    data["evaluation_mode"][2] = "xp_masked_ablation"
    # Fixed-width input strings must be explicitly wide enough in real inputs.
    data["evaluation_mode"] = np.array(
        ["natural", "natural", "xp_masked_ablation", "natural"]
    )
    with pytest.raises(ValueError, match="only a paired XP-star"):
        report.validate_predictions(data, "gaussian")
    data["xp_status"][2] = "complete_xp"
    checked = report.validate_predictions(data, "gaussian")
    assert report.populations(checked)["natural_no_xp"].sum() == 1
    assert report.populations(checked)["xp_masked_ablation"].sum() == 1
    no_evidence = {k: v for k, v in data.items() if k != "xp_status"}
    with pytest.raises(ValueError, match="input missingness is not proof"):
        report.validate_predictions(no_evidence, "gaussian")


def test_duplicate_ids_allowed_only_across_paired_modes():
    data = fixture()
    data["source_id"][1] = 10
    with pytest.raises(ValueError, match="Duplicate source"):
        report.validate_predictions(data, "gaussian")
    data["evaluation_mode"] = np.array(
        ["natural", "xp_masked_ablation", "natural", "natural"]
    )
    report.validate_predictions(data, "gaussian")


def test_sparse_reference_tail_and_quantile_metrics():
    data = fixture()
    data["observed"][:, 4:] = False
    data["target_native"][:, 4:] = np.nan
    checked = report.validate_predictions(data, "quantile3")
    rows = report.metric_rows(checked, "synthetic")
    row = next(
        r
        for r in rows
        if r["population"] == "natural_no_xp"
        and r["scope"] == "metal_poor"
        and r["target"] == "fe_h"
        and r["label_source"] == "all_sources"
    )
    assert row["count"] == 1
    assert row["mae"] == pytest.approx(0.05)
    assert row["bias"] == pytest.approx(0.05)
    assert row["predictive_90_coverage"] == 1
    age = next(
        r
        for r in rows
        if r["population"] == "natural_no_xp"
        and r["scope"] == "metal_poor"
        and r["target"] == "age"
    )
    assert age["count"] == 0 and age["mae"] is None


def test_observed_law_excludes_unknown_errors_without_mutating_reference_mask():
    data = fixture()
    data["intrinsic_pit"] = np.full((4, 6), 0.5)
    data["observed_pit"] = np.full((4, 6), 0.6)
    data["error_observed"] = np.zeros((4, 6), bool)
    data["error_observed"][:, :3] = True
    data["observed_crps_native"] = np.ones((4, 6))
    data["intrinsic_crps_native"] = np.ones((4, 6)) * 0.5
    data["observed_energy_full6_fit_standardized"] = np.ones(4)
    checked = report.validate_predictions(data, "gaussian")
    before = checked["observed"].copy()
    rows = report.distribution_score_rows(checked, "synthetic")
    assert np.array_equal(before, checked["observed"])
    age = next(
        r
        for r in rows
        if r["metric"] == "observed_crps_native"
        and r["target"] == "age"
        and r["population"] == "natural_xp"
        and r["scope"] == "global"
    )
    assert age["count"] == 0 and age["mean"] is None
    assert not checked["pit_observed_valid"][:, 4:].any()
    intrinsic = next(
        r
        for r in rows
        if r["metric"] == "intrinsic_crps_native"
        and r["target"] == "age"
        and r["population"] == "natural_xp"
        and r["scope"] == "global"
    )
    assert intrinsic["count"] == 2 and intrinsic["mean"] == 0.5


def test_corner_mass_is_never_renormalized_to_plotting_window():
    counts = np.array([[4.0, 0.0], [0.0, 0.0]])
    assert report.histogram_mass_levels(counts, 10) == [None, None]
    assert report.histogram_mass_levels(counts, 4) == [4.0, 4.0]
    with pytest.raises(ValueError, match="accounting"):
        report.histogram_mass_levels(counts, 3)


def test_andrae_same_object_nontraining_scope_and_metallicity_proxy(tmp_path):
    data = fixture()
    data["andrae_prediction_native"] = data["target_native"][:, :3].copy()
    data["andrae_matched"] = np.array([True, True, False, False])
    data["andrae_training_membership_known"] = np.ones(4, bool)
    data["andrae_in_training"] = np.array([True, False, False, False])
    checked = report.validate_predictions(data, "gaussian")
    result = report.andrae_comparison(checked, tmp_path, "synthetic")
    assert result["paired_rows"] == 2 and result["known_nontraining_rows"] == 1
    assert all(r["mh_vs_feh_proxy"] for r in result["rows"] if r["target"] == "fe_h")
    checked["andrae_matched"][2] = True
    with pytest.raises(ValueError, match="certified no-XP"):
        report.andrae_comparison(checked, tmp_path, "synthetic")


def test_end_to_end_quantile_six_label_panels_without_fabricated_pit(tmp_path):
    summary = report.make_report(
        fixture(),
        tmp_path,
        run_id="synthetic",
        head_kind="quantile3",
        metadata={"point_functional": "intrinsic conditional median"},
    )
    assert summary["pit_status"] == "unsupported_point_or_quantile_baseline"
    assert summary["corners"]["status"] == "unsupported"
    assert len(summary["figures"]) == 3
    for stem in summary["figures"]:
        assert (tmp_path / (stem + ".png")).is_file() and (
            tmp_path / (stem + ".pdf")
        ).is_file()
    assert (
        "no PIT, PDF or full corner plot is invented"
        in (tmp_path / "index.html").read_text()
    )


def test_saved_actual_draw_corner_preserves_full_samples_and_fixed_axes(tmp_path):
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(1, 256, 6))
    samples[:, :, 4:] = np.exp(samples[:, :, 4:])
    reference = np.array([[5000.0, 3.0, -2.5, 0.2, np.nan, np.nan]])
    corners = dict(
        source_id=np.array([10]),
        samples_native=samples,
        target_native=reference,
        observed=np.array([[True, True, True, True, False, False]]),
        plot_limits=np.broadcast_to(np.array([-3.0, 3.0]), (1, 6, 2)).copy(),
    )
    summary = report.corner_figures(
        corners, tmp_path, "synthetic", "factorized_nsf", {}
    )
    assert summary["common_axes_source_frozen"]
    assert len(summary["rows"]) == 1
    with np.load(tmp_path / "corners/10-samples.npz") as saved:
        np.testing.assert_array_equal(saved["samples_native"], samples[0])
        assert not saved["observed"][4:].any()
        assert np.isnan(saved["target_native"][4:]).all()


def test_ancestry_slices_and_native_nll_sign():
    data = fixture()
    data["ancestor_feh_supervised_fit_exposed"] = np.array([True, False, False, False])
    data["ancestor_150m_reconstruction_exposed"] = np.array([True, True, True, False])
    data["intrinsic_log_prob_native"] = np.full((4, 6), -2.0)
    checked = report.validate_predictions(data, "gaussian")
    slices = report.populations(checked, ancestry=True)
    assert slices["natural_xp__ancestor_feh_supervised_fit_exposed__exposed"].sum() == 1
    assert (
        slices["natural_xp__ancestor_150m_reconstruction_exposed__exposed"].sum() == 2
    )
    score = next(
        r
        for r in report.distribution_score_rows(checked, "synthetic")
        if r["population"] == "natural_xp"
        and r["scope"] == "global"
        and r["target"] == "teff"
    )
    assert score["metric"] == "intrinsic_nll_native" and score["mean"] == 2.0


def test_pit_panels_explain_unknown_errors_without_inventing_observed_cdf(
    tmp_path, monkeypatch
):
    captured = {}

    def capture(fig, stem):
        captured[stem.name] = [
            [text.get_text() for text in ax.texts] for ax in fig.axes
        ]
        import matplotlib.pyplot as plt

        plt.close(fig)

    monkeypatch.setattr(report, "_save", capture)
    mask = np.ones((3, 6), bool)
    valid = mask.copy()
    valid[:, 4:] = False
    data = dict(
        evaluation_mode=np.array(["natural"] * 3),
        xp_status=np.array(["xp_present"] * 3),
        observed=mask,
        target_native=np.ones((3, 6)),
        pit_observed=np.full((3, 6), 0.5),
        pit_observed_valid=valid,
    )
    report.plot_pits(data, tmp_path, "synthetic")
    assert "quoted errors unknown" in captured["pit_observed-global"][4][0]
    assert "No eligible reference labels" in captured["pit_observed-metalpoor"][4][0]
