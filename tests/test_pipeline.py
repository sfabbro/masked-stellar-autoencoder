"""Pipeline: Jacobian, masked covariance NLL, and source_id splits."""

from pathlib import Path

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.pipeline.batch import (
    FeatureGroup,
    MemoryReader,
    StellarBatch,
    assemble,
)
from masked_stellar_autoencoder.pipeline.model import StellarNet
from masked_stellar_autoencoder.pipeline.normalize import (
    OrderScaler,
    clipped_snr,
    diagonal_covariance,
    fit_order_scaler,
    fit_vector_scaler,
    inverse_vector,
    pogson_factor,
    reservoir_matrix,
    transform_coefficients,
    transform_vector,
)
from masked_stellar_autoencoder.pipeline.registry import (
    Column,
    ColumnRegistry,
    gaia_dr3_registry,
)
from masked_stellar_autoencoder.pipeline.splits import (
    SourceLeak,
    four_way_split,
    healpix_holdout,
    leave_one_survey_out,
)
from masked_stellar_autoencoder.pipeline.steps import (
    artificial_xp_masks,
    build_optimizer,
    checkpoint_score,
    choose_joint,
    covariance_nll,
    coverage_after_inverse,
    cqr_intervals,
    diagonal_weighted_l1,
    eval_forward,
    finetune_loss,
    fit_cqr,
    gaussian_joint_coverage,
    joint_nll,
    metal_poor_mask,
    prepare,
    pretrain_epoch,
    pretrain_loss,
    rare_label_weights,
    regime_nll,
    selection_score,
    set_encoder_lr,
    structural_missing,
)
from masked_stellar_autoencoder.pipeline.train import load_config

ROOT = Path(__file__).resolve().parents[1]


def _tiny_registry() -> ColumnRegistry:
    columns = [Column(f"bp_{i}", "xp_bp", "coeff", f"bpe_{i}", "xp") for i in (1, 2)]
    columns += [Column(f"rp_{i}", "xp_rp", "coeff", f"rpe_{i}", "xp") for i in (1, 2)]
    columns += [
        Column("G", "photometry", "mag", None, "robust"),
        Column("EBV", "photometry", "ebv", None, "robust"),
        Column("PARALLAX", "astrometry", "mas", "e_parallax", "snr"),
        Column("fe_h", "labels", "dex", "e_fe_h", "robust"),
        Column("teff", "labels", "K", "e_teff", "log10"),
    ]
    return ColumnRegistry(tuple(columns), g_ref=8.5)


def test_jacobian_masked_nll_and_shared_registry():
    registry = _tiny_registry()
    values = np.array([[1.0, -0.5]])
    cov = np.array([[[2.0, 0.3], [0.3, 0.5]]])
    g_mag = np.array([12.0])
    scaler = OrderScaler(median=np.zeros(2), iqr=np.ones(2), g_ref=8.5)
    z, cov_z, jacobian = transform_coefficients(
        values, cov, g_mag, scaler, flux_state="raw"
    )

    eps = 1e-6
    numerical = np.zeros(2)
    for k in range(2):
        bumped = values.copy()
        bumped[0, k] += eps
        z_bumped, _, _ = transform_coefficients(
            bumped, None, g_mag, scaler, flux_state="raw"
        )
        numerical[k] = (z_bumped[0, k] - z[0, k]) / eps
    assert np.allclose(jacobian[0], numerical, rtol=1e-4, atol=1e-5)

    factor = pogson_factor(g_mag, scaler.g_ref)
    jac = np.diag(jacobian[0])
    assert np.allclose(cov_z[0], jac @ cov[0] @ jac.T)

    scaled = values / factor[:, None]
    cov_scaled = cov / factor[:, None, None] ** 2
    z_file, cov_file, _ = transform_coefficients(
        scaled, cov_scaled, g_mag, scaler, flux_state="already_scaled"
    )
    assert np.allclose(z, z_file)
    assert np.allclose(cov_z, cov_file)

    # A second application of the raw tag would move z. The flag is what blocks it.
    z_twice, _, _ = transform_coefficients(
        scaled, None, g_mag, scaler, flux_state="raw"
    )
    assert not np.allclose(z, z_twice)

    fit = fit_order_scaler(np.vstack([scaled, scaled * 0.5, scaled * 1.5]), g_ref=8.5)
    assert fit.iqr.shape == (2,)
    assert fit.flux_tag == scaler.flux_tag

    z_t = torch.tensor(z, dtype=torch.float32)
    cov_t = torch.tensor(cov_z, dtype=torch.float32)
    loss_mask = torch.tensor([[True, False]])
    ancillary = torch.zeros(1, 3)
    miss_anc = torch.zeros(1, 3, dtype=torch.bool)
    missing = torch.ones(1, 2, dtype=torch.bool)
    model = StellarNet(registry, width=8, ensemble_size=2)
    loss = pretrain_loss(
        model,
        z_bp=z_t,
        cov_bp=cov_t,
        input_missing_bp=missing,
        loss_bp=loss_mask,
        z_rp=z_t,
        cov_rp=cov_t,
        input_missing_rp=missing,
        loss_rp=loss_mask,
        ancillary=ancillary,
        miss_anc=miss_anc,
    )
    loss.backward()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.recon_bp.weight.grad is not None
    assert torch.isfinite(model.recon_bp.weight.grad).all()

    # The sliced covariance that entered the loss is the masked order's variance.
    sliced = covariance_nll(
        torch.zeros_like(z_t),
        z_t.detach(),
        cov_t,
        loss_mask,
    )
    assert torch.isfinite(sliced)

    other = StellarNet(registry, width=8, ensemble_size=2)
    other.load_state_dict(model.state_dict())
    assert torch.equal(other.recon_bp.weight, model.recon_bp.weight)

    full = gaia_dr3_registry(n_bp=10, n_rp=12)
    assert len(full.names("xp_bp")) == 10
    assert len(full.names("xp_rp")) == 12
    batch = assemble(
        {"bp_1": np.array([1.0]), "G": np.array([15.0])},
        full,
        source_id=np.array([1]),
        healpix=np.array([0]),
        survey=np.array(["gaia"]),
    )
    assert batch.groups["xp_bp"].values.shape == (1, 10)
    assert batch.groups["xp_bp"].missing[0, 1]
    assert batch.groups["photometry"].missing[0, full.index("photometry", "J")]

    sigma = np.array([[0.2, 0.4]])
    packed = diagonal_covariance(sigma)
    assert np.allclose(packed[0], np.diag([0.04, 0.16]))
    l1 = diagonal_weighted_l1(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.tensor([[True, False]]),
    )
    assert float(l1) == 0.0

    structural = structural_missing(2, np.array([1]), 1)
    observed = ~structural
    hidden_bp, _, loss_bp, _ = artificial_xp_masks(
        observed,
        observed,
        np.random.default_rng(0),
        span=2,
        p_span=0.0,
        p_drop_bp=0.0,
        p_drop_rp=0.0,
        p_drop_both=1.0,
    )
    assert hidden_bp[0, 0] and not hidden_bp[0, 1]
    assert loss_bp[0, 0] and not loss_bp[0, 1]


def test_leave_one_survey_refuses_train_source():
    source_id = np.array([7, 7, 8])
    survey = np.array(["apogee", "galah", "apogee"])
    with pytest.raises(SourceLeak, match="already in train"):
        leave_one_survey_out(source_id, survey, "galah", train_source_ids=np.array([7]))

    train_idx, test_idx = leave_one_survey_out(source_id, survey, "galah")
    assert 7 not in source_id[train_idx]
    assert set(source_id[test_idx].tolist()) == {7}
    assert set(source_id[train_idx].tolist()) == {8}

    masks = four_way_split(np.arange(8), seed=0)
    assigned = [set(np.arange(8)[mask].tolist()) for mask in masks.values()]
    for i, left in enumerate(assigned):
        for right in assigned[i + 1 :]:
            assert left.isdisjoint(right)
    assert sum(mask.sum() for mask in masks.values()) == 8

    healpix = np.array([10, 10, 20, 30, 40])
    holdout = healpix_holdout(healpix, fraction=0.25, seed=1)
    assert holdout[0] == holdout[1]
    assert 0 < holdout.sum() < len(holdout)
    with pytest.raises(ValueError, match="at least 2"):
        healpix_holdout(np.array([10, 10]), fraction=0.25, seed=0)


def test_finetune_cqr_and_joint_gate():
    registry = _tiny_registry()
    torch.manual_seed(0)
    model = StellarNet(registry, width=8, ensemble_size=2)
    batch = 6
    z = torch.randn(batch, 2)
    cov = torch.eye(2).expand(batch, 2, 2).contiguous()
    missing = torch.zeros(batch, 2, dtype=torch.bool)
    loss_mask = torch.ones(batch, 2, dtype=torch.bool)
    ancillary = torch.randn(batch, 3)
    miss_anc = torch.zeros(batch, 3, dtype=torch.bool)
    labels = torch.zeros(batch, 2)
    labels[:, 0] = 0
    labels[0, 0] = -3
    labels[:, 1] = 3.7
    weights = rare_label_weights(labels, column=0)
    assert weights[0, 0] > weights[1, 0]
    label_mask = torch.ones(batch, 2, dtype=torch.bool)
    logged = finetune_loss(
        model,
        z_bp=z,
        cov_bp=cov,
        input_missing_bp=missing,
        loss_bp=loss_mask,
        z_rp=z,
        cov_rp=cov,
        input_missing_rp=missing,
        loss_rp=loss_mask,
        ancillary=ancillary,
        miss_anc=miss_anc,
        labels=labels,
        label_mask=label_mask,
        label_weights=weights,
        recon_weight=0.1,
    )
    logged.total.backward()
    assert torch.isfinite(logged.total)
    assert logged.recon.ndim == 0
    assert logged.joint.ndim == 0
    assert model.joint_mean.weight.grad is not None
    assert float(model.joint_mean.weight.grad.abs().sum()) > 0
    poison = labels.clone()
    poison[0, 1] = float("nan")
    poisoned = finetune_loss(
        model,
        z_bp=z,
        cov_bp=cov,
        input_missing_bp=missing,
        loss_bp=loss_mask,
        z_rp=z,
        cov_rp=cov,
        input_missing_rp=missing,
        loss_rp=loss_mask,
        ancillary=ancillary,
        miss_anc=miss_anc,
        labels=poison,
        label_mask=label_mask,
        label_weights=weights,
        recon_weight=0.1,
    )
    assert torch.isfinite(poisoned.total)

    opt = build_optimizer(
        model, encoder_lr=1e-5, head_lr=1e-3, encoder_weight_decay=0.0
    )
    assert opt.param_groups[0]["weight_decay"] == 0.0
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert set_encoder_lr(opt, 0, freeze_epochs=2, encoder_lr=1e-5) == 0.0
    assert opt.param_groups[0]["lr"] == 0.0
    assert set_encoder_lr(opt, 2, freeze_epochs=2, encoder_lr=1e-5) == 1e-5
    assert opt.param_groups[0]["lr"] == 1e-5

    y = torch.linspace(-1, 1, 20).unsqueeze(1)
    lower = y - 0.5
    upper = y + 0.5
    y_hole = y.clone()
    y_hole[0, 0] = float("nan")
    predictors = fit_cqr(lower, upper, y_hole, alpha=0.32)
    lo, hi = cqr_intervals(predictors, lower, upper)
    assert torch.isfinite(lo).all() and torch.isfinite(hi).all()
    scaled_coverage = coverage_after_inverse(lo, hi, y, lambda t: t)
    physical = coverage_after_inverse(lo, hi, 3 * y, lambda t: 3 * t)
    assert physical == pytest.approx(scaled_coverage)
    assert (
        coverage_after_inverse(
            torch.tensor([[-1.0], [-1.0]]),
            torch.tensor([[1.0], [1.0]]),
            torch.tensor([[0.0], [float("nan")]]),
            lambda t: t,
        )
        == 1.0
    )
    assert np.isnan(
        coverage_after_inverse(
            torch.tensor([[0.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[float("nan")]]),
            lambda t: t,
        )
    )

    photo = fit_vector_scaler(
        np.array([[15.0, 0.1], [14.0, 0.2], [16.0, 0.05]]), ("robust", "robust")
    )
    label_scaler = fit_vector_scaler(
        np.array([[0.0, 5000.0], [-1.0, 4000.0], [-2.0, 6000.0]]), ("robust", "log10")
    )
    order = OrderScaler(median=np.zeros(2), iqr=np.ones(2), g_ref=8.5)
    star = StellarBatch(
        source_id=np.array([1]),
        healpix=np.array([0]),
        survey=np.array(["gaia"]),
        g_mag=np.array([12.0]),
        n_relevant_bp=np.array([1]),
        groups={
            "xp_bp": FeatureGroup(
                values=np.array([[1.0, -0.5]]),
                missing=np.zeros((1, 2), dtype=bool),
                covariance=np.array([[[2.0, 0.3], [0.3, 0.5]]]),
            ),
            "xp_rp": FeatureGroup(
                values=np.array([[0.2, 0.1]]),
                missing=np.zeros((1, 2), dtype=bool),
                errors=np.array([[0.1, 0.2]]),
            ),
            "photometry": FeatureGroup(
                values=np.array([[15.0, np.nan]]),
                missing=np.array([[False, True]]),
            ),
            "astrometry": FeatureGroup(
                values=np.array([[100.0]]),
                missing=np.array([[False]]),
                errors=np.array([[1.0]]),
            ),
            "labels": FeatureGroup(
                values=np.array([[-0.5, 5000.0]]),
                missing=np.zeros((1, 2), dtype=bool),
            ),
        },
    )
    prepared = prepare(star, order, order, photo, label_scaler)
    assert prepared.missing_bp[0, 1]
    assert not prepared.missing_bp[0, 0]
    assert np.isnan(prepared.ancillary[0, 1])
    assert prepared.ancillary[0, -1] == 10.0
    assert np.allclose(inverse_vector(prepared.labels, label_scaler)[0, 1], 5000.0)
    assert prepared.cov_rp is not None and prepared.cov_rp[0, 0, 1] == 0.0
    blind = prepare(
        StellarBatch(
            source_id=star.source_id,
            healpix=star.healpix,
            survey=star.survey,
            g_mag=np.array([np.nan]),
            n_relevant_bp=star.n_relevant_bp,
            groups=star.groups,
        ),
        order,
        order,
        photo,
        label_scaler,
    )
    assert blind.missing_bp.all() and blind.missing_rp.all()
    no_error = dict(star.groups)
    no_error["astrometry"] = FeatureGroup(
        values=np.array([[100.0]]), missing=np.array([[False]])
    )
    bare = prepare(
        StellarBatch(
            source_id=star.source_id,
            healpix=star.healpix,
            survey=star.survey,
            g_mag=star.g_mag,
            groups=no_error,
        ),
        order,
        order,
        photo,
        label_scaler,
    )
    assert bare.missing_ancillary[0, -1] and np.isnan(bare.ancillary[0, -1])
    assert np.isnan(
        clipped_snr(np.array([100.0, 1.0, 1.0]), np.array([0.0, -1.0, np.nan]))
    ).all()
    zero_groups = dict(star.groups)
    zero_groups["astrometry"] = FeatureGroup(
        values=np.array([[100.0]]),
        missing=np.array([[False]]),
        errors=np.array([[0.0]]),
    )
    zero_groups["xp_rp"] = FeatureGroup(
        values=np.array([[0.2, 0.1]]),
        missing=np.zeros((1, 2), dtype=bool),
        errors=np.array([[0.0, 0.2]]),
    )
    zero_groups["xp_bp"] = FeatureGroup(
        values=np.array([[np.nan, -0.5]]),
        missing=np.zeros((1, 2), dtype=bool),
        covariance=star.groups["xp_bp"].covariance,
    )
    zero_groups["photometry"] = FeatureGroup(
        values=np.array([[np.nan, 0.1]]),
        missing=np.zeros((1, 2), dtype=bool),
    )
    zeroed = prepare(
        StellarBatch(
            source_id=star.source_id,
            healpix=star.healpix,
            survey=star.survey,
            g_mag=star.g_mag,
            n_relevant_bp=star.n_relevant_bp,
            groups=zero_groups,
        ),
        order,
        order,
        photo,
        label_scaler,
    )
    assert zeroed.missing_ancillary[0, -1]
    assert zeroed.missing_ancillary[0, 0]
    assert not zeroed.missing_ancillary[0, 1]
    assert zeroed.missing_bp[0, 0]
    assert zeroed.missing_rp[0, 0] and not zeroed.missing_rp[0, 1]
    cold_groups = dict(star.groups)
    cold_groups["labels"] = FeatureGroup(
        values=np.array([[-0.5, 0.0]]),
        missing=np.zeros((1, 2), dtype=bool),
    )
    cold_prepared = prepare(
        StellarBatch(
            source_id=star.source_id,
            healpix=star.healpix,
            survey=star.survey,
            g_mag=star.g_mag,
            n_relevant_bp=star.n_relevant_bp,
            groups=cold_groups,
        ),
        order,
        order,
        photo,
        label_scaler,
    )
    assert cold_prepared.label_missing[0, 1]
    assert not cold_prepared.label_missing[0, 0]
    cold = transform_vector(
        np.array([[-0.5, 0.0]]), np.zeros((1, 2), dtype=bool), label_scaler
    )
    assert np.isnan(cold[0, 1])
    dropped = eval_forward(model, prepared, drop_xp=True)
    assert dropped.quantiles.shape == (1, 3, 2)
    epoch_loss = pretrain_epoch(
        model,
        MemoryReader([star, star]),
        opt,
        order,
        order,
        photo,
        label_scaler,
        np.random.default_rng(0),
        p_drop_both=1.0,
        p_span=0.0,
        p_drop_bp=0.0,
        p_drop_rp=0.0,
    )
    assert np.isfinite(epoch_loss)

    def _copy(source: StellarBatch, **overrides) -> StellarBatch:
        fields = dict(
            source_id=source.source_id,
            healpix=source.healpix,
            survey=source.survey,
            g_mag=source.g_mag,
            n_relevant_bp=source.n_relevant_bp,
            groups=source.groups,
        )
        fields.update(overrides)
        return StellarBatch(**fields)

    def _stack(left: StellarBatch, right: StellarBatch) -> StellarBatch:
        groups = {}
        for name in left.groups:
            a = left.groups[name]
            b = right.groups[name]
            groups[name] = FeatureGroup(
                values=np.concatenate([a.values, b.values]),
                missing=np.concatenate([a.missing, b.missing]),
                errors=None
                if a.errors is None
                else np.concatenate([a.errors, b.errors]),
                covariance=None
                if a.covariance is None
                else np.concatenate([a.covariance, b.covariance]),
            )
        return StellarBatch(
            source_id=np.concatenate([left.source_id, right.source_id]),
            healpix=np.concatenate([left.healpix, right.healpix]),
            survey=np.concatenate([left.survey, right.survey]),
            g_mag=np.concatenate([left.g_mag, right.g_mag]),
            n_relevant_bp=None
            if left.n_relevant_bp is None
            else np.concatenate([left.n_relevant_bp, right.n_relevant_bp]),
            groups=groups,
        )

    wide_groups = dict(star.groups)
    wide_groups["xp_bp"] = FeatureGroup(
        values=np.array([[50.0, -50.0]]),
        missing=np.zeros((1, 2), dtype=bool),
        covariance=np.array([[[0.01, 0.0], [0.0, 0.01]]]),
    )
    wide = _copy(star, groups=wide_groups)
    blank = _copy(star, g_mag=np.array([np.nan]))
    sparse = _stack(wide, blank)
    opt0 = build_optimizer(model, encoder_lr=0.0, head_lr=0.0, encoder_weight_decay=0.0)
    mask_kw = dict(
        p_drop_both=1.0, p_span=0.0, p_drop_bp=0.0, p_drop_rp=0.0, p_ancillary=0.0
    )
    solo = pretrain_epoch(
        model,
        MemoryReader([star]),
        opt0,
        order,
        order,
        photo,
        label_scaler,
        np.random.default_rng(0),
        **mask_kw,
    )
    half = pretrain_epoch(
        model,
        MemoryReader([sparse]),
        opt0,
        order,
        order,
        photo,
        label_scaler,
        np.random.default_rng(0),
        **mask_kw,
    )
    both = pretrain_epoch(
        model,
        MemoryReader([star, sparse]),
        opt0,
        order,
        order,
        photo,
        label_scaler,
        np.random.default_rng(0),
        **mask_kw,
    )
    scored_mean = (solo * 1 + half * 1) / 2
    length_mean = (solo * 1 + half * 2) / 3
    assert both == pytest.approx(scored_mean)
    assert solo != pytest.approx(half)
    assert both != pytest.approx(length_mean)
    span_nll, drop_nll = regime_nll(model, prepared, np.random.default_rng(1), span=2)
    assert np.isfinite(span_nll) and np.isfinite(drop_nll)
    assert model.training
    assert selection_score(model, prepared, 0, np.array([-2.0])) == float("inf")
    assert model.training

    assert np.isfinite(
        checkpoint_score(
            torch.tensor([0.0, 0.1]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.2]),
            torch.tensor([1.0, 1.1]),
        )
    )
    assert checkpoint_score(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0, 1.2]),
        torch.tensor([1.0, 1.1]),
    ) == float("inf")
    assert metal_poor_mask(np.array([0.0, -1.2, np.nan])).tolist() == [
        False,
        True,
        False,
    ]

    mean = torch.zeros(8, 2, requires_grad=True)
    factor = torch.zeros(8, 2, 1, requires_grad=True)
    diag = torch.ones(8, 2)
    target = torch.zeros(8, 2)
    nll = joint_nll(mean, factor, diag, target)
    nll.backward()
    assert mean.grad is not None
    coverage = gaussian_joint_coverage(mean.detach(), factor.detach(), diag, target)
    assert coverage == 1.0
    dirty = target.clone()
    dirty[0, 0] = float("nan")
    assert gaussian_joint_coverage(mean.detach(), factor.detach(), diag, dirty) == 1.0
    assert choose_joint(None, coverage, minimum=0.9) == "low_rank"
    assert choose_joint(0.5, coverage, minimum=0.9) == "low_rank"
    assert choose_joint(0.95, coverage, minimum=0.9) == "flow"
    if model.flow is not None:
        context = torch.randn(8, 8)
        flow_loss = model.flow(context, target)
        flow_loss.backward()
        assert torch.isfinite(flow_loss)

    rows = np.arange(6, dtype=np.float64)[:, None] * np.ones((1, 2))

    def _batch(values: np.ndarray) -> StellarBatch:
        n = len(values)
        group = FeatureGroup(values=values, missing=np.zeros(values.shape, dtype=bool))
        return StellarBatch(
            source_id=np.arange(n),
            healpix=np.zeros(n, dtype=np.int64),
            survey=np.array(["gaia"] * n),
            g_mag=np.full(n, 15.0),
            groups={"xp_bp": group},
        )

    kept = reservoir_matrix(
        MemoryReader([_batch(rows[:4]), _batch(rows[4:])]), "xp_bp", max_rows=3
    )
    assert kept.shape == (3, 2)
    assert np.allclose(kept, rows[:3])

    cfg = load_config(ROOT / "configs" / "pipeline.yaml")
    assert cfg["encoder_weight_decay"] == 0.0
    assert "feature_cols" not in cfg
