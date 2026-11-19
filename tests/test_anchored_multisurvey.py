import math

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from masked_stellar_autoencoder.training.anchored_multisurvey import (
    AnchoredMultiSurveyWrapper,
    AnchoredRecords,
    dedupe_spectra_groups,
)
from masked_stellar_autoencoder.training.anchored_survey_map import (
    FULL,
    IDENTITY,
    LINEAR,
    OFFSET,
    AnchoredSurveyMaps,
    StudentTHead,
    assign_tiers,
    condition_on_observation,
    student_t_nll,
)


def test_student_t_nll_matches_scipy_and_ignores_padding():
    from scipy.stats import multivariate_t

    torch.manual_seed(0)
    P, D, nu = 3, 4, 5.0
    A = torch.randn(1, D, P, dtype=torch.float64)
    L = torch.tril(torch.randn(1, P, P, dtype=torch.float64)) + 2 * torch.eye(P)
    noise = torch.rand(1, D, dtype=torch.float64) + 0.1
    resid = torch.randn(1, D, dtype=torch.float64)
    valid = torch.ones(1, D, dtype=torch.bool)
    nll, d = student_t_nll(resid, A, L, noise, valid, nu)
    cov = (A[0] @ L[0] @ L[0].T @ A[0].T + torch.diag(noise[0])).numpy()
    ref = -multivariate_t(np.zeros(D), cov + 1e-6 * np.eye(D), df=nu).logpdf(
        resid[0].numpy()
    )
    assert float(d) == D
    np.testing.assert_allclose(float(nll), ref, rtol=1e-6)
    # Two padded components (arbitrary values) change nothing.
    pad = lambda t, v: torch.cat([t, v], 1)  # noqa: E731
    nll2, d2 = student_t_nll(
        pad(resid, torch.randn(1, 2, dtype=torch.float64)),
        torch.cat([A, torch.randn(1, 2, P, dtype=torch.float64)], 1),
        L,
        pad(noise, torch.rand(1, 2, dtype=torch.float64)),
        pad(valid, torch.zeros(1, 2, dtype=torch.bool)),
        nu,
    )
    assert float(d2) == D
    np.testing.assert_allclose(float(nll2), float(nll), rtol=1e-9)


def test_assign_tiers():
    counts = np.array([[10, 10, 0], [6000, 6000, 6000], [60000, 60000, 60000]])
    anchor = np.zeros((3, 3), bool)
    anchor[0, 1] = True
    core = np.zeros((3, 3), bool)
    core[2, 2] = True
    t = assign_tiers(counts, anchor, core)
    assert t[0].tolist() == [OFFSET, IDENTITY, IDENTITY]
    assert t[1].tolist() == [LINEAR] * 3
    assert t[2].tolist() == [FULL, FULL, OFFSET]


def test_maps_respect_tiers_core_and_jacobian():
    torch.manual_seed(0)
    tier = np.array([[IDENTITY] * 3, [OFFSET] * 3, [FULL] * 3, [OFFSET] * 3])
    core = np.zeros((4, 3), bool)
    core[[1, 3], 2] = True
    maps = AnchoredSurveyMaps(
        tier, core, np.array([[1] * 3, [3] * 3, [1] * 3, [1] * 3])
    )
    with torch.no_grad():
        for p in (maps.bias, maps.linear, maps.mix):
            p.uniform_(-0.5, 0.5)
    x = torch.randn(6, 3)
    for s in range(4):
        y, jac = maps(x, torch.full((6,), s))
        if s == 0:
            torch.testing.assert_close(y, x)
            torch.testing.assert_close(jac, torch.eye(3).expand(6, 3, 3))
        if s in (1, 3):
            torch.testing.assert_close(jac, torch.eye(3).expand(6, 3, 3))
        if s == 2:
            num = torch.autograd.functional.jacobian(
                lambda z: maps(z[None], torch.tensor([2]))[0][0], x[0]
            )
            torch.testing.assert_close(jac[0], num, atol=1e-5, rtol=1e-4)
    bias, _, _ = maps.coefficients()
    # star-weighted core offsets sum to zero for [Fe/H]
    assert abs(float((3 * bias[1, 2] + bias[3, 2]).detach())) < 1e-6


def test_dedupe_keeps_first_pipeline_per_parameter():
    names = ["a_main", "a_vmp", "other"]
    row = np.array([0, 0, 0, 1])
    survey = np.array([1, 0, 2, 0])
    value = np.array(
        [[np.nan, np.nan, -2.5], [5000, 4.5, -2.0], [5100, 4.4, -2.2], [6000, 4, 0]]
    )
    out = dedupe_spectra_groups(row, survey, value, names, {"a": ["a_vmp", "a_main"]})
    assert np.isnan(out[1, 2]) and out[0, 2] == -2.5  # vmp [Fe/H] wins
    assert out[1, 0] == 5000  # main keeps Teff (vmp has none)
    np.testing.assert_array_equal(out[2:], value[2:])


def test_student_t_head_quantiles():
    head = StudentTHead(16, 5, torch.nn.ReLU(), nu=5.0)
    z = torch.randn(4, 16)
    q = head(z)
    mu, L = head.params(z)
    assert q.shape == (4, 5, 3)
    torch.testing.assert_close(q[:, :, 1], mu)
    sd = L.square().sum(-1).sqrt()
    torch.testing.assert_close(q[:, :, 2] - mu, head.q84 * sd)
    assert torch.all(torch.triu(L, 1) == 0)


def _synthetic_records(rng, n=600):
    """Truth x ~ N(0, 1) (scaled units); see test below."""
    x = rng.normal(size=(n, 3))
    rows, surveys, values = [], [], []

    def add(s, idx, val):
        v = np.full((len(idx), 4), np.nan)
        v[:, :3] = val
        rows.append(idx)
        surveys.append(np.full(len(idx), s))
        values.append(v)

    seis = rng.choice(n, 300, replace=False)
    v = np.full((300, 3), np.nan)
    v[:, 1] = x[seis, 1] + rng.normal(scale=0.05, size=300)
    add(0, seis, v)
    irfm = rng.choice(n, 300, replace=False)
    v = np.full((300, 3), np.nan)
    v[:, 0] = x[irfm, 0] + rng.normal(scale=0.05, size=300)
    add(1, irfm, v)
    for s, off in ((2, 0.05), (3, -0.05)):  # [Fe/H] core references
        idx = rng.choice(n, 200, replace=False)
        v = x[idx] + rng.normal(scale=0.05, size=(200, 3))
        v[:, 2] += off
        add(s, idx, v)
    idx = np.arange(n)  # large survey with a distorted scale
    v = np.column_stack([x[:, 0] + 0.3, 1.2 * x[:, 1], 1.5 * x[:, 2] + 0.2])
    add(4, idx, v + rng.normal(scale=0.1, size=(n, 3)))
    rec = dict(
        row=np.concatenate(rows),
        survey=np.concatenate(surveys),
        value=np.concatenate(values),
        error=np.full((sum(map(len, rows)), 4), np.nan),
        survey_names=np.array(
            ["anchor_seis", "anchor_irfm", "core_a", "core_b", "big"]
        ),
        survey_is_anchor=np.array([True, True, False, False, False]),
        parameters=np.array(["teff", "logg", "fe_h", "alpha"]),
    )
    return x, rec


def _wrapper(rec, n, **record_kw):
    scalers = [StandardScaler().fit(np.array([[-1.0], [1.0]])) for _ in range(3)]
    for s in scalers:
        s.mean_[:] = 0.0
        s.scale_[:] = 1.0
    names = ["teff", "logg", "fe_h"]
    records = AnchoredRecords(
        rec, scalers, names, n, "cpu", core=("core_",), linear_min=100,
        full_min=10**6, **record_kw,
    )  # fmt: skip
    w = AnchoredMultiSurveyWrapper.__new__(AnchoredMultiSurveyWrapper)
    torch.nn.Module.__init__(w) if isinstance(w, torch.nn.Module) else None
    w.device = "cpu"
    w.feature_cols = []
    w.label_scalers = scalers
    w.configure_anchored(records, names, map_reg_weight=1e-4)
    return w, records


def test_anchors_and_core_fix_the_common_scale():
    """Anchors + core pin the scale; the big survey's map and noise are learned.

    The "network" is a free linear function of a noisy proxy of the truth
    (like a photometric estimate), so it cannot fit any survey's label noise,
    and nothing but the anchors and core fixes its scale.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 600
    x, rec = _synthetic_records(rng, n)
    w, records = _wrapper(rec, n)
    assert records.tier[4].tolist() == [LINEAR] * 3
    u = torch.as_tensor(x + rng.normal(scale=0.1, size=x.shape), dtype=torch.float32)
    W = torch.nn.Parameter(0.5 * torch.eye(3))  # starts off-scale on purpose
    c = torch.nn.Parameter(torch.zeros(3))
    raw = torch.nn.Parameter(torch.full((3,), -2.0))
    opt = torch.optim.Adam([W, c, raw, *w.survey_maps.parameters()], lr=0.02)
    rows = torch.arange(n)
    y = torch.zeros(n, 0)
    for _ in range(2000):
        opt.zero_grad()
        mu = u @ W.T + c
        L = torch.diag_embed(torch.nn.functional.softplus(raw)).expand(n, 3, 3)
        loss = w._joint_loss(mu, L, y, None, rows)
        loss.backward()
        opt.step()
    mu = (u @ W.T + c).detach()
    est = mu.numpy()
    for j in range(3):
        slope, icpt = np.polyfit(x[:, j], est[:, j], 1)
        assert abs(slope - 1) < 0.05, (j, slope)
        assert abs(icpt) < 0.03, (j, icpt)
    _, jac = w.survey_maps(mu, torch.full((n,), 4))
    diag = torch.diagonal(jac, dim1=1, dim2=2).mean(0)
    np.testing.assert_allclose(diag.detach().numpy(), [1.0, 1.2, 1.5], atol=0.06)
    sigma = w.survey_maps.sigma(torch.tensor([4]))[0].detach().numpy()
    np.testing.assert_allclose(sigma, 0.1, atol=0.03)


def test_kept_rows_returns_table_rows(tmp_path):
    from astropy.table import Table

    from masked_stellar_autoencoder.training.finetune_data import kept_rows

    t = Table(
        dict(
            fe_h=[-1.0, -2.0, 0.0, -0.5, -3.0],
            e_fe_h=[0.1, np.nan, 0.1, 0.1, 0.1],
            teff=[5000.0] * 5,
        )
    )
    path = tmp_path / "t.fits"
    t.write(path)
    cfg = {"finetuning": {"metal_poor": {"require_finite_e_fe_h": True}}}
    assert kept_rows(str(path), cfg).tolist() == [0, 2, 3, 4]


def test_reddening_labels_recover_injected_reddening():
    from masked_stellar_autoencoder.training.reddening_labels import (
        extinction_coefficients,
        fit_intrinsic_colours,
        reddening_labels,
    )

    rng = np.random.default_rng(0)
    n = 40000
    teff = rng.uniform(4000, 7000, n)
    logg = rng.uniform(1.5, 4.8, n)
    feh = rng.uniform(-2, 0.4, n)
    theta = 5040 / teff
    c0 = np.column_stack(
        [1.6 * theta - 0.6 + 0.05 * feh, 1.1 * theta - 0.35 + 0.02 * logg]
    )
    ebv = np.where(rng.random(n) < 0.5, rng.uniform(0, 0.01, n), rng.uniform(0, 1, n))
    ratio = extinction_coefficients(c0[:, 0])
    obs = c0 + ratio * ebv[:, None] + rng.normal(scale=0.01, size=(n, 2))
    model = fit_intrinsic_colours(teff, logg, feh, obs, ebv, max_map_ebv=0.02)
    est, err = reddening_labels(model, teff, logg, feh, obs)
    ok = np.isfinite(est)
    assert ok.mean() > 0.9
    d = est[ok] - ebv[ok]
    assert abs(np.median(d)) < 0.01
    assert 1.4826 * np.median(np.abs(d - np.median(d))) < 0.015
    assert 0.5 < np.median(np.abs(d) / err[ok]) < 1.5  # errors are sensible


@pytest.mark.parametrize("n", [1, 2])
def test_unlabelled_loss_is_finite_for_tiny_batches(n):
    from masked_stellar_autoencoder.models.model import FinetuneContext
    from masked_stellar_autoencoder.training.multisurvey_finetune import (
        MultiSurveyWrapper,
    )

    w = MultiSurveyWrapper.__new__(MultiSurveyWrapper)
    w.device = "cpu"
    w.unl_X = torch.randn(10, 4)
    w.unl_E = torch.ones(10, 4)
    w.ft = lambda z: torch.stack([z - 1, z, z + 1], -1)
    w._apply_batch_masking = lambda x, e, ctx: (
        x + 0.1 * torch.randn_like(x),
        torch.zeros_like(x, dtype=torch.bool),
        torch.ones_like(x, dtype=torch.bool),
    )
    w.model = type("M", (), {"encoder": staticmethod(lambda x: x)})()
    ctx = FinetuneContext(**{k: None for k in FinetuneContext.__dataclass_fields__})
    ctx.multitask = False
    _, cons = w._unlabelled_loss(n, ctx)
    assert torch.isfinite(cons)


def _fit_proxy(w, x, rng, steps=2000, degree=1):
    """Fit a polynomial proxy "network" of a noisy truth; return its estimates."""
    n = len(x)
    u = torch.as_tensor(x + rng.normal(scale=0.1, size=x.shape), dtype=torch.float32)
    feats = torch.cat([u**k for k in range(1, degree + 1)], dim=1)
    W = torch.nn.Parameter(torch.zeros(3, feats.shape[1]))
    with torch.no_grad():
        W[:, :3] = 0.5 * torch.eye(3)
    c = torch.nn.Parameter(torch.zeros(3))
    raw = torch.nn.Parameter(torch.full((3,), -2.0))
    opt = torch.optim.Adam([W, c, raw, *w.survey_maps.parameters()], lr=0.02)
    rows = torch.arange(n)
    for _ in range(steps):
        opt.zero_grad()
        mu = feats @ W.T + c
        L = torch.diag_embed(torch.nn.functional.softplus(raw)).expand(n, 3, 3)
        w._joint_loss(mu, L, torch.zeros(n, 0), None, rows).backward()
        opt.step()
    return (feats @ W.T + c).detach().numpy()


def test_censoring_a_survey_floor_removes_the_metal_poor_bias():
    """A big survey that cannot report [Fe/H] below a floor biases the scale's
    low end upward when its floored values are taken at face value; treating
    them as upper limits recovers the core references' scale. (Flexible proxy
    network, few core stars, as for the real metal-poor tail.)"""
    floor = -0.8
    bias = {}
    for censor in (None, {"big": {"fe_h": floor + 0.2}}):
        torch.manual_seed(0)
        rng = np.random.default_rng(1)
        n = 600
        x, rec = _synthetic_records(rng, n)
        core = np.isin(rec["survey"], [2, 3])
        keep = ~core | (np.cumsum(core) % 5 == 0)  # 40 stars per core survey
        rec = {k: (v[keep] if k in ("row", "survey", "value", "error") else v)
               for k, v in rec.items()}  # fmt: skip
        big = rec["survey"] == 4
        v = rec["value"][big, 2]
        rec["value"][big, 2] = np.maximum(
            v, floor + rng.normal(scale=0.05, size=v.shape)
        )
        w, records = _wrapper(rec, n, censor=censor)
        assert records.n_censored == (0 if censor is None else int(
            (rec["value"][big, 2] < floor + 0.2).sum()))  # fmt: skip
        est = _fit_proxy(w, x, rng, steps=3000, degree=3)
        low = x[:, 2] < -1.0
        bias[censor is None] = float(np.median(est[low, 2] - x[low, 2]))
    print("bias at the low end (face value, censored):", bias)
    assert bias[True] > 0.15  # face value: low end pulled up
    assert abs(bias[False]) < 0.05  # censored: unbiased


def test_record_cuts_and_min_records():
    rng = np.random.default_rng(0)
    _, rec = _synthetic_records(rng, 600)
    # A cut on the big survey's Teff drops whole records outside [lo, hi).
    w_cut, r_cut = _wrapper(rec, 600, cuts={"big": {"teff": [None, 0.0]}})
    big_t = rec["value"][rec["survey"] == 4, 0]
    assert r_cut.counts[4, 2] == int((big_t < 0.0).sum())
    # min_records drops small non-anchor surveys, keeping the core by default.
    _, r_keep = _wrapper(rec, 600, min_records=250)
    assert r_keep.dropped == []  # core_a/core_b (200) are core; anchors kept
    _, r_all = _wrapper(rec, 600, min_records=250, min_records_keep_core=False)
    assert set(r_all.dropped) == {"core_a", "core_b"}
    assert r_all.counts[2].sum() == 0 and r_all.counts[0, 1] == 300
    # exclude drops a source entirely.
    _, r_ex = _wrapper(rec, 600, exclude=["big"])
    assert r_ex.counts[4].sum() == 0 and r_ex.counts[2].sum() > 0


def test_condition_on_observation_matches_gaussian_conditioning():
    torch.manual_seed(0)
    B, P = 4, 3
    mu = torch.randn(B, P, dtype=torch.float64)
    L = torch.tril(torch.randn(B, P, P, dtype=torch.float64)) + 2 * torch.eye(P)
    z = torch.randn(B, dtype=torch.float64)
    var = torch.full((B,), 0.3, dtype=torch.float64)
    z[3] = float("nan")
    loc, sd = condition_on_observation(mu, L, 2, z, var, nu=1e9)
    for b in range(3):
        S = (L[b] @ L[b].T).numpy()
        m = mu[b].numpy()
        s22 = S[2, 2] + 0.3
        ref_loc = m + S[:, 2] * (z[b].item() - m[2]) / s22
        ref_cov = S - np.outer(S[:, 2], S[:, 2]) / s22
        np.testing.assert_allclose(loc[b].numpy(), ref_loc, rtol=1e-6)
        np.testing.assert_allclose(sd[b].numpy() ** 2, np.diag(ref_cov), rtol=1e-5)
    # No observation: unchanged location and marginal scale.
    np.testing.assert_allclose(loc[3].numpy(), mu[3].numpy())
    np.testing.assert_allclose(
        sd[3].numpy(), (L[3] @ L[3].T).diagonal().sqrt().numpy(), rtol=1e-12
    )
    # Student-t: a surprising observation widens the conditional scale.
    _, sd_t = condition_on_observation(mu, L, 2, mu[:, 2] + 10.0, var, nu=5.0)
    _, sd_n = condition_on_observation(mu, L, 2, mu[:, 2] + 10.0, var, nu=1e9)
    assert (sd_t[:, :2] > sd_n[:, :2]).all()


def test_galactic_unit_vector_matches_astropy():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from masked_stellar_autoencoder.training.context_inputs import (
        galactic_unit_vector,
    )

    rng = np.random.default_rng(0)
    ra, dec = rng.uniform(0, 360, 50), np.degrees(np.arcsin(rng.uniform(-1, 1, 50)))
    g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    ref = g.cartesian.xyz.value.T
    got = galactic_unit_vector(
        torch.as_tensor(ra, dtype=torch.float64),
        torch.as_tensor(dec, dtype=torch.float64),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-6)  # astropy matrix: 1e-7 apart


def test_context_head_starts_identical_and_context_is_finite():
    from sklearn.preprocessing import RobustScaler

    from masked_stellar_autoencoder.training.context_inputs import ContextInputs

    torch.manual_seed(0)
    cols = ["G", "PARALLAX", "EBV", "RA", "DEC"]
    raw = np.column_stack([
        np.random.default_rng(0).normal(15, 1, 64),
        np.random.default_rng(1).normal(3, 2, 64),
        np.random.default_rng(2).uniform(0, 1, 64),
        np.random.default_rng(3).uniform(0, 360, 64),
        np.random.default_rng(4).uniform(-80, 80, 64),
    ])  # fmt: skip
    raw[0, 2] = np.nan
    scaler = RobustScaler().fit(raw)
    X = torch.as_tensor(scaler.transform(raw), dtype=torch.float32)
    labels = ["teff", "logg", "fe_h", "ebv_star", "parallax"]
    lsc = [StandardScaler().fit(np.array([[0.0], [1.0]])) for _ in labels]
    y = torch.randn(64, 5)
    spec = dict(features=["EBV"], posenc=dict(n=4), dust=dict(n=4, hidden=16))
    ctx = ContextInputs(spec, cols, scaler, labels, lsc)
    (values, visible), dust = ctx(X, None, y, torch.full((64, 5), 0.1))
    assert values.shape == (64, ctx.dim) and visible.shape == values.shape
    assert not bool(visible[0, 0]) and bool(visible[1, 0])
    assert torch.isfinite(values[visible]).all()
    assert torch.isnan(dust[0][0]) and torch.isfinite(dust[0][1:]).all()
    loss = ctx.dust_loss(dust, y, torch.full((64, 5), 0.1))
    loss.backward()
    assert ctx.dust.net[0].weight.grad is not None
    # The context starts as a no-op on the head.
    act = torch.nn.ReLU()
    h0 = StudentTHead(8, 5, act)
    h1 = StudentTHead(8, 5, act, context_dim=ctx.dim)
    h1.load_state_dict(h0.state_dict(), strict=False)
    z = torch.randn(64, 8)
    with torch.no_grad():
        np.testing.assert_allclose(
            h0(z).numpy(), h1(z, (values.detach(), visible)).numpy(), rtol=1e-6
        )


def test_balance_weights_upweight_the_tail_within_the_cap():
    from masked_stellar_autoencoder.training.anchored_multisurvey import (
        balance_weights,
    )

    rng = np.random.default_rng(0)
    feh = np.concatenate(
        [rng.normal(-0.25, 0.3, 50_000), rng.uniform(-4, -2, 300), [np.nan]]
    )
    w, centres, ratio = balance_weights(feh, power=0.5, cap=10.0)
    assert w[-1] == 1.0 and np.all(w >= 1.0) and np.all(w <= 10.0)
    tail = feh < -2
    assert w[tail].mean() > 5 * np.median(w[~tail & np.isfinite(feh)])
    assert ratio.shape == centres.shape and np.all(ratio >= 1.0 - 1e-9)


def test_balanced_loader_oversamples_and_reference_pool_trains():
    rng = np.random.default_rng(1)
    x, rec = _synthetic_records(rng)
    w, _ = _wrapper(rec, len(x))
    # Loader: rows with weight 9 are drawn about 9x as often.
    weights = np.ones(400)
    weights[:40] = 9.0
    w.configure_balance(weights, np.zeros(2), np.ones(2))
    X = np.arange(400, dtype="f4")[:, None]
    loader = w._prepare_finetune_loader(X, X, X, X, 400)
    drawn = torch.cat([b[0][:, 0] for _ in range(20) for b in loader])
    share = float((drawn < 40).float().mean())
    assert 0.45 < share < 0.55  # 360 / (360 + 360)
    # Reference pool: finite loss, gradients reach the head and the pool map.
    w.drop_idx, w.nu = [], 5.0
    w.ft = StudentTHead(3, 3, torch.nn.ReLU())
    w.model = type("M", (), {"encoder": staticmethod(lambda x: x)})()
    xr = rng.normal(size=(50, 3)).astype("f4")
    w.configure_reference_pool(xr, xr, xr[:, 2] + 0.3, np.full(50, 0.05))
    loss = w._reference_pool_loss(16)
    loss.backward()
    assert torch.isfinite(loss)
    assert w.survey_maps.ref_bias.grad is not None
    assert w.ft.loc.weight.grad is not None


def test_prior_free_feh_recovers_the_likelihood():
    from scipy.stats import norm

    from masked_stellar_autoencoder.training.feh_prior import (
        CENTRES,
        feh_extras,
        smoothed_histogram,
    )

    rng = np.random.default_rng(0)
    prior = smoothed_histogram(
        np.concatenate([rng.normal(-0.3, 0.3, 200_000), rng.normal(-1.2, 0.5, 5_000)])
    )
    # A star at -2.8 measured with 0.45 dex: the posterior median sits higher;
    # dividing the (t-approximated) posterior by the prior moves it back.
    like = norm.pdf(CENTRES, -2.8, 0.45)
    post = like * prior
    post /= post.sum()
    cdf = np.cumsum(post)
    med = CENTRES[np.argmax(cdf >= 0.5)]
    sd = 0.5 * (CENTRES[np.argmax(cdf >= 0.84)] - CENTRES[np.argmax(cdf >= 0.16)])
    q, pb = feh_extras(np.array([med, np.nan]), np.array([sd, 0.1]), 30.0, prior)
    assert med > -2.6  # shrunk toward the bulk
    assert abs(q[0, 1] - (-2.8)) < abs(med - (-2.8))
    assert q[0, 0] < q[0, 1] < q[0, 2]
    assert 0.0 < pb[0] < 1.0 and np.isnan(pb[1]) and np.isnan(q[1]).all()
    # Near-solar stars are left alone (flat prior above its peak).
    qs, _ = feh_extras(np.array([0.3]), np.array([0.07]), 30.0, prior)
    assert abs(qs[0, 1] - 0.3) < 0.03


def test_mixture_of_identical_components_equals_the_single_student_t():
    rng = np.random.default_rng(2)
    x, rec = _synthetic_records(rng)
    w, _ = _wrapper(rec, len(x))
    B = 64
    rows = torch.as_tensor(rng.choice(len(x), B, replace=False))
    mu = torch.randn(B, 3)
    L = torch.diag_embed(0.3 + torch.rand(B, 3))
    y = torch.full((B, 3), float("nan"))
    single = w._joint_loss(mu, L, y, None, rows)
    K = 3
    mix = w._joint_loss(
        mu[:, None].expand(B, K, 3), L[:, None].expand(B, K, 3, 3), y, None, rows,
        logw=torch.full((B, K), -math.log(K)),
    )  # fmt: skip
    # The weak weight prior adds mix_prior * log K for uniform weights.
    torch.testing.assert_close(mix, single + w.mix_prior * math.log(K))
    # A mixture head (K > 1) runs through the same loss and gives gradients.
    head = StudentTHead(4, 3, torch.nn.ReLU(), n_components=3, min_scale=1e-2)
    logw, m, Lh = head.mixture(torch.randn(B, 4))
    assert logw.shape == (B, 3) and m.shape == (B, 3, 3) and Lh.shape == (B, 3, 3, 3)
    loss = w._joint_loss(m, Lh, y, None, rows, logw=logw)
    loss.backward()
    assert torch.isfinite(loss) and head.logit.weight.grad is not None
    with pytest.raises(ValueError):
        head.params(torch.randn(2, 4))


def test_mixture_quantiles_match_sampling():
    from scipy.stats import t as student_t

    from masked_stellar_autoencoder.training.anchored_predict import (
        mixture_quantiles,
    )

    logw = np.log(np.array([[0.8, 0.2]]))
    loc = np.array([[[0.0], [-2.0]]])
    scale = np.array([[[0.2], [0.4]]])
    q = mixture_quantiles(logw, loc, scale, np.array([5.0]))[0, 0]
    rng = np.random.default_rng(0)
    n = 400_000
    comp = rng.random(n) < 0.2
    s = np.where(comp, -2.0 + 0.4 * rng.standard_t(5, n), 0.2 * rng.standard_t(5, n))
    np.testing.assert_allclose(q, np.quantile(s, [0.16, 0.5, 0.84]), atol=0.01)
    # One component: the closed form.
    q1 = mixture_quantiles(np.zeros((1, 1)), loc[:, :1], scale[:, :1], np.array([5.0]))
    np.testing.assert_allclose(q1[0, 0, 2], 0.2 * student_t.ppf(0.84, 5), rtol=1e-9)


def test_condition_mixture_moves_weight_to_the_matching_component():
    from masked_stellar_autoencoder.training.anchored_survey_map import (
        condition_mixture,
    )

    # Two components that differ in label 1 (e.g. parallax: giant vs dwarf).
    mu = torch.tensor([[[0.0, -1.0], [0.5, 1.0]]])
    L = torch.tensor([[[0.3, 0.0], [0.2, 0.3]]]).expand(1, 2, 2, 2).contiguous()
    logw = torch.log(torch.tensor([[0.5, 0.5]]))
    lw, loc, sd = condition_mixture(
        logw, mu, L, 1, torch.tensor([1.0]), torch.tensor([0.01]), 5.0
    )
    assert lw.exp()[0, 1] > 0.99
    # K = 1 reduces to condition_on_observation.
    loc1, sd1 = condition_on_observation(
        mu[:, 0], L[:, 0], 1, torch.tensor([0.2]), torch.tensor([0.01]), 5.0
    )
    lw1, locm, sdm = condition_mixture(
        torch.zeros(1, 1), mu[:, :1], L[:, :1], 1, torch.tensor([0.2]),
        torch.tensor([0.01]), 5.0,
    )  # fmt: skip
    torch.testing.assert_close(locm[:, 0], loc1)
    torch.testing.assert_close(sdm[:, 0], sd1)
    # Unobserved rows keep their weights.
    lw2, _, _ = condition_mixture(
        logw, mu, L, 1, torch.tensor([float("nan")]), torch.tensor([0.01]), 5.0
    )
    torch.testing.assert_close(lw2, logw)


def test_feh_extras_mixture_of_copies_equals_single():
    from masked_stellar_autoencoder.training.feh_prior import (
        feh_extras,
        smoothed_histogram,
    )

    rng = np.random.default_rng(0)
    prior = smoothed_histogram(rng.normal(-0.3, 0.4, 100_000))
    loc, scale = np.array([-1.8, 0.1]), np.array([0.3, 0.1])
    q1, p1 = feh_extras(loc, scale, 5.0, prior)
    qk, pk = feh_extras(
        np.repeat(loc[:, None], 3, 1), np.repeat(scale[:, None], 3, 1), 5.0, prior,
        logw=np.log(np.full((2, 3), 1 / 3)),
    )  # fmt: skip
    np.testing.assert_allclose(qk, q1)
    np.testing.assert_allclose(pk, p1)


def test_diagonal_mixture_head_has_no_cross_terms():
    head = StudentTHead(4, 3, torch.nn.ReLU(), n_components=2, full_covariance=False)
    _, _, L = head.mixture(torch.randn(5, 4))
    off = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1))
    assert torch.count_nonzero(off) == 0


def test_feh_population_undoes_the_sampling_ratio():
    from masked_stellar_autoencoder.training.feh_prior import (
        CENTRES,
        feh_population,
    )

    # A flat ratio changes nothing; an up-weighted tail moves stars back up.
    loc, scale = np.array([-2.4]), np.array([0.3])
    q_flat, _ = feh_population(loc, scale, 30.0, CENTRES, np.ones(len(CENTRES)))
    assert abs(q_flat[0, 1] + 2.4) < 0.03
    ratio = np.exp(np.clip(-1.5 * (CENTRES + 0.5), 0.0, None))  # rises below -0.5
    q_tail, p = feh_population(loc, scale, 30.0, CENTRES, ratio)
    assert q_tail[0, 1] > q_flat[0, 1] + 0.1 and 0.0 <= p[0] <= 1.0
