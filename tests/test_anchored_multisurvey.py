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


def _wrapper(rec, n):
    scalers = [StandardScaler().fit(np.array([[-1.0], [1.0]])) for _ in range(3)]
    for s in scalers:
        s.mean_[:] = 0.0
        s.scale_[:] = 1.0
    names = ["teff", "logg", "fe_h"]
    records = AnchoredRecords(
        rec, scalers, names, n, "cpu", core=("core_",), linear_min=100,
        full_min=10**6,
    )  # fmt: skip
    w = AnchoredMultiSurveyWrapper.__new__(AnchoredMultiSurveyWrapper)
    torch.nn.Module.__init__(w) if isinstance(w, torch.nn.Module) else None
    w.device = "cpu"
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
