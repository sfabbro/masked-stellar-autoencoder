"""v11 model shapes, absent-band handling, masking invariants and the
heteroscedastic reconstruction loss."""

from __future__ import annotations

import math

import pytest
import torch

from masked_stellar_autoencoder.v11.config import MaskingConfig, ModelConfig
from masked_stellar_autoencoder.v11.losses import ReconstructionLoss, student_t_nll
from masked_stellar_autoencoder.v11.masking import BandGroups, sample_masks, view
from masked_stellar_autoencoder.v11.model import MSAv11
from masked_stellar_autoencoder.v11.schema import load_registry

torch.manual_seed(0)
REG = load_registry()
SMALL = ModelConfig(
    d_model=32,
    n_heads=4,
    n_latents=4,
    n_layers=2,
    cross_attn_layers=[0, 1],
    latent_dim=256,
    value_frequencies=4,
    band_frequencies=4,
    decoder_hidden=32,
    decoder_layers=1,
)


def _rows(n=6, xp=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.full((n, len(REG)), float("nan"))
    e = torch.full((n, len(REG)), float("nan"))
    names = [
        "gaia_g",
        "gaia_bp",
        "gaia_rp",
        "ps1_g",
        "ps1_r",
        "ps1_i",
        "tmass_j",
        "tmass_h",
        "catwise_w1",
        "parallax",
        "pmra",
        "pmdec",
        "sfd_ebv",
    ]
    idx = torch.as_tensor([REG.index(x) for x in names])
    v[:, idx] = torch.randn(n, len(idx), generator=g)
    e[:, idx] = 0.03
    if xp:
        xi = torch.as_tensor(REG.xp_indices)
        v[: n // 2, xi] = torch.randn(n // 2, len(xi), generator=g)
        e[: n // 2, xi] = 0.1
    return v, e


def _model():
    m = MSAv11(REG, SMALL)
    m.eval()
    return m


def test_shapes_and_parameter_count():
    m = _model()
    v, e = _rows()
    z = m.encode(v, e)
    assert z.shape == (6, 256) and torch.isfinite(z).all()
    g = m.decoder.features(z)
    out = m.decoder.dense(g, torch.as_tensor(REG.xp_indices))
    assert out.shape == (6, 110)
    rows = torch.tensor([0, 0, 5])
    bands = torch.tensor(
        [REG.index("ps1_g"), REG.index("xp_rp_10"), REG.index("galex_fuv")]
    )
    assert m.decoder.pairs(g, rows, bands).shape == (3,)


def test_absent_values_never_reach_the_encoder():
    """Values in absent slots (NaN or garbage behind a False visible mask) change nothing."""
    m = _model()
    v, e = _rows()
    vis = torch.isfinite(v)
    z0 = m.encode(v, e, vis)
    v2, e2 = v.clone(), e.clone()
    v2[~vis] = 1e6
    e2[~vis] = -3.0
    z1 = m.encode(v2, e2, vis)
    torch.testing.assert_close(z0, z1)
    # hiding a band equals the band being absent
    j = REG.index("ps1_r")
    v3 = v.clone()
    v3[:, j] = float("nan")
    vis_h = vis.clone()
    vis_h[:, j] = False
    torch.testing.assert_close(m.encode(v, e, vis_h), m.encode(v3, e))


def test_permutation_and_padding_invariance():
    """A row's latent does not depend on the other rows (padding length) or token order."""
    m = _model()
    v, e = _rows(n=6)
    z_batch = m.encode(v, e)
    for i in (0, 4):  # an XP row (long) and a photometry-only row (short)
        z_one = m.encode(v[i : i + 1], e[i : i + 1])
        torch.testing.assert_close(z_batch[i : i + 1], z_one, atol=2e-5, rtol=1e-4)
    vs, es = m.scale_inputs(v, e)
    band, val, err, mask = m.pack(vs, es, torch.isfinite(v))
    perm = torch.randperm(band.shape[1])
    tok_a = m.tokens(val, err, m.bands.table()[band])
    tok_b = m.tokens(val[:, perm], err[:, perm], m.bands.table()[band[:, perm]])
    torch.testing.assert_close(tok_a[:, perm], tok_b)


def test_row_without_visible_token_fails_loudly():
    m = _model()
    v, e = _rows(n=2, xp=False)
    with pytest.raises(ValueError, match="no visible token"):
        m.encode(v, e, torch.zeros_like(v, dtype=torch.bool))


def test_gradient_checkpointing_matches():
    cfg = ModelConfig(**{**SMALL.__dict__, "gradient_checkpointing": True})
    torch.manual_seed(1)
    a = MSAv11(REG, SMALL)
    b = MSAv11(REG, cfg)
    b.load_state_dict(a.state_dict())
    a.train()
    b.train()
    v, e = _rows()
    za, zb = a.encode(v, e), b.encode(v, e)
    torch.testing.assert_close(za, zb)
    za.square().sum().backward()
    zb.square().sum().backward()
    torch.testing.assert_close(a.latents.grad, b.latents.grad, atol=1e-6, rtol=1e-4)


# ------------------------------------------------------------------ masking


def test_masking_invariants():
    groups = BandGroups(REG)
    v, _ = _rows(n=400, seed=3)
    cfg = MaskingConfig(
        token_ratio=0.3,
        xp_token_ratio=0.5,
        survey_dropout=0.2,
        xp_dropout=0.4,
        astrometry_hide=0.5,
        sfd_hide=0.5,
        min_visible=2,
    )
    gen = torch.Generator().manual_seed(0)
    plan = sample_masks(v, groups, cfg, gen)
    present = torch.isfinite(v)
    assert not (plan.visible & ~present).any()  # only present tokens are visible
    assert not (plan.target & ~present).any()  # only present tokens are targets
    assert not (plan.visible & plan.target).any()  # a target is never visible
    assert not plan.target[:, groups.is_cond].any()  # conditioning is never a target
    recon_vis = (plan.visible & groups.reconstruct).sum(1)
    assert (recon_vis >= 2).all()
    # whole-XP dropout: the XP block is all hidden or partly visible, never dropped
    # coefficient-by-coefficient for rows that lost it
    xp = groups.is_xp
    has_xp = (present & xp).any(1)
    vis_xp = (plan.visible & xp).sum(1)
    frac_dropped = float((has_xp & (vis_xp == 0)).sum() / has_xp.sum())
    assert 0.25 < frac_dropped < 0.55
    kept = has_xp & (vis_xp > 0)
    frac_tok = float((plan.target & xp)[kept].sum() / (present & xp)[kept].sum())
    assert 0.4 < frac_tok < 0.6
    # astrometry group hidden together
    ast = groups.is_astrometry
    va = (plan.visible & ast).sum(1)
    assert set(va.unique().tolist()) <= {0, 3}
    # survey dropout hides every band of the survey
    ps1 = torch.as_tensor([REG.index(f"ps1_{b}") for b in "gri"])
    hidden = (~plan.visible[:, ps1]).sum(1)
    assert (hidden == 3).float().mean() > 0.15


def test_masking_deterministic_and_views():
    groups = BandGroups(REG)
    v, _ = _rows(n=50)
    cfg = MaskingConfig()
    p1 = sample_masks(v, groups, cfg, torch.Generator().manual_seed(5))
    p2 = sample_masks(v, groups, cfg, torch.Generator().manual_seed(5))
    assert torch.equal(p1.visible, p2.visible) and torch.equal(p1.target, p2.target)
    vis = view(v, groups, hide=("astrometry",))
    assert not vis[:, groups.is_astrometry].any()
    assert vis[:, REG.index("gaia_g")].all()
    assert not view(v, groups, hide=("xp",))[:, groups.is_xp].any()
    with pytest.raises(ValueError):
        view(v, groups, hide=("parallax",))


def test_min_visible_restores_tokens():
    groups = BandGroups(REG)
    v = torch.full((3, len(REG)), float("nan"))
    v[:, [REG.index("gaia_g"), REG.index("gaia_bp"), REG.index("ps1_g")]] = 1.0
    cfg = MaskingConfig(token_ratio=0.99, survey_dropout=0.99, min_visible=2)
    plan = sample_masks(v, groups, cfg, torch.Generator().manual_seed(0))
    assert ((plan.visible & groups.reconstruct).sum(1) >= 2).all()


# ------------------------------------------------------------------ loss


def test_student_t_nll_matches_torch():
    r = torch.tensor([0.0, 0.3, -2.0, 7.0])
    var = torch.tensor([0.01, 0.2, 1.5, 0.5])
    for nu in (1.0, 5.0, 30.0):
        ref = -torch.distributions.StudentT(nu, 0.0, var.sqrt()).log_prob(r)
        torch.testing.assert_close(student_t_nll(r, var, nu), ref)


def test_heteroscedastic_reconstruction_uses_measured_and_model_sigma():
    groups = BandGroups(REG)
    loss = ReconstructionLoss(
        len(REG),
        groups.is_xp,
        nu=5.0,
        sigma_init=0.05,
        floor=1e-3,
        group_weights={"xp": 1.0, "phot": 1.0},
    )
    bands = torch.tensor(
        [REG.index("ps1_g"), REG.index("xp_bp_03"), REG.index("ps1_g")]
    )
    pred = torch.tensor([0.1, -0.2, 0.0])
    target = torch.tensor([0.0, 0.1, 0.5])
    err = torch.tensor([0.2, float("nan"), 0.1])  # NaN error -> sigma_meas = 0
    sums = loss(pred, target, err, bands)
    sm = loss.sigma_model()
    s_g = sm[bands[0]]
    exp_phot = student_t_nll(torch.tensor(-0.1), 0.2**2 + s_g**2, 5.0) + student_t_nll(
        torch.tensor(0.5), 0.1**2 + s_g**2, 5.0
    )
    exp_xp = student_t_nll(torch.tensor(0.3), sm[bands[1]] ** 2, 5.0)
    torch.testing.assert_close(sums["phot"][0], exp_phot)
    torch.testing.assert_close(sums["xp"][0], exp_xp)
    assert sums["phot"][1] == 2 and sums["xp"][1] == 1
    total = loss.combine({k: v[0] for k, v in sums.items()}, {"xp": 1, "phot": 2})
    total.backward()
    g = loss.log_sigma.grad
    assert g[bands[0]] != 0 and g[bands[1]] != 0
    assert int((g != 0).sum()) == 2  # only the bands seen get a gradient


def test_sigma_model_fit_recovers_scatter():
    """With sigma_meas known, fitting sigma_model on residuals recovers the extra scatter."""
    groups = BandGroups(REG)
    loss = ReconstructionLoss(
        len(REG),
        groups.is_xp,
        nu=1e6,
        sigma_init=0.5,
        floor=1e-4,
        group_weights={"xp": 1.0, "phot": 1.0},
    )
    gen = torch.Generator().manual_seed(0)
    n = 40000
    meas, extra = 0.3, 0.4
    r = torch.randn(n, generator=gen) * math.sqrt(meas**2 + extra**2)
    band = torch.full((n,), REG.index("ps1_z"))
    opt = torch.optim.Adam(loss.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        s = loss(r, torch.zeros(n), torch.full((n,), meas), band)
        (s["phot"][0] / n).backward()
        opt.step()
    assert abs(float(loss.sigma_model()[REG.index("ps1_z")].detach()) - extra) < 0.02


def test_micro_batch_normalisation_is_global():
    """Summing per-micro-batch NLL sums divided by the global count equals one batch."""
    groups = BandGroups(REG)
    loss = ReconstructionLoss(
        len(REG), groups.is_xp, 5.0, 0.05, 1e-3, {"xp": 1.0, "phot": 1.0}
    )
    gen = torch.Generator().manual_seed(2)
    bands = torch.randint(0, len(REG) - 4, (100,), generator=gen)
    pred, tgt, err = (
        torch.randn(100, generator=gen),
        torch.randn(100, generator=gen),
        torch.rand(100, generator=gen),
    )
    full = loss(pred, tgt, err, bands)
    counts = {k: full[k][1] for k in full}
    one = loss.combine({k: full[k][0] for k in full}, counts)
    a = loss(pred[:37], tgt[:37], err[:37], bands[:37])
    b = loss(pred[37:], tgt[37:], err[37:], bands[37:])
    two = loss.combine({k: a[k][0] for k in a}, counts) + loss.combine(
        {k: b[k][0] for k in b}, counts
    )
    torch.testing.assert_close(one, two)
