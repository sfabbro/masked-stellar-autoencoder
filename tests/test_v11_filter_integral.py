"""Filter-integral operators against torchgaiaxp, the exact XP variance, and the
filter-integral likelihood (zero points fitted from data, never set)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("torchgaiaxp")
from torchgaiaxp import Sampled, XPSpectra, generate  # noqa: E402

from masked_stellar_autoencoder.v11 import filters as F  # noqa: E402
from masked_stellar_autoencoder.v11.losses import (  # noqa: E402
    MAG,
    FilterIntegralLoss,
    student_t_nll,
)
from masked_stellar_autoencoder.v11.schema import N_XP, load_registry  # noqa: E402

REG = load_registry()
WL = torch.linspace(330.0, 1050.0, 721, dtype=torch.float64)


def _spectra(n=5, seed=0):
    """Synthetic XP spectra (fits of blackbodies) with errors and AR(1) correlations."""
    g = torch.Generator().manual_seed(seed)
    teff = torch.linspace(3600.0, 9500.0, n, dtype=torch.float64)
    lam = WL[None, :] * 1e-9
    f = 1.0 / (
        lam**5
        * torch.expm1(6.62607e-34 * 2.99792e8 / (lam * 1.380649e-23 * teff[:, None]))
    )
    f = f / f.mean(-1, keepdim=True) * 1e-16
    coef = XPSpectra.fit(Sampled(WL, f), WL).coefficients
    err = coef.abs() * 0.01 + coef.abs().amax(-1, keepdim=True) * 1e-3
    r, c = np.tril_indices(N_XP, -1)
    rho = torch.rand(n, 2, 1, generator=g, dtype=torch.float64) * 0.5 - 0.2
    corr = rho ** torch.as_tensor(np.abs(r - c), dtype=torch.float64)
    return XPSpectra(coefficients=coef, errors=err, correlations=corr)


def test_band_selection_follows_passband_support():
    sel = F.select_fi_bands(REG, (336.0, 1020.0), 0.01)
    for name in (
        "ps1_g",
        "ps1_r",
        "ps1_i",
        "ps1_z",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "skymapper_g",
        "skymapper_r",
        "skymapper_i",
        "des_z",
        "ls_s_g",
        "delve_i",
        "splus_j0395",
    ):
        assert name in sel, name
    for name in (
        "sdss_u",
        "sdss_z",
        "ps1_y",
        "skymapper_u",
        "skymapper_z",
        "des_y",
        "gaia_g",
        "tmass_j",
        "pristine_cahk0",
    ):
        assert name not in sel, name
    with pytest.raises(ValueError, match="sdss_z"):
        F.select_fi_bands(REG, (336.0, 1020.0), 0.01, candidates=["sdss_g", "sdss_z"])


def test_packaged_operator_matches_torchgaiaxp_generate():
    xp = _spectra()
    op = F.build_operator(
        REG, ["ps1_g", "ps1_r", "ps1_i", "ps1_z", "sdss_g", "skymapper_v"]
    )
    flat = xp.coefficients.reshape(len(xp), 2 * N_XP)
    mags = op.magnitudes_raw(flat)
    var = F.exact_band_variance(torch.as_tensor(op.matrix), xp.errors, xp.correlations)
    s = op.relative_flux(flat)
    ps1 = generate(xp, ["PanSTARRS1"])["PanSTARRS1"]
    torch.testing.assert_close(mags[:, :4], ps1["mag"][:, :4], atol=1e-9, rtol=0)
    torch.testing.assert_close(
        var[:, :4].sqrt() / s[:, :4],
        ps1["flux_error"][:, :4] / ps1["flux"][:, :4],
        atol=1e-10,
        rtol=1e-8,
    )
    sdss = generate(xp, ["SDSS"])["SDSS"]
    torch.testing.assert_close(mags[:, 4], sdss["mag"][:, 1], atol=1e-9, rtol=0)
    gv = generate(xp, ["Gaia_Vega"])["Gaia_Vega"]
    c0, c1 = op.colour_columns
    torch.testing.assert_close(
        mags[:, c0] - mags[:, c1], gv["mag"][:, 1] - gv["mag"][:, 2], atol=1e-9, rtol=0
    )


@pytest.mark.parametrize(
    "name", ["ls_n_g", "ls_n_z", "atlas_i", "cfis_r", "splus_j0395", "splus_j0861"]
)
def test_svo_operator_matches_xpspectra_integrate(name):
    """a' . c and sqrt(a^T C a) equal XPSpectra.integrate(passband, wl) with the exact
    covariance (flux and error) on a few synthetic spectra."""
    xp = _spectra(seed=1)
    op = F.build_operator(REG, [name])
    wl_f, t = F.load_svo_curve(REG[name].svo)
    pb = Sampled(torch.as_tensor(wl_f), torch.as_tensor(t), fill=0.0)
    lo, hi = max(wl_f.min(), 330.0), min(wl_f.max(), 1050.0)
    grid = torch.linspace(
        lo, hi, int(math.ceil((hi - lo) / 0.25)) + 1, dtype=torch.float64
    )
    flux, err = xp.integrate(pb, grid)
    tg = pb(grid)
    lam2 = torch.trapezoid(tg * grid, grid) / torch.trapezoid(tg / grid, grid)
    mag_ref = -2.5 * torch.log10(flux * lam2 / F.C_NM_PER_S) + F.AB_ZP_SI
    flat = xp.coefficients.reshape(len(xp), 2 * N_XP)
    torch.testing.assert_close(
        op.magnitudes_raw(flat)[:, 0], mag_ref, atol=1e-9, rtol=0
    )
    var = F.exact_band_variance(torch.as_tensor(op.matrix), xp.errors, xp.correlations)[
        :, 0
    ]
    s = op.relative_flux(flat)[:, 0]
    torch.testing.assert_close(var.sqrt() / s, err / flux, atol=1e-12, rtol=1e-9)


def test_packaged_and_svo_routes_agree_for_the_same_filter():
    """GaiaXPy's packaged PS1 r and the SVO PS1 r curve integrate to the same magnitude
    within the filter-curve version differences (units and zero points consistent)."""
    xp = _spectra(seed=2)
    flat = xp.coefficients.reshape(len(xp), 2 * N_XP)
    packaged = F.build_operator(REG, ["ps1_r"]).magnitudes_raw(flat)[:, 0]
    a = F.svo_integrate_vector("PAN-STARRS/PS1.r")
    svo = 8.5 - 2.5 * torch.log10(
        flat @ torch.as_tensor(a * 10 ** (-0.4 * (F.AB_ZP_SI - 8.5)))
    )
    assert float((packaged - svo).abs().max()) < 0.02


def test_exact_variance_diagonal_and_missing():
    xp = _spectra(n=3)
    a = torch.as_tensor(F.build_operator(REG, ["ps1_i"]).matrix)
    diag = F.exact_band_variance(a, xp.errors, None)
    ref = ((xp.errors.reshape(3, 110) ** 2) @ (a**2)).squeeze()
    torch.testing.assert_close(diag, ref.reshape(diag.shape))
    e = xp.errors.clone()
    e[1] = float("nan")
    assert torch.isnan(F.exact_band_variance(a, e, xp.correlations)[1]).all()


def test_g_normalisation_cancels_in_observed_synthetic_magnitudes():
    """m = G - 2.5 log10(a'.c_norm) equals 8.5 - 2.5 log10(a'.c_raw) for any G used."""
    xp = _spectra(n=3)
    op = F.build_operator(REG, ["ps1_g"])
    flat = xp.coefficients.reshape(3, 110)
    for g in (9.0, 15.0, 19.0):
        f = 10 ** ((8.5 - g) / 2.5)
        m = g - 2.5 * torch.log10(op.relative_flux(flat / f)[:, 0])
        torch.testing.assert_close(m, op.magnitudes_raw(flat)[:, 0])


def _fi_data(n=4000, seed=0, offsets=(0.03, -0.02, 0.05), k_true=(0.0, 0.02, -0.03)):
    gen = torch.Generator().manual_seed(seed)
    s = torch.exp(torch.randn(n, 5, generator=gen) * 0.3)  # 3 bands + BP + RP
    var = (0.01 * s) ** 2
    g = 12 + 6 * torch.rand(n, generator=gen)
    col = -2.5 * torch.log10(s[:, 3] / s[:, 4])
    m_syn = g[:, None] - 2.5 * torch.log10(s[:, :3])
    err = torch.full((n, 3), 0.02)
    obs = m_syn + torch.tensor(offsets) + torch.tensor(k_true) * (col[:, None] - 1.0)
    obs = obs + torch.randn(n, 3, generator=gen) * torch.sqrt(
        err**2 + (MAG * 0.01) ** 2 + 0.01**2
    )
    obs[torch.rand(n, 3, generator=gen) < 0.3] = float("nan")
    return obs, err, s, var, g


def test_calibrate_mode_fits_zero_points_and_colour_terms_from_data():
    obs, err, s, var, g = _fi_data()
    fi = FilterIntegralLoss(
        3, nu=5.0, sigma_sys_init=0.05, colour_reference=1.0, linear_below_fraction=0.1
    )
    assert torch.equal(
        fi.zp.detach(), torch.zeros(3)
    )  # starts at 0: never ground truth
    opt = torch.optim.Adam(fi.parameters(), lr=0.02)
    for _ in range(400):
        opt.zero_grad()
        tot, cnt = fi.calibrate(obs, err, s, var, g)
        (tot / cnt).backward()
        opt.step()
    np.testing.assert_allclose(fi.zp.detach().numpy(), [0.03, -0.02, 0.05], atol=0.004)
    np.testing.assert_allclose(fi.k.detach().numpy(), [0.0, 0.02, -0.03], atol=0.01)
    assert (fi.sigma_sys().detach() < 0.03).all()


def test_calibrate_likelihood_formula():
    obs, err, s, var, g = _fi_data(n=50)
    fi = FilterIntegralLoss(3, 5.0, 0.02, 1.0, 0.1)
    with torch.no_grad():
        fi.zp.copy_(torch.tensor([0.01, 0.0, -0.01]))
        fi.k.copy_(torch.tensor([0.1, 0.0, 0.0]))
    tot, cnt = fi.calibrate(obs, err, s, var, g)
    col = -2.5 * torch.log10(s[:, 3] / s[:, 4])
    m_syn = g[:, None] - 2.5 * torch.log10(s[:, :3])
    r = obs - (m_syn + fi.zp + fi.k * (col[:, None] - 1.0))
    v = err**2 + (MAG * var[:, :3].sqrt() / s[:, :3]) ** 2 + fi.sigma_sys() ** 2
    ok = torch.isfinite(obs)
    ref = student_t_nll(r[ok], v[ok], 5.0).sum()
    assert cnt == int(ok.sum())
    torch.testing.assert_close(tot, ref)


def test_sed_and_phot_modes_train_the_network_side_only():
    obs, err, s, var, g = _fi_data(n=200)
    fi = FilterIntegralLoss(3, 5.0, 0.02, 1.0, 0.1)
    dec = s.clone().requires_grad_(True)
    tot, cnt = fi.sed(
        obs, err, dec, torch.full((1, 5), 1e-6), g, torch.full_like(g, 0.001)
    )
    tot.backward()
    assert cnt > 0 and dec.grad.abs().sum() > 0
    assert fi.zp.grad is None and fi.log_sigma_sys.grad is None  # nuisances detached
    # non-positive decoded fluxes still give a finite, restoring gradient
    dec2 = (-s).clone().requires_grad_(True)
    tot2, _ = fi.sed(
        obs, err, dec2, torch.full((1, 5), 1e-6), g, torch.full_like(g, 0.001)
    )
    tot2.backward()
    assert torch.isfinite(tot2) and torch.isfinite(dec2.grad).all()
    assert (dec2.grad[torch.isfinite(obs)[:, 0], 0] < 0).all()  # pushes fluxes up
    dm = (g[:, None] - 2.5 * torch.log10(s[:, :3]) + 0.1).clone().requires_grad_(True)
    use = torch.ones(200, 3, dtype=torch.bool)
    tot3, cnt3 = fi.phot(dm, torch.full((3,), 0.05), use, s, var, g)
    tot3.backward()
    assert cnt3 == 600 and dm.grad.abs().sum() > 0 and fi.zp.grad is None


def test_extended_magnitude_is_c1():
    tau = torch.tensor([0.1])
    s = torch.tensor(
        [0.1 - 1e-6, 0.1, 0.1 + 1e-6], dtype=torch.float64, requires_grad=True
    )
    m = FilterIntegralLoss.extended_magnitude(s, tau.double())
    m.sum().backward()
    assert abs(float((m[0] - m[2]).detach())) < 1e-4
    assert float((s.grad - s.grad[1]).abs().max()) < 1e-3
