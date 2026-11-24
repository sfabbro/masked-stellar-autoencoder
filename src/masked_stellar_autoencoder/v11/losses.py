"""Losses of the v11 pretraining.

Reconstruction (per masked, present, reconstructable entry ``(i, b)``, scaled units)::

    r = x_ib - x_hat_ib,   s2 = sigma_meas,ib^2 + sigma_model,b^2
    NLL = log G(nu/2) - log G((nu+1)/2) + 1/2 log(nu pi s2) + (nu+1)/2 log(1 + r^2 / (nu s2))

``sigma_model,b = floor + exp(log_sigma_b)`` is fitted per band (per coefficient for
XP). Entries without a quoted error use ``sigma_meas = 0``. XP coefficient errors enter
diagonally (the per-token likelihood ignores the BP/RP coefficient correlations; the
filter-integral term below uses them exactly). The loss is
``sum_g w_g (sum_{entries in g} NLL) / N_g`` over the groups g = xp, phot, with
``N_g`` counted over the whole optimiser batch so micro-batching does not change it.

Filter-integral consistency (bands b inside the XP range, :mod:`.filters`)::

    m_syn,b(c) = G - 2.5 log10(a'_b . c_norm)          (c_norm = c_raw / f_G)
    r = m_b - [m_syn,b(c) + zp_b + k_b (BP-RP_syn(c) - c_ref)]
    s2 = sigma_meas^2 + sigma_XP^2 + sigma_sys,b^2,     NLL_t(r; s2, nu)

with three modes:

* ``calibrate``: observed XP (exact covariance -> sigma_XP) vs observed survey
  magnitude (pairs with sigma_XP <= ``max_sigma_xp_mag``: the first-order magnitude
  error of a low-S/N integral is not valid, and its log is biased). Both sides are data, so only ``zp_b, k_b, sigma_sys,b`` learn here; they
  start at 0, 0 and ``sigma_sys_init`` (fitted, never set to literature values);
* ``sed``: decoded XP vs observed survey magnitude, for every star with an observed
  in-range band and G (photometry-only stars included). sigma_XP comes from the fitted
  decoder scatter of the coefficients, sigma_meas includes sigma_G (the decoded XP is
  G-normalised). Gradients reach the encoder/decoder;
* ``phot``: observed XP (exact covariance) vs the decoded survey magnitude of bands the
  encoder did not see (masked or absent). sigma_meas is the decoder's fitted scatter
  of that band. Gradients reach the encoder/decoder.

In ``sed`` and ``phot`` the nuisances and the decoder scatters are detached, so the
calibration is identified only by data. A decoded relative flux ``s`` below
``tau = linear_below_fraction x F_obs`` (``F_obs = 10**(-0.4 (m_b - G))``, the observed
relative flux) continues the magnitude linearly, ``m = -2.5 log10(tau) - MAG (s - tau) /
tau`` (C1 at ``tau``): the NLL stays a magnitude NLL where the decoded SED is sensible,
and a non-positive decoded flux (early in training) still gets a finite, restoring
gradient. The ``sed`` colour is computed from the decoded BP/RP fluxes floored at 1e-3
and detached (``k_b`` is detached there anyway).
"""

from __future__ import annotations

import math

import torch
from torch import nn

LOG10 = math.log(10.0)
MAG = 2.5 / LOG10


def student_t_nll(r: torch.Tensor, var: torch.Tensor, nu: float) -> torch.Tensor:
    """Elementwise univariate Student-t NLL with scale^2 = var."""
    return (
        math.lgamma(nu / 2)
        - math.lgamma((nu + 1) / 2)
        + 0.5 * torch.log(nu * math.pi * var)
        + 0.5 * (nu + 1) * torch.log1p(r * r / (nu * var))
    )


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        n_bands: int,
        is_xp: torch.Tensor,
        nu: float,
        sigma_init: float,
        floor: float,
        group_weights: dict,
    ):
        super().__init__()
        self.nu = float(nu)
        self.floor = float(floor)
        self.log_sigma = nn.Parameter(
            torch.full((n_bands,), math.log(max(sigma_init - floor, 1e-6)))
        )
        self.register_buffer("is_xp", is_xp.clone().bool(), persistent=False)
        self.w = {
            "xp": float(group_weights["xp"]),
            "phot": float(group_weights["phot"]),
        }

    def sigma_model(self) -> torch.Tensor:
        return self.floor + self.log_sigma.exp()

    def forward(self, pred, target, err, bands):
        """Per-group NLL sums: ``{"xp": (sum, count), "phot": (sum, count)}``."""
        sm = self.sigma_model()[bands]
        meas = torch.where(torch.isfinite(err), err, torch.zeros_like(err))
        nll = student_t_nll(target - pred, meas * meas + sm * sm, self.nu)
        xp = self.is_xp[bands]
        return {
            "xp": (nll[xp].sum(), int(xp.sum())),
            "phot": (nll[~xp].sum(), int((~xp).sum())),
        }

    def combine(self, sums: dict, counts: dict) -> torch.Tensor:
        total = 0.0
        for g in ("xp", "phot"):
            if counts[g] > 0:
                total = total + self.w[g] * sums[g] / counts[g]
        return total


class FilterIntegralLoss(nn.Module):
    """Per-band zero point, colour term and systematic scatter + the three modes."""

    def __init__(
        self,
        n_bands: int,
        nu: float,
        sigma_sys_init: float,
        colour_reference: float,
        linear_below_fraction: float,
        max_sigma_xp_mag: float = 0.2,
    ):
        super().__init__()
        self.max_sigma_xp = float(max_sigma_xp_mag)
        self.nu = float(nu)
        self.c_ref = float(colour_reference)
        self.linear_below = float(linear_below_fraction)
        self.zp = nn.Parameter(torch.zeros(n_bands))
        self.k = nn.Parameter(torch.zeros(n_bands))
        self.log_sigma_sys = nn.Parameter(
            torch.full((n_bands,), math.log(sigma_sys_init))
        )

    def sigma_sys(self):
        return 1e-3 + self.log_sigma_sys.exp()

    @staticmethod
    def extended_magnitude(s, tau):
        """``-2.5 log10(s)`` for ``s >= tau``, linear in ``s`` below (C1)."""
        safe = torch.maximum(s, tau)
        return torch.where(
            s >= tau,
            -2.5 * torch.log10(safe),
            -2.5 * torch.log10(tau) - MAG * (s - tau) / tau,
        )

    def _nuisance(self, detach: bool):
        zp, k, ss = self.zp, self.k, self.sigma_sys()
        if detach:
            zp, k, ss = zp.detach(), k.detach(), ss.detach()
        return zp, k, ss

    @staticmethod
    def _colour(s, n_fi):
        sb, sr = s[:, n_fi], s[:, n_fi + 1]
        return -2.5 * torch.log10(sb / sr)

    def _observed(self, xp_syn, xp_var, g_obs, n_fi):
        """Observed-XP synthetic magnitudes, their sigma_XP and colour, and validity."""
        ok = observed_xp_valid(xp_syn, xp_var, g_obs, n_fi, self.max_sigma_xp)
        s = torch.where(ok, xp_syn[:, :n_fi], 1.0)
        m_syn = g_obs.nan_to_num()[:, None] - 2.5 * torch.log10(s)
        sig_xp2 = (
            MAG * torch.where(ok, xp_var[:, :n_fi], 0.0).clamp_min(0).sqrt() / s
        ) ** 2
        row = ok.any(1)
        col = torch.where(
            row, self._colour(torch.where(row[:, None], xp_syn, 1.0), n_fi), 0.0
        )
        return m_syn, sig_xp2, col, ok

    def calibrate(self, obs_mag, obs_err, xp_syn, xp_var, g_obs):
        """Observed XP vs observed magnitude. Returns (sum, count)."""
        n_fi = obs_mag.shape[1]
        m_syn, sig_xp2, col, ok = self._observed(xp_syn, xp_var, g_obs, n_fi)
        ok = ok & torch.isfinite(obs_mag)
        if not bool(ok.any()):
            return obs_mag.new_zeros(()), 0
        zp, k, ss = self._nuisance(False)
        r = obs_mag - (m_syn + zp + k * (col[:, None] - self.c_ref))
        var = torch.where(ok, obs_err, 0.0).nan_to_num() ** 2 + sig_xp2 + ss**2
        nll = student_t_nll(torch.where(ok, r, 0.0), var, self.nu)
        return nll[ok].sum(), int(ok.sum())

    def sed(self, obs_mag, obs_err, dec_syn, dec_var, g_obs, g_err):
        """Decoded XP (relative fluxes ``dec_syn`` [n, n_fi+2]) vs observed magnitude."""
        n_fi = obs_mag.shape[1]
        ok = torch.isfinite(obs_mag) & torch.isfinite(g_obs)[:, None]
        if not bool(ok.any()):
            return obs_mag.new_zeros(()), 0
        obs0 = obs_mag.nan_to_num()
        g0 = g_obs.nan_to_num()
        tau = (
            self.linear_below * torch.pow(10.0, -0.4 * (obs0 - g0[:, None]))
        ).detach()
        s = dec_syn[:, :n_fi]
        col = self._colour(dec_syn.detach().clamp_min(1e-3), n_fi)
        m_syn = g0[:, None] + self.extended_magnitude(s, tau)
        sig_xp2 = (
            MAG * dec_var[:, :n_fi].clamp_min(0).sqrt() / torch.maximum(s, tau)
        ).detach() ** 2
        zp, k, ss = self._nuisance(True)
        r = obs0 - (m_syn + zp + k * (col[:, None] - self.c_ref))
        ge = torch.where(torch.isfinite(g_err), g_err, torch.zeros_like(g_err))
        var = obs_err.nan_to_num() ** 2 + ge[:, None] ** 2 + sig_xp2 + ss**2
        nll = student_t_nll(torch.where(ok, r, 0.0), var, self.nu)
        return nll[ok].sum(), int(ok.sum())

    def phot(self, dec_mag, dec_sigma, use, xp_syn, xp_var, g_obs):
        """Observed XP vs decoded magnitude where ``use`` [n, n_fi] (band unseen)."""
        n_fi = dec_mag.shape[1]
        m_syn, sig_xp2, col, ok = self._observed(xp_syn, xp_var, g_obs, n_fi)
        ok = ok & use
        if not bool(ok.any()):
            return dec_mag.new_zeros(()), 0
        zp, k, ss = self._nuisance(True)
        r = dec_mag - (m_syn + zp + k * (col[:, None] - self.c_ref))
        var = dec_sigma.detach()[None, :] ** 2 + sig_xp2 + ss**2
        nll = student_t_nll(torch.where(ok, r, 0.0), var, self.nu)
        return nll[ok].sum(), int(ok.sum())


def observed_xp_valid(
    xp_syn, xp_var, g_obs, n_fi: int, max_sigma_mag: float
) -> torch.Tensor:
    """[n, n_fi]: observed-XP synthetic magnitude usable: G and the BP/RP colour fluxes
    finite and positive, band flux positive, and sigma_XP <= max_sigma_mag (beyond it
    the first-order magnitude error of a noisy integral is not valid)."""
    s = xp_syn[:, :n_fi]
    row = (
        torch.isfinite(xp_syn).all(1)
        & (xp_syn[:, n_fi:] > 0).all(1)
        & torch.isfinite(g_obs)
    )
    pos = torch.isfinite(s) & (s > 0)
    sig = MAG * xp_var[:, :n_fi].clamp_min(0).sqrt() / torch.where(pos, s, 1.0)
    return row[:, None] & pos & torch.isfinite(sig) & (sig <= max_sigma_mag)


def calibrate_count(obs_mag, xp_syn, xp_var, g_obs, max_sigma_mag: float) -> int:
    ok = observed_xp_valid(xp_syn, xp_var, g_obs, obs_mag.shape[1], max_sigma_mag)
    return int((ok & torch.isfinite(obs_mag)).sum())


def sed_count(obs_mag, g_obs) -> int:
    return int((torch.isfinite(obs_mag) & torch.isfinite(g_obs)[:, None]).sum())


def phot_count(use, xp_syn, xp_var, g_obs, max_sigma_mag: float) -> int:
    ok = observed_xp_valid(xp_syn, xp_var, g_obs, use.shape[1], max_sigma_mag)
    return int((ok & use).sum())
