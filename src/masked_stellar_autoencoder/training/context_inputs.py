"""Per-star metadata context for the anchored head (``StudentTHead``).

Builds the (values, visible) pair that ``ContextLatent`` adds to the latent,
from clean (unmasked, unperturbed) inputs. Parts, all optional (``spec``):

* ``features``: scaled input columns (e.g. ``EBV``, the SFD map value);
  ``errors``: their scaled error columns (visible where the feature is).
* ``labels``: own-label values and errors as context (e.g. ``parallax``,
  log10 mas). Only for experiments: a label used as context is copied by the
  head, and prediction needs the label column.
* ``posenc``: Galactic position as random Fourier features of the unit
  vector on the sphere (multi-scale, no RA wrap or pole artefacts).
* ``dust``: a dust branch predicting the star's E(B-V) as a fraction of the
  SFD value along its line of sight, from Galactic position, distance
  (parallax label) and SFD. It is fitted to the E(B-V) labels only (its
  inputs are data, and its output enters the context detached), so it learns
  dust, not stellar populations; the context gets its E(B-V) and scale.

Sky position enters only through ``posenc`` or the dust branch, so its use is
explicit; drop RA/Dec from the encoder (``drop_input_features``) to keep it
out of the stellar-parameter path.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

# ICRS -> Galactic rotation (Hipparcos convention, as astropy).
_ICRS_TO_GAL = (
    (-0.0548755604162154, -0.8734370902348850, -0.4838350155487132),
    (+0.4941094278755837, -0.4448296299600112, +0.7469822444972189),
    (-0.8676661490190047, -0.1980763734312015, +0.4559837761750669),
)


def galactic_unit_vector(ra_deg: torch.Tensor, dec_deg: torch.Tensor) -> torch.Tensor:
    """(N, 3) Galactic Cartesian unit vectors from ICRS RA/Dec in degrees."""
    ra, dec = torch.deg2rad(ra_deg), torch.deg2rad(dec_deg)
    v = torch.stack(
        [
            torch.cos(dec) * torch.cos(ra),
            torch.cos(dec) * torch.sin(ra),
            torch.sin(dec),
        ],
        dim=-1,
    )
    R = torch.as_tensor(_ICRS_TO_GAL, dtype=v.dtype, device=v.device)
    return v @ R.T


class SphereFourier(nn.Module):
    """Random Fourier features of unit vectors at several angular scales.

    For each scale s (radians^-1), ``n`` frequencies ~ N(0, s^2 I_3); the
    features are sin/cos of their projections. Scale s resolves structure of
    roughly 1/s radians (s = 64: about 1 degree).
    """

    def __init__(self, scales=(1.0, 4.0, 16.0, 64.0), n: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        freq = torch.cat(
            [float(s) * torch.randn(int(n), 3, generator=g) for s in scales]
        )
        self.register_buffer("freq", freq)
        self.dim = 2 * len(freq)

    def forward(self, unit: torch.Tensor) -> torch.Tensor:
        p = unit @ self.freq.T
        return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)


class DustBranch(nn.Module):
    """E(B-V) of a star = 1.2 SFD sigmoid(f(position, distance, SFD)), with scale.

    Inputs: Galactic unit vector, log10 parallax (mas; missing -> sentinel),
    log10 SFD. The fraction of the line-of-sight dust in front of the star is
    learned; 1.2 allows for the SFD calibration (Schlafly & Finkbeiner 2011).
    """

    def __init__(self, scales=(1.0, 4.0, 16.0, 64.0), n=32, hidden=128, seed=1):
        super().__init__()
        self.enc = SphereFourier(scales, n, seed)
        self.net = nn.Sequential(
            nn.Linear(self.enc.dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, unit, log_plx, sfd):
        has = torch.isfinite(log_plx)
        lp = torch.where(has, log_plx, torch.full_like(log_plx, -1.5))
        lsfd = torch.log10(sfd.clamp_min(1e-3))
        h = self.net(
            torch.cat(
                [self.enc(unit), lp[:, None], has[:, None].float(), lsfd[:, None]], -1
            )
        )
        ebv = 1.2 * sfd * torch.sigmoid(h[:, 0])
        sd = nn.functional.softplus(h[:, 1]) + 0.005
        return ebv, sd


class ContextInputs(nn.Module):
    def __init__(
        self, spec: dict, feature_cols, scaler, label_names=(), label_scalers=()
    ):
        super().__init__()
        self.spec = dict(spec)
        cols = list(feature_cols)
        self.f_idx = [cols.index(c) for c in spec.get("features", [])]
        self.e_idx = [cols.index(c) for c in spec.get("errors", [])]
        labels = list(label_names)
        self.l_idx = [labels.index(c) for c in spec.get("labels", [])]
        center = np.asarray(getattr(scaler, "center_", np.zeros(len(cols))), "f4")
        scale = np.asarray(getattr(scaler, "scale_", np.ones(len(cols))), "f4")
        scale = np.where(scale <= 0, 1.0, scale)
        self.register_buffer("center", torch.as_tensor(center))
        self.register_buffer("scale", torch.as_tensor(scale))
        self.need_sky = bool(spec.get("posenc") or spec.get("dust"))
        if self.need_sky:
            self.ra, self.dec = cols.index("RA"), cols.index("DEC")
        self.posenc = SphereFourier(**spec["posenc"]) if spec.get("posenc") else None
        self.dust = None
        if spec.get("dust"):
            self.dust = DustBranch(**spec["dust"])
            self.sfd = cols.index("EBV")
            self.plx_label = labels.index("parallax")
            self.ebv_label = labels.index("ebv_star")
        if label_scalers:
            lm = [float(s.mean_[0]) for s in label_scalers]
            ls = [float(s.scale_[0]) for s in label_scalers]
        else:
            lm, ls = [0.0] * len(labels), [1.0] * len(labels)
        self.register_buffer("label_mean", torch.as_tensor(lm, dtype=torch.float32))
        self.register_buffer("label_scale", torch.as_tensor(ls, dtype=torch.float32))
        self.dim = len(self.f_idx) + len(self.e_idx) + 2 * len(self.l_idx)
        self.dim += self.posenc.dim if self.posenc is not None else 0
        self.dim += 2 if self.dust is not None else 0

    @classmethod
    def from_spec(cls, spec, feature_cols, scaler, label_names=(), label_scalers=()):
        return cls(spec, feature_cols, scaler, label_names, label_scalers)

    def raw(self, X, j):
        return X[:, j] * self.scale[j] + self.center[j]

    def label_raw(self, y, j):
        return y[:, j] * self.label_scale[j] + self.label_mean[j]

    def forward(self, X, E=None, y=None, e_y=None):
        """((values, visible) for ContextLatent, dust (E(B-V), sd) or None)."""
        vals, vis = [], []
        for j in self.f_idx:
            vals.append(X[:, j])
            vis.append(torch.isfinite(X[:, j]))
        for j in self.e_idx:
            if E is None:
                raise ValueError("This context needs the feature errors")
            vals.append(E[:, j])
            vis.append(torch.isfinite(X[:, j]) & torch.isfinite(E[:, j]))
        for j in self.l_idx:
            if y is None:
                raise ValueError("This context needs label values")
            v, e = (
                y[:, j],
                (e_y[:, j] if e_y is not None else torch.zeros_like(y[:, j])),
            )
            ok = torch.isfinite(v)
            vals += [v, e]
            vis += [ok, ok & torch.isfinite(e)]
        dust = None
        if self.need_sky:
            unit = galactic_unit_vector(self.raw(X, self.ra), self.raw(X, self.dec))
            sky_ok = torch.isfinite(unit).all(-1)
            unit = torch.nan_to_num(unit)
            if self.posenc is not None:
                f = self.posenc(unit)
                vals += list(f.T)
                vis += [sky_ok] * f.shape[1]
            if self.dust is not None:
                if y is None:
                    raise ValueError("The dust branch needs the parallax label")
                log_plx = self.label_raw(y, self.plx_label)
                sfd = self.raw(X, self.sfd)
                ok = sky_ok & torch.isfinite(sfd)
                ebv, sd = self.dust(unit, log_plx, torch.nan_to_num(sfd))
                dust = (torch.where(ok, ebv, torch.nan), sd)
                vals += [ebv.detach(), torch.log(sd.detach())]
                vis += [ok, ok]
        values = torch.stack(vals, dim=-1) if vals else X.new_zeros(len(X), 0)
        visible = torch.stack(vis, dim=-1) if vis else values.bool()
        return (values, visible), dust

    def dust_loss(self, dust, y, e_y):
        """Gaussian NLL of the E(B-V) labels under the dust branch."""
        ebv, sd = dust
        target = self.label_raw(y, self.ebv_label)
        err = e_y[:, self.ebv_label] * self.label_scale[self.ebv_label]
        ok = torch.isfinite(target) & torch.isfinite(ebv) & torch.isfinite(err)
        if not ok.any():
            return ebv.new_zeros(())
        var = sd[ok] ** 2 + err[ok] ** 2
        r = target[ok] - ebv[ok]
        return 0.5 * (r * r / var + torch.log(var) + math.log(2 * math.pi)).mean()
