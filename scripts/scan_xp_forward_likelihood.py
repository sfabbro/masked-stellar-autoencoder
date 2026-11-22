"""Likelihood scans of a trained XP forward model around the labels of real stars.

Inference of run 2 (20261122) fails on real stars although the exact likelihood
prefers the labels and the passes converge when started there. This script maps
the likelihood around the labels of ``--n`` held-out stars to find out why the
start ranking fails and whether a spurious optimum at low ``E`` exists:

* 1-D scans in Teff (``--teff-step``, +-``--teff-span``) and [Fe/H] with the other
  two parameters at their labels and ``(E, distance)`` profiled, under (a) the
  exact likelihood (learned covariance at each point's own prediction, one
  covariance refresh in the profile) and (b) the known covariance alone (the
  start ranking's covariance).
* a 2-D scan in (Teff, E) with log g and [Fe/H] at their labels and the
  distance profiled, under the exact likelihood.

Summaries (``scan.md``, ``scan.json``) and the raw scans (``scans.npz``) go to
``--output``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import masked_stellar_autoencoder.training.xp_forward_model as xf


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--covariance-sidecar")
    p.add_argument("--reddening-labels")
    p.add_argument("--core-rows")
    p.add_argument("--bands", default=",".join(xf.LIKELIHOOD_BANDS))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--teff-span", type=float, default=1000.0)
    p.add_argument("--teff-step", type=float, default=50.0)
    p.add_argument("--feh-lo", type=float, default=-1.5)
    p.add_argument("--feh-hi", type=float, default=1.0)
    p.add_argument("--feh-step", type=float, default=0.1)
    p.add_argument("--e-lo", type=float, default=-0.3)
    p.add_argument("--e-step", type=float, default=0.02)
    p.add_argument("--grid-teff-step", type=float, default=100.0)
    p.add_argument("--grid-teff-span", type=float, default=800.0)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _profile(model, data, theta, device, *, known_only: bool, steps: int = 12):
    """(E, parallax, exact NLL, chi2_xp, dof) with (E, mu) profiled at ``theta``.

    Three E starts (0, mid-box, upper bound); the best end point is kept. With
    ``known_only`` the profile uses the known covariance alone and the returned
    NLL is that covariance's; otherwise the covariance is refreshed once at the
    end point and the exact NLL (:func:`star_nll`) is returned.
    """
    dt = torch.float64
    out = {k: [] for k in ("ebv", "parallax", "nll", "chi2", "dof")}
    with torch.no_grad():
        for s in range(0, len(data), 2048):
            idx = np.arange(s, min(s + 2048, len(data)))
            batch = data.batch(idx, device, dt)
            th = torch.as_tensor(theta[idx], device=device, dtype=dt)
            n = len(idx)
            e_min, e_max = batch["e_min"], batch["e_max"]
            starts = torch.stack(
                [
                    torch.zeros_like(e_max).clamp(e_min, e_max),
                    0.5 * (e_min + e_max),
                    e_max,
                ]
            ).reshape(-1)
            rows = torch.arange(n, device=device).repeat(3)
            b3 = {k: v[rows] for k, v in batch.items()}
            th3 = th[rows]
            ref = None if known_only else xf.model_reference(model, th3, starts)
            prep = xf.prepare_stars(model, b3, reference=ref, dtype=dt)
            mu0 = xf.distance_modulus_start(model, th3, starts, prep)
            ebv, mu, f = xf.fit_nuisances(
                model, th3, prep, starts, mu0, steps=steps, anchor=False
            )
            best = torch.nan_to_num(f.view(3, n), nan=math.inf).argmin(0)
            pick = best * n + torch.arange(n, device=device)
            ebv, mu = ebv[pick], mu[pick]
            if known_only:
                prep1 = prep.take(pick)
            else:
                prep1 = xf.prepare_stars(
                    model, batch, reference=xf.model_reference(model, th, ebv), dtype=dt
                )
                ebv, mu, _ = xf.fit_nuisances(
                    model, th, prep1, ebv, mu, steps=steps, anchor=False
                )
            plx = 100.0 * torch.pow(10.0, -mu / 5.0)
            if known_only:
                parts = xf.star_nll_prepared(model, th, ebv, plx, prep1, parts=True)
            else:
                parts = xf.star_nll(model, th, ebv, plx, batch, parts=True)
            out["ebv"].append(ebv.cpu().numpy())
            out["parallax"].append(plx.cpu().numpy())
            out["nll"].append(parts["nll"].cpu().numpy())
            out["chi2"].append(parts["chi2_xp"].cpu().numpy())
            out["dof"].append(parts["dof_xp"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _grid_nll(model, data, theta, ebv, device, steps: int = 8):
    """Exact NLL at fixed (theta, E) with the distance profiled (E held)."""
    dt = torch.float64
    out = []
    with torch.no_grad():
        for s in range(0, len(data), 4096):
            idx = np.arange(s, min(s + 4096, len(data)))
            batch = data.batch(idx, device, dt)
            th = torch.as_tensor(theta[idx], device=device, dtype=dt)
            e = torch.as_tensor(ebv[idx], device=device, dtype=dt)
            prep = xf.prepare_stars(
                model, batch, reference=xf.model_reference(model, th, e), dtype=dt
            )
            mu0 = xf.distance_modulus_start(model, th, e, prep)
            _, mu, _ = xf.fit_nuisances(model, th, prep, e, mu0, steps=steps, anchor=False,
                                        hold_ebv=True)  # fmt: skip
            plx = 100.0 * torch.pow(10.0, -mu / 5.0)
            out.append(xf.star_nll(model, th, e, plx, batch).cpu().numpy())
    return np.concatenate(out)


def _local_minima(curve: np.ndarray) -> int:
    c = np.where(np.isfinite(curve), curve, np.inf)
    inner = (c[1:-1] < c[:-2]) & (c[1:-1] < c[2:])
    return int(inner.sum() + (c[0] < c[1]) + (c[-1] < c[-2]))


def _summarise(name, offsets, nll, ebv, e_min) -> dict:
    """Per-scan summaries over stars: argmin offset, minima count, E at argmin."""
    i0 = int(np.argmin(np.abs(offsets)))
    am = np.nanargmin(np.where(np.isfinite(nll), nll, np.inf), axis=1)
    off = offsets[am]
    gap = nll[:, i0] - nll[np.arange(len(nll)), am]
    e_at = ebv[np.arange(len(nll)), am]
    step = float(np.median(np.diff(offsets)))
    return {
        "scan": name,
        "argmin_offset_median": float(np.median(off)),
        "argmin_within_2_steps": float(np.mean(np.abs(off) <= 2 * step + 1e-9)),
        "argmin_at_scan_edge": float(np.mean((am == 0) | (am == len(offsets) - 1))),
        "label_minus_min_nll_median": float(np.median(gap)),
        "local_minima_median": float(np.median([_local_minima(r) for r in nll])),
        "multimodal_fraction": float(np.mean([_local_minima(r) > 1 for r in nll])),
        "E_at_argmin_on_lower_bound": float(np.mean(e_at <= e_min + 1e-6)),
    }


def main(argv=None) -> None:
    a = parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = xf.load_labels(a.labels)
    core = (
        np.load(a.core_rows).astype(np.int64) if a.core_rows else np.zeros(0, np.int64)
    )
    sel = xf.select_rows(
        labels, core, max_train=0, max_valid=0, max_heldout=a.n, seed=a.seed
    )
    sidecar = (
        xf.SidecarXPCovariance(a.covariance_sidecar) if a.covariance_sidecar else None
    )
    reddening = xf.ReddeningLabels(a.reddening_labels) if a.reddening_labels else None
    bands = xf.bands_by_name([b for b in a.bands.split(",") if b])
    data, _ = xf.load_star_data(
        a.table, sel["heldout"], labels=labels, bands=bands, sidecar=sidecar,
        reddening=reddening, zero_point="sidecar" if sidecar else "constant", log=log,
    )  # fmt: skip
    data = data.subset(np.flatnonzero(np.isfinite(data.theta).all(axis=1)))
    n = len(data)
    lab = data.theta
    log(f"{n} labelled held-out stars")
    model, _ = xf.load_model(a.checkpoint)
    model = model.to(device=device, dtype=torch.float64).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def scan(param: int, offsets: np.ndarray, *, absolute: bool = False):
        m = len(offsets)
        rep = data.subset(np.repeat(np.arange(n), m))
        theta = np.repeat(lab, m, axis=0)
        grid = np.tile(offsets, n)
        theta[:, param] = grid if absolute else theta[:, param] + grid
        res = {}
        for mode in ("exact", "known"):
            r = _profile(model, rep, theta, device, known_only=mode == "known")
            res[mode] = {k: v.reshape(n, m) for k, v in r.items()}
        return res

    teff_off = np.arange(-a.teff_span, a.teff_span + 1e-6, a.teff_step)
    feh_off = np.arange(a.feh_lo, a.feh_hi + 1e-6, a.feh_step)
    t0 = time.time()
    st = scan(0, teff_off)
    log(f"Teff scans done ({time.time() - t0:.0f} s)")
    sf = scan(2, feh_off)
    log(f"[Fe/H] scans done ({time.time() - t0:.0f} s)")
    # 2-D (Teff, E) grid, distance profiled, exact likelihood.
    g_teff = np.arange(-a.grid_teff_span, a.grid_teff_span + 1e-6, a.grid_teff_step)
    e_hi = float(
        np.quantile(
            data.batch(np.arange(n), torch.device("cpu"), torch.float64)[
                "e_max"
            ].numpy(),
            0.9,
        )
    )
    g_e = np.arange(a.e_lo, max(e_hi, 0.3) + 1e-6, a.e_step)
    tt, ee = np.meshgrid(g_teff, g_e, indexing="ij")
    m2 = tt.size
    rep = data.subset(np.repeat(np.arange(n), m2))
    theta = np.repeat(lab, m2, axis=0)
    theta[:, 0] += np.tile(tt.ravel(), n)
    grid_nll = _grid_nll(model, rep, theta, np.tile(ee.ravel(), n), device).reshape(
        n, *tt.shape
    )
    log(f"(Teff, E) grid done ({time.time() - t0:.0f} s)")

    e_min = xf.E_MIN
    summary = {
        "n": n,
        "teff": {
            m: _summarise(f"teff/{m}", teff_off, st[m]["nll"], st[m]["ebv"], e_min)
            for m in st
        },
        "fe_h": {
            m: _summarise(f"fe_h/{m}", feh_off, sf[m]["nll"], sf[m]["ebv"], e_min)
            for m in sf
        },
    }
    flat = grid_nll.reshape(n, -1)
    am = np.nanargmin(np.where(np.isfinite(flat), flat, np.inf), axis=1)
    it, ie = np.unravel_index(am, tt.shape)
    lab_e = data.ebv_label if data.ebv_label is not None else np.full(n, np.nan)
    summary["grid"] = {
        "argmin_teff_offset_median": float(np.median(g_teff[it])),
        "argmin_teff_within_100K": float(np.mean(np.abs(g_teff[it]) <= 100 + 1e-6)),
        "argmin_E_median": float(np.median(g_e[ie])),
        "argmin_E_below_E_MIN": float(np.mean(g_e[ie] < e_min)),
        "argmin_E_minus_label_median": float(np.nanmedian(g_e[ie] - lab_e)),
        "teff_grid": g_teff.tolist(),
        "e_grid": g_e.tolist(),
    }
    g = data.g_mag
    for tag, mask in (("G<=16", g <= 16), ("G>16", g > 16)):
        if mask.any():
            summary[f"teff_exact_{tag}"] = _summarise(
                f"teff/exact {tag}",
                teff_off,
                st["exact"]["nll"][mask],
                st["exact"]["ebv"][mask],
                e_min,
            )
    np.savez_compressed(
        out / "scans.npz", teff_offsets=teff_off, feh_offsets=feh_off, grid_teff=g_teff, grid_e=g_e,
        label=lab, g_mag=g, ebv_label=lab_e, source_id=data.source_id,
        **{f"teff_{m}_{k}": v for m in st for k, v in st[m].items()},
        **{f"feh_{m}_{k}": v for m in sf for k, v in sf[m].items()},
        grid_nll=grid_nll,
    )  # fmt: skip
    (out / "scan.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# XP forward-model likelihood scans", "", f"{n} held-out stars; E_MIN = {e_min}", "",
             "| scan | argmin offset (median) | argmin within 2 steps | argmin at scan edge | "
             "NLL(label) - min (median) | local minima (median) | multimodal | E on lower bound at argmin |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    rows = [summary["teff"]["exact"], summary["teff"]["known"], summary["fe_h"]["exact"],
            summary["fe_h"]["known"]] + [summary[k] for k in summary if k.startswith("teff_exact_")]  # fmt: skip
    for r in rows:
        lines.append(
            f"| {r['scan']} | {r['argmin_offset_median']:+.3g} | {r['argmin_within_2_steps']:.2f} | "
            f"{r['argmin_at_scan_edge']:.2f} | {r['label_minus_min_nll_median']:.1f} | "
            f"{r['local_minima_median']:.0f} | {r['multimodal_fraction']:.2f} | "
            f"{r['E_at_argmin_on_lower_bound']:.2f} |"
        )
    gs = summary["grid"]
    lines += ["", f"(Teff, E) grid, exact likelihood, distance profiled: argmin Teff offset median "
              f"{gs['argmin_teff_offset_median']:+.0f} K, within 100 K {gs['argmin_teff_within_100K']:.2f}; "
              f"argmin E median {gs['argmin_E_median']:+.3f}, below E_MIN {gs['argmin_E_below_E_MIN']:.2f}, "
              f"minus label {gs['argmin_E_minus_label_median']:+.3f}", ""]  # fmt: skip
    (out / "scan.md").write_text("\n".join(lines))
    log("done")


if __name__ == "__main__":
    main()
