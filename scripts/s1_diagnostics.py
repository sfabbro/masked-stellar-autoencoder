"""Diagnose the S1 likelihood-head failure on frozen v8-without-pool latents.

usage: s1_diagnostics.py --s1-dir DIR --fits FINETUNE.fits --output DIR [--device cuda]

S1 failed two ways (report in DIR/report.md): (1) for ~11 % of stars the MLE
ran to the upper [Fe/H] edge of the box, outside the training labels' support,
where the ratio classifier is unconstrained; (2) metal-poor stars stayed noisy
and biased (+0.15 to +0.21 dex) even away from the edges, unlike a 1-D
prototype on v7 latents (+0.11 / -0.01). This job separates the causes, all on
the held-out frozen core (valid + test stars with a high-resolution [Fe/H]):

- D1: 1-D [Fe/H] head, raw high-resolution reference labels (``fe_h_ref``) —
  the prototype replayed on the v8-without-pool latents (latents);
- D2: 1-D [Fe/H] head, the same labels mapped to the common scale
  (``fe_h_core``) (label mapping);
- D3: 3-D (Teff, log g, [Fe/H]) head, raw labels (table Teff/log g + raw
  reference) and D4: 3-D, mapped labels (dimension, with either label kind);
- D5: S1's trained ``all`` head, inference re-run (support fix only).

Every MLE is restricted to the support of the training labels: cells of a
dilated occupancy grid of the training theta (no prior shape; outside the
labels' support the learned ratio is undefined). 1-D arms use an exact grid
MLE; 3-D arms use ``torchregress.inference.maximize_score`` with the score set
to NaN outside the support. "At edge": 1-D, MLE at the support's lowest or
highest [Fe/H]; 3-D, MLE on a box bound. Writes ``diagnostics.md`` and
``diagnostics.json``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from masked_stellar_autoencoder.training.likelihood_head import (
    RatioHead,
    embed_rows,
    load_head,
    train_ratio_head,
)

BOUNDS3 = ((3000.0, 8000.0), (-0.5, 5.5), (-4.5, 1.0))
WIDTHS3 = (100.0, 0.1, 0.1)
FEH_BINS = (-9, -3, -2.5, -2, -1.5, -1, -0.5, 9)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nmad(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


class Support:
    """Dilated occupancy grid of training labels: ``inside(theta)`` -> bool."""

    def __init__(self, theta, bounds, widths, device, dilate=1, min_count=1):
        self.lo = torch.tensor(
            [b[0] for b in bounds], dtype=torch.float64, device=device
        )
        self.w = torch.tensor(widths, dtype=torch.float64, device=device)
        self.n = [
            int(np.ceil((b[1] - b[0]) / w)) for b, w in zip(bounds, widths, strict=True)
        ]
        idx = np.floor((theta - self.lo.cpu().numpy()) / np.asarray(widths)).astype(int)
        idx = np.clip(idx, 0, np.asarray(self.n) - 1)
        counts = np.zeros(self.n, dtype=np.int64)
        np.add.at(counts, tuple(idx.T), 1)
        occ = counts >= min_count
        for _ in range(dilate):  # max filter over the 3^P neighbourhood
            grown = occ.copy()
            for ax in range(occ.ndim):
                grown |= np.roll(occ, 1, axis=ax) | np.roll(occ, -1, axis=ax)
            occ = grown
        self.grid = torch.as_tensor(occ, device=device)
        self.fraction = float(occ.mean())

    def inside(self, theta):
        idx = torch.floor((theta.double() - self.lo) / self.w).long()
        n = torch.tensor(self.n, device=theta.device)
        ok = ((idx >= 0) & (idx < n)).all(-1)
        idx = torch.minimum(idx.clamp_min(0), n - 1)
        return ok & self.grid[tuple(idx.T)]


def grid_mle_1d(head, emb, support, lo=-4.5, hi=1.0, step=0.01, chunk=2048):
    """Exact grid MLE and Wilks 68 % interval for a 1-D head."""
    grid = torch.arange(lo, hi + step / 2, step, dtype=torch.float32, device=emb.device)
    ok = support.inside(grid[:, None])
    mle, lo68, hi68, atb = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(emb), chunk):
            e = emb[i : i + chunk]
            th = grid[None, :, None].expand(len(e), -1, -1).reshape(-1, 1)
            ee = e[:, None, :].expand(-1, len(grid), -1).reshape(-1, e.shape[1])
            s = head.score_embedded(th, ee).view(len(e), len(grid))
            s = torch.where(ok[None, :], s, torch.full_like(s, -torch.inf))
            smax, j = s.max(1)
            mle.append(grid[j])
            inset = s >= (smax[:, None] - 0.5)
            first = inset.float().argmax(1)
            last = len(grid) - 1 - inset.flip(1).float().argmax(1)
            lo68.append(grid[first])
            hi68.append(grid[last])
            okidx = torch.nonzero(ok).squeeze(1)
            atb.append((j == okidx.min()) | (j == okidx.max()))
    cat = lambda xs: torch.cat(xs).cpu().numpy()  # noqa: E731
    return cat(mle), cat(lo68), cat(hi68), cat(atb)


def mle_3d(head, emb, support, init, block=100_000, seed=0):
    from torchregress.inference import maximize_score

    box = torch.as_tensor(BOUNDS3, dtype=emb.dtype, device=emb.device)

    def score(theta, e):
        s = head.score_embedded(theta, e)
        return torch.where(support.inside(theta), s, torch.full_like(s, torch.nan))

    out_t, out_b = [], []
    for i in range(0, len(emb), block):
        sl = slice(i, i + block)
        ini = torch.as_tensor(init[sl], dtype=emb.dtype, device=emb.device)
        ini = torch.minimum(torch.maximum(ini, box[:, 0]), box[:, 1])
        ini = torch.where(torch.isfinite(ini), ini, box.mean(1))
        res = maximize_score(score, emb[sl], box, init=ini,
                             generator=torch.Generator().manual_seed(seed + i))  # fmt: skip
        out_t.append(res.theta.cpu())
        out_b.append(res.at_bound.cpu())
    return torch.cat(out_t).double().numpy(), torch.cat(out_b).numpy()


def offsets(est, ref, gmag):
    rows = {}
    for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:], strict=True):
        cell = {}
        for gname, gm in (("G<=16", gmag <= 16), ("G>16", gmag > 16)):
            m = (ref >= lo) & (ref < hi) & gm & np.isfinite(est)
            d = est[m] - ref[m]
            cell[gname] = dict(n=int(m.sum()), bias=float(np.median(d)) if m.any() else None,
                               nmad=nmad(d))  # fmt: skip
        rows[f"[{lo},{hi})"] = cell
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s1-dir", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bulk-sample", type=int, default=100_000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    lab = np.load(a.s1_dir / "labels.npz")
    pred = np.load(a.s1_dir / "predictions.npz")
    lat = np.load(a.s1_dir / "latent.npy", mmap_mode="r")
    split = lab["split"]
    with fits.open(a.fits, memmap=True) as h:
        d = h[1].data
        gmag = np.asarray(d["G"], dtype=float)
        teff_raw = np.asarray(d["teff"], dtype=float)
        logg_raw = np.asarray(d["logg"], dtype=float)
    rows_lab = lab["table_row"]
    if not np.array_equal(rows_lab, np.arange(len(rows_lab))) or len(lat) != len(
        rows_lab
    ):
        raise ValueError("labels.npz / latent.npy are not aligned with the table rows")
    gmag, teff_raw, logg_raw = gmag[rows_lab], teff_raw[rows_lab], logg_raw[rows_lab]
    post = np.asarray(pred["median"] if "median" in pred.files else pred["post_median"])
    post = post[:, :3] if post.ndim == 2 else post
    has_core = lab["has_core_feh"].astype(bool)
    ref_raw, ref_map = lab["fe_h_ref"], lab["fe_h_core"]
    held = has_core & np.isin(split, (1, 2)) & np.isfinite(ref_raw)
    log(f"held-out core stars: {held.sum()}")
    report = dict(posterior=offsets(post[held, 2], ref_raw[held], gmag[held]), arms={})

    def arm_data(kind, raw):
        feh = ref_raw if raw else ref_map
        if kind == "1d":
            th = feh[:, None]
        else:
            th = np.column_stack([teff_raw, logg_raw, feh] if raw else
                                 [lab["teff"], lab["logg"], feh])  # fmt: skip
        box = np.asarray(BOUNDS3 if kind == "3d" else BOUNDS3[2:])
        ok = has_core & np.isfinite(th).all(1)
        ok &= np.all((th >= box[:, 0]) & (th <= box[:, 1]), axis=1)
        return th, ok

    for name, kind, raw in (("D1_1d_raw", "1d", True), ("D2_1d_mapped", "1d", False),
                            ("D3_3d_raw", "3d", True), ("D4_3d_mapped", "3d", False)):  # fmt: skip
        th, ok = arm_data(kind, raw)
        tr, va = np.flatnonzero(ok & (split == 0)), np.flatnonzero(ok & (split == 1))
        ev = np.flatnonzero(held)
        log(f"{name}: {len(tr)} train, {len(va)} valid, {len(ev)} evaluated")
        P = th.shape[1]
        support = Support(th[tr], BOUNDS3 if P == 3 else BOUNDS3[2:],
                          WIDTHS3 if P == 3 else WIDTHS3[2:], dev)  # fmt: skip
        seeds = []
        for seed in range(a.seeds):
            rows = np.concatenate([tr, va])
            Z = torch.as_tensor(np.asarray(lat[rows_lab[rows]]), device=dev)
            T = torch.as_tensor(th[rows], dtype=torch.float32, device=dev)
            head = RatioHead(lat.shape[1], n_params=P)
            train_ratio_head(head, Z, T, torch.arange(len(tr), device=dev),
                             torch.arange(len(tr), len(rows), device=dev), seed=seed,
                             log=lambda m: None)  # fmt: skip
            head.eval()
            emb = embed_rows(head, lat, rows_lab[ev], device=dev)
            if P == 1:
                mle, lo68, hi68, atb = grid_mle_1d(head, emb, support)
                feh_mle = mle
                truth = ref_raw[ev] if raw else ref_map[ev]
                cov = float(np.mean((truth >= lo68) & (truth <= hi68)))
            else:
                mle, atbm = mle_3d(head, emb, support, post[ev], seed=seed)
                feh_mle, atb, cov = mle[:, 2], atbm.any(1), None
            seeds.append(dict(seed=seed, at_bound=float(np.mean(atb)), wilks68_cov=cov,
                              offsets=offsets(feh_mle, ref_raw[ev], gmag[ev])))  # fmt: skip
            log(f"  seed {seed}: at bound {np.mean(atb):.3f}, cov {cov}")
            del Z, T, emb
        report["arms"][name] = dict(n_train=len(tr), support_fraction=support.fraction,
                                    seeds=seeds)  # fmt: skip

    # D5: S1's all-arm head, inference re-run with the support restriction.
    head, _ = load_head(a.s1_dir / "head_all.pt", device=dev)
    th_all = np.column_stack([lab["teff"], lab["logg"], lab["fe_h"]])
    box = np.asarray(BOUNDS3)
    ok = np.isfinite(th_all).all(1) & np.all(
        (th_all >= box[:, 0]) & (th_all <= box[:, 1]), 1
    )
    support = Support(th_all[ok & (split == 0)], BOUNDS3, WIDTHS3, dev)
    rng = np.random.default_rng(0)
    other = np.flatnonzero(np.isin(split, (1, 2)) & ~has_core & ok)
    bulk = np.sort(
        rng.choice(other, size=min(a.bulk_sample, len(other)), replace=False)
    )
    for tag, ev in (("core", np.flatnonzero(held)), ("bulk", bulk)):
        emb = embed_rows(head, lat, rows_lab[ev], device=dev)
        mle, atbm = mle_3d(head, emb, support, post[ev])
        entry = dict(n=len(ev), at_bound=float(atbm.any(1).mean()),
                     at_feh_upper=float(np.isclose(mle[:, 2], BOUNDS3[2][1]).mean()))  # fmt: skip
        if tag == "core":
            entry["offsets"] = offsets(mle[:, 2], ref_raw[ev], gmag[ev])
        else:
            for j, nm_ in enumerate(("teff", "logg", "fe_h")):
                dlt = mle[:, j] - th_all[ev, j]
                g16 = gmag[ev] <= 16
                entry[nm_] = dict(bias=float(np.median(dlt[g16])), nmad=nmad(dlt[g16]),
                                  post_nmad=nmad(post[ev][g16, j] - th_all[ev][g16, j]))  # fmt: skip
        report["arms"][f"D5_all_support_{tag}"] = entry
        log(f"D5 {tag}: at bound {entry['at_bound']:.3f}")
    report["support_fraction_all"] = support.fraction
    (a.output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")

    # Markdown summary: median over seeds of bias / NMAD per bin (G <= 16 and > 16).
    lines = [
        "# S1 diagnostics (held-out frozen core; MLE restricted to the training support)",
        "",
    ]
    hdr = (
        "| estimator | "
        + " | ".join(f"{b} G<=16" for b in report["posterior"])
        + " | at edge |"
    )
    lines += [hdr, "|" + "---|" * (len(report["posterior"]) + 2)]

    def row(name, offs_list, atb):
        cells = []
        for b in report["posterior"]:
            bs = [
                o[b]["G<=16"]["bias"]
                for o in offs_list
                if o[b]["G<=16"]["bias"] is not None
            ]
            ns = [o[b]["G<=16"]["nmad"] for o in offs_list]
            n = offs_list[0][b]["G<=16"]["n"]
            cells.append(
                f"{np.median(bs):+.2f}/{np.nanmedian(ns):.2f} ({n})" if bs else "-"
            )
        return f"| {name} | " + " | ".join(cells) + f" | {atb} |"

    lines.append(row("posterior head", [report["posterior"]], "-"))
    for name, arm in report["arms"].items():
        if "seeds" in arm:
            offs = [s["offsets"] for s in arm["seeds"]]
            atb = f"{np.median([s['at_bound'] for s in arm['seeds']]):.3f}"
            cov = [
                s["wilks68_cov"] for s in arm["seeds"] if s["wilks68_cov"] is not None
            ]
            label = name + (f" (Wilks68 cov {np.median(cov):.2f})" if cov else "")
            lines.append(row(label, offs, atb))
        elif "offsets" in arm:
            lines.append(row(name, [arm["offsets"]], f"{arm['at_bound']:.3f}"))
    b = report["arms"]["D5_all_support_bulk"]
    lines += [
        "",
        f"D5 bulk (held-out non-core sample, n={b['n']}, G<=16, vs mapped labels): "
        + ", ".join(
            f"{k} MLE {b[k]['bias']:+.3f}/{b[k]['nmad']:.3f} vs posterior NMAD "
            f"{b[k]['post_nmad']:.3f}"
            for k in ("teff", "logg", "fe_h")
        )
        + f"; at bound {b['at_bound']:.3f}, at [Fe/H] upper edge {b['at_feh_upper']:.3f}.",
    ]
    (a.output / "diagnostics.md").write_text("\n".join(lines) + "\n")
    log("wrote diagnostics.md")


if __name__ == "__main__":
    main()
