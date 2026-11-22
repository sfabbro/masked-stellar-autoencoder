"""S3: joint distance and extinction p(s, E | z) on frozen latents, trained on Gaia parallaxes.

usage: s3_distance_extinction.py --inputs S1_PLX_DIR --config CFG --output DIR
       [--labels LABELS.npz] [--fits FITS] [--sidecar SIDECAR.h5]
       [--reddening REDDENING.npz] [--core-rows ROWS.npy]
       [--members MEMBERS.fits --clusters-table CLUSTERS.fits]
       [--families gmix,flow] [--ensemble 5] [--max-train 3000000]
       [--stages data,clusters,train,predict,noxp,report] [--checkpoint CKPT]
       [--noxp-inputs DIR] [--predict-rows eval|all] [--device cuda]

Design note ``docs/design-20261122-data-driven-inference.md`` (sections 1-3, 6):
distance and E(B-V) as first-class, data-driven outputs, with no stellar or
Galactic model. Distance is geometry, not a label: the heads are trained on
every star's own Gaia parallax through its measurement likelihood (no
inversion, no parallax / error cut: low-S/N and negative parallaxes stay), the
empirical E(B-V) labels enter with their quoted errors (errors in variables),
and (s = ln(d / kpc), E) are modelled jointly so their degeneracy is kept
(``masked_stellar_autoencoder.training.distance_extinction``)::

    L_i = int int p(s, E | z_i) N(plx_i - zp_i | e^-s, sig_eff^2) [N(E_obs | E, e^2)] ds dE

The latents (``S1_PLX_DIR``: ``latent.npy``, ``latent_meta.json``,
``predictions.npz``) must have been computed with the PARALLAX input hidden
(S1 ``--parallax-nuisance``), so the target cannot leak through the input.

Stages (each writes into DIR; a stage whose outputs exist is skipped unless
``--force``):

1. ``data``: per-row observations (``observations.npz``): the table's parallax
   and error with the Lindegren+2021 zero point (sidecar ``parallax_zpt_l21``,
   else -0.017 mas; S1 ``read_parallax``), G, Galactic latitude, SFD E(B-V)
   (the table's ``EBV``, validation only), the empirical reddening labels
   (``--reddening``: ``ebv_star``, ``e_ebv_star``; error floor/cap 0.01/0.5 mag),
   [Fe/H] labels and the frozen high-resolution core. Split: the labels' split
   (0/1/2 = train/valid/test, defining E's held-out stars); rows the labels did
   not assign (-1) get a deterministic hash split (80/13.4/6.6 %) so that every
   star with a parallax can train while held-out stars stay held out. Training
   rows: split 0 with a parallax or an E label, natural distribution (no
   balancing), up to ``--max-train``; stop rows (early stopping): split 1, up
   to ``--max-stop`` (``selection.npz``).
2. ``clusters``: Hunt & Reffert (2024) open-cluster members (``--members``,
   ``--clusters-table``; Prob >= ``--cluster-min-prob``, Type in
   ``--cluster-types``) matched to table rows (``clusters.npz``): a label-free
   check (members share distance and E).
3. ``train``: K members per family (``heads/<family>_<k>.pt``): ``gmix`` (2-D
   full-covariance Gaussian mixture) and ``flow`` (2-D conditional spline flow,
   order (s, E)), each fitted by maximum marginal likelihood with a fitted
   parallax-error inflation per G bin; early stopping on the stop rows.
4. ``predict``: per family, ensemble (equal-weight mixture) predictions
   (``s3_predictions_<family>.npz``): predictive s and E median and 16/84 %,
   correlation, predicted parallax mean and sd, the spectrophotometric +
   Gaia-parallax combination (predictive x parallax likelihood, normalised),
   log scores of the observations, the parallax PIT under the convolved
   predictive and the E-label PIT; rows: ``--predict-rows eval`` (test split,
   held-out core, held-out cluster members) or ``all``. Also example grids of
   faint stars (``examples_<family>.npz``), a quadrature-convergence check
   (``quadrature_<family>.json``) and throughput (``speed.json``).
5. ``noxp`` (optional, "beyond XP"): latents with the XP coefficients (and
   PARALLAX) hidden (``encode_table(..., hide=...)``; needs ``--checkpoint``,
   or ``--noxp-inputs DIR`` with an existing ``latent.npy``), and the same heads
   evaluated on the test rows (``s3_noxp_predictions_<family>.npz``): how much
   distance/E precision survives for photometry-only stars.
6. ``report``: ``metrics.json``, ``report.md`` and QA plots (``qa_*.png``).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import fnmatch
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_likelihood_head import log, read_parallax  # noqa: E402
from s2_distributional_heads import COLORS, SURFACE, _figure, _save  # noqa: E402

FAMILIES = ("gmix", "flow")
G_BINS = (-math.inf, 13.0, 15.0, 16.0, 17.0, 18.0, math.inf)
G_NAMES = ("G<=13", "13-15", "15-16", "16-17", "17-18", "G>18")
SNR_BINS = (-math.inf, 1.0, 3.0, 10.0, math.inf)
SNR_NAMES = ("pred. plx/err<1", "1-3", "3-10", ">10")
PRECISE_SNR = 10.0
HIGH_LAT = 60.0
LEVELS = (0.68, 0.95)
COV_TOL = 0.03
MIN_N = 20
SPLIT_FRACTIONS = (
    0.8,
    0.934,
)  # hash split of unassigned rows: train < 0.8 <= valid < 0.934
NGP_RA, NGP_DEC = 192.85948, 27.12825  # ICRS north Galactic pole [deg]
STYLE = {"gmix": (COLORS[0], "-"), "flow": (COLORS[1], "-"),
         "gmix_noxp": (COLORS[0], "--"), "flow_noxp": (COLORS[1], "--")}  # fmt: skip
LABELS = {"gmix": "Gaussian mixture", "flow": "spline flow",
          "gmix_noxp": "Gaussian mixture, XP hidden",
          "flow_noxp": "spline flow, XP hidden"}  # fmt: skip


def _ints(text):
    return [int(x) for x in str(text).split(",") if x]


def head_paths(a, family):
    return [a.output / "heads" / f"{family}_{k}.pt" for k in range(a.ensemble)]


def _style(name):
    return STYLE.get(name, (COLORS[2], ":"))


def _label(name):
    return LABELS.get(name, name)


# ── helpers ─────────────────────────────────────────────────────────────────


def galactic_latitude(ra, dec):
    """Galactic latitude [deg] of ICRS (ra, dec) [deg]."""
    r, d = np.radians(ra), np.radians(dec)
    rg, dg = math.radians(NGP_RA), math.radians(NGP_DEC)
    sb = np.sin(d) * math.sin(dg) + np.cos(d) * math.cos(dg) * np.cos(r - rg)
    return np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0)))


def _hash_unit(source_id):
    """Deterministic uniform [0, 1) per source_id (murmur3 64-bit finalizer)."""
    x = np.asarray(source_id, dtype=np.int64).view(np.uint64).copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def s3_split(label_split, source_id):
    """The labels' split; unassigned rows (-1) get a hash split (train/valid/test)."""
    split = np.asarray(label_split).astype("i1").copy()
    free = split < 0
    u = _hash_unit(np.asarray(source_id)[free])
    split[free] = np.where(
        u < SPLIT_FRACTIONS[0], 0, np.where(u < SPLIT_FRACTIONS[1], 1, 2)
    )
    return split


def read_table_columns(path, names, chunk_rows=1 << 20):
    """Columns of a FITS table (HDU 1), float64, read in row chunks; a missing
    column is NaN (logged)."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        n = len(data)
        have = {c.upper(): c for c in data.columns.names}
        out = {}
        for name in names:
            out[name] = np.full(n, np.nan)
            if name.upper() not in have:
                log(f"table: no column {name}; NaN")
        for start in range(0, n, chunk_rows):
            chunk = data[start : start + chunk_rows]
            for name in names:
                if name.upper() in have:
                    out[name][start : start + len(chunk)] = np.asarray(
                        chunk[have[name.upper()]], dtype="f8"
                    )
    return out


def read_fits_table(path, names):
    """Whole columns of a FITS table (binary or ASCII extension), as numpy."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        have = {c.upper(): c for c in data.columns.names}
        missing = [c for c in names if c.upper() not in have]
        if missing:
            raise SystemExit(
                f"{path}: columns {missing} missing (have {list(have)[:20]})"
            )
        return {c: np.asarray(data[have[c.upper()]]) for c in names}


def _cap(rng, rows, k):
    return np.sort(rng.choice(rows, k, replace=False)) if k and len(rows) > k else rows


def _expand_hide(cols, patterns):
    out = []
    for pat in patterns:
        hit = [c for c in cols if fnmatch.fnmatchcase(c, pat)]
        if not hit:
            raise SystemExit(f"--noxp-hide pattern {pat!r} matches no input feature")
        out += [c for c in hit if c not in out]
    return out


def _latent_meta(path):
    meta = Path(path) / "latent_meta.json"
    return json.loads(meta.read_text()) if meta.exists() else None


def _bin(values, edges):
    v = np.asarray(values, dtype=np.float64)
    idx = np.searchsorted(np.asarray(edges[1:-1]), v, side="left")
    return np.where(np.isfinite(v), idx, -1)


def _nmad(x):
    x = np.asarray(x, "f8")
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else math.nan


def _summary(d):
    d = np.asarray(d, "f8")
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0, bias=math.nan, nmad=math.nan)
    return dict(n=int(len(d)), bias=float(np.median(d)), nmad=_nmad(d))


def _mean_se(x):
    x = np.asarray(x, "f8")
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(n=int(len(x)), mean=math.nan, se=math.nan, sd=math.nan)
    return dict(n=int(len(x)), mean=float(x.mean()), se=float(x.std(ddof=1) / math.sqrt(len(x))),
                sd=float(x.std(ddof=1)))  # fmt: skip


def _coverage_by(bins, names, covered, nominal):
    """Coverage per bin with Bonferroni Wilson 95 % intervals (torchregress
    ``binned_conditional_coverage``)."""
    from torchregress.inference import binned_conditional_coverage

    ok = bins >= 0
    out = dict(all=dict(n=int(ok.sum()), coverage=float(np.mean(covered[ok])) if ok.any()
                        else math.nan), bins=[])  # fmt: skip
    if ok.sum() < 2:
        return out
    edges = torch.arange(len(names) + 1, dtype=torch.float64) - 0.5
    r = binned_conditional_coverage(
        torch.as_tensor(bins[ok], dtype=torch.float64)[:, None],
        torch.as_tensor(covered[ok], dtype=torch.float64), nominal=nominal, bins=[edges],
    )[0]  # fmt: skip
    for b, c, v, lo, hi in zip(names, r.count.tolist(), r.coverage.tolist(),
                               r.lower.tolist(), r.upper.tolist(), strict=True):  # fmt: skip
        out["bins"].append(dict(bin=b, n=int(c), coverage=None if math.isnan(v) else float(v),
                                lower=None if math.isnan(lo) else float(lo),
                                upper=None if math.isnan(hi) else float(hi)))  # fmt: skip
    return out


def _coverage_fails(tab, nominal):
    """Bins (n >= MIN_N) whose Wilson interval misses nominal +- COV_TOL."""
    return [(r["bin"], r["n"], round(r["coverage"], 3)) for r in tab["bins"]
            if r["n"] >= MIN_N and r["coverage"] is not None
            and (r["upper"] < nominal - COV_TOL or r["lower"] > nominal + COV_TOL)]  # fmt: skip


# ── stage data ──────────────────────────────────────────────────────────────


def stage_data(a, config, out):
    from torchregress.losses.nflows import recommended_tail_bound

    from masked_stellar_autoencoder.training import distance_extinction as de
    from masked_stellar_autoencoder.training import distributional_heads as dh

    lat = np.load(a.inputs / "latent.npy", mmap_mode="r")
    meta = _latent_meta(a.inputs)
    hidden = (meta or {}).get("hidden_inputs", [])
    if "PARALLAX" not in hidden:
        msg = (f"{a.inputs}: latents were not computed with PARALLAX hidden "
               f"(latent_meta.json hidden_inputs={hidden}); the parallax target would "
               "leak through the input")  # fmt: skip
        if not a.allow_parallax_input:
            raise SystemExit(msg)
        log("WARNING " + msg)
    pred = np.load(a.inputs / "predictions.npz")
    lab = np.load(a.labels)
    n = lat.shape[0]
    sid = np.asarray(pred["source_id"], "i8")
    if (
        len(sid) != n
        or len(lab["split"]) != n
        or not np.array_equal(lab["source_id"], sid)
    ):
        raise SystemExit("latent.npy / predictions.npz / labels.npz rows differ")
    par = read_parallax(config, a.fits, a.sidecar, chunk_rows=a.chunk_rows, log=log)
    if not np.array_equal(par["source_id"], sid):
        raise SystemExit("table rows differ from predictions.npz")
    cols = read_table_columns(a.fits, ("RA", "DEC", "EBV"), a.chunk_rows)
    G = np.asarray(pred["G"], "f8")
    ebv, e_ebv = np.full(n, np.nan), np.full(n, np.nan)
    if a.reddening:
        from masked_stellar_autoencoder.training.xp_forward_model import ReddeningLabels

        ebv, e_ebv = ReddeningLabels(a.reddening).lookup(sid)
        e_ebv = np.where(np.isfinite(e_ebv), np.clip(e_ebv, de.E_ERROR_FLOOR, de.E_ERROR_CAP),
                         np.nan)  # fmt: skip
    split = s3_split(lab["split"], sid)
    feh = np.asarray(lab["fe_h"], "f8")
    ref = (
        np.asarray(lab["fe_h_ref"], "f8")
        if "fe_h_ref" in lab.files
        else np.full(n, np.nan)
    )
    core = np.isfinite(ref)
    if a.core_rows:
        core &= np.isin(np.arange(n), np.load(a.core_rows))
    np.savez(
        out / "observations.npz", source_id=sid, split=split,
        label_split=np.asarray(lab["split"], "i1"), G=G, parallax=par["parallax"],
        parallax_error=par["parallax_error"], zero_point=par["zero_point"],
        zero_point_l21=par["zero_point_l21"], parallax_corrected=par["parallax_corrected"],
        kappa_bin=de.kappa_bin(G), ebv=ebv, e_ebv=e_ebv, ra=cols["RA"], dec=cols["DEC"],
        b=galactic_latitude(cols["RA"], cols["DEC"]), sfd=cols["EBV"], fe_h=feh,
        fe_h_ref=ref, core=core,
    )  # fmt: skip
    plx, err = par["parallax_corrected"], par["parallax_error"]
    has_plx = np.isfinite(plx) & np.isfinite(err) & (err > 0)
    has_e = np.isfinite(ebv) & np.isfinite(e_ebv)
    usable = has_plx | has_e
    rng = np.random.default_rng(a.seed)
    train = _cap(rng, np.flatnonzero(usable & (split == 0)), a.max_train)
    stop = _cap(rng, np.flatnonzero(usable & (split == 1)), a.max_stop)
    test = np.flatnonzero(split == 2)
    if len(train) < a.min_train or len(stop) < 2:
        raise SystemExit(f"too few rows: train {len(train)}, stop {len(stop)}")
    sub = _cap(rng, train, a.max_stats)
    z = dh.gather_rows(lat, sub, chunk_rows=a.chunk_rows).astype("f8")
    lab_tr = ebv[train][has_e[train]]
    e_loc = float(np.mean(lab_tr)) if len(lab_tr) > 10 else 0.2
    e_scale = float(np.std(lab_tr)) if len(lab_tr) > 10 else 0.3
    e_scale = max(e_scale, 0.02)
    e_bound = recommended_tail_bound((lab_tr - e_loc) / e_scale, minimum=6.0) if len(
        lab_tr) else 6.0  # fmt: skip
    std = dh.Standardizer(z.mean(0), np.maximum(z.std(0), 1e-6), np.array([0.0, e_loc]),
                          np.array([1.0, e_scale]))  # fmt: skip
    np.savez(out / "selection.npz", train=train, stop=stop, test=test, e_bound=e_bound,
             **std.state())  # fmt: skip
    snr = np.where(has_plx, plx / np.where(has_plx, err, 1.0), np.nan)
    tr = np.zeros(n, bool)
    tr[train] = True
    counts = dict(
        rows=int(n), with_parallax=int(has_plx.sum()),
        negative_parallax=int((has_plx & (plx < 0)).sum()),
        parallax_snr_lt_1=int((has_plx & (snr < 1)).sum()),
        parallax_snr_ge_10=int((has_plx & (snr >= 10)).sum()),
        with_ebv_label=int(has_e.sum()), with_l21_zero_point=int(par["zero_point_l21"].sum()),
        split_counts=[int((split == k).sum()) for k in (0, 1, 2)],
        hash_split_rows=int((np.asarray(lab["split"]) < 0).sum()),
        train=len(train), stop=len(stop), test=len(test),
        train_with_parallax=int((tr & has_plx).sum()), train_with_ebv=int((tr & has_e).sum()),
        train_negative_parallax=int((tr & has_plx & (plx < 0)).sum()),
        train_snr_lt_5=int((tr & has_plx & (snr < 5)).sum()),
        test_with_parallax=int(has_plx[test].sum()), test_with_ebv=int(has_e[test].sum()),
        heldout_core=int((core & np.isin(split, (1, 2))).sum()),
    )  # fmt: skip
    (out / "selection.json").write_text(json.dumps(dict(
        counts=counts, e_loc=e_loc, e_scale=e_scale, e_bound=float(e_bound),
        latent_hidden_inputs=hidden, latent_dropped_inputs=(meta or {}).get("dropped_inputs", []),
        latent_dim=int(lat.shape[1]), sidecar=str(a.sidecar or ""),
        reddening=str(a.reddening or ""),
        ebv_error_median=float(np.nanmedian(e_ebv)) if has_e.any() else None,
    ), indent=1) + "\n")  # fmt: skip
    log(f"data: {counts}")


# ── stage clusters ──────────────────────────────────────────────────────────


def stage_clusters(a, config, out):
    if not a.members:
        log("clusters: no --members table, skipped")
        return
    obs = np.load(out / "observations.npz")
    sid = obs["source_id"]
    mc = [a.member_columns[0], a.member_columns[1], a.member_columns[2]]
    mem = read_fits_table(a.members, mc)
    msid = np.asarray(mem[mc[0]], "i8")
    cid = np.asarray(mem[mc[1]], "i8")
    prob = np.asarray(mem[mc[2]], "f8")
    keep = np.isfinite(prob) & (prob >= a.cluster_min_prob)
    names, types, ids = np.array([], "U1"), np.array([], "U1"), np.array([], "i8")
    if a.clusters_table:
        cc = a.cluster_columns
        cl = read_fits_table(a.clusters_table, cc)
        ids = np.asarray(cl[cc[0]], "i8")
        names = np.char.strip(np.asarray(cl[cc[1]]).astype(str))
        types = np.char.strip(np.asarray(cl[cc[2]]).astype(str))
        ok_types = set(a.cluster_types)
        good = ids[np.isin(types, list(ok_types))]
        keep &= np.isin(cid, good)
    order = np.argsort(sid, kind="stable")
    pos = np.clip(np.searchsorted(sid[order], msid), 0, len(sid) - 1)
    found = keep & (sid[order][pos] == msid)
    rows = order[pos[found]]
    np.savez(out / "clusters.npz", table_row=rows, cluster_id=cid[found], prob=prob[found],
             ids=ids, names=names, types=types, members=str(a.members),
             clusters_table=str(a.clusters_table or ""))  # fmt: skip
    nclus = len(np.unique(cid[found]))
    log(f"clusters: {int(keep.sum())} members kept of {len(msid)}; {int(found.sum())} in "
        f"the table ({nclus} clusters)")  # fmt: skip


# ── stage train ─────────────────────────────────────────────────────────────


def _member_kwargs(a, sel, family):
    std_e = sel["y_std"][1]
    return dict(hidden=a.hidden, context=a.context, transforms=a.flow_transforms,
                flow_hidden=a.flow_hidden, bins=a.flow_bins, s_bound=a.s_bound,
                e_bound=float(sel["e_bound"]), components=a.components,
                e_loc=float(sel["y_mean"][1]), e_scale=float(std_e),
                fit_kappa=a.parallax_inflation == "fit")  # fmt: skip


def stage_train(a, config, out):
    from masked_stellar_autoencoder.training import distance_extinction as de
    from masked_stellar_autoencoder.training import distributional_heads as dh

    todo = {f: [k for k, p in enumerate(head_paths(a, f)) if a.force or not p.exists()]
            for f in a.families}  # fmt: skip
    todo = {f: ks for f, ks in todo.items() if ks}
    if not todo:
        log("train: all heads exist, skipped")
        return
    (out / "heads").mkdir(exist_ok=True)
    sel = np.load(out / "selection.npz")
    obs = dict(np.load(out / "observations.npz"))
    std = dh.Standardizer.from_state(sel)
    lat = np.load(a.inputs / "latent.npy", mmap_mode="r")
    dev = torch.device(a.device)
    t0 = time.time()
    z_tr = torch.as_tensor(
        dh.gather_rows(lat, sel["train"], chunk_rows=a.chunk_rows), device=dev
    )
    z_st = torch.as_tensor(
        dh.gather_rows(lat, sel["stop"], chunk_rows=a.chunk_rows), device=dev
    )
    log(f"train: latents {len(z_tr)} + {len(z_st)} rows in {time.time() - t0:.0f} s")
    o_tr = de.observation_tensors(obs, sel["train"], dev)
    o_st = de.observation_tensors(obs, sel["stop"], dev)
    for family, ks in todo.items():
        for k in ks:
            log(f"train {family} member {k} ({len(z_tr)} rows)")
            member = de.Member(
                family, z_tr.shape[1], **_member_kwargs(a, sel, family)
            ).to(dev)
            hist = de.train_member(
                member, std, z_tr, o_tr, z_st, o_st, n_s=a.train_ns, n_e=a.train_ne,
                zp_sigma=a.zp_sigma, batch=a.batch, max_epochs=a.epochs, patience=a.patience,
                lr=a.lr, seed=a.seed + k, log=log,
            )  # fmt: skip
            de.save_member(member, head_paths(a, family)[k], std, family=family, member=k,
                           history=hist)  # fmt: skip
            (out / "heads" / f"history_{family}_{k}.json").write_text(
                json.dumps(hist, indent=1) + "\n"
            )
            del member
            if dev.type == "cuda":
                torch.cuda.empty_cache()


# ── stage predict ───────────────────────────────────────────────────────────


def _ensemble(a, family, dev):
    from masked_stellar_autoencoder.training import distance_extinction as de
    from masked_stellar_autoencoder.training import distributional_heads as dh

    paths = head_paths(a, family)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        log(f"{family}: missing members {missing}; skipped")
        return None
    members, std = [], None
    for p in paths:
        m, blob = de.load_member(p, device=dev)
        members.append(m)
        std = dh.Standardizer.from_state(blob["standardizer"])
    return de.EnsembleDE(members, std, zp_sigma=a.zp_sigma)


def eval_rows(a, out, obs):
    """Rows with predictions: all rows, or the test split + held-out core +
    held-out cluster members (``--predict-rows eval``)."""
    n = len(obs["split"])
    if a.predict_rows == "all":
        return np.arange(n)
    split = obs["split"]
    keep = split == 2
    keep |= obs["core"] & (split == 1)
    cp = out / "clusters.npz"
    if cp.exists():
        cr = np.load(cp)["table_row"]
        m = np.zeros(n, bool)
        m[cr] = True
        keep |= m & np.isin(split, (1, 2))
    return np.flatnonzero(keep)


def example_rows(obs, rows, k=4, seed=0):
    """Faint test stars with a parallax and an E label, one per parallax S/N
    class (negative, 0-1, 1-3, 3-10) where available, then the faintest."""
    rng = np.random.default_rng(seed)
    r = rows[obs["split"][rows] == 2]
    plx, err = obs["parallax_corrected"][r], obs["parallax_error"][r]
    ok = np.isfinite(plx) & np.isfinite(err) & (err > 0) & np.isfinite(obs["ebv"][r])
    gmag = obs["G"][r]
    if not ok.any():
        return np.array([], "i8")
    gcut = min(17.0, float(np.nanpercentile(gmag[ok], 80)))
    faint = ok & (gmag >= gcut)
    snr = np.where(ok, plx / np.where(ok, err, 1.0), np.nan)
    pick = []
    for lo, hi in ((-np.inf, 0), (0, 1), (1, 3), (3, 10)):
        c = np.flatnonzero(faint & (snr >= lo) & (snr < hi))
        if len(c):
            pick.append(int(rng.choice(c)))
    rest = [i for i in np.flatnonzero(faint)[np.argsort(-gmag[faint])] if i not in pick]
    pick += rest[: max(0, k - len(pick))]
    return np.sort(r[np.array(pick[:k], "i8")])


def _predict(a, ens, lat, rows, obs, dev, label):
    """Ensemble predictions of ``rows`` (dict of arrays) and stars/s."""
    from masked_stellar_autoencoder.training import distance_extinction as de
    from masked_stellar_autoencoder.training import distributional_heads as dh

    parts, t_pred, t0 = [], 0.0, time.time()
    for i in range(0, len(rows), a.chunk_rows):
        blk = rows[i : i + a.chunk_rows]
        zb = dh.gather_rows(lat, blk, chunk_rows=a.chunk_rows)
        for j in range(0, len(blk), a.eval_chunk):
            r = blk[j : j + a.eval_chunk]
            z = torch.as_tensor(zb[j : j + a.eval_chunk], device=dev)
            ob = de.observation_tensors(obs, r, dev)
            t1 = time.time()
            parts.append(ens.predict(z, ob, n_fine=a.pred_ns, n_joint=a.pred_nsj,
                                     n_e=a.pred_ne, n_s_lik=a.train_ns, n_e_lik=a.train_ne,
                                     device=dev))  # fmt: skip
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_pred += time.time() - t1
        log(f"  {label}: rows {min(i + a.chunk_rows, len(rows))}/{len(rows)} "
            f"({time.time() - t0:.0f} s)")  # fmt: skip
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
    return res, len(rows) / max(t_pred, 1e-9)


ROW_INDEPENDENT = ("kappa", "kappa_g_edges")


def _write_predictions(path, rows, obs, res, ens):
    from masked_stellar_autoencoder.training.distance_extinction import KAPPA_G_EDGES

    np.savez(path, table_row=rows, source_id=obs["source_id"][rows], split=obs["split"][rows],
             G=obs["G"][rows], kappa=ens.kappas(), kappa_g_edges=np.array(KAPPA_G_EDGES),
             **{k: v.astype("f4") if v.dtype.kind == "f" else v for k, v in res.items()})  # fmt: skip


def stage_predict(a, config, out):
    from masked_stellar_autoencoder.training import distance_extinction as de
    from masked_stellar_autoencoder.training import distributional_heads as dh

    obs = dict(np.load(out / "observations.npz"))
    sel = np.load(out / "selection.npz")
    lat = np.load(a.inputs / "latent.npy", mmap_mode="r")
    rows = eval_rows(a, out, obs)
    dev = torch.device(a.device)
    ex = example_rows(obs, rows, a.n_examples, a.seed)
    speed_path = out / "speed.json"
    speed = json.loads(speed_path.read_text()) if speed_path.exists() else {}
    for family in a.families:
        path = out / f"s3_predictions_{family}.npz"
        if path.exists() and not a.force:
            log(f"predict {family}: exists, skipped")
            continue
        ens = _ensemble(a, family, dev)
        if ens is None:
            continue
        log(f"predict {family}: {len(rows)} rows")
        res, rate = _predict(a, ens, lat, rows, obs, dev, f"predict {family}")
        _write_predictions(path, rows, obs, res, ens)
        speed[family] = dict(device=str(dev), rows=int(len(rows)), stars_per_s=rate,
                             members=len(ens.members))  # fmt: skip
        speed_path.write_text(json.dumps(speed, indent=1) + "\n")
        # Quadrature convergence of the training objective (n vs 4n nodes).
        test = sel["test"]
        rng = np.random.default_rng(a.seed + 3)
        qr = np.sort(rng.choice(test, min(a.quad_rows, len(test)), replace=False))
        diffs = []
        for j in range(0, len(qr), a.eval_chunk // 2):
            r = qr[j : j + a.eval_chunk // 2]
            z = torch.as_tensor(
                dh.gather_rows(lat, r, chunk_rows=a.chunk_rows), device=dev
            )
            diffs.append(de.quadrature_check(ens, z, de.observation_tensors(obs, r, dev),
                                             n_s=a.train_ns, n_e=a.train_ne, device=dev))  # fmt: skip
        d = np.concatenate(diffs) if diffs else np.zeros(0)
        q = dict(rows=int(len(d)), n_s=a.train_ns, n_e=a.train_ne, factor=4,
                 median=float(np.median(d)) if len(d) else None,
                 p99=float(np.percentile(d, 99)) if len(d) else None,
                 max=float(d.max()) if len(d) else None)  # fmt: skip
        (out / f"quadrature_{family}.json").write_text(json.dumps(q, indent=1) + "\n")
        log(f"predict {family}: {rate:.0f} stars/s; quadrature |dlogL| median "
            f"{q['median']}, p99 {q['p99']}, max {q['max']}")  # fmt: skip
        if len(ex):
            z = torch.as_tensor(
                dh.gather_rows(lat, ex, chunk_rows=a.chunk_rows), device=dev
            )
            g = ens.grids(z, de.observation_tensors(obs, ex, dev), device=dev)
            np.savez(out / f"examples_{family}.npz", table_row=ex, G=obs["G"][ex],
                     parallax=obs["parallax_corrected"][ex],
                     parallax_error=obs["parallax_error"][ex], ebv=obs["ebv"][ex],
                     e_ebv=obs["e_ebv"][ex], **g)  # fmt: skip
        del ens
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ── stage noxp ──────────────────────────────────────────────────────────────


def stage_noxp(a, config, out):
    from masked_stellar_autoencoder.training.frozen_latents import encode_table

    cols = config["data"]["feature_cols"]
    hide = _expand_hide(cols, a.noxp_hide)
    if a.noxp_inputs:
        lat_dir = a.noxp_inputs
    else:
        lat_dir = out / "noxp"
        if a.force or not (lat_dir / "latent.npy").exists():
            if not a.checkpoint:
                raise SystemExit("noxp: --checkpoint (or --noxp-inputs) is required")
            from s1_likelihood_head import load_state

            log(f"noxp: encoding the table with {len(hide)} inputs hidden")
            encode_table(config, load_state(a.checkpoint), a.fits, lat_dir, device=a.device,
                         chunk_rows=a.chunk_rows, batch=a.encoder_batch,
                         condition_parallax=5.0, hide=tuple(hide), log=log)  # fmt: skip
    meta = _latent_meta(lat_dir) or {}
    hidden = set(meta.get("hidden_inputs", []))
    if not set(hide) <= hidden:
        raise SystemExit(
            f"noxp: {lat_dir} latents do not hide {sorted(set(hide) - hidden)}"
        )
    lat = np.load(lat_dir / "latent.npy", mmap_mode="r")
    main = np.load(a.inputs / "latent.npy", mmap_mode="r")
    if lat.shape != main.shape:
        raise SystemExit(f"noxp: latent shape {lat.shape} != {main.shape}")
    obs = dict(np.load(out / "observations.npz"))
    rows = np.load(out / "selection.npz")["test"]
    dev = torch.device(a.device)
    for family in a.families:
        ens = _ensemble(a, family, dev)
        if ens is None:
            continue
        log(f"noxp {family}: {len(rows)} test rows")
        res, rate = _predict(a, ens, lat, rows, obs, dev, f"noxp {family}")
        _write_predictions(
            out / f"s3_noxp_predictions_{family}.npz", rows, obs, res, ens
        )
    (out / "noxp_meta.json").write_text(json.dumps(dict(
        latent_dir=str(lat_dir), hidden_inputs=sorted(hidden)), indent=1) + "\n")  # fmt: skip


# ── stage report: metrics ───────────────────────────────────────────────────


def _kappa_rows(P, obs, rows):
    """Ensemble-mean fitted inflation kappa_g of each row."""
    kap = np.asarray(P["kappa"], "f8").mean(0)
    return kap[np.asarray(obs["kappa_bin"])[rows]]


def distance_metrics(P, obs, mask_fn):
    """Parallax calibration, parallax-space bias and precise-parallax distance
    moduli of the rows of ``P`` selected by ``mask_fn(rows)``."""
    rows = P["table_row"]
    use = mask_fn(rows)
    plx = obs["parallax_corrected"][rows]
    err = obs["parallax_error"][rows]
    has = use & np.isfinite(plx) & np.isfinite(err) & (err > 0)
    snr = np.where(has, plx / np.where(has, err, 1.0), np.nan)
    G = obs["G"][rows]
    sig = np.where(has, _kappa_rows(P, obs, rows) * err, np.nan)
    # S/N classes by the PREDICTED parallax over its error (a function of z and
    # the error only): classes of the observed parallax would select on the
    # measurement and bias its PIT and pull by construction.
    gb, sb = _bin(G, G_BINS), _bin(P["plx_mean"] / sig, SNR_BINS)
    pit = P["pit_plx"].astype("f8")
    res = plx - P["plx_mean"]
    tot = np.sqrt(P["plx_sd"].astype("f8") ** 2 + sig**2)
    pull = res / tot
    precise = has & (snr > PRECISE_SNR)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_geo = np.where(
            precise, 10.0 - 5.0 * np.log10(np.where(precise, plx, 1.0)), np.nan
        )
    mu_sp = 10.0 + 5.0 * P["s_q"][:, 1].astype("f8") / math.log(10.0)
    dmu = mu_sp - mu_geo
    half_s = 0.5 * (P["s_q"][:, 2] - P["s_q"][:, 0]).astype("f8")
    half_c = 0.5 * (P["comb_s_q"][:, 2] - P["comb_s_q"][:, 0]).astype("f8")
    feh = obs["fe_h"][rows]
    out = dict(n=int(has.sum()), n_precise=int(precise.sum()), by_G=[], by_snr=[], by_feh=[],
               coverage={}, pit_hist=None)  # fmt: skip
    for name, lv in zip(("68", "95"), LEVELS, strict=True):
        cov = np.abs(pit - 0.5) <= lv / 2
        out["coverage"][f"G_{name}"] = _coverage_by(
            np.where(has, gb, -1), G_NAMES, cov, lv
        )
        out["coverage"][f"snr_{name}"] = _coverage_by(
            np.where(has, sb, -1), SNR_NAMES, cov, lv
        )
    edges = np.linspace(0, 1, 21)
    out["pit_hist"] = {nm: np.histogram(pit[has & (sb == b)], edges, density=True)[0].tolist()
                       for b, nm in enumerate(SNR_NAMES)}  # fmt: skip
    out["pit_hist"]["all"] = np.histogram(pit[has], edges, density=True)[0].tolist()

    def cell(m):
        d = dict(n=int(m.sum()), dmu=_summary(dmu[m & precise]),
                 resid_uas=_mean_se(1e3 * res[m]), pull=_mean_se(pull[m]),
                 frac_parallax_bias=float(np.nansum(res[m]) / np.nansum(P["plx_mean"][m]))
                 if m.any() else math.nan,
                 median_sigma_s=float(np.nanmedian(half_s[m])) if m.any() else math.nan,
                 median_sigma_s_combined=float(np.nanmedian(half_c[m])) if m.any() else math.nan)  # fmt: skip
        for name, lv in zip(("68", "95"), LEVELS, strict=True):
            d[f"cov{name}"] = (
                float(np.mean(np.abs(pit[m] - 0.5) <= lv / 2)) if m.any() else math.nan
            )
        return d

    for b, nm in enumerate(G_NAMES):
        out["by_G"].append(dict(bin=nm, **cell(has & (gb == b))))
    for b, nm in enumerate(SNR_NAMES):
        out["by_snr"].append(dict(bin=nm, **cell(has & (sb == b))))
    from masked_stellar_autoencoder.training import likelihood_validation as V

    fb = _bin(feh, V.FEH_BINS)
    for b in range(len(V.FEH_BINS) - 1):
        nm = f"[{V.FEH_BINS[b]:g},{V.FEH_BINS[b + 1]:g})"
        out["by_feh"].append(dict(bin=nm, **cell(has & (fb == b))))
    out["all"] = cell(has)
    out["window_fallback"] = (
        float(np.mean(P["window_fallback"][has])) if has.any() else math.nan
    )
    return out


def core_metrics(P, obs):
    """Held-out frozen core: distance-modulus offsets and parallax residuals per
    high-resolution [Fe/H]_ref bin (the metal-poor stars, as in S1/S2)."""
    from masked_stellar_autoencoder.training import likelihood_validation as V

    rows = P["table_row"]
    held = obs["core"][rows] & np.isin(obs["split"][rows], (1, 2))
    plx, err = obs["parallax_corrected"][rows], obs["parallax_error"][rows]
    has = held & np.isfinite(plx) & np.isfinite(err) & (err > 0)
    snr = np.where(has, plx / np.where(has, err, 1.0), np.nan)
    precise = has & (snr > PRECISE_SNR)
    with np.errstate(divide="ignore", invalid="ignore"):
        dmu = (10.0 + 5.0 * P["s_q"][:, 1] / math.log(10.0)) - np.where(
            precise, 10.0 - 5.0 * np.log10(np.where(precise, plx, 1.0)), np.nan)  # fmt: skip
    sig = _kappa_rows(P, obs, rows) * err
    pull = (plx - P["plx_mean"]) / np.sqrt(P["plx_sd"].astype("f8") ** 2 + sig**2)
    ref = obs["fe_h_ref"][rows]
    fb = _bin(ref, V.FEH_BINS)
    out = []
    for b in range(len(V.FEH_BINS) - 1):
        m = has & (fb == b)
        out.append(dict(bin=f"[{V.FEH_BINS[b]:g},{V.FEH_BINS[b + 1]:g})", n=int(m.sum()),
                        dmu=_summary(dmu[m & precise]), pull=_mean_se(pull[m]),
                        cov68=float(np.mean(np.abs(P["pit_plx"][m] - 0.5) <= 0.34))
                        if m.any() else math.nan))  # fmt: skip
    return out


def ebv_metrics(P, obs, mask_fn):
    rows = P["table_row"]
    use = mask_fn(rows)
    e, ee = obs["ebv"][rows], obs["e_ebv"][rows]
    has = use & np.isfinite(e) & np.isfinite(ee)
    med = P["e_q"][:, 1].astype("f8")
    d = med - e
    pull = (P["e_mean"] - e) / np.sqrt(P["e_sd"].astype("f8") ** 2 + ee**2)
    gb = _bin(obs["G"][rows], G_BINS)
    out = dict(
        all=dict(**_summary(d[has]), pull=_mean_se(pull[has])), by_G=[], coverage={}
    )
    for name, lv in zip(("68", "95"), LEVELS, strict=True):
        cov = np.abs(P["pit_e"] - 0.5) <= lv / 2
        out["coverage"][f"G_{name}"] = _coverage_by(
            np.where(has, gb, -1), G_NAMES, cov, lv
        )
    for b, nm in enumerate(G_NAMES):
        m = has & (gb == b)
        out["by_G"].append(dict(bin=nm, **_summary(d[m]), pull=_mean_se(pull[m]),
                                median_sigma_e=float(np.nanmedian(0.5 * (P["e_q"][m, 2] - P["e_q"][m, 0])))
                                if m.any() else math.nan))  # fmt: skip
    hl = use & (np.abs(obs["b"][rows]) > HIGH_LAT) & np.isfinite(obs["sfd"][rows])
    out["high_lat_vs_sfd"] = dict(prediction=_summary(med[hl] - obs["sfd"][rows][hl]),
                                  label=_summary((e - obs["sfd"][rows])[hl & has]),
                                  median_sfd=float(np.nanmedian(obs["sfd"][rows][hl]))
                                  if hl.any() else math.nan)  # fmt: skip
    return out


def score_metrics(P, obs, mask_fn):
    rows = P["table_row"]
    use = mask_fn(rows)
    gb = _bin(obs["G"][rows], G_BINS)
    out = {}
    for key in ("lp_obs", "lp_plx", "lp_e"):
        v = -P[key].astype("f8")
        m = use & np.isfinite(v)
        out[key] = dict(n=int(m.sum()), nll=float(np.mean(v[m])) if m.any() else math.nan,
                        by_G={nm: float(np.mean(v[m & (gb == b)])) if (m & (gb == b)).any()
                              else math.nan for b, nm in enumerate(G_NAMES)})  # fmt: skip
    return out


def cluster_metrics(P, obs, cl, eval_mask, n_min):
    """Members of a cluster share s and E: their predictive medians scatter by
    their predictive widths (chi2/dof ~ 1 when calibrated); the cluster's
    mean spectrophotometric distance modulus against the geometric one from
    all its members' parallaxes (precision-weighted mean, plx/err > 20)."""
    pos = {int(r): i for i, r in enumerate(P["table_row"])}
    rows, cid = cl["table_row"], cl["cluster_id"]
    kap = np.asarray(P["kappa"], "f8").mean(0)
    plx, err = obs["parallax_corrected"], obs["parallax_error"]
    out, pulls_s, pulls_e = [], [], []
    for c in np.unique(cid):
        r = rows[cid == c]
        ev = np.array([x for x in r if eval_mask[x] and int(x) in pos], "i8")
        if len(ev) < n_min:
            continue
        i = np.array([pos[int(x)] for x in ev])
        m = P["s_q"][i, 1].astype("f8")
        sd = 0.5 * (P["s_q"][i, 2] - P["s_q"][i, 0]).astype("f8")
        w = 1.0 / sd**2
        sbar = float(np.sum(w * m) / np.sum(w))
        chi = (m - sbar) / sd
        pulls_s += list(chi * math.sqrt(len(m) / (len(m) - 1)))
        me = P["e_q"][i, 1].astype("f8")
        sde = np.maximum(0.5 * (P["e_q"][i, 2] - P["e_q"][i, 0]).astype("f8"), 1e-4)
        we = 1.0 / sde**2
        ebar = float(np.sum(we * me) / np.sum(we))
        chie = (me - ebar) / sde
        pulls_e += list(chie * math.sqrt(len(me) / (len(me) - 1)))
        hp = np.isfinite(plx[r]) & np.isfinite(err[r]) & (err[r] > 0)
        se = kap[obs["kappa_bin"][r[hp]]] * err[r[hp]]
        d = dict(cluster_id=int(c), n_eval=int(len(ev)), n_members=int(len(r)),
                 s_mean=sbar, s_sd_mean=float(1 / math.sqrt(np.sum(w))),
                 chi2_s=float(np.sum(chi**2) / (len(m) - 1)), e_mean=ebar,
                 chi2_e=float(np.sum(chie**2) / (len(me) - 1)), mu_sp=float(10 + 5 * sbar / math.log(10)))  # fmt: skip
        if hp.sum():
            wp = 1.0 / se**2
            pbar = float(np.sum(wp * plx[r[hp]]) / np.sum(wp))
            pse = float(1.0 / math.sqrt(np.sum(wp)))
            d.update(plx_mean=pbar, plx_se=pse)
            if pbar > 20 * pse:
                d["mu_geo"] = float(10 - 5 * math.log10(pbar))
                d["dmu"] = d["mu_sp"] - d["mu_geo"]
        out.append(d)
    dm = np.array([d["dmu"] for d in out if "dmu" in d])
    return dict(
        clusters=out, n_clusters=len(out), n_with_geometric=int(len(dm)),
        dmu=_summary(dm), median_chi2_s=float(np.median([d["chi2_s"] for d in out])) if out else math.nan,
        median_chi2_e=float(np.median([d["chi2_e"] for d in out])) if out else math.nan,
        pull_sd_s=_nmad(pulls_s), pull_sd_e=_nmad(pulls_e), n_members_eval=len(pulls_s),
    )  # fmt: skip


def stage_report(a, config, out):
    obs = dict(np.load(out / "observations.npz"))
    selj = json.loads((out / "selection.json").read_text())
    n = len(obs["split"])
    test = np.zeros(n, bool)
    test[obs["split"] == 2] = True

    def is_test(rows):
        return test[rows]

    preds = {}
    for f in a.families:
        p = out / f"s3_predictions_{f}.npz"
        if p.exists():
            preds[f] = dict(np.load(p))
        p = out / f"s3_noxp_predictions_{f}.npz"
        if p.exists():
            preds[f"{f}_noxp"] = dict(np.load(p))
    m = dict(summary="", selection=selj, scores={}, distance={}, core={}, ebv={},
             clusters={}, kappa={}, quadrature={}, training={}, speed={}, noxp={},
             decision={}, files=[], notes=[])  # fmt: skip
    for name, P in preds.items():
        m["scores"][name] = score_metrics(P, obs, is_test)
        m["distance"][name] = distance_metrics(P, obs, is_test)
        m["ebv"][name] = ebv_metrics(P, obs, is_test)
        if not name.endswith("_noxp"):
            m["core"][name] = core_metrics(P, obs)
            m["kappa"][name] = np.asarray(P["kappa"]).tolist()
    # Same rows for the beyond-XP comparison: test rows of the noxp run.
    for f in a.families:
        if f in preds and f"{f}_noxp" in preds:
            rows_x = preds[f"{f}_noxp"]["table_row"]
            keep = np.isin(preds[f]["table_row"], rows_x)
            sub = {
                k: v if k in ROW_INDEPENDENT else v[keep] for k, v in preds[f].items()
            }
            m["noxp"][f] = dict(full=distance_metrics(sub, obs, is_test),
                                noxp=m["distance"][f"{f}_noxp"],
                                ebv_full=ebv_metrics(sub, obs, is_test),
                                ebv_noxp=m["ebv"][f"{f}_noxp"])  # fmt: skip
    cp = out / "clusters.npz"
    if cp.exists():
        cl = dict(np.load(cp))
        held = (
            np.isin(obs["split"], (1, 2))
            if a.cluster_rows == "heldout"
            else np.ones(n, bool)
        )
        for f in a.families:
            if f in preds:
                m["clusters"][f] = cluster_metrics(
                    preds[f], obs, cl, held, a.cluster_min_n
                )
        m["clusters_source"] = dict(members=str(cl["members"]), clusters_table=str(cl["clusters_table"]),
                                    matched_member_rows=int(len(cl["table_row"])),
                                    min_prob=a.cluster_min_prob, types=a.cluster_types,
                                    rows=a.cluster_rows)  # fmt: skip
    for f in a.families:
        q = out / f"quadrature_{f}.json"
        if q.exists():
            m["quadrature"][f] = json.loads(q.read_text())
        hs = sorted((out / "heads").glob(f"history_{f}_*.json"))
        m["training"][f] = [
            {k: h[k] for k in ("best_valid", "epochs", "n_train", "n_valid", "seed", "n_s",
                               "n_e", "kappa_final", "train_seconds") if k in h}
            for h in (json.loads(p.read_text()) for p in hs)
        ]  # fmt: skip
    sp = out / "speed.json"
    m["speed"] = json.loads(sp.read_text()) if sp.exists() else {}
    _decide(m, a)
    _notes(a, m)
    m["files"] = _plots(a, out, m, preds, obs)
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float) + "\n")
    (out / "report.md").write_text(render_report(m))
    log(m["summary"])


def _decide(m, a):
    fams = [f for f in a.families if f in m["scores"]]
    if not fams:
        m["summary"] = "No predictions."
        return
    best = min(fams, key=lambda f: m["scores"][f]["lp_obs"]["nll"])
    for f in fams:
        d = m["distance"][f]
        fails = {lv: _coverage_fails(d["coverage"][f"G_{lv}"], float(lv) / 100)
                 for lv in ("68", "95")}  # fmt: skip
        snr_fails = {lv: _coverage_fails(d["coverage"][f"snr_{lv}"], float(lv) / 100)
                     for lv in ("68", "95")}  # fmt: skip
        e = m["ebv"][f]
        efails = {lv: _coverage_fails(e["coverage"][f"G_{lv}"], float(lv) / 100)
                  for lv in ("68", "95")}  # fmt: skip
        bias = [(r["bin"], round(r["pull"]["mean"], 3), round(r["pull"]["se"], 3))
                for r in d["by_G"] if r["n"] >= MIN_N and np.isfinite(r["pull"]["mean"])
                and abs(r["pull"]["mean"]) > max(0.05, 3 * r["pull"]["se"])]  # fmt: skip
        m["decision"][f] = dict(
            best_log_score=f == best, nll_obs=m["scores"][f]["lp_obs"]["nll"],
            parallax_coverage_G_fails=fails, parallax_coverage_snr_fails=snr_fails,
            ebv_coverage_G_fails=efails, parallax_bias_G_fails=bias,
            calibrated=not any(fails.values()) and not any(snr_fails.values()) and not bias,
        )  # fmt: skip
    cal = [f for f in fams if m["decision"][f]["calibrated"]]
    d = m["distance"][best]["all"]
    m["summary"] = (
        f"Families {fams}; best held-out log score of the measurements: {best} "
        f"(NLL {m['scores'][best]['lp_obs']['nll']:.3f}). Parallax calibration "
        f"(PIT coverage per G and S/N bin, mean parallax pull per G bin) passes for: "
        f"{cal or 'none'}. {best}: precise-parallax distance modulus offset "
        f"{d['dmu']['bias']:+.3f} mag, NMAD {d['dmu']['nmad']:.3f} mag ({d['dmu']['n']} stars); "
        f"68 % parallax coverage {d['cov68']:.3f} over {d['n']} test stars."
    )


def _notes(a, m):
    sel = m["selection"]
    m["notes"] = [
        "Likelihood per star: L = int int p(s, E | z) N(plx - zp | exp(-s), sig_eff^2) "
        "[N(E_obs | E, e_E^2)] ds dE, s = ln(d/kpc), zp = Lindegren+2021 (sidecar; else "
        "-0.017 mas), sig_eff^2 = (kappa_g sig)^2 + sig_zp^2 (kappa_g fitted per G bin "
        f"{'(fit)' if a.parallax_inflation == 'fit' else '(fixed at 1)'}, sig_zp = "
        f"{a.zp_sigma:g} mas). No parallax cut or inversion: every parallax "
        f"(train: {sel['counts']['train_with_parallax']}, of which "
        f"{sel['counts']['train_negative_parallax']} negative and "
        f"{sel['counts']['train_snr_lt_5']} with plx/err < 5) enters the likelihood.",
        "E(B-V) labels: empirical reddening labels (intrinsic colours learned from the "
        "spectroscopic labels; BP-RP based), errors in variables with their quoted errors "
        f"(floor 0.01, cap 0.5 mag; median {sel.get('ebv_error_median')}). Missing label: E "
        "marginalised exactly. The SFD map value is an encoder input, never a label; the "
        "|b| > 60 deg comparison with SFD is therefore a zero-point sanity check, not an "
        "independent validation (SFD informs the prediction). The labels derive from BP-RP "
        "(an encoder input) and the spectroscopic labels, so E is learned on their scale.",
        "Quadrature: trapezoid rule on the intersection of the predictive window (base "
        "quantiles +-5.5) and the measurement window (+-7 errors), "
        f"{a.train_ns} s nodes in training and scores, {a.train_ne} inner E nodes (flow); "
        "gmix integrates E in closed form. 'quadrature' reports |log L(n) - log L(4n)| on "
        "test rows.",
        "Calibration: PIT of the observed parallax under the convolved predictive "
        "P(plx_rep <= plx_obs | z) for every test star with a parallax (no selection on the "
        "parallax); 68/95 % central-interval coverage = |PIT - 0.5| <= 0.34/0.475; Wilson "
        "95 % intervals with a Bonferroni split (torchregress binned_conditional_coverage). "
        "Parallax pull = (plx - E[h]) / sqrt(Var h + sig_eff^2): its mean per G or [Fe/H] "
        "bin is a selection-free bias test that works for faint stars.",
        "Parallax S/N classes are of the predicted parallax over its error, E[h] / "
        "sig_eff (a function of z and the error), never of the observed parallax: classes "
        "of the observed value select on the measurement and bias its PIT and pull.",
        "Precise-parallax test (plx/err > 10): mu_sp (median of the predictive s) against "
        "mu_geo = 10 - 5 log10(plx - zp); the inversion is negligible at this S/N, but the "
        "S/N cut selects on the parallax, so these offsets carry a small selection bias; the "
        "pull test above does not.",
        "Clusters: Hunt & Reffert (2024) members (Prob >= "
        f"{a.cluster_min_prob:g}, types {a.cluster_types}); evaluation members are "
        f"{a.cluster_rows} rows; chi2/dof of the members' predictive medians about the "
        "precision-weighted cluster mean (members at one distance; differential reddening "
        "inflates the E chi2); cluster geometric distance from the precision-weighted mean "
        "of all its members' corrected parallaxes.",
        f"Latents ({sel.get('latent_dim')}-d): inputs hidden for these latents "
        f"{sel.get('latent_hidden_inputs')}, dropped by the checkpoint "
        f"{sel.get('latent_dropped_inputs')} (latent_meta.json); any other input (photometry, "
        "XP, the SFD value) informs the prediction. Kinematics or positions among the "
        "inputs would make it product B1 (design note section 4).",
    ]


# ── report rendering ────────────────────────────────────────────────────────


def _fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render_report(m):
    L = ["# S3: joint distance and extinction from Gaia parallaxes (frozen latents)\n",
         m["summary"] + "\n"]  # fmt: skip
    c = m["selection"]["counts"]
    L.append("## Data\n")
    L.append(
        f"Rows {c['rows']}; with a parallax {c['with_parallax']} (negative "
        f"{c['negative_parallax']}, plx/err < 1: {c['parallax_snr_lt_1']}, >= 10: "
        f"{c['parallax_snr_ge_10']}); with an E(B-V) label {c['with_ebv_label']}; L21 zero "
        f"point {c['with_l21_zero_point']}. Train {c['train']} ({c['train_with_parallax']} "
        f"parallaxes, {c['train_with_ebv']} E labels), stop {c['stop']}, test {c['test']}; "
        f"held-out core {c['heldout_core']}; hash-split rows {c['hash_split_rows']}.\n"
    )
    if m["decision"]:
        L.append("## Decision\n")
        L.append("| family | held-out NLL of the measurements | parallax coverage fails (G; S/N) "
                 "| parallax pull bias fails (G) | E coverage fails (G) | calibrated |")  # fmt: skip
        L.append("|---|---|---|---|---|---|")
        for f, d in m["decision"].items():
            gf = sum(len(v) for v in d["parallax_coverage_G_fails"].values())
            sf = sum(len(v) for v in d["parallax_coverage_snr_fails"].values())
            ef = sum(len(v) for v in d["ebv_coverage_G_fails"].values())
            L.append(f"| {f} | {_fmt(d['nll_obs'])}{' best' if d['best_log_score'] else ''} | "
                     f"{gf}; {sf} | {len(d['parallax_bias_G_fails'])} | {ef} | "
                     f"{'YES' if d['calibrated'] else 'no'} |")  # fmt: skip
        L.append("")
        for f, d in m["decision"].items():
            for k in ("parallax_coverage_G_fails", "parallax_coverage_snr_fails",
                      "ebv_coverage_G_fails"):  # fmt: skip
                for lv, v in d[k].items():
                    if v:
                        L.append(f"- `{f}` {k} {lv} %: {v}")
            if d["parallax_bias_G_fails"]:
                L.append(
                    f"- `{f}` mean parallax pull (bin, mean, se): {d['parallax_bias_G_fails']}"
                )
        L.append("")
    L.append("## Scores (test split): mean -log p\n")
    L.append("| predictor | measurements (n) | parallax only (n) | E label only (n) | "
             + " | ".join(f"meas. {g}" for g in G_NAMES) + " |")  # fmt: skip
    L.append("|---" * (4 + len(G_NAMES)) + "|")
    for nm, s in m["scores"].items():
        L.append(f"| {nm} | {_fmt(s['lp_obs']['nll'])} ({s['lp_obs']['n']}) | "
                 f"{_fmt(s['lp_plx']['nll'])} ({s['lp_plx']['n']}) | {_fmt(s['lp_e']['nll'])} "
                 f"({s['lp_e']['n']}) | "
                 + " | ".join(_fmt(s["lp_obs"]["by_G"][g]) for g in G_NAMES) + " |")  # fmt: skip
    L.append("")
    L.append("## Distances (test split)\n")
    L.append("Per bin: n with a parallax; precise (plx/err > 10) distance-modulus offset "
             "mu_sp - mu_geo median / NMAD (n); mean parallax residual [uas] +- se; mean / sd "
             "of the parallax pull; PIT coverage 68 / 95 %; median predictive half-width of s "
             "(16-84 %, ~ fractional distance error), alone / combined with the parallax.\n")  # fmt: skip
    for nm, d in m["distance"].items():
        L.append(f"**{nm}** (window fallback {_fmt(d['window_fallback'], 4)})\n")
        L.append("| bin | n | dmu bias / NMAD (n) | resid [uas] | pull mean / sd | cov 68 / 95 "
                 "| sigma_s / combined |")  # fmt: skip
        L.append("|---|---|---|---|---|---|---|")
        for r in [dict(bin="all", **d["all"])] + d["by_G"] + d["by_snr"] + d["by_feh"]:
            if not r["n"]:
                continue
            L.append(
                f"| {r['bin']} | {r['n']} | {_fmt(r['dmu']['bias'], 3, True)} / "
                f"{_fmt(r['dmu']['nmad'])} ({r['dmu']['n']}) | {_fmt(r['resid_uas']['mean'], 1, True)}"
                f" +- {_fmt(r['resid_uas']['se'], 1)} | {_fmt(r['pull']['mean'], 3, True)} / "
                f"{_fmt(r['pull']['sd'])} | {_fmt(r['cov68'])} / {_fmt(r['cov95'])} | "
                f"{_fmt(r['median_sigma_s'])} / {_fmt(r['median_sigma_s_combined'])} |"
            )
        L.append("")
    if m["core"]:
        L.append("## Held-out frozen core by [Fe/H]_ref (metal-poor stars)\n")
        L.append("| family | [Fe/H]_ref | n | dmu bias / NMAD (n, plx/err > 10) | pull mean +- se "
                 "| cov 68 |\n|---|---|---|---|---|---|")  # fmt: skip
        for f, rows in m["core"].items():
            for r in rows:
                if r["n"]:
                    L.append(f"| {f} | {r['bin']} | {r['n']} | {_fmt(r['dmu']['bias'], 3, True)} / "
                             f"{_fmt(r['dmu']['nmad'])} ({r['dmu']['n']}) | "
                             f"{_fmt(r['pull']['mean'], 3, True)} +- {_fmt(r['pull']['se'])} | "
                             f"{_fmt(r['cov68'])} |")  # fmt: skip
        L.append("")
    L.append("## E(B-V) (test split)\n")
    L.append("| predictor | vs labels bias / NMAD (n) | pull mean / sd | cov 68 / 95 | "
             "|b| > 60: pred - SFD bias / NMAD (n) | label - SFD bias / NMAD (n) |")  # fmt: skip
    L.append("|---|---|---|---|---|---|")
    for nm, e in m["ebv"].items():
        a_, hl = e["all"], e["high_lat_vs_sfd"]
        c68 = e["coverage"]["G_68"]["all"]["coverage"]
        c95 = e["coverage"]["G_95"]["all"]["coverage"]
        L.append(f"| {nm} | {_fmt(a_['bias'], 3, True)} / {_fmt(a_['nmad'])} ({a_['n']}) | "
                 f"{_fmt(a_['pull']['mean'], 3, True)} / {_fmt(a_['pull']['sd'])} | {_fmt(c68)} / "
                 f"{_fmt(c95)} | {_fmt(hl['prediction']['bias'], 3, True)} / "
                 f"{_fmt(hl['prediction']['nmad'])} ({hl['prediction']['n']}) | "
                 f"{_fmt(hl['label']['bias'], 3, True)} / {_fmt(hl['label']['nmad'])} "
                 f"({hl['label']['n']}) |")  # fmt: skip
    L.append("\nPer G bin: bias / NMAD vs labels, median half-width of E (16-84 %)\n")
    L.append("| predictor | " + " | ".join(G_NAMES) + " |")
    L.append("|---" * (len(G_NAMES) + 1) + "|")
    for nm, e in m["ebv"].items():
        L.append(f"| {nm} | " + " | ".join(
            f"{_fmt(r['bias'], 3, True)} / {_fmt(r['nmad'])}, {_fmt(r['median_sigma_e'])} ({r['n']})"
            for r in e["by_G"]) + " |")  # fmt: skip
    L.append("")
    if m["clusters"]:
        L.append("## Open clusters (label-free)\n")
        L.append("| family | clusters (members) | chi2/dof s median | chi2/dof E median | pull NMAD "
                 "s / E | clusters with geometric distance | dmu bias / NMAD |")  # fmt: skip
        L.append("|---|---|---|---|---|---|---|")
        for f, c_ in m["clusters"].items():
            L.append(f"| {f} | {c_['n_clusters']} ({c_['n_members_eval']}) | "
                     f"{_fmt(c_['median_chi2_s'], 2)} | {_fmt(c_['median_chi2_e'], 2)} | "
                     f"{_fmt(c_['pull_sd_s'], 2)} / {_fmt(c_['pull_sd_e'], 2)} | "
                     f"{c_['n_with_geometric']} | {_fmt(c_['dmu']['bias'], 3, True)} / "
                     f"{_fmt(c_['dmu']['nmad'])} |")  # fmt: skip
        L.append("")
    else:
        L.append("## Open clusters (label-free)\n\nNo membership table or no cluster with "
                 "enough evaluated members.\n")  # fmt: skip
    if m["noxp"]:
        L.append(
            "## Beyond XP: XP coefficients hidden (photometry-only latents, test rows)\n"
        )
        L.append("| family | latents | dmu NMAD (n) | median sigma_s | cov 68 | pull mean | E NMAD vs "
                 "labels | E cov 68 |\n|---|---|---|---|---|---|---|---|")  # fmt: skip
        for f, d in m["noxp"].items():
            for kind in ("full", "noxp"):
                r = d[kind]["all"]
                e = d[f"ebv_{kind}"]
                L.append(f"| {f} | {kind} | {_fmt(r['dmu']['nmad'])} ({r['dmu']['n']}) | "
                         f"{_fmt(r['median_sigma_s'])} | {_fmt(r['cov68'])} | "
                         f"{_fmt(r['pull']['mean'], 3, True)} | {_fmt(e['all']['nmad'])} | "
                         f"{_fmt(e['coverage']['G_68']['all']['coverage'])} |")  # fmt: skip
        L.append("")
    if m["kappa"]:
        L.append("## Fitted parallax-error inflation kappa_g (members x G bins "
                 "<=13, 13-15, 15-17, 17-18, 18-19, >19)\n")  # fmt: skip
        for f, k in m["kappa"].items():
            L.append(f"- `{f}`: " + "; ".join("[" + ", ".join(f"{x:.3f}" for x in row) + "]"
                                             for row in k))  # fmt: skip
        L.append("")
    if m["quadrature"]:
        L.append("## Quadrature convergence (|log L(n) - log L(4n)|, test rows)\n")
        for f, q in m["quadrature"].items():
            L.append(
                f"- `{f}`: {q['rows']} rows, n_s {q['n_s']}, n_e {q['n_e']}: median "
                f"{_fmt(q['median'], 6)}, p99 {_fmt(q['p99'], 6)}, max {_fmt(q['max'], 6)}"
            )
        L.append("")
    if m["training"]:
        L.append("## Training (per member: best stop NLL, epochs, minutes)\n")
        for f, rows in m["training"].items():
            L.append(f"- `{f}`: " + "; ".join(
                f"{_fmt(r.get('best_valid'))}, {r.get('epochs')} ep, "
                f"{r.get('train_seconds', 0) / 60:.1f} min" for r in rows))  # fmt: skip
        L.append("")
    if m["speed"]:
        L.append("## Speed\n")
        for f, s in m["speed"].items():
            L.append(f"- `{f}`: {s['stars_per_s']:.0f} stars/s ({s['device']}, {s['members']} "
                     f"members, all summaries and scores)")  # fmt: skip
        L.append("")
    if m["files"]:
        L.append("## QA plots\n")
        L.extend(f"- `{x}`" for x in m["files"])
        L.append("")
    L.append("## Notes\n")
    L.extend(f"- {x}" for x in m["notes"])
    return "\n".join(L) + "\n"


# ── plots ───────────────────────────────────────────────────────────────────


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "s3", [SURFACE, "#9cc3ef", COLORS[0], "#123e73"]
    )


def _finish(fig, path, ax=None, handles=None):
    """tight layout, one legend above the figure (>= 2 series), save."""
    fig.tight_layout()
    if handles is None and ax is not None:
        handles = ax.get_legend_handles_labels()[0]
    if handles and len(handles) >= 2:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   ncol=min(len(handles), 4), frameon=False, fontsize=8)  # fmt: skip
    _save(fig, path)


def _plots(a, out, m, preds, obs):
    files = []
    names = [n for n in preds if not n.endswith("_noxp")]
    if not names:
        return files
    x = np.arange(len(G_NAMES))
    # (1) spectrophotometric vs geometric distance modulus (precise parallaxes).
    fig, axes = _figure(1, len(names), (4.2 * len(names), 4.0))
    for ax, nm in zip(axes[0], names, strict=True):
        P = preds[nm]
        r = P["table_row"]
        plx, err = obs["parallax_corrected"][r], obs["parallax_error"][r]
        ok = (
            (obs["split"][r] == 2)
            & np.isfinite(plx)
            & (err > 0)
            & (plx > PRECISE_SNR * err)
        )
        mg = 10 - 5 * np.log10(plx[ok])
        ms = 10 + 5 * P["s_q"][ok, 1] / math.log(10)
        if ok.sum() > 1:
            lo, hi = np.percentile(np.r_[mg, ms], [0.5, 99.5])
            ax.hexbin(mg, ms, gridsize=60, extent=(lo, hi, lo, hi), bins="log", cmap=_cmap(),
                      mincnt=1, linewidths=0)  # fmt: skip
            ax.plot([lo, hi], [lo, hi], color="#2b2b28", linewidth=0.8)
        ax.set_xlabel("geometric distance modulus [mag] (plx/err > 10)")
        ax.set_ylabel("spectrophotometric distance modulus [mag]")
        ax.set_title(f"{_label(nm)}: offset {m['distance'][nm]['all']['dmu']['bias']:+.3f}, "
                     f"NMAD {m['distance'][nm]['all']['dmu']['nmad']:.3f} mag", fontsize=9)  # fmt: skip
    _finish(fig, out / "qa_distance_hexbin.png")
    files.append("qa_distance_hexbin.png")
    # (2) bias / scatter vs G and [Fe/H]; parallax-space residual vs G.
    from masked_stellar_autoencoder.training import likelihood_validation as V

    fig, axes = _figure(2, 2, (11, 7.4))
    fx = np.arange(len(V.FEH_BINS) - 1)
    for nm in preds:
        d = m["distance"][nm]
        c, ls = _style(nm)
        kw = dict(
            color=c,
            linestyle=ls,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=_label(nm),
        )

        def ys(rows, f):
            return np.array([f(r) if r["n"] >= MIN_N else np.nan for r in rows])

        axes[0, 0].plot(x, ys(d["by_G"], lambda r: r["dmu"]["bias"]
                              if r["dmu"]["n"] >= MIN_N else np.nan), **kw)  # fmt: skip
        axes[0, 1].plot(x, ys(d["by_G"], lambda r: r["dmu"]["nmad"]
                              if r["dmu"]["n"] >= MIN_N else np.nan), **kw)  # fmt: skip
        axes[1, 0].plot(fx, ys(d["by_feh"], lambda r: r["dmu"]["bias"]
                               if r["dmu"]["n"] >= MIN_N else np.nan), **kw)  # fmt: skip
        mres = ys(d["by_G"], lambda r: r["resid_uas"]["mean"])
        se = ys(d["by_G"], lambda r: r["resid_uas"]["se"])
        axes[1, 1].errorbar(x, mres, yerr=se, capsize=2, **kw)
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.axhline(0, color="#8a8a84", linewidth=0.8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xticks(x)
        ax.set_xticklabels(G_NAMES, fontsize=8)
        ax.set_xlabel("G [mag]")
    axes[1, 0].set_xticks(fx)
    axes[1, 0].set_xticklabels([r["bin"] for r in m["distance"][names[0]]["by_feh"]],
                               rotation=30, fontsize=7)  # fmt: skip
    axes[1, 0].set_xlabel("[Fe/H] label bin [dex]")
    axes[0, 0].set_ylabel("mu_sp - mu_geo median [mag]\n(plx/err > 10)")
    axes[0, 1].set_ylabel("mu_sp - mu_geo NMAD [mag]")
    axes[1, 0].set_ylabel("mu_sp - mu_geo median [mag]")
    axes[1, 1].set_ylabel("mean plx_obs - E[plx] [uas]\n(all test stars, no S/N cut)")
    _finish(fig, out / "qa_distance_bias.png", axes[0, 1])
    files.append("qa_distance_bias.png")
    # (3) parallax calibration: PIT histograms per S/N class, coverage vs G.
    fig, axes = _figure(2, 4, (14, 6.4))
    edges = np.linspace(0, 1, 21)
    for j, sn in enumerate(SNR_NAMES):
        ax = axes[0, j]
        ax.axhline(1.0, color="#8a8a84", linewidth=0.8)
        for nm in preds:
            h = m["distance"][nm]["pit_hist"][sn]
            c, ls = _style(nm)
            ax.step(edges, np.r_[h, h[-1]], where="post", color=c, linestyle=ls, linewidth=1.6,
                    label=_label(nm))  # fmt: skip
        ax.set_xlabel(f"PIT of the parallax ({sn})", fontsize=8)
        ax.set_ylabel("density")
    for j, (lv, key) in enumerate(((0.68, "cov68"), (0.95, "cov95"))):
        ax = axes[1, j]
        ax.axhspan(-COV_TOL, COV_TOL, color="#ecece8", zorder=0)
        ax.axhline(0, color="#8a8a84", linewidth=0.8)
        for nm in preds:
            c, ls = _style(nm)
            y = [
                r[key] - lv if r["n"] >= MIN_N else np.nan
                for r in m["distance"][nm]["by_G"]
            ]
            ax.plot(x, y, color=c, linestyle=ls, marker="o", markersize=4, linewidth=1.6,
                    label=_label(nm))  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(G_NAMES, fontsize=7)
        ax.set_xlabel("G [mag]")
        ax.set_ylabel(f"{round(lv * 100)} % coverage - nominal")
        ax.set_ylim(-0.15, 0.15)
    xs = np.arange(len(SNR_NAMES))
    for j, (key, ylab) in enumerate(((lambda r: r["cov68"] - 0.68, "68 % coverage - nominal"),
                                     (lambda r: r["pull"]["mean"], "mean parallax pull"))):  # fmt: skip
        ax = axes[1, 2 + j]
        ax.axhline(0, color="#8a8a84", linewidth=0.8)
        for nm in preds:
            c, ls = _style(nm)
            y = [
                key(r) if r["n"] >= MIN_N else np.nan
                for r in m["distance"][nm]["by_snr"]
            ]
            ax.plot(xs, y, color=c, linestyle=ls, marker="o", markersize=4, linewidth=1.6,
                    label=_label(nm))  # fmt: skip
        ax.set_xticks(xs)
        ax.set_xticklabels(SNR_NAMES, fontsize=7)
        ax.set_xlabel("predicted parallax / error")
        ax.set_ylabel(ylab)
    _finish(fig, out / "qa_parallax_calibration.png", axes[0, 0])
    files.append("qa_parallax_calibration.png")
    # (4) E(B-V): predicted vs label; bias vs G; high latitude vs SFD.
    best = min(names, key=lambda f: m["scores"][f]["lp_obs"]["nll"])
    fig, axes = _figure(1, 4, (17, 4.0))
    P = preds[best]
    r = P["table_row"]
    e = obs["ebv"][r]
    ok = (obs["split"][r] == 2) & np.isfinite(e)
    ax = axes[0, 0]
    if ok.sum() > 1:
        lo, hi = np.percentile(np.r_[e[ok], P["e_q"][ok, 1]], [0.5, 99.5])
        ax.hexbin(e[ok], P["e_q"][ok, 1], gridsize=60, extent=(lo, hi, lo, hi), bins="log",
                  cmap=_cmap(), mincnt=1, linewidths=0)  # fmt: skip
        ax.plot([lo, hi], [lo, hi], color="#2b2b28", linewidth=0.8)
    ax.set_xlabel("E(B-V) label [mag]")
    ax.set_ylabel(f"predicted E(B-V) median [mag] ({_label(best)})")
    for ax, key, ylab in ((axes[0, 1], "bias", "median predicted - label E(B-V) [mag]"),
                          (axes[0, 2], "nmad", "NMAD predicted - label E(B-V) [mag]")):  # fmt: skip
        if key == "bias":
            ax.axhline(0, color="#8a8a84", linewidth=0.8)
        for nm in preds:
            c, ls = _style(nm)
            y = [rr[key] if rr["n"] >= MIN_N else np.nan for rr in m["ebv"][nm]["by_G"]]
            ax.plot(x, y, color=c, linestyle=ls, marker="o", markersize=4, linewidth=1.6,
                    label=_label(nm))  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(G_NAMES, fontsize=7)
        ax.set_xlabel("G [mag]")
        ax.set_ylabel(ylab)
    ax = axes[0, 3]
    bins = np.linspace(-0.15, 0.15, 41)
    for nm in preds:
        P_ = preds[nm]
        rr = P_["table_row"]
        hl = (
            (obs["split"][rr] == 2)
            & (np.abs(obs["b"][rr]) > HIGH_LAT)
            & np.isfinite(obs["sfd"][rr])
        )
        if hl.sum():
            c, ls = _style(nm)
            ax.hist(P_["e_q"][hl, 1] - obs["sfd"][rr][hl], bins, histtype="step", color=c,
                    linestyle=ls, linewidth=1.6, density=True)  # fmt: skip
    ax.axvline(0, color="#8a8a84", linewidth=0.8)
    ax.set_xlabel("predicted E(B-V) - SFD [mag] (|b| > 60 deg)")
    ax.set_ylabel("density")
    _finish(fig, out / "qa_ebv.png", axes[0, 1])
    files.append("qa_ebv.png")
    # (5) example faint stars: 2-D (s, E) predictive and combined densities.
    exs = {f: dict(np.load(out / f"examples_{f}.npz")) for f in names
           if (out / f"examples_{f}.npz").exists()}  # fmt: skip
    if exs:
        k = len(next(iter(exs.values()))["table_row"])
        if k:
            from matplotlib.lines import Line2D

            fig, axes = _figure(1, k, (3.6 * k, 3.6))
            for i, ax in enumerate(axes[0]):
                box = []
                for f, ex in exs.items():
                    c = _style(f)[0]
                    for key, ls in (("log_p", "-"), ("log_p_comb", ":")):
                        box.append(_contours(ax, ex["s"][:, i], ex["E"][:, i], ex[key][:, :, i],
                                             c, ls))  # fmt: skip
                box = np.array([b_ for b_ in box if b_ is not None])
                if len(box):
                    lo_, hi_ = box[:, :2].min(0), box[:, 2:].max(0)
                    pad = 0.15 * (hi_ - lo_)
                    ax.set_xlim(lo_[0] - pad[0], hi_[0] + pad[0])
                    ax.set_ylim(lo_[1] - pad[1], hi_[1] + pad[1])
                ex = next(iter(exs.values()))
                plx, err = ex["parallax"][i], ex["parallax_error"][i]
                for v in (plx - err, plx + err):
                    if v > 0:
                        ax.axvline(10 + 5 * -math.log(v) / math.log(10), color="#8a8a84",
                                   linewidth=0.8, linestyle="--")  # fmt: skip
                if np.isfinite(ex["ebv"][i]):
                    ax.axhspan(ex["ebv"][i] - ex["e_ebv"][i], ex["ebv"][i] + ex["e_ebv"][i],
                               color="#ecece8", zorder=0)  # fmt: skip
                ax.set_xlabel("distance modulus [mag]")
                if i == 0:
                    ax.set_ylabel("E(B-V) [mag]")
                ax.set_title(
                    f"G = {ex['G'][i]:.2f}, plx = {plx:.3f} +- {err:.3f} mas",
                    fontsize=8,
                )
            handles = [Line2D([], [], color=_style(f)[0], linestyle=ls, label=f"{_label(f)} {t}")
                       for f in exs for ls, t in (("-", "predictive"), (":", "+ parallax"))]  # fmt: skip
            handles.append(Line2D([], [], color="#8a8a84", linestyle="--",
                                  label="parallax +- 1 sigma"))  # fmt: skip
            _finish(fig, out / "qa_examples.png", handles=handles)
            files.append("qa_examples.png")
    # (6) open clusters.
    if m["clusters"] and any(c["n_clusters"] for c in m["clusters"].values()):
        fig, axes = _figure(1, 2, (10, 3.8))
        for f, c in m["clusters"].items():
            col, ls = _style(f)
            g = [d for d in c["clusters"] if "dmu" in d]
            if g:
                axes[0, 0].scatter([d["mu_geo"] for d in g], [d["dmu"] for d in g], s=12,
                                   color=col, label=_label(f), alpha=0.8, linewidths=0)  # fmt: skip
            axes[0, 1].hist([d["chi2_s"] for d in c["clusters"]], np.linspace(0, 5, 26),
                            histtype="step", color=col, linestyle=ls, linewidth=1.6,
                            label=_label(f))  # fmt: skip
        axes[0, 0].axhline(0, color="#8a8a84", linewidth=0.8)
        axes[0, 0].set_xlabel("cluster geometric distance modulus [mag]")
        axes[0, 0].set_ylabel("mean mu_sp - mu_geo [mag]")
        axes[0, 1].axvline(1, color="#8a8a84", linewidth=0.8)
        axes[0, 1].set_xlabel("chi2/dof of members' s about the cluster mean")
        axes[0, 1].set_ylabel("clusters")
        _finish(fig, out / "qa_clusters.png", axes[0, 1])
        files.append("qa_clusters.png")
    # (7) beyond XP: precision and calibration vs G with and without XP.
    if m["noxp"]:
        fig, axes = _figure(1, 2, (10, 3.8))
        for f, d in m["noxp"].items():
            for kind, nm in (("full", f), ("noxp", f"{f}_noxp")):
                c, ls = _style(nm)
                rows = d[kind]["by_G"]
                axes[0, 0].plot(x, [r["median_sigma_s"] if r["n"] >= MIN_N else np.nan for r in rows],
                                color=c, linestyle=ls, marker="o", markersize=4, linewidth=1.6,
                                label=_label(nm))  # fmt: skip
                axes[0, 1].plot(x, [r["cov68"] - 0.68 if r["n"] >= MIN_N else np.nan for r in rows],
                                color=c, linestyle=ls, marker="o", markersize=4, linewidth=1.6,
                                label=_label(nm))  # fmt: skip
        axes[0, 1].axhspan(-COV_TOL, COV_TOL, color="#ecece8", zorder=0)
        axes[0, 1].axhline(0, color="#8a8a84", linewidth=0.8)
        for ax in axes[0]:
            ax.set_xticks(x)
            ax.set_xticklabels(G_NAMES, fontsize=7)
            ax.set_xlabel("G [mag]")
        axes[0, 0].set_ylabel("median half-width of s (16-84 %)")
        axes[0, 1].set_ylabel("68 % parallax coverage - nominal")
        _finish(fig, out / "qa_noxp.png", axes[0, 0])
        files.append("qa_noxp.png")
    return files


def _contours(ax, s, E, lp, color, ls):
    """68 / 95 % HPD contours of a grid log density (s nodes [Ns], E [Ne])."""
    mu = 10 + 5 * s / math.log(10)
    p = np.exp(lp - np.nanmax(lp))
    mass = (p * np.outer(np.gradient(s), np.gradient(E))).ravel()
    order = np.argsort(mass)[::-1]
    cum = np.cumsum(mass[order]) / mass.sum()
    flat = p.ravel()[order]
    levels = np.unique(
        [flat[min(np.searchsorted(cum, q), len(cum) - 1)] for q in (0.95, 0.68)]
    )
    if len(levels) == 2 and np.isfinite(levels).all():
        ax.contour(
            mu, E, p.T, levels=levels, colors=[color], linestyles=ls, linewidths=1.3
        )
    inside = p >= flat[min(np.searchsorted(cum, 0.99), len(cum) - 1)]
    if not inside.any():
        return None
    i, j = np.nonzero(inside)
    return np.array([mu[i].min(), E[j].min(), mu[i].max(), E[j].max()])


STAGES = dict(
    data=(stage_data, ("observations.npz", "selection.npz")),
    clusters=(stage_clusters, ("clusters.npz",)),
    train=(stage_train, ()),
    predict=(stage_predict, ()),
    noxp=(stage_noxp, ()),
    report=(stage_report, ("report.md", "metrics.json")),
)


def main():
    import yaml

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--inputs", type=Path, required=True,
                   help="S1 run with PARALLAX-hidden latents: latent.npy, latent_meta.json, "
                   "predictions.npz")  # fmt: skip
    p.add_argument("--config", type=Path, required=True, help="fine-tune config (yaml)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--labels", type=Path, help="labels.npz (default: INPUTS/labels.npz)"
    )
    p.add_argument("--fits", type=Path, help="table (default: data.ft_datafile)")
    p.add_argument("--sidecar", type=Path, help="covariance sidecar (parallax_zpt_l21)")
    p.add_argument(
        "--reddening", type=Path, help="npz: source_id, ebv_star, e_ebv_star"
    )
    p.add_argument("--core-rows", type=Path, help="frozen core rows (rows.npy)")
    p.add_argument("--members", type=Path, help="open-cluster members FITS (Hunt+24)")
    p.add_argument(
        "--clusters-table", type=Path, help="open-cluster table FITS (Hunt+24)"
    )
    p.add_argument("--member-columns", default="GaiaDR3,ID,Prob",
                   help="members: source_id, cluster id, membership probability")  # fmt: skip
    p.add_argument("--cluster-columns", default="ID,Name,Type")
    p.add_argument(
        "--cluster-types", default="o", help="comma list of cluster Type values"
    )
    p.add_argument("--cluster-min-prob", type=float, default=0.7)
    p.add_argument("--cluster-min-n", type=int, default=5)
    p.add_argument("--cluster-rows", choices=["heldout", "all"], default="heldout")
    p.add_argument(
        "--checkpoint", type=Path, help="fine-tune checkpoint (noxp encoding)"
    )
    p.add_argument(
        "--noxp-inputs", type=Path, help="existing XP-hidden latents (latent.npy)"
    )
    p.add_argument("--noxp-hide", default="PARALLAX,bp_*,rp_*",
                   help="inputs hidden for the beyond-XP latents (fnmatch patterns)")  # fmt: skip
    p.add_argument("--allow-parallax-input", action="store_true",
                   help="do not refuse latents whose input included the parallax")  # fmt: skip
    p.add_argument("--stages", default="data,clusters,train,predict,report")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--force", action="store_true")
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--max-train", type=int, default=3_000_000)
    p.add_argument("--max-stop", type=int, default=200_000)
    p.add_argument("--max-stats", type=int, default=200_000)
    p.add_argument("--min-train", type=int, default=50)
    p.add_argument("--predict-rows", choices=["eval", "all"], default="eval")
    p.add_argument("--parallax-inflation", choices=["fit", "none"], default="fit")
    p.add_argument("--zp-sigma", type=float, default=0.0,
                   help="zero-point uncertainty added in quadrature [mas]")  # fmt: skip
    p.add_argument("--hidden", type=_ints, default=[256, 256])
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--flow-transforms", type=int, default=3)
    p.add_argument("--flow-hidden", type=_ints, default=[128, 128])
    p.add_argument("--flow-bins", type=int, default=16)
    p.add_argument(
        "--s-bound", type=float, default=7.5, help="spline bound in s = ln(d/kpc)"
    )
    p.add_argument("--components", type=int, default=4)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-ns", type=int, default=40, help="s nodes of the likelihood")
    p.add_argument("--train-ne", type=int, default=24, help="inner E nodes (flow)")
    p.add_argument(
        "--pred-ns", type=int, default=256, help="s nodes of the s marginals"
    )
    p.add_argument(
        "--pred-nsj", type=int, default=48, help="s nodes of the joint grids"
    )
    p.add_argument("--pred-ne", type=int, default=48, help="E nodes of the joint grids")
    p.add_argument("--quad-rows", type=int, default=20_000)
    p.add_argument("--eval-chunk", type=int, default=2048)
    p.add_argument("--n-examples", type=int, default=4)
    p.add_argument("--chunk-rows", type=int, default=262_144)
    p.add_argument("--encoder-batch", type=int, default=8192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    config = yaml.safe_load(a.config.read_text())
    a.fits = a.fits or Path(config["data"]["ft_datafile"])
    a.labels = a.labels or a.inputs / "labels.npz"
    a.families = [f for f in a.families.split(",") if f]
    unknown = [f for f in a.families if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; choose from {FAMILIES}")
    a.member_columns = [c for c in a.member_columns.split(",") if c]
    a.cluster_columns = [c for c in a.cluster_columns.split(",") if c]
    a.cluster_types = [c for c in a.cluster_types.split(",") if c]
    a.noxp_hide = [c for c in a.noxp_hide.split(",") if c]
    if len(a.member_columns) != 3 or len(a.cluster_columns) != 3:
        raise SystemExit("--member-columns and --cluster-columns take three names each")
    if a.ensemble < 1:
        raise SystemExit("--ensemble must be >= 1")
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in a.stages.split(",") if s]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s!r}; choose from {list(STAGES)}")
    torch.manual_seed(a.seed)
    t0 = time.time()
    for s in STAGES:
        if s not in stages:
            continue
        fn, outputs = STAGES[s]
        if outputs and not a.force and all((out / f).exists() for f in outputs):
            log(f"stage {s}: outputs exist, skipped")
            continue
        log(f"stage {s} ({a.device})")
        fn(a, config, out)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    log(f"done in {time.time() - t0:.0f} s; peak RSS {rss:.1f} GB")
    if torch.cuda.is_available() and a.device.startswith("cuda"):
        log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
