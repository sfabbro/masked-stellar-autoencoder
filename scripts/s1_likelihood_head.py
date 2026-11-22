"""S1: likelihood (ratio) head on frozen latents, scored against the posterior head.

usage: s1_likelihood_head.py --config CFG --checkpoint CKPT --output DIR
       [--fits FITS] [--records REC] [--core-split ROWS_SPLIT.npz]
       [--stages latents,labels,parallax,train,infer,report] [--device cuda]
       [--parallax-nuisance] [--ensemble K] [--labels LABELS.npz]
       [--sidecar SIDECAR.h5] [--baseline S1_DIR]

Design note ``docs/design-20261120-likelihood-inference.md`` (S1, §5, §7).
Stages (each writes into DIR; a stage whose outputs exist is skipped unless
``--force``):

1. ``latents``: frozen encoder over every table row (preprocessing as
   ``predict_fits_rows``), latents as float16 ``latent.npy`` and the posterior
   head's (Teff, log g, [Fe/H]) median and 16/84 % (parallax-conditioned) in
   ``predictions.npz``.
2. ``labels``: common-scale labels per row (``common_scale_labels``): survey
   records through the checkpoint's maps, combined by inverse variance;
   ``labels.npz`` (contract fields ``LABEL_FIELDS`` plus the frozen core
   reference).
3. ``train``: two CNRE ratio heads on the train split, early stopping on the
   valid split: ``core`` (stars with a high-resolution [Fe/H] reference and
   Teff, log g) and ``all`` (all stars with complete mapped labels).
   Negatives are other training stars' labels (natural population).
4. ``infer``: MLE (``maximize_score``, init = posterior median) on held-out
   (valid + test) and core rows; profile intervals at 0.5 and 1.92 on test and
   core rows. Neyman critical values C(theta, G) at 68/95 % are learned on the
   valid split's labels (``CriticalValueRegressor``) for the joint statistic
   lambda = s_max - s(theta) (the joint set, scored by membership of the
   label) and for each parameter's profile statistic (per-parameter Neyman
   intervals: profile intervals at C_j(MLE, G)).
5. ``report``: the fixed validation table (``likelihood_validation``),
   ``report.md``, ``metrics.json`` and per-star ``s1_predictions.npz``
   (arm ``all``) and ``s1_predictions_<arm>.npz``.

Variant (``--parallax-nuisance``, arm ``plx``): parallax as a parameter with
its measurement likelihood.

- ``latents``: the encoder sees ``PARALLAX`` as missing (the NaN token), so z
  carries no parallax information (``latent_meta.json``); the posterior head
  (the comparator) still uses the full input.
- ``labels``: ``--labels`` reuses an existing ``labels.npz`` (copied).
- ``parallax``: ``parallax.npz``, the table's parallax and error and the
  Lindegren+2021 zero point (``--sidecar``, dataset ``parallax_zpt_l21``
  aligned to the table rows; NaN or no sidecar: -0.017 mas).
- ``train``: the heads take theta' = (Teff, log g, [Fe/H], u = log10 parallax)
  and train on stars whose corrected parallax has parallax/error >=
  ``--parallax-label-snr`` (10), u being the label.
- ``infer``: the score is s(theta', z) + log N(parallax_obs - zpt | 10**u,
  error^2) (constant terms dropped; no measurement: flat term), maximised over
  (theta, u) inside the 4-D label support; u is searched in a per-row window
  of +-``--parallax-window`` errors around the measurement (the whole box
  without one). Neyman calibration, label statistics and profiles are for the
  three atmospheric parameters with u profiled out.

Ensemble (``--ensemble K``): K heads per arm (seeds seed .. seed+K-1, same
data; ``head_<arm>.pt``, ``head_<arm>_<k>.pt``); the score is the mean of
their log-ratios (geometric mean of the ratios) and the Neyman critical
values are fitted on that statistic. With K > 1 member 0 alone is also
scored (MLE and test-label statistics, ``infer_<arm>_single.npz``) to show
the ensemble's effect on the MLE scatter and on the raw Wilks coverage.
``--baseline DIR`` adds another S1 run's arms (``--baseline-arms``) to every
report table.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ARMS = ("core", "all")
PLX_ARM = "plx"  # 'all' labels + parallax nuisance parameter
KNOWN_ARMS = (*ARMS, PLX_ARM)
BOUNDS = ((3000.0, 8000.0), (-0.5, 5.5), (-4.5, 1.0))
SUPPORT_WIDTHS = (100.0, 0.1, 0.1)  # occupancy-grid cells of the label support
U_BOUNDS = (-1.5, 2.5)  # log10(parallax / mas) of the plx arm (0.03-316 mas)
U_WIDTH = 0.1  # support cell in log10 parallax
LEVELS = (("68", 0.5, 0.32), ("95", 1.92, 0.05))  # Wilks threshold, Neyman alpha
WILKS68_JOINT = 1.7649  # wilks_threshold(0.6827, df=3): joint 3-parameter set
L21_CONSTANT_ZP = -0.017  # mas, Lindegren+2021 quasar median (rows without L21)
# A label with no score inside the support (e.g. its theta outside the 4-D
# support at every parallax of its window) is rejected: not in any set, and an
# exceedance of every quantile in the critical-value fit (which needs finite
# values; the quantile loss does not depend on how far above it lies).
LAMBDA_OUTSIDE = 1e3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state(path):
    from masked_stellar_autoencoder.models.checkpoint_load import (
        torch_load_finetune_checkpoint,
    )

    return torch_load_finetune_checkpoint(path, map_location="cpu")


def arm_theta(labels, arm, bounds=BOUNDS):
    """(theta (N, 3) physical, mask) of an arm's labels (inside the box).

    The ``plx`` arm calibrates and is scored on the ``all`` labels; only its
    training labels add the parallax (``training_theta``).
    """
    if arm == "core":
        theta = np.column_stack([labels["teff"], labels["logg"], labels["fe_h_core"]])
    elif arm in ("all", PLX_ARM):
        theta = np.column_stack([labels["teff"], labels["logg"], labels["fe_h"]])
    else:
        raise ValueError(arm)
    box = np.asarray(bounds)
    ok = np.isfinite(theta).all(1)
    ok &= np.all((theta >= box[:, 0]) & (theta <= box[:, 1]), axis=1)
    return theta, ok


def parallax_labels(par, snr, u_bounds=U_BOUNDS):
    """(u = log10 corrected parallax / mas, mask): parallax / error >= ``snr``."""
    plx = np.asarray(par["parallax_corrected"], "f8")
    err = np.asarray(par["parallax_error"], "f8")
    ok = np.isfinite(plx) & np.isfinite(err) & (err > 0)
    ok &= plx >= snr * np.where(ok, err, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(ok, np.log10(np.where(ok, plx, 1.0)), np.nan)
    ok &= (u >= u_bounds[0]) & (u <= u_bounds[1])
    return u, ok


def training_theta(labels, arm, out, a):
    """(theta, mask, bounds, support widths) of an arm's training labels."""
    theta, ok = arm_theta(labels, arm)
    if arm != PLX_ARM:
        return theta, ok, BOUNDS, SUPPORT_WIDTHS
    u, oku = parallax_labels(np.load(out / "parallax.npz"), a.parallax_label_snr)
    return (np.column_stack([theta, u]), ok & oku, (*BOUNDS, U_BOUNDS),
            (*SUPPORT_WIDTHS, U_WIDTH))  # fmt: skip


def head_paths(out, arm, k):
    """Ensemble member files: member 0 is ``head_<arm>.pt`` (the K = 1 head)."""
    return [out / f"head_{arm}.pt"] + [out / f"head_{arm}_{i}.pt" for i in range(1, k)]


def history_path(out, arm, i):
    return out / (f"history_{arm}.json" if i == 0 else f"history_{arm}_{i}.json")


def _latent_hidden(out):
    meta = out / "latent_meta.json"
    return (
        json.loads(meta.read_text()).get("hidden_inputs", []) if meta.exists() else []
    )


# ── stages ──────────────────────────────────────────────────────────────────


def stage_latents(a, config, out):
    from masked_stellar_autoencoder.training.frozen_latents import encode_table

    state = load_state(a.checkpoint)
    encode_table(
        config, state, a.fits, out, device=a.device, chunk_rows=a.chunk_rows,
        batch=a.encoder_batch, condition_parallax=a.condition_parallax,
        hide=("PARALLAX",) if a.parallax_nuisance else (), log=log,
    )  # fmt: skip


def stage_labels(a, config, out):
    from masked_stellar_autoencoder.training.common_scale_labels import write_labels

    if a.labels:
        src = np.load(a.labels)["source_id"]
        sid = np.load(out / "predictions.npz")["source_id"]
        if len(src) != len(sid) or not np.array_equal(src, sid):
            raise SystemExit(f"{a.labels}: rows do not match this table's rows")
        shutil.copyfile(a.labels, out / "labels.npz")
        diag = a.labels.with_name("labels_diagnostics.json")
        if diag.exists():
            shutil.copyfile(diag, out / "labels_diagnostics.json")
        log(f"labels: reused {a.labels}")
        return
    diag = write_labels(
        config, load_state(a.checkpoint), dict(np.load(a.records, allow_pickle=False)),
        np.load(out / "predictions.npz"), a.fits, out / "labels.npz",
        iterations=a.map_iterations, device=a.device, core_split=a.core_split, log=log,
    )  # fmt: skip
    (out / "labels_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    log(f"labels: per split (train/valid/test/not kept) {diag['split_counts']}")


def read_parallax(config, fits_path, sidecar=None, chunk_rows=1 << 20, log=print):
    """Per-row parallax, error, L21 zero point and corrected parallax (mas).

    Table columns ``PARALLAX`` and its error column (``data.error_cols``);
    zero point ``parallax_zpt_l21`` from a covariance sidecar aligned to the
    table rows (source_id checked), else (NaN, no sidecar) ``L21_CONSTANT_ZP``.
    """
    from astropy.io import fits

    cols = config["data"]["feature_cols"]
    ecol = config["data"]["error_cols"][cols.index("PARALLAX")]
    if not ecol:
        raise ValueError("The config has no PARALLAX error column")
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        n = len(data)
        sid, plx, err = np.zeros(n, "i8"), np.full(n, np.nan), np.full(n, np.nan)
        for start in range(0, n, chunk_rows):
            chunk = data[start : start + chunk_rows]
            sl = slice(start, start + len(chunk))
            sid[sl] = np.asarray(chunk["source_id"], "i8")
            plx[sl] = np.asarray(chunk["PARALLAX"], "f8")
            err[sl] = np.asarray(chunk[ecol], "f8")
    zpt = np.full(n, np.nan)
    if sidecar is not None:
        from masked_stellar_autoencoder.data.covariance_sidecar import (
            CovarianceSidecar,
        )

        with CovarianceSidecar(sidecar) as sc:
            r = sc.read(["source_id", "parallax_zpt_l21"])
        if len(r["source_id"]) != n or not np.array_equal(r["source_id"], sid):
            raise ValueError(f"{sidecar} is not aligned to the table rows")
        zpt = np.asarray(r["parallax_zpt_l21"], "f8")
    l21 = np.isfinite(zpt)
    zpt = np.where(l21, zpt, L21_CONSTANT_ZP)
    log(f"parallax: {n} rows, {int(np.isfinite(plx).sum())} with a parallax, "
        f"{int(l21.sum())} with the L21 zero point (else {L21_CONSTANT_ZP} mas)")  # fmt: skip
    return dict(source_id=sid, parallax=plx, parallax_error=err, zero_point=zpt,
                zero_point_l21=l21, parallax_corrected=plx - zpt)  # fmt: skip


def stage_parallax(a, config, out):
    if PLX_ARM not in a.arms:
        log("parallax: no parallax arm, skipped")
        return
    par = read_parallax(config, a.fits, a.sidecar, log=log)
    sid = np.load(out / "predictions.npz")["source_id"]
    if not np.array_equal(par["source_id"], sid):
        raise SystemExit("parallax: table rows differ from predictions.npz")
    np.savez(out / "parallax.npz", sidecar=str(a.sidecar or ""), **par)


def stage_train(a, config, out):
    from masked_stellar_autoencoder.training.likelihood_head import (
        RatioHead,
        save_head,
        train_ratio_head,
    )

    lab = np.load(out / "labels.npz")
    lat = np.load(out / "latent.npy", mmap_mode="r")
    split = lab["split"]
    dev = torch.device(a.device)
    for arm in a.arms:
        paths = head_paths(out, arm, a.ensemble)
        todo = [i for i, p in enumerate(paths) if a.force or not p.exists()]
        if not todo:
            log(f"train {arm}: heads exist, skipped")
            continue
        if arm == PLX_ARM and "PARALLAX" not in _latent_hidden(out):
            raise SystemExit("train plx: latents were not computed with PARALLAX "
                             "hidden (--parallax-nuisance)")  # fmt: skip
        theta, ok, bounds, _ = training_theta(lab, arm, out, a)
        tr = np.flatnonzero(ok & (split == 0))
        va = np.flatnonzero(ok & (split == 1))
        log(f"train {arm}: {len(tr)} train, {len(va)} valid stars, "
            f"{len(paths)} member(s)")  # fmt: skip
        if len(tr) < a.min_train or len(va) < 2:
            log(f"train {arm}: too few labelled stars, arm skipped")
            continue
        rows = np.concatenate([tr, va])
        Z = torch.as_tensor(np.asarray(lat[rows]), device=dev)
        T = torch.as_tensor(theta[rows], dtype=torch.float32, device=dev)
        itr = torch.arange(len(tr), device=dev)
        iva = torch.arange(len(tr), len(rows), device=dev)
        for i in todo:
            head = RatioHead(
                lat.shape[1], n_params=theta.shape[1], embed_out=a.embed_dim
            )
            hist = train_ratio_head(
                head, Z, T, itr, iva, negatives=a.negatives, max_epochs=a.epochs,
                patience=a.patience, lr=a.lr, seed=a.seed + i, log=log,
            )  # fmt: skip
            hist["seed"] = a.seed + i
            save_head(head, paths[i], history=hist, arm=arm, bounds=bounds, member=i)
            history_path(out, arm, i).write_text(json.dumps(hist, indent=2) + "\n")
        del Z, T
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def _fit_critical(theta, g, stat, seed):
    """C(theta, G) at the 68 and 95 % levels (``CriticalValueRegressor``).

    Non-finite statistics (label outside the support) enter as
    ``LAMBDA_OUTSIDE``.
    """
    from torchregress.inference import CriticalValueRegressor

    stat = np.where(np.isfinite(stat), stat, LAMBDA_OUTSIDE)
    return CriticalValueRegressor(
        alpha=[lv[2] for lv in LEVELS], hidden=(64, 64), batch_size=4096,
        max_epochs=200, seed=seed,
    ).fit(torch.as_tensor(np.column_stack([theta, g])), torch.as_tensor(stat))  # fmt: skip


def _predict_critical(reg, theta, g):
    """(N, len(LEVELS)) critical values; NaN without a regressor."""
    if reg is None or not len(theta):
        return np.full((len(theta), len(LEVELS)), np.nan)
    c = reg.predict(torch.as_tensor(np.column_stack([theta, g]))).numpy()
    return np.asarray(c, "f8").reshape(len(theta), len(LEVELS))


def _rows_t(idx, dev):
    return torch.as_tensor(np.asarray(idx), device=dev)


def _joint_statistic(a, score, x, box, rows, th, smax, mle):
    """Joint statistic of the labels ``th`` at ``rows`` [len(rows)].

    lambda = s_max - s(theta_label) for a score over theta only; the profile
    statistic over the nuisance coordinates (the parallax) when the box has
    more coordinates than theta.
    """
    from masked_stellar_autoencoder.training import likelihood_head as lh

    n_atm = th.shape[1]
    if not len(rows):
        return np.empty(0)
    xr = x[_rows_t(rows, x.device)]
    if len(box) > n_atm:
        return lh.profile_statistic(
            score, xr, list(range(n_atm)), th[rows], smax[rows], mle[rows], box,
            n_candidates=16, n_starts=2, seed=a.seed, block=a.row_block,
            chunk_size=a.chunk_size,
        )  # fmt: skip
    return lh.lr_statistic(score, xr, th[rows], smax[rows], chunk_size=a.chunk_size)


def _param_statistics(a, score, x, box, rows, th, smax, mle):
    """Per-parameter profile statistics of the labels at ``rows`` [len(rows), 3]."""
    from masked_stellar_autoencoder.training import likelihood_head as lh

    n_atm = th.shape[1]
    out = np.full((len(rows), n_atm), np.nan)
    if not len(rows):
        return out
    xr = x[_rows_t(rows, x.device)]
    for j in range(n_atm):
        out[:, j] = lh.profile_statistic(
            score, xr, j, th[rows, j], smax[rows], mle[rows], box, seed=a.seed,
            block=a.row_block, chunk_size=a.chunk_size,
        )  # fmt: skip
    return out


def _infer_score(a, score, x, box, init, th, g, sp, ok, profiled, label):
    """MLE, label statistics, Neyman critical values and profiles of one score.

    ``x`` observations of the target rows, ``box`` the search box (theta, then
    nuisance coordinates), ``th``/``g``/``sp``/``ok`` the rows' 3-parameter
    labels, G, split and label mask. Returns the ``infer_<arm>.npz`` dict
    (``mle`` holds theta, ``mle_search`` every search coordinate).
    """
    from masked_stellar_autoencoder.training import likelihood_head as lh

    kw = dict(block=a.row_block, chunk_size=a.chunk_size)
    n_atm = th.shape[1]
    dev = x.device
    t0 = time.time()
    mle_s, smax, atb = lh.maximize(
        score, x, box, init=init, n_candidates=a.n_candidates, seed=a.seed, log=log,
        **kw,
    )  # fmt: skip
    mle = mle_s[:, :n_atm]
    t_mle = time.time() - t0
    found = np.isfinite(smax)
    if not found.all():
        log(f"infer {label}: {int((~found).sum())} rows found no score inside the "
            "label support; left out of statistics and profiles")  # fmt: skip
    has = ok & found
    # Calibration (valid) and coverage (test) rows with the arm's label.
    cal = np.flatnonzero(has & (sp == 1))
    if len(cal) > a.max_calib:
        rng = np.random.default_rng(a.seed)
        cal = np.sort(rng.choice(cal, a.max_calib, replace=False))
    need = np.union1d(cal, np.flatnonzero(has & (sp == 2)))
    t1 = time.time()
    # Joint statistic (every labelled row for a theta-only score, as in S1;
    # the calibration and test rows when the parallax is profiled).
    jrows = need if len(box) > n_atm else np.flatnonzero(has)
    lam = np.full(len(th), np.nan)
    lam[jrows] = _joint_statistic(a, score, x, box, jrows, th, smax, mle_s)
    enough = len(cal) >= a.min_calib
    if not enough:
        log(f"infer {label}: {len(cal)} calibration stars, no Neyman sets")
    reg = _fit_critical(th[cal], g[cal], lam[cal], a.seed) if enough else None
    res = dict(table_row=None, mle=mle, mle_search=mle_s, smax=smax, at_bound=atb,
               lam_label=lam, profiled=profiled, n_calib=len(cal), t_mle=t_mle,
               label_outside=int((~np.isfinite(lam[need])).sum()),
               n_label_statistics=len(need))  # fmt: skip
    clab = np.full((len(th), len(LEVELS)), np.nan)
    clab[has] = _predict_critical(reg, th[has], g[has])
    p = np.flatnonzero(profiled & found)
    cmle = np.full((len(th), len(LEVELS)), np.nan)
    cmle[p] = _predict_critical(reg, mle[p], g[p])
    for k, (name, _, _) in enumerate(LEVELS):
        res[f"in_neyman{name}"] = has & (lam <= clab[:, k])
        res[f"crit{name}_mle"] = cmle[:, k]
    # Per-parameter profile statistic at the label, its critical values
    # C_j(theta, G), membership, and the thresholds at the MLE.
    lamj = np.full((len(th), n_atm), np.nan)
    lamj[need] = _param_statistics(a, score, x, box, need, th, smax, mle_s)
    thr_mle = {name: np.full((len(p), n_atm), np.nan) for name, _, _ in LEVELS}
    inj = {name: np.zeros((len(th), n_atm), bool) for name, _, _ in LEVELS}
    for j in range(n_atm):
        regj = _fit_critical(th[cal], g[cal], lamj[cal, j], a.seed) if enough else None
        cj = _predict_critical(regj, th[need], g[need])
        cm = _predict_critical(regj, mle[p], g[p])
        for k, (name, _, _) in enumerate(LEVELS):
            inj[name][need, j] = lamj[need, j] <= cj[:, k]
            thr_mle[name][:, j] = cm[:, k]
    res["lam_label_param"] = lamj
    res["label_outside_param"] = (~np.isfinite(lamj[need])).sum(0)
    for name, _, _ in LEVELS:
        res[f"in_neyman{name}_param"] = inj[name]
        res[f"neyman{name}_crit"] = np.full((len(th), n_atm), np.nan)
        res[f"neyman{name}_crit"][p] = thr_mle[name]
    t_stat = time.time() - t1
    # Profile intervals of theta (nuisance coordinates profiled out): Wilks
    # thresholds and per-parameter Neyman values.
    xp = x[_rows_t(p, dev)]
    t2 = time.time()
    for name, wilks, _ in LEVELS:
        for kind, thr in (("", wilks), ("neyman", thr_mle[name])):
            lo = np.full((len(th), n_atm), np.nan)
            hi = np.full((len(th), n_atm), np.nan)
            lob = np.zeros((len(th), n_atm), bool)
            hib = np.zeros((len(th), n_atm), bool)
            if len(p) and np.isfinite(thr).all():
                lo[p], hi[p], lob[p], hib[p] = lh.profiles(
                    score, xp, mle_s[p], smax[p], thr, box, params=range(n_atm),
                    log=log, label=f"{label} {kind}{name}", **kw,
                )  # fmt: skip
            if kind:
                res[f"neyman{name}_lo"], res[f"neyman{name}_hi"] = lo, hi
            else:
                res[f"lo{name}"], res[f"hi{name}"] = lo, hi
                res[f"lo{name}_at_bound"], res[f"hi{name}_at_bound"] = lob, hib
    res["t_stat"], res["t_profiles"] = t_stat, time.time() - t2
    log(f"infer {label}: {len(th)} rows, {len(p)} profiled, {len(need)} "
        f"label statistics; MLE {t_mle:.0f} s, label statistics {t_stat:.0f} s,"
        f" profiles {res['t_profiles']:.0f} s")  # fmt: skip
    return res


def _single_member(a, score, x, box, init, th, sp, ok, label):
    """Member 0 alone: MLE of every row, test-label statistics (Wilks coverage)."""
    from masked_stellar_autoencoder.training import likelihood_head as lh

    one, x1 = score.member(0), score.member_observation(x, 0)
    t0 = time.time()
    mle_s, smax, atb = lh.maximize(
        one, x1, box, init=init, n_candidates=a.n_candidates, seed=a.seed, log=log,
        block=a.row_block, chunk_size=a.chunk_size,
    )  # fmt: skip
    rows = np.flatnonzero(ok & (sp == 2))
    if a.single_max_rows and len(rows) > a.single_max_rows:
        rng = np.random.default_rng(a.seed + 1)
        rows = np.sort(rng.choice(rows, a.single_max_rows, replace=False))
    stat_rows = np.zeros(len(th), bool)
    stat_rows[rows] = True  # rows with no score in the support count as not covered
    st = rows[np.isfinite(smax[rows])]
    lam = np.full(len(th), np.nan)
    lamj = np.full((len(th), th.shape[1]), np.nan)
    lam[st] = _joint_statistic(a, one, x1, box, st, th, smax, mle_s)
    lamj[st] = _param_statistics(a, one, x1, box, st, th, smax, mle_s)
    log(f"infer {label} member 0: MLE + {len(rows)} test-label statistics "
        f"({time.time() - t0:.0f} s)")  # fmt: skip
    return dict(mle=mle_s[:, : th.shape[1]], mle_search=mle_s, smax=smax, at_bound=atb,
                lam_label=lam, lam_label_param=lamj, stat_rows=stat_rows,
                t_single=time.time() - t0)  # fmt: skip


def _parallax_outputs(score, x, res, window):
    """Decoded parallax MLE (mas), its pull against the measurement, window."""
    a_, b_, plx, w = window
    n_atm = len(BOUNDS)
    u = a_ + b_ * res["mle_search"][:, n_atm]
    with np.errstate(invalid="ignore", over="ignore"):
        varpi = 10.0**u
        pull = np.where(w > 0, (varpi - plx) * w, np.nan)
    return dict(mle_log_parallax=u, mle_parallax=varpi, parallax_pull=pull,
                parallax_window=np.column_stack([a_, b_]),
                parallax_measured=w > 0)  # fmt: skip


def _on_support_edge(score, x, mle_search, found):
    out = np.zeros(len(found), bool)
    if score.support is None or not found.any():
        return out
    rows = np.flatnonzero(found)
    th = torch.as_tensor(mle_search[rows], dtype=x.dtype, device=x.device)
    tp = score.physical(th, x[_rows_t(rows, x.device)])
    out[rows] = score.support.on_edge(tp).cpu().numpy()
    return out


def stage_infer(a, config, out):
    from masked_stellar_autoencoder.training import likelihood_head as lh

    lab = np.load(out / "labels.npz")
    pred = np.load(out / "predictions.npz")
    lat = np.load(out / "latent.npy", mmap_mode="r")
    split, G = lab["split"], pred["G"].astype("f8")
    G = np.where(np.isfinite(G), G, np.nanmedian(G))
    target = np.flatnonzero(np.isin(split, (1, 2)) | lab["has_core_feh"])
    profiled = lab["has_core_feh"][target].copy()
    if a.profile_rows == "test+core":
        profiled |= split[target] == 2
    elif a.profile_rows == "none":
        profiled[:] = False
    if a.profile_valid:
        profiled[:] = True
    dev = torch.device(a.device)
    for arm in a.arms:
        paths = head_paths(out, arm, a.ensemble)
        if not paths[0].exists():
            log(f"infer {arm}: no head, skipped")
            continue
        if (out / f"infer_{arm}.npz").exists() and not a.force:
            log(f"infer {arm}: outputs exist, skipped")
            continue
        missing = [str(q) for q in paths if not q.exists()]
        if missing:
            raise SystemExit(f"infer {arm}: missing ensemble members {missing}")
        heads = [lh.load_head(q, device=dev)[0] for q in paths]
        plx = arm == PLX_ARM
        theta, ok = arm_theta(lab, arm)
        support = None
        if a.support:
            tt, tok, sb, sw = training_theta(lab, arm, out, a)
            support = lh.Support(tt[tok & (split == 0)], sb, sw, dev)
            log(f"infer {arm}: label support covers {support.fraction:.3f} of the box")
        score = lh.EnsembleScore(heads, support=support, parallax=plx)
        th_t, g_t, sp_t = theta[target], G[target], split[target]
        extra = lh.N_PARALLAX_COLUMNS if plx else 0
        emb = lh.embed_rows(score, lat, target, device=dev, extra=extra)
        init = pred["post_median"][target].astype("f8")
        box, window = [list(b) for b in BOUNDS], None
        if plx:
            par = np.load(out / "parallax.npz")
            window = lh.parallax_window(
                par["parallax_corrected"][target], par["parallax_error"][target],
                U_BOUNDS, a.parallax_window,
            )  # fmt: skip
            box.append([0.0, 1.0])
        x = score.observation(emb, window)
        del emb
        if plx:
            mid = np.asarray(BOUNDS).mean(1)
            t0 = lh.window_start(score, x, np.where(np.isfinite(init), init, mid))
            init = np.column_stack([init, t0])
        res = _infer_score(a, score, x, box, init, th_t, g_t, sp_t, ok[target],
                           profiled, arm)  # fmt: skip
        res["table_row"] = target
        res["n_members"] = len(heads)
        res["on_support_edge"] = _on_support_edge(
            score, x, res["mle_search"], np.isfinite(res["smax"])
        )
        if plx:
            res.update(_parallax_outputs(score, x, res, window))
        np.savez(out / f"infer_{arm}.npz", **res)
        if len(heads) > 1 and a.single_member:
            one = _single_member(a, score, x, box, init, th_t, sp_t, ok[target], arm)
            one["table_row"] = target
            one["on_support_edge"] = _on_support_edge(
                score, x, one["mle_search"], np.isfinite(one["smax"])
            )
            if plx:
                one.update(_parallax_outputs(score, x, one, window))
            np.savez(out / f"infer_{arm}_single.npz", **one)
        del x
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def _posterior_p_below(pred, limit=-2.5):
    from scipy.stats import t as student_t

    nu = pred["post_nu"].astype("f8") if "post_nu" in pred.files else 5.0
    med = pred["post_median"][:, 2].astype("f8")
    half = 0.5 * (pred["post_q84"][:, 2] - pred["post_q16"][:, 2]).astype("f8")
    scale = half / student_t.ppf(0.84, nu)
    return student_t.cdf((limit - med) / scale, nu)


def _load_arms(a, out, lab):
    """infer dicts of this run's arms, the baseline's arms, and member-0 runs."""
    inf, source, single = {}, {}, {}
    for arm in KNOWN_ARMS:
        path = out / f"infer_{arm}.npz"
        if path.exists():
            inf[arm], source[arm] = dict(np.load(path)), str(out)
        path = out / f"infer_{arm}_single.npz"
        if path.exists():
            single[arm] = dict(np.load(path))
    if a.baseline:
        blab = a.baseline / "labels.npz"
        if blab.exists() and not np.array_equal(
            np.load(blab)["source_id"], lab["source_id"]
        ):
            raise SystemExit(f"baseline {a.baseline}: table rows differ")
        for arm in a.baseline_arms:
            path = a.baseline / f"infer_{arm}.npz"
            if arm in inf:
                log(f"report: arm {arm} run here; baseline {arm} not added")
            elif path.exists():
                inf[arm], source[arm] = dict(np.load(path)), str(a.baseline)
            else:
                log(f"report: no {path}")
    # Baseline arms first in every table, then this run's.
    inf = {k: inf[k] for k in sorted(inf, key=lambda k: source[k] == str(out))}
    return inf, source, single


def _coverage(r, theta, ok, split, G, pred):
    """Coverage tables of one arm's test rows with G <= 16 (each kind present)."""
    from masked_stellar_autoencoder.training import likelihood_validation as V

    rows = r["table_row"]
    sel = ok[rows] & (split[rows] == 2)
    if "stat_rows" in r:
        sel &= r["stat_rows"]
    th, g = theta[rows][sel], G[rows][sel]
    lowg = g <= V.G_MAX
    cov = {}

    def add(name, covered, nominal):
        cov[name] = V.coverage_table(th[lowg], g[lowg], np.asarray(covered)[lowg],
                                     nominal)  # fmt: skip

    if "in_neyman68" in r:
        for name, _, alpha in LEVELS:
            add(f"neyman{name}", r[f"in_neyman{name}"][sel], 1 - alpha)
    with np.errstate(invalid="ignore"):
        add("wilks68_joint", r["lam_label"][sel] <= WILKS68_JOINT, 0.6827)
        for j, prm in enumerate(V.PARAMS):
            if "lo68" in r and r["profiled"][sel].all():
                inside = (r["lo68"][sel, j] <= th[:, j]) & (
                    th[:, j] <= r["hi68"][sel, j]
                )
                add(f"profile68_{prm}", inside, 0.6827)
            add(f"wilks68_{prm}", r["lam_label_param"][sel, j] <= 0.5, 0.6827)
            if "in_neyman68_param" in r:
                add(f"neyman68_{prm}", r["in_neyman68_param"][sel, j], 0.68)
            hq = (pred["post_q16"][rows][sel, j] <= th[:, j]) & (
                th[:, j] <= pred["post_q84"][rows][sel, j]
            )
            add(f"posterior68_{prm}", hq, 0.68)
    return cov


def _edges(r, split, G, on_edge):
    """Fractions of rows left out / at the box, support or window edges."""
    rows = r["table_row"]
    sp, g = split[rows], G[rows]
    found = np.isfinite(r["smax"])
    atb = np.asarray(r["at_bound"], bool)
    n_atm = len(BOUNDS)
    groups = (("held-out", np.isin(sp, (1, 2))), ("test G<=16", (sp == 2) & (g <= 16)),
              ("test G>16", (sp == 2) & (g > 16)))  # fmt: skip
    out = {}
    for name, m in groups:
        if not m.any():
            continue
        d = dict(n=int(m.sum()), left_out=float(np.mean(~found[m])),
                 at_box_bound=float(np.mean(found[m] & atb[m, :n_atm].any(1))))  # fmt: skip
        for j, prm in enumerate(("teff", "logg", "fe_h")):
            d[f"at_bound_{prm}"] = float(np.mean(found[m] & atb[m, j]))
        if on_edge is not None:
            d["support_edge"] = float(np.mean(found[m] & on_edge[m]))
        if atb.shape[1] > n_atm:
            d["parallax_window_edge"] = float(np.mean(found[m] & atb[m, n_atm]))
        if "parallax_pull" in r:
            pull = r["parallax_pull"][m]
            with np.errstate(invalid="ignore"):
                d["parallax_pull_gt3"] = float(np.mean(found[m] & (np.abs(pull) > 3)))
            d["parallax_measured"] = float(np.mean(np.isfinite(pull)))
        out[name] = d
    return out


def _baseline_edge(r, lab, arm):
    """Support-edge flags of a 3-parameter arm without stored ones (recomputed)."""
    from masked_stellar_autoencoder.training.likelihood_head import Support

    theta, ok = arm_theta(lab, arm)
    sup = Support(theta[ok & (lab["split"] == 0)], BOUNDS, SUPPORT_WIDTHS)
    found = np.isfinite(r["smax"])
    edge = np.zeros(len(found), bool)
    edge[found] = sup.on_edge(torch.as_tensor(r["mle"][found, :3])).numpy()
    return edge


def _precision(metrics):
    """Precision summary per estimator: NMAD by [Fe/H] bin, anchors, labels,
    raw Wilks and Neyman coverage (test, G <= 16)."""
    out = {}
    cells = (("[Fe/H] NMAD core [-0.5,9) G<=16", -0.5), ("[Fe/H] NMAD core [-1,-0.5) G<=16", -1.0),
             ("[Fe/H] NMAD core [-2.5,-2) G<=16", -2.5), ("[Fe/H] NMAD core [-3,-2.5) G<=16", -3.0),
             ("[Fe/H] NMAD core [-9,-3) G<=16", -9.0))  # fmt: skip
    for est, tab in metrics["frozen_core"].items():
        d = {}
        for name, lo in cells:
            r = next((r for r in tab if r["feh_lo"] == lo and r["g"] == "G<=16"), None)
            d[name] = r["nmad"] if r else None
        d["[Fe/H] worst |bias| core G<=16"] = max(
            (abs(r["bias"]) for r in tab if r["g"] == "G<=16" and r["n"] >= 20),
            default=None,
        )
        for key, name in (("teff_anchor", "Teff NMAD IRFM anchors"),
                          ("logg_anchor", "log g NMAD seismic anchors")):  # fmt: skip
            if est in metrics["anchors"].get(key, {}):
                d[name] = metrics["anchors"][key][est]["nmad"]
        for prm, rows in metrics["test_labels"].get(est, {}).items():
            d[f"test {prm} NMAD vs labels G<=16"] = rows["G<=16"]["nmad"]
        arm = est[4:] if est.startswith("mle_") else None
        cov = metrics["coverage"].get(arm, {}) if arm else {}
        for k, name in (("wilks68_joint", "raw Wilks 68 joint (df 3)"),
                        ("wilks68_teff", "raw Wilks 68 Teff"),
                        ("wilks68_logg", "raw Wilks 68 log g"),
                        ("wilks68_fe_h", "raw Wilks 68 [Fe/H]"),
                        ("neyman68", "Neyman 68 joint")):  # fmt: skip
            if cov.get(k):
                d[name] = cov[k]["all"]["coverage"]
        out[est] = d
    return out


def _score_note(arm, r, a):
    k = int(r.get("n_members", 1))
    s = "one ratio head"
    if k > 1:
        s = f"mean log-ratio of {k} heads (geometric mean of ratios)"
    if arm == PLX_ARM:
        s += (
            " over theta' = (Teff, log g, [Fe/H], log10 parallax) + "
            "log N(parallax_obs - zpt | parallax, error^2), parallax profiled "
            f"in a +-{a.parallax_window:g}-error window; latents without PARALLAX"
        )
    return s


def stage_report(a, config, out):
    from masked_stellar_autoencoder.training import likelihood_validation as V

    lab = np.load(out / "labels.npz")
    pred = np.load(out / "predictions.npz")
    rec = dict(np.load(a.records, allow_pickle=False))
    n = len(lab["split"])
    split, G = lab["split"], pred["G"].astype("f8")
    post = pred["post_median"].astype("f8")
    est = {"posterior": post}
    inf, source, single = _load_arms(a, out, lab)

    def full(r):
        f = np.full((n, 3), np.nan)
        f[r["table_row"]] = r["mle"][:, :3]
        return f

    for arm, r in inf.items():  # member 0 alone next to its ensemble
        if arm in single:
            est[f"mle_{arm}_single"] = full(single[arm])
        est[f"mle_{arm}"] = full(r)
    ref = lab["fe_h_ref"]
    core = np.isin(split, (1, 2)) & np.isfinite(ref)
    metrics = dict(frozen_core={}, anchors={}, coverage={}, selection={},
                   decision={}, test_labels={}, runs={}, edges={})  # fmt: skip
    for name, p in est.items():
        metrics["frozen_core"][name] = V.offset_table(p[core, 2], ref[core], G[core])
    metrics["anchors"] = V.anchor_table(est, V.anchor_records(rec, split == 2))
    # Coverage on test stars, each arm with its own labels.
    for arm, r in inf.items():
        theta, ok = arm_theta(lab, arm)
        metrics["coverage"][arm] = _coverage(r, theta, ok, split, G, pred)
        on_edge = r.get("on_support_edge")
        if on_edge is None and arm in ARMS:
            on_edge = _baseline_edge(r, lab, arm)
        metrics["edges"][arm] = _edges(r, split, G, on_edge)
        run = dict(
            source=source[arm], rows=int(len(r["table_row"])),
            profiled=int(r["profiled"].sum()), calibration=int(r["n_calib"]),
            t_mle_s=float(r["t_mle"]), t_profiles_s=float(r["t_profiles"]),
            members=int(r.get("n_members", 1)), score=_score_note(arm, r, a),
            at_bound_fraction=[float(x) for x in np.asarray(r["at_bound"]).mean(0)],
        )  # fmt: skip
        if "t_stat" in r:
            run["t_stat_s"] = float(r["t_stat"])
        if "label_outside" in r:
            run["labels_outside_support"] = dict(
                rows=int(r["n_label_statistics"]), joint=int(r["label_outside"]),
                per_parameter=[int(v) for v in r["label_outside_param"]],
            )  # fmt: skip
        metrics["runs"][arm] = run
    for arm, r in single.items():
        theta, ok = arm_theta(lab, arm)
        metrics["coverage"][f"{arm}_single"] = _coverage(r, theta, ok, split, G, pred)
        metrics["edges"][f"{arm}_single"] = _edges(
            r, split, G, r.get("on_support_edge")
        )
        metrics["runs"][f"{arm}_single"] = dict(
            source=str(out), members=1, t_s=float(r["t_single"]),
            score=f"member 0 of arm {arm} alone (same support"
            + (", parallax term" if arm == PLX_ARM else "") + ")",
        )  # fmt: skip
    # Metal-poor selection on the held-out core.
    selected = {"posterior P>0.9": _posterior_p_below(pred)[core] > 0.9}
    for arm, r in inf.items():
        hi = np.full(n, np.nan)
        hi[r["table_row"]] = r["hi95"][:, 2]
        selected[f"{arm}: LR test (hi95 < -2.5)"] = hi[core] < V.MP_LIMIT
        if "neyman95_hi" in r:
            hn = np.full(n, np.nan)
            hn[r["table_row"]] = r["neyman95_hi"][:, 2]
            selected[f"{arm}: Neyman 95 (hi < -2.5)"] = hn[core] < V.MP_LIMIT
    metrics["selection"] = V.selection_table(ref[core], G[core], selected)
    # Test split against the mapped labels, by G.
    test = split == 2
    for name, p in est.items():
        tab = {}
        for j, prm in enumerate(V.PARAMS):
            y = lab[prm]
            tab[prm] = {
                g: V._summary(p[m, j] - y[m])
                for g, m in (("G<=16", test & (G <= 16)), ("G>16", test & (G > 16)))
            }
        metrics["test_labels"][name] = tab
    for arm, r in inf.items():
        if int(r["n_calib"]) < a.min_calib:
            continue
        metrics["decision"][arm] = V.decide(
            metrics["frozen_core"][f"mle_{arm}"], metrics["frozen_core"]["posterior"],
            metrics["coverage"][arm]["neyman68"],
            metrics["anchors"], f"mle_{arm}",
        )  # fmt: skip
    metrics["precision"] = _precision(metrics)
    diag = out / "labels_diagnostics.json"
    metrics["labels"] = json.loads(diag.read_text()) if diag.exists() else {}
    for arm in inf:
        base = Path(source[arm])
        for i in range(int(inf[arm].get("n_members", 1))):
            h = history_path(base, arm, i)
            if not h.exists():
                continue
            hist = json.loads(h.read_text())
            keep = {k: hist[k] for k in ("best_valid", "epochs", "n_train", "n_valid",
                                         "batch_size", "seed") if k in hist}  # fmt: skip
            run = metrics["runs"].setdefault(arm, {})
            if i == 0:
                run["training"] = keep
            run.setdefault("training_members", []).append(keep)
    if (out / "parallax.npz").exists() and PLX_ARM in inf:
        par = np.load(out / "parallax.npz")
        theta, ok = arm_theta(lab, PLX_ARM)
        _, oku = parallax_labels(par, a.parallax_label_snr)
        tr = ok & oku & (split == 0)
        feh = theta[:, 2]
        metrics["runs"][PLX_ARM]["parallax"] = dict(
            label_snr=a.parallax_label_snr, window_errors=a.parallax_window,
            sidecar=str(par["sidecar"]),
            rows_with_l21_zero_point=int(par["zero_point_l21"].sum()),
            train_by_feh={f"[{lo:g},{hi:g})": [int((tr & (feh >= lo) & (feh < hi)).sum()),
                                               int((ok & (split == 0) & (feh >= lo)
                                                    & (feh < hi)).sum())]
                          for lo, hi in zip(V.FEH_BINS[:-1], V.FEH_BINS[1:], strict=True)},
        )  # fmt: skip
    adopted = [arm for arm, d in metrics["decision"].items() if d["adopt"]]
    metrics["summary"] = (
        f"Table {a.fits}; checkpoint {a.checkpoint}. Held-out frozen core: "
        f"{int(core.sum())} stars. Arms: {sorted(inf)}"
        + (f" (baseline arms from {a.baseline})" if a.baseline else "")
        + (f"; member-0 runs: {sorted(single)}" if single else "")
        + ". Decision: "
        + (f"adopt the likelihood product of arm(s) {adopted}." if adopted else
           "do not adopt the likelihood product (no arm passes every criterion).")
    )  # fmt: skip
    metrics["notes"] = [
        "Frozen-core offsets use the raw high-resolution reference (one record per "
        "star, fixed priority) on held-out (valid + test) core stars; the 'core' "
        "arm trains on the same records mapped to the common scale (train split).",
        "Neyman sets: lambda = s_max - s(theta) <= C(theta, G), C fitted on the valid "
        "split's labels (CriticalValueRegressor, 68/95 %); the decision uses the "
        "joint set. Per-parameter Neyman intervals calibrate each parameter's "
        "profile statistic the same way and are profile intervals at C_j(MLE, G).",
        "Coverage is of the noisy label, not the true parameter: approximate, and "
        "conservative for the Neyman sets (label noise is part of the calibration).",
        "Raw Wilks coverage: 'Wilks 68 <p>' = profile statistic of the label "
        "lambda_j <= 0.5 (the 68.27 % profile interval holds the label when the "
        "profile set is connected); 'Wilks 68 joint' = joint statistic <= "
        f"{WILKS68_JOINT} (chi2_3, 68.27 %). Rows with no score inside the support "
        "count as not covered.",
        "Posterior head: the checkpoint's Student-t head conditioned on parallax "
        f"(parallax/error >= {a.condition_parallax}).",
        "LR test: [Fe/H] < -2.5 is accepted when the 95 % profile (Wilks 1.92) or "
        "Neyman interval lies below -2.5; posterior: P([Fe/H] < -2.5) > 0.9.",
    ]
    for arm, run in metrics["runs"].items():
        metrics["notes"].append(f"Score of arm `{arm}`: {run['score']}.")
    if PLX_ARM in inf:
        metrics["notes"].append(
            "Arm `plx`: heads trained on stars with corrected parallax / error >= "
            f"{a.parallax_label_snr:g} (u label = log10 parallax); the Neyman "
            "calibration, coverage and profiles use every labelled star, with the "
            "parallax profiled out (no measurement: flat term). Parallax zero "
            f"point: Lindegren+2021 from the sidecar, else {L21_CONSTANT_ZP} mas."
        )
    if "post_balanced" in pred.files and bool(pred["post_balanced"]):
        metrics["notes"].append(
            "The posterior head was trained with balanced [Fe/H] sampling: its "
            "posterior (median, 16/84 %, P) is under the sampling prior, not the "
            "training population."
        )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=1, default=float) + "\n"
    )
    (out / "report.md").write_text(V.render_report(metrics))
    for arm, r in inf.items():
        if source[arm] != str(out):
            continue
        rows = r["table_row"]
        blob = dict(
            source_id=lab["source_id"][rows], table_row=rows, split=split[rows],
            mle=r["mle"], smax=r["smax"], at_bound=r["at_bound"],
            post_median=pred["post_median"][rows], post_q16=pred["post_q16"][rows],
            post_q84=pred["post_q84"][rows],
        )  # fmt: skip
        for name, _, _ in LEVELS:
            for k in (f"lo{name}", f"hi{name}", f"lo{name}_at_bound",
                      f"hi{name}_at_bound", f"neyman{name}_lo", f"neyman{name}_hi",
                      f"crit{name}_mle", f"in_neyman{name}", f"neyman{name}_crit",
                      f"in_neyman{name}_param"):  # fmt: skip
                if k in r:
                    blob[k] = r[k]
        for k in ("mle_parallax", "mle_log_parallax", "parallax_pull",
                  "on_support_edge", "profiled"):  # fmt: skip
            if k in r:
                blob[k] = r[k]
        name = "s1_predictions.npz" if arm == "all" else f"s1_predictions_{arm}.npz"
        np.savez(out / name, **blob)
    log(metrics["summary"])


STAGES = dict(
    latents=(stage_latents, ("latent.npy", "predictions.npz")),
    labels=(stage_labels, ("labels.npz",)),
    parallax=(stage_parallax, ("parallax.npz",)),
    train=(stage_train, ()),
    infer=(stage_infer, ()),
    report=(stage_report, ("report.md", "metrics.json")),
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fits", type=Path, help="table (default: data.ft_datafile)")
    p.add_argument(
        "--records", type=Path, help="records (default: multisurvey.records)"
    )
    p.add_argument("--core-split", type=Path, help="rows_split.npz to cross-check")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument(
        "--arms", help=f"comma list of {KNOWN_ARMS} (default: core,all; plx with "
        "--parallax-nuisance)",
    )  # fmt: skip
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--parallax-nuisance", action="store_true",
        help="variant: latents without PARALLAX, arm plx (parallax as a parameter "
        "with its measurement likelihood)",
    )  # fmt: skip
    p.add_argument(
        "--ensemble", type=int, default=1,
        help="ratio heads per arm (mean log-ratio score)",
    )  # fmt: skip
    p.add_argument(
        "--labels", type=Path, help="reuse this labels.npz (same table rows)"
    )
    p.add_argument(
        "--sidecar", type=Path,
        help="covariance sidecar (parallax_zpt_l21 aligned to the table rows)",
    )  # fmt: skip
    p.add_argument("--parallax-label-snr", type=float, default=10.0)
    p.add_argument(
        "--parallax-window", type=float, default=8.0,
        help="parallax search window: measurement +- this many errors",
    )  # fmt: skip
    p.add_argument(
        "--baseline", type=Path, help="S1 run whose arms join the report tables"
    )
    p.add_argument("--baseline-arms", default="all")
    p.add_argument(
        "--profile-rows", choices=["test+core", "core", "none"], default="test+core",
        help="rows with profile intervals (coverage uses the label statistics; "
        "the metal-poor selection needs the core rows)",
    )  # fmt: skip
    p.add_argument(
        "--no-single-member", dest="single_member", action="store_false",
        help="with --ensemble > 1, do not also score member 0 alone",
    )  # fmt: skip
    p.add_argument(
        "--single-max-rows", type=int, default=None,
        help="cap on the test rows of the member-0 label statistics",
    )  # fmt: skip
    p.add_argument(
        "--no-support", dest="support", action="store_false",
        help="do not restrict MLE/profiles to the training labels' support",
    )  # fmt: skip
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition-parallax", type=float, default=5.0)
    p.add_argument("--chunk-rows", type=int, default=262144)
    p.add_argument("--encoder-batch", type=int, default=8192)
    p.add_argument("--map-iterations", type=int, default=5)
    p.add_argument("--negatives", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--min-train", type=int, default=50)
    p.add_argument("--max-calib", type=int, default=300_000)
    p.add_argument(
        "--min-calib", type=int, default=50,
        help="fewest calibration stars for Neyman sets",
    )  # fmt: skip
    p.add_argument("--n-candidates", type=int, default=256)
    p.add_argument("--row-block", type=int, default=200_000)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="max (theta, x) pairs per score call (torchregress default 32768)",
    )
    p.add_argument(
        "--profile-valid",
        action="store_true",
        help="also profile valid rows (default: test and core rows)",
    )
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    config = yaml.safe_load(a.config.read_text())
    a.fits = a.fits or Path(config["data"]["ft_datafile"])
    a.records = a.records or Path(config["multisurvey"]["records"])
    if a.arms is None:
        a.arms = PLX_ARM if a.parallax_nuisance else ",".join(ARMS)
    a.arms = [x for x in a.arms.split(",") if x]
    a.baseline_arms = [x for x in a.baseline_arms.split(",") if x]
    unknown = [x for x in a.arms if x not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {KNOWN_ARMS}")
    if PLX_ARM in a.arms and not a.parallax_nuisance:
        raise SystemExit("arm plx needs --parallax-nuisance")
    if a.parallax_nuisance and set(a.arms) - {PLX_ARM}:
        raise SystemExit("--parallax-nuisance latents hide PARALLAX: run arm plx only")
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
