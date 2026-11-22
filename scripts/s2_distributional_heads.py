"""S2: distributional heads (product B) on frozen latents, compared under one protocol.

usage: s2_distributional_heads.py --inputs S1_DIR --output DIR
       [--records REC] [--core-rows ROWS.npy] [--baseline S1_DIR ...]
       [--heads flow,tmix,t,flow_noeiv] [--ensemble 5] [--max-train 2000000]
       [--stages data,train,evaluate,calibrate,report] [--device cuda]

Design note ``docs/design-20261122-data-driven-inference.md`` (sections 2, 6;
plan step 4, B0: intrinsic data only). Competitive multi-target distributional
predictions p(Teff, log g, [Fe/H] | star) on the FROZEN latents of an S1 run
(``S1_DIR``: ``latent.npy``, ``predictions.npz``, ``labels.npz``), each head a
K-member ensemble whose density is the equal-weight mixture of its members
(``masked_stellar_autoencoder.training.distributional_heads``):

* ``flow``: conditional rational-quadratic spline flow (torchregress NSF), EIV;
* ``tmix``: mixture of 4 multivariate Student-t (full scale), EIV;
* ``t``: one multivariate Student-t, EIV;
* ``flow_noeiv``: the flow fitted to the noisy labels directly (EIV off).

The checkpoint's posterior Student-t head (``predictions.npz``) is scored with
the same protocol as independent Student-t marginals; S1 ratio-head MLEs
(``--baseline``, ``infer_<arm>.npz``) join the point-estimate tables.

Stages (each writes into DIR; a stage whose outputs exist is skipped unless
``--force``):

1. ``data``: rows and standardization (``selection.npz``). Labels: the S1
   common-scale labels inside the S1 box (arm ``all``). Natural training
   distribution (no balanced [Fe/H] sampling): up to ``--max-train`` random
   split-0 rows. Split 1 minus the held-out frozen core is cut in two disjoint
   random halves: the stop set (early stopping, <= ``--max-stop``) and the
   calibration set (conformal, <= ``--max-calib``). Evaluation: every split-2
   row (label metrics on the labelled ones) and the held-out frozen core
   (``--core-rows`` rows with a raw [Fe/H] reference in splits 1 + 2).
   Label errors for EIV: ``e_teff``, ``e_logg``, ``e_fe_h`` (missing ->
   train-median, then floor/cap ``ERROR_FLOOR``/``ERROR_CAP``).
2. ``train``: K members per head (seeds seed .. seed+K-1), early stopping on
   the stop set's noisy-label NLL (``heads/<head>_<k>.pt``).
3. ``evaluate``: per head and the posterior baseline (``eval_<name>.npz``):
   log densities of the label (clean and label-predictive = EIV convolution),
   median and 16/84 % quantiles, gradient-refined mode, P([Fe/H] < -2.5) for
   every evaluation row; on <= ``--n-score`` test rows, ``--score-samples``
   draws per star: energy and variogram scores (standardized units), CRPS
   and PIT per parameter, the HPD level of the label (raw coverage), the
   secondary-mode mass (flow, tmix). Examples (``examples.npz``: faint stars
   with the largest secondary mass) and throughput (``speed.json``).
4. ``calibrate``: split-conformal level sets of -log p of the label
   predictive (torchregress ``ConformalizedDensity``): marginal, Mondrian by
   G bin, Mondrian by G bin x predicted-[Fe/H] bin (sparse cells fall back to
   their G bin), at the nominal levels ``CURVE_LEVELS`` (``conformal_<name>.npz``,
   thresholds in ``conformal_<name>.json``).
5. ``report``: ``metrics.json``, ``report.md``, QA plots (``qa_*.png``) and
   per-star ``s2_predictions_<head>.npz`` (test + core rows).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_likelihood_head import arm_theta, log  # noqa: E402

ARMS = {
    "flow": dict(kind="flow", eiv=True),
    "tmix": dict(kind="tmix", eiv=True),
    "t": dict(kind="t", eiv=True),
    "flow_noeiv": dict(kind="flow", eiv=False),
}
POSTERIOR = "posterior"
SUMMARY_Q = (0.16, 0.5, 0.84)
KEY_LEVELS = (0.68, 0.95)
CURVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95)
VARIANTS = ("marginal", "mondrian_g", "mondrian_gfeh")
COV_TOL = 0.03  # decision: coverage within +-0.03 of nominal in every stratum
BIAS_TOL, NMAD_TOL = 0.02, 0.01  # dex: "no worse than the posterior" per cell
MP_LIMIT = -2.5
SURFACE = "#fcfcfb"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
# Fixed series styles: four primary series, then dotted variants.
STYLE = {
    "flow": (COLORS[0], "-"), "tmix": (COLORS[1], "-"), "t": (COLORS[2], "-"),
    POSTERIOR: (COLORS[3], "-"), "flow_noeiv": (COLORS[0], ":"),
    "s1_all": (COLORS[3], ":"), "s1_plx": (COLORS[1], ":"),
}  # fmt: skip
LABELS = {
    "flow": "flow (EIV)", "tmix": "t-mixture (EIV)", "t": "Student-t (EIV)",
    POSTERIOR: "posterior head", "flow_noeiv": "flow (no EIV)", "s1_all": "S1 MLE",
    "s1_plx": "S1 plx MLE",
}  # fmt: skip


def _style(name):
    return STYLE.get(name, (COLORS[2], ":"))


def _label(name):
    return LABELS.get(name, name)


def _ints(text):
    return [int(x) for x in str(text).split(",") if x]


def _G(pred):
    G = pred["G"].astype("f8")
    return np.where(np.isfinite(G), G, np.nanmedian(G))


def _theta_err(a, lab):
    theta, ok = arm_theta(lab, "all")
    sel = np.load(a.output / "selection.npz")
    return theta, ok, sel["err"].astype("f8")


def _latents(a, rows):
    """Latents (float16) of ``rows``: the union of every role's rows is read
    once per process in contiguous chunks and kept in memory."""
    from masked_stellar_autoencoder.training.distributional_heads import gather_rows

    if getattr(a, "_lat", None) is None:
        sel = np.load(a.output / "selection.npz")
        union = np.unique(np.concatenate([sel[k] for k in ("train", "stop", "cal",
                                                           "test", "core")]))  # fmt: skip
        lat = np.load(a.inputs / "latent.npy", mmap_mode="r")
        t0 = time.time()
        a._lat = gather_rows(lat, union, chunk_rows=a.chunk_rows)
        a._lat_rows = union
        log(f"latents: {len(union)} rows x {lat.shape[1]} read in "
            f"{time.time() - t0:.0f} s ({a._lat.nbytes / 2**30:.1f} GiB)")  # fmt: skip
    pos = np.searchsorted(a._lat_rows, rows)
    if not np.array_equal(a._lat_rows[pos], rows):
        raise KeyError("rows outside the selection")
    return a._lat[pos]


def head_paths(a, arm):
    return [a.output / "heads" / f"{arm}_{k}.pt" for k in range(a.ensemble)]


# ── stage data ──────────────────────────────────────────────────────────────


def stage_data(a, out):
    from torchregress.losses.nflows import recommended_tail_bound

    from masked_stellar_autoencoder.training import distributional_heads as dh

    lab = np.load(a.inputs / "labels.npz")
    pred = np.load(a.inputs / "predictions.npz")
    lat = np.load(a.inputs / "latent.npy", mmap_mode="r")
    n = len(lab["split"])
    if lat.shape[0] != n or not np.array_equal(pred["source_id"], lab["source_id"]):
        raise SystemExit("latent.npy / predictions.npz / labels.npz rows differ")
    split = lab["split"]
    theta, ok = arm_theta(lab, "all")
    ref = lab["fe_h_ref"]
    core = np.isin(split, (1, 2)) & np.isfinite(ref)
    if a.core_rows:
        core &= np.isin(np.arange(n), np.load(a.core_rows))
    rng = np.random.default_rng(a.seed)

    def cap(rows, k):
        return (
            np.sort(rng.choice(rows, k, replace=False)) if k and len(rows) > k else rows
        )

    train = cap(np.flatnonzero(ok & (split == 0)), a.max_train)
    valid = rng.permutation(np.flatnonzero(ok & (split == 1) & ~core))
    n_stop = min(a.max_stop, len(valid) // 2)
    stop = np.sort(valid[:n_stop])
    cal = np.sort(valid[n_stop:][: a.max_calib])
    test = np.flatnonzero(split == 2)
    if a.max_test and len(test) > a.max_test:
        test = np.union1d(cap(test, a.max_test), np.flatnonzero(core & (split == 2)))
    core_rows = np.flatnonzero(core)
    if len(train) < a.min_train or len(stop) < 2 or len(cal) < 2:
        raise SystemExit(f"too few rows: train {len(train)}, stop {len(stop)}, "
                         f"calibration {len(cal)}")  # fmt: skip
    sub = cap(train, a.max_stats)
    std = dh.Standardizer.fit(dh.gather_rows(lat, sub, chunk_rows=a.chunk_rows),
                              theta[train])  # fmt: skip
    bound = recommended_tail_bound(std.y(theta[train]))
    errs = np.column_stack([lab[f"e_{p}"] for p in dh.PARAMS])
    err = dh.label_errors(errs, reference=np.isin(np.arange(n), train))
    weights = np.ones(0)
    if a.weights:
        w = np.asarray(np.load(a.weights)[a.weights_key], "f8")
        if len(w) != n:
            raise SystemExit(f"{a.weights}: {len(w)} weights for {n} rows")
        weights = w[train] / np.mean(w[train])
    np.savez(
        out / "selection.npz", train=train, stop=stop, cal=cal, test=test,
        core=core_rows, err=err.astype("f4"), tail_bound=bound, weights=weights,
        **std.state(),
    )  # fmt: skip
    counts = dict(train=len(train), stop=len(stop), calibration=len(cal),
                  test=len(test), test_labelled=int(ok[test].sum()),
                  core_heldout=len(core_rows), labelled=int(ok.sum()))  # fmt: skip
    (out / "selection.json").write_text(json.dumps(dict(
        counts=counts, tail_bound=float(bound), y_mean=std.y_mean.tolist(),
        y_std=std.y_std.tolist(), weights=str(a.weights or ""),
        error_median=np.median(err[train], 0).tolist(),
    ), indent=1) + "\n")  # fmt: skip
    log(f"data: {counts}; spline tail bound {bound:.1f} (standardized)")


# ── stage train ─────────────────────────────────────────────────────────────


def stage_train(a, out):
    from masked_stellar_autoencoder.training import distributional_heads as dh

    todo = {arm: [k for k, p in enumerate(head_paths(a, arm)) if a.force or not p.exists()]
            for arm in a.heads}  # fmt: skip
    todo = {arm: ks for arm, ks in todo.items() if ks}
    if not todo:
        log("train: all heads exist, skipped")
        return
    (out / "heads").mkdir(exist_ok=True)
    lab = np.load(a.inputs / "labels.npz")
    sel = np.load(out / "selection.npz")
    std = dh.Standardizer.from_state(sel)
    theta, _, err = _theta_err(a, lab)
    dev = torch.device(a.device)

    def tensors(rows):
        z = torch.as_tensor(_latents(a, rows), device=dev)
        y = torch.as_tensor(std.y(theta[rows]), dtype=torch.float32, device=dev)
        s = torch.as_tensor(std.sigma(err[rows]), dtype=torch.float32, device=dev)
        return z, y, s

    ztr, ytr, s_tr = tensors(sel["train"])
    zst, yst, s_st = tensors(sel["stop"])
    w = sel["weights"] if len(sel["weights"]) else None
    for arm, ks in todo.items():
        cfg = ARMS[arm]
        for k in ks:
            log(f"train {arm} member {k} ({len(ztr)} rows, EIV {cfg['eiv']})")
            member = dh.Member(
                cfg["kind"], ztr.shape[1], hidden=a.hidden, context=a.context,
                transforms=a.flow_transforms, flow_hidden=a.flow_hidden,
                bins=a.flow_bins, tail_bound=float(sel["tail_bound"]),
                components=a.components, df=a.df,
            ).to(dev)  # fmt: skip
            hist = dh.train_member(
                member, std, ztr, ytr, s_tr, zst, yst, s_st, eiv=cfg["eiv"],
                eiv_samples=a.eiv_samples, stop_mc=a.stop_mc, weights=w,
                batch=a.batch, max_epochs=a.epochs, patience=a.patience, lr=a.lr,
                seed=a.seed + k, log=log,
            )  # fmt: skip
            dh.save_member(member, head_paths(a, arm)[k], std, arm=arm, member=k,
                           eiv=cfg["eiv"], history=hist)  # fmt: skip
            (out / "heads" / f"history_{arm}_{k}.json").write_text(
                json.dumps(hist, indent=1) + "\n"
            )
            del member
            if dev.type == "cuda":
                torch.cuda.empty_cache()


# ── stage evaluate ──────────────────────────────────────────────────────────


def _predictors(a, out, rows, pred, std, dev):
    """name -> (factory(idx, device) -> prediction, kind, eiv) for rows[idx]."""
    from masked_stellar_autoencoder.training import distributional_heads as dh

    nu = pred["post_nu"].astype("f8") if "post_nu" in pred.files else 5.0
    nu = np.broadcast_to(nu, (len(pred["G"]),))
    post = {k: pred[k] for k in ("post_median", "post_q16", "post_q84")}

    def posterior(idx, device, z=None):
        r = rows[idx]
        return dh.posterior_prediction(post["post_median"][r], post["post_q16"][r],
                                       post["post_q84"][r], nu[r], std, device)  # fmt: skip

    out_ = {POSTERIOR: (posterior, "student_t_marginals", False, None)}
    for arm in a.heads:
        paths = head_paths(a, arm)
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            log(f"evaluate {arm}: missing members {missing}; skipped")
            continue
        members = [dh.load_member(p, device=dev)[0] for p in paths]
        ens = dh.EnsemblePredictor(members, std, ARMS[arm]["eiv"])

        def factory(idx, device, z=None, ens=ens):
            zz = _latents(a, rows[idx]) if z is None else z
            return ens.predict(zz, device)

        out_[arm] = (factory, ens.kind, ens.eiv, ens)
    return out_


def _safe(lp):
    return torch.nan_to_num(lp, nan=-1e30, neginf=-1e30, posinf=1e30)


def _evaluate_one(a, make, yl, sg, has, score_idx, std, dev, modes):
    """Per-row summaries and scores of one predictor (standardized math,
    physical outputs)."""
    from torchregress.metrics.distribution import (
        crps_from_samples,
        energy_score,
        sample_hpd_level,
        variogram_score,
    )

    from masked_stellar_autoencoder.training import distributional_heads as dh

    n = len(yl)
    torch.manual_seed(a.seed)
    f32 = dict(dtype=torch.float32, device=dev)
    ym = torch.as_tensor(std.y_mean, **f32)
    ys = torch.as_tensor(std.y_std, **f32)
    res = dict(lp_label=np.full(n, np.nan), lp_clean=np.full(n, np.nan),
               quant=np.full((n, 3, 3), np.nan), mode=np.full((n, 3), np.nan),
               p_mp=np.full(n, np.nan))  # fmt: skip
    mp_std = float((MP_LIMIT - std.y_mean[2]) / std.y_std[2])
    t0 = time.time()
    for i in range(0, n, a.eval_chunk):
        idx = np.arange(i, min(i + a.eval_chunk, n))
        with torch.no_grad():
            P = make(idx, dev)
            y = torch.as_tensor(np.where(has[idx, None], yl[idx], 0.0), **f32)
            s = torch.as_tensor(sg[idx], **f32)
            lab = has[idx]
            res["lp_clean"][idx[lab]] = P.log_prob(y).cpu().numpy()[lab]
            res["lp_label"][idx[lab]] = (
                P.label_log_prob(y, s, a.eval_mc).cpu().numpy()[lab]
            )
            q, smp = P.quantiles(SUMMARY_Q, n_samples=a.summary_samples)
            res["quant"][idx] = (q * ys[:, None] + ym[:, None]).cpu().numpy()
            res["p_mp"][idx] = P.prob_below(2, mp_std, smp).cpu().numpy()
            starts = P.mode_starts(smp)
        md, _ = dh.find_mode(P, starts, steps=a.mode_steps)
        res["mode"][idx] = (md * ys + ym).cpu().numpy()
        if (i // a.eval_chunk) % 50 == 0:
            log(f"  rows {idx[-1] + 1}/{n} ({time.time() - t0:.0f} s)")
    res["t_rows"] = time.time() - t0
    m = len(score_idx)
    sc = dict(es=np.full(m, np.nan), vs=np.full(m, np.nan), crps=np.full((m, 3), np.nan),
              pit=np.full((m, 3), np.nan), hpd_label=np.full(m, np.nan),
              hpd_clean=np.full(m, np.nan), second=np.full(m, np.nan),
              peak=np.full((m, 3), np.nan))  # fmt: skip
    t1 = time.time()
    for i in range(0, m, a.score_chunk):
        sl = slice(i, min(i + a.score_chunk, m))
        idx = score_idx[sl]
        with torch.no_grad():
            P = make(idx, dev)
            y = torch.as_tensor(yl[idx], **f32)
            s = torch.as_tensor(sg[idx], **f32)
            th = P.sample(a.score_samples)
            lp_th = _safe(P.log_prob_many(th))
            sc["hpd_clean"][sl] = (
                sample_hpd_level(_safe(P.log_prob(y)), lp_th).cpu().numpy()
            )
            if P.eiv:
                yt = th + torch.randn_like(th) * s[:, None]
                lab_y = _safe(P.label_log_prob(y, s, a.hpd_mc))
                lab_t = _safe(P.label_log_prob(yt, s, a.hpd_mc))
                sc["hpd_label"][sl] = sample_hpd_level(lab_y, lab_t).cpu().numpy()
            else:
                yt = th
                sc["hpd_label"][sl] = sc["hpd_clean"][sl]
            ytt = yt.transpose(0, 1)
            sc["es"][sl] = energy_score(ytt, y, reduction="none").cpu().numpy()
            sc["vs"][sl] = (
                variogram_score(ytt, y, p=0.5, reduction="none").cpu().numpy()
            )
            for j in range(3):
                sc["crps"][sl, j] = (crps_from_samples(ytt[..., j], y[:, j], reduction="none")
                                     * std.y_std[j]).cpu().numpy()  # fmt: skip
            sc["pit"][sl] = (yt <= y[:, None]).float().mean(1).cpu().numpy()
            if modes:
                sec, peak = dh.secondary_mode(
                    th, lp_th, lambda p, P=P: _safe(P.log_prob_many(p))
                )
                sc["second"][sl] = sec.cpu().numpy()
                sc["peak"][sl] = (peak * ys + ym).cpu().numpy()
    res["t_score"] = time.time() - t1
    res.update({f"score_{k}": v for k, v in sc.items()})
    return res


def _speed(a, name, entry, z, std, rows_t):
    """Stars/s of the per-star summaries (median, 16/84 %) and of the HPD
    membership test (one log density against a conformal threshold)."""
    from masked_stellar_autoencoder.training import distributional_heads as dh

    factory, kind, eiv, ens = entry
    devices = [a.device] if a.device.startswith("cuda") else []
    devices.append("cpu")
    res = {}
    for device in devices:
        n = min(len(z), a.speed_rows if device != "cpu" else a.speed_rows_cpu)
        if not n:
            continue
        if ens is not None:
            members = [copy.deepcopy(m).to(device) for m in ens.members]
            local = dh.EnsemblePredictor(members, std, eiv)

            def make(sl, local=local, device=device):
                return local.predict(z[sl], device)

        else:

            def make(sl, device=device):
                return factory(np.arange(sl.start, sl.stop), device)

        def sync(device=device):
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        chunk = a.eval_chunk
        with torch.no_grad():
            make(slice(0, min(n, 256))).quantiles(
                SUMMARY_Q, n_samples=a.summary_samples
            )
            sync()
            t0 = time.time()
            for i in range(0, n, chunk):
                P = make(slice(i, min(i + chunk, n)))
                q, _ = P.quantiles(SUMMARY_Q, n_samples=a.summary_samples)
            sync()
            t_sum = time.time() - t0
            t0 = time.time()
            for i in range(0, n, chunk):
                sl = slice(i, min(i + chunk, n))
                P = make(sl)
                inside = P.log_prob(torch.as_tensor(rows_t[sl], device=device)) >= -1.0
            sync()
            t_hpd = time.time() - t0
        del q, inside
        res[device] = dict(rows=int(n), summaries_per_s=n / t_sum, hpd_test_per_s=n / t_hpd,
                           threads=torch.get_num_threads())  # fmt: skip
        log(f"speed {name} {device}: summaries {n / t_sum:.0f}/s, HPD test "
            f"{n / t_hpd:.0f}/s")  # fmt: skip
    return res


def stage_evaluate(a, out):
    from masked_stellar_autoencoder.training import distributional_heads as dh
    from masked_stellar_autoencoder.training import likelihood_validation as V

    lab = np.load(a.inputs / "labels.npz")
    pred = np.load(a.inputs / "predictions.npz")
    sel = np.load(out / "selection.npz")
    std = dh.Standardizer.from_state(sel)
    theta, ok, err = _theta_err(a, lab)
    G = _G(pred)
    dev = torch.device(a.device)
    rows = np.unique(np.concatenate([sel["cal"], sel["test"], sel["core"]]))
    has = ok[rows]
    yl = np.where(has[:, None], std.y(np.where(ok[:, None], theta, 0.0)[rows]), np.nan)
    sg = std.sigma(err[rows])
    rng = np.random.default_rng(a.seed + 7)
    scorable = np.flatnonzero(has & np.isin(rows, sel["test"]))
    score_idx = np.sort(
        rng.choice(scorable, min(a.n_score, len(scorable)), replace=False)
    )
    preds = _predictors(a, out, rows, pred, std, dev)
    speed_path = out / "speed.json"
    speed = json.loads(speed_path.read_text()) if speed_path.exists() else {}
    for name, entry in preds.items():
        path = out / f"eval_{name}.npz"
        if path.exists() and not a.force:
            log(f"evaluate {name}: exists, skipped")
            continue
        factory, kind, eiv, _ = entry
        log(f"evaluate {name}: {len(rows)} rows, {len(score_idx)} scored")
        res = _evaluate_one(a, factory, yl, sg, has, score_idx, std, dev,
                            modes=kind in ("flow", "tmix"))  # fmt: skip
        np.savez(path, table_row=rows, score_idx=score_idx, kind=kind, eiv=eiv, **res)
        log(
            f"evaluate {name}: rows {res['t_rows']:.0f} s, scoring {res['t_score']:.0f} s"
        )
        zs = rows[score_idx[: a.speed_rows]]
        speed[name] = _speed(a, name, entry, _latents(a, zs), std,
                             np.nan_to_num(yl[score_idx[: a.speed_rows]]).astype("f4"))  # fmt: skip
        speed_path.write_text(json.dumps(speed, indent=1) + "\n")
    # Examples: faint test stars with the largest secondary-mode mass.
    ex_path = out / "examples.npz"
    src = next((n for n in ("flow", "flow_noeiv", "tmix") if (out / f"eval_{n}.npz").exists()),
               None)  # fmt: skip
    if src and (a.force or not ex_path.exists()):
        e = np.load(out / f"eval_{src}.npz")
        sec = e["score_second"]
        faint = G[rows[score_idx]] > V.G_MAX
        cand = np.flatnonzero(faint & np.isfinite(sec))
        if not len(cand):
            cand = np.flatnonzero(np.isfinite(sec))
        pick = cand[np.argsort(-sec[cand])][: a.n_examples]
        idx = score_idx[pick]
        blob = dict(table_row=rows[idx], G=G[rows[idx]], label=theta[rows[idx]],
                    second=sec[pick], source=src)  # fmt: skip
        for name, (factory, *_rest) in preds.items():
            with torch.no_grad():
                smp = factory(idx, dev).sample(a.example_samples)
            blob[f"samples_{name}"] = std.y_phys(smp.cpu().numpy()).astype("f4")
        np.savez(ex_path, **blob)


# ── stage calibrate ─────────────────────────────────────────────────────────


def stage_calibrate(a, out):
    from masked_stellar_autoencoder.training import distributional_heads as dh
    from masked_stellar_autoencoder.training import likelihood_validation as V

    lab = np.load(a.inputs / "labels.npz")
    pred = np.load(a.inputs / "predictions.npz")
    sel = np.load(out / "selection.npz")
    _, ok, _ = _theta_err(a, lab)
    G = _G(pred)
    for name in _eval_names(a, out):
        e = np.load(out / f"eval_{name}.npz")
        rows = e["table_row"]
        lp = np.nan_to_num(e["lp_label"], nan=-1e30, neginf=-1e30)
        cal = np.isin(rows, sel["cal"]) & ok[rows]
        test = np.isin(rows, sel["test"]) & ok[rows]
        gb = dh.bin_index(G[rows], V.G_BINS)
        fb = dh.bin_index(e["quant"][:, 2, 1], V.FEH_BINS)
        groups = dict(marginal=None,
                      mondrian_g=dh.mondrian_groups(gb, None, cal, a.mondrian_min),
                      mondrian_gfeh=dh.mondrian_groups(gb, fb, cal, a.mondrian_min))  # fmt: skip
        blob = dict(test_rows=rows[test], levels=np.array(CURVE_LEVELS))
        thr = {}
        for v, g in groups.items():
            cov = np.zeros((int(test.sum()), len(CURVE_LEVELS)), bool)
            thr[v] = {}
            for j, level in enumerate(CURVE_LEVELS):
                inside, q = dh.conformal_level_sets(
                    lp[cal], lp[test], level, None if g is None else g[cal],
                    None if g is None else g[test],
                )  # fmt: skip
                cov[:, j] = inside
                thr[v][f"{level:g}"] = {str(k): val for k, val in q.items()}
            blob[f"covered_{v}"] = cov
            if g is not None:
                blob[f"groups_{v}"] = g
        np.savez(out / f"conformal_{name}.npz", **blob)
        (out / f"conformal_{name}.json").write_text(json.dumps(dict(
            score="-log p of the label predictive (EIV convolution for EIV heads)",
            calibration_rows=int(cal.sum()), min_count=a.mondrian_min,
            groups="mondrian_g: 1000 + G bin; mondrian_gfeh: 100 * G bin + predicted "
                   "[Fe/H] bin (fallback 1000 + G bin, then -1 pooled)",
            thresholds=thr), indent=1) + "\n")  # fmt: skip
        log(
            f"calibrate {name}: {int(cal.sum())} calibration, {int(test.sum())} test rows"
        )


def _eval_names(a, out):
    return [n for n in (POSTERIOR, *a.heads) if (out / f"eval_{n}.npz").exists()]


# ── stage report ────────────────────────────────────────────────────────────


def _coverage_strata(covered, gbin, feh_label, feh_pred, nominal):
    """Coverage per G bin, label-[Fe/H] bin and predicted-[Fe/H] bin
    (torchregress ``binned_conditional_coverage``: Bonferroni Wilson 95 %)."""
    from torchregress.inference import binned_conditional_coverage

    from masked_stellar_autoencoder.training import likelihood_validation as V

    feats = np.column_stack([gbin, feh_label, feh_pred]).astype("f8")
    good = np.isfinite(feats).all(1)
    out = dict(all=dict(n=int(good.sum()), coverage=float(np.mean(covered[good]))
                        if good.any() else math.nan))  # fmt: skip
    if good.sum() < 2:
        return out
    edges = [torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=torch.float64),
             torch.tensor(V.FEH_BINS, dtype=torch.float64),
             torch.tensor(V.FEH_BINS, dtype=torch.float64)]  # fmt: skip
    res = binned_conditional_coverage(
        torch.as_tensor(feats[good]), torch.as_tensor(covered[good], dtype=torch.float64),
        nominal=nominal, bins=edges,
    )  # fmt: skip
    names = (list(V.G_NAMES),
             [f"[{lo:g},{hi:g})" for lo, hi in zip(V.FEH_BINS[:-1], V.FEH_BINS[1:], strict=True)])  # fmt: skip
    for fam, r, nm in zip(("G", "feh_label", "feh_pred"), res,
                          (names[0], names[1], names[1]), strict=True):  # fmt: skip
        out[fam] = [dict(bin=b, n=int(c), coverage=None if math.isnan(v) else float(v),
                         lower=None if math.isnan(lo) else float(lo),
                         upper=None if math.isnan(hi) else float(hi))
                    for b, c, v, lo, hi in zip(nm, r.count.tolist(), r.coverage.tolist(),
                                               r.lower.tolist(), r.upper.tolist(),
                                               strict=True)]  # fmt: skip
    return out


def _coverage_fails(tab, nominal, families=("G", "feh_label")):
    """Strata whose coverage is inconsistent with nominal +- COV_TOL (the
    Wilson interval misses the band) and those whose point estimate is outside
    the band (bins with >= MIN_N stars)."""
    from masked_stellar_autoencoder.training import likelihood_validation as V

    sig, point = [], []
    for fam in families:
        for r in tab.get(fam, []):
            if r["n"] < V.MIN_N or r["coverage"] is None:
                continue
            if abs(r["coverage"] - nominal) > COV_TOL:
                point.append((fam, r["bin"], r["n"], round(r["coverage"], 3)))
                if r["upper"] < nominal - COV_TOL or r["lower"] > nominal + COV_TOL:
                    sig.append((fam, r["bin"], r["n"], round(r["coverage"], 3)))
    return sig, point


def _cell_fails(arm_tab, post_tab):
    """[Fe/H]ref x G cells where the head is worse than the posterior beyond
    the tolerance (and twice the sampling error of a median / NMAD)."""
    from masked_stellar_autoencoder.training import likelihood_validation as V

    fails = []
    for r in arm_tab:
        if r["g"] == "all" or r["n"] < V.MIN_N:
            continue
        p = next(x for x in post_tab if x["feh_lo"] == r["feh_lo"] and x["g"] == r["g"])
        se = r["nmad"] / math.sqrt(r["n"])
        if abs(r["bias"]) - abs(p["bias"]) > max(BIAS_TOL, 2 * 1.2533 * se):
            fails.append(("bias", r["feh_lo"], r["g"], r["n"], round(r["bias"], 3),
                          round(p["bias"], 3)))  # fmt: skip
        if r["nmad"] - p["nmad"] > max(NMAD_TOL, 2 * 0.8 * se):
            fails.append(("nmad", r["feh_lo"], r["g"], r["n"], round(r["nmad"], 3),
                          round(p["nmad"], 3)))  # fmt: skip
    return fails


def _mean(x):
    x = np.asarray(x, "f8")
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else math.nan


def _median(x):
    x = np.asarray(x, "f8")
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else math.nan


def _load_baselines(a, lab):
    out = {}
    for i, base in enumerate(a.baseline or []):
        blab = base / "labels.npz"
        if blab.exists() and not np.array_equal(
            np.load(blab)["source_id"], lab["source_id"]
        ):
            raise SystemExit(f"baseline {base}: table rows differ")
        for arm in a.baseline_arms:
            path = base / f"infer_{arm}.npz"
            if not path.exists():
                log(f"report: no {path}")
                continue
            name = f"s1_{arm}" if f"s1_{arm}" not in out else f"s1_{arm}_{i}"
            r = np.load(path)
            full = np.full((len(lab["split"]), 3), np.nan)
            full[r["table_row"]] = r["mle"][:, :3]
            out[name] = full
    return out


def stage_report(a, out):
    from torchregress.metrics.distribution import probability_integral_transform

    from masked_stellar_autoencoder.training import distributional_heads as dh
    from masked_stellar_autoencoder.training import likelihood_validation as V

    lab = np.load(a.inputs / "labels.npz")
    pred = np.load(a.inputs / "predictions.npz")
    sel = np.load(out / "selection.npz")
    theta, ok, err = _theta_err(a, lab)
    G = _G(pred)
    split, ref = lab["split"], lab["fe_h_ref"]
    n = len(split)
    core = np.zeros(n, bool)
    core[sel["core"]] = True
    names = _eval_names(a, out)
    ev = {nm: dict(np.load(out / f"eval_{nm}.npz")) for nm in names}
    conf = {nm: dict(np.load(out / f"conformal_{nm}.npz")) for nm in names
            if (out / f"conformal_{nm}.npz").exists()}  # fmt: skip
    m = dict(summary="", selection=json.loads((out / "selection.json").read_text()),
             scores={}, coverage={}, frozen_core={}, test_labels={}, anchors={},
             metal_poor={}, bimodality={}, decision={}, training={}, speed={},
             files=[], notes=[])  # fmt: skip
    # Point estimates per table row.
    est = {POSTERIOR: pred["post_median"].astype("f8")}
    est.update(_load_baselines(a, lab))
    for nm in names:
        if nm == POSTERIOR:
            continue
        for kind, key in (("median", "quant"), ("mode", "mode")):
            full = np.full((n, 3), np.nan)
            v = ev[nm][key]
            full[ev[nm]["table_row"]] = v[:, :, 1] if key == "quant" else v
            est[f"{nm}_{kind}"] = full
    for e_name, p in est.items():
        m["frozen_core"][e_name] = V.offset_table(p[core, 2], ref[core], G[core])
    if a.records:
        rec = dict(np.load(a.records, allow_pickle=False))
        m["anchors"] = V.anchor_table(est, V.anchor_records(rec, split == 2))
    test_lab = np.zeros(n, bool)
    test_lab[sel["test"]] = True
    test_lab &= ok
    for e_name, p in est.items():
        m["test_labels"][e_name] = {
            prm: {g: V._summary(p[msk, j] - theta[msk, j])
                  for g, msk in (("G<=16", test_lab & (G <= V.G_MAX)),
                                 ("G>16", test_lab & (G > V.G_MAX)))}
            for j, prm in enumerate(V.PARAMS)
        }  # fmt: skip
    # Distributional scores, coverage, bimodality.
    for nm in names:
        e = ev[nm]
        rows = e["table_row"]
        tl = test_lab[rows]
        si = e["score_idx"]
        sr = rows[si]
        gb_rows = dh.bin_index(G[rows], V.G_BINS)
        sc = dict(n_test=int(tl.sum()), nll_label=-_mean(e["lp_label"][tl]),
                  nll_label_median=-_median(e["lp_label"][tl]),
                  nll_clean=-_mean(e["lp_clean"][tl]), n_scored=len(si),
                  energy=_mean(e["score_es"]), variogram=_mean(e["score_vs"]),
                  crps={p: _mean(e["score_crps"][:, j]) for j, p in enumerate(V.PARAMS)},
                  pit_ks={}, by_G={})  # fmt: skip
        for j, prm in enumerate(V.PARAMS):
            pit = e["score_pit"][:, j]
            pit = pit[np.isfinite(pit)]
            if len(pit):
                r = probability_integral_transform(lambda y, v=pit: v, np.zeros(len(pit)),
                                                   n_bins=20, return_histogram=True)  # fmt: skip
                sc["pit_ks"][prm] = float(r["uniformity_ks"])
        gb_s = dh.bin_index(G[sr], V.G_BINS)
        for b, gname in enumerate(V.G_NAMES):
            msk = tl & (gb_rows == b)
            ms = gb_s == b
            sc["by_G"][gname] = dict(n=int(msk.sum()), nll_label=-_mean(e["lp_label"][msk]),
                                     energy=_mean(e["score_es"][ms]))  # fmt: skip
        m["scores"][nm] = sc
        fpred_s = e["quant"][si, 2, 1]
        cov = {}
        for level in KEY_LEVELS:
            key = f"{level:g}"
            cov[f"raw_{key}"] = _coverage_strata(
                e["score_hpd_label"] <= level, gb_s, theta[sr, 2], fpred_s, level
            )
            if bool(e["eiv"]):
                cov[f"raw_clean_{key}"] = _coverage_strata(
                    e["score_hpd_clean"] <= level, gb_s, theta[sr, 2], fpred_s, level
                )
        if nm in conf:
            c = conf[nm]
            tpos = np.searchsorted(rows, c["test_rows"])
            gb_t = gb_rows[tpos]
            fp_t = e["quant"][tpos, 2, 1]
            fl_t = theta[c["test_rows"], 2]
            for v in VARIANTS:
                for level in KEY_LEVELS:
                    j = list(CURVE_LEVELS).index(level)
                    cov[f"{v}_{level:g}"] = _coverage_strata(
                        c[f"covered_{v}"][:, j], gb_t, fl_t, fp_t, level
                    )
        m["coverage"][nm] = cov
        if np.isfinite(e["score_second"]).any():
            sec = e["score_second"]
            m["bimodality"][nm] = {
                gname: dict(n=int((gb_s == b).sum()),
                            fraction=_mean((sec[gb_s == b] >= 0.1).astype(float)))
                for b, gname in enumerate(V.G_NAMES)
            } | {"all": dict(n=len(sec), fraction=_mean((sec >= 0.1).astype(float)))}  # fmt: skip
        pmp = np.full(n, np.nan)
        pmp[rows] = e["p_mp"]
        m["metal_poor"][f"{nm}: P([Fe/H]<-2.5) > 0.9"] = pmp[core] > 0.9
    m["metal_poor"] = V.selection_table(ref[core], G[core], m["metal_poor"])
    sp = out / "speed.json"
    m["speed"] = json.loads(sp.read_text()) if sp.exists() else {}
    for arm in a.heads:
        hs = sorted((out / "heads").glob(f"history_{arm}_*.json"))
        m["training"][arm] = [
            {k: h[k] for k in ("best_valid", "epochs", "n_train", "n_valid", "seed",
                               "eiv", "eiv_samples", "train_seconds") if k in h}
            for h in (json.loads(p.read_text()) for p in hs)
        ]  # fmt: skip
    _decide(m, names)
    _notes(a, m, pred)
    m["files"] = _plots(a, out, m, ev, conf, theta, G, names)
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float) + "\n")
    (out / "report.md").write_text(render_report(m))
    _write_predictions(a, out, lab, ev, conf, names)
    log(m["summary"])


def _decide(m, names):
    """Decision table per head (see ``render_report``)."""
    heads = [nm for nm in names if nm != POSTERIOR]
    if not heads or POSTERIOR not in m["scores"]:
        return
    ps = m["scores"][POSTERIOR]
    best_ll = min(heads, key=lambda h: m["scores"][h]["nll_label"])
    best_es = min(heads, key=lambda h: m["scores"][h]["energy"])
    for h in heads:
        s = m["scores"][h]
        cov = m["coverage"].get(h, {})
        crit = dict(
            log_score=dict(passed=bool(s["nll_label"] < ps["nll_label"]),
                           value=s["nll_label"], posterior=ps["nll_label"],
                           best=h == best_ll),
            energy=dict(passed=bool(s["energy"] < ps["energy"]), value=s["energy"],
                        posterior=ps["energy"], best=h == best_es),
        )  # fmt: skip
        for kind in ("mondrian_gfeh", "raw"):
            sig, point = [], []
            for level in KEY_LEVELS:
                tab = cov.get(f"{kind}_{level:g}")
                if tab:
                    s_, p_ = _coverage_fails(tab, level)
                    sig += [(level, *x) for x in s_]
                    point += [(level, *x) for x in p_]
            crit[f"coverage_{kind}"] = dict(
                passed=bool(cov.get(f"{kind}_0.68")) and not sig,
                failing=sig, outside_band_point=point,
            )  # fmt: skip
        fails = _cell_fails(
            m["frozen_core"][f"{h}_median"], m["frozen_core"][POSTERIOR]
        )
        crit["bias_nmad"] = dict(passed=not fails, failing=fails)
        need = ("log_score", "energy", "coverage_mondrian_gfeh", "bias_nmad")
        m["decision"][h] = dict(
            criteria=crit, competitive=all(crit[k]["passed"] for k in need)
        )
    good = [h for h, d in m["decision"].items() if d["competitive"]]
    m["summary"] = (
        f"Heads {heads} on frozen latents vs the posterior head. Best log score: "
        f"{best_ll}; best energy score: {best_es}. "
        + (f"Competitive (all four criteria): {good}." if good else
           "No head passes all four criteria.")
    )  # fmt: skip


def _notes(a, m, pred):
    sel = m["selection"]
    m["notes"] = [
        "Training distribution: natural (split-0 rows sampled uniformly, no "
        "balanced [Fe/H] sampling)"
        + (
            f"; importance weights from {a.weights}"
            if a.weights
            else "; no importance weights"
        )
        + f". Rows: {sel['counts']}.",
        f"Ensembles: K = {a.ensemble} members per head (seeds {a.seed}.."
        f"{a.seed + a.ensemble - 1}), density = equal-weight mixture; trunk MLP "
        f"{a.hidden} on the standardized latent; flow: NSF, {a.flow_transforms} "
        f"transforms, hidden {a.flow_hidden}, {a.flow_bins} bins, tail bound "
        f"{sel['tail_bound']:.1f}; tmix: {a.components} components; t: 1 component; "
        f"full scale matrices, one learned df per member (initial {a.df:g}).",
        "Errors in variables: flows by torchregress NoiseAwareDensityLoss (Monte Carlo "
        f"over label noise, {a.eiv_samples} draws per row in training, {a.stop_mc} for "
        f"early stopping, {a.eval_mc} for the label scores), mixtures in closed form "
        "(StudentTMixture.component_observation_log_prob, exact for Gaussians, "
        "Gaussian noise inside the t for finite df). Label errors: e_teff, e_logg, "
        f"e_fe_h (train medians {[round(x, 3) for x in sel['error_median']]}), floor "
        "(10 K, 0.01, 0.01), cap (500 K, 0.5, 0.5).",
        "Log score: mean -log p of the label under the label predictive (the EIV "
        "convolution for EIV heads, the head density otherwise): the label is a noisy "
        "measurement, so this scores the prediction of the label. 'clean NLL' is the "
        "head density at the label (EIV heads: of theta, so it is not a score of the "
        "label).",
        "Energy and variogram scores: label-predictive draws (EIV heads: theta draws "
        f"plus label noise), {a.score_samples} per star, standardized units "
        f"(y / {[round(x, 3) for x in sel['y_std']]}); CRPS per parameter in physical "
        "units.",
        "Raw coverage: the sample HPD level of the label (torchregress "
        "sample_hpd_level) under the label predictive; 'raw clean' (EIV heads): the "
        "level under the theta density, which should under-cover the noisy label.",
        "Conformal: level sets of -log p of the label predictive calibrated on the "
        "calibration half of split 1 (disjoint from early stopping and from the "
        "held-out core); Mondrian by G bin and by G bin x predicted-[Fe/H] bin "
        f"(cells with < {a.mondrian_min} calibration stars fall back to the G bin). "
        "Coverage is of the noisy label; for an EIV head the conformal set is a "
        "set for the label, its theta-space HPD set is the raw model set.",
        "Coverage strata: G bins (rows beyond 17.65 in the last bin), label-[Fe/H] "
        "and predicted-[Fe/H] bins; Wilson 95 % intervals with a Bonferroni split "
        "over the bins (torchregress binned_conditional_coverage). A stratum fails "
        f"when its interval misses nominal +- {COV_TOL} (bins with >= 20 stars).",
        "Posterior head: independent Student-t marginals from post_median/q16/q84 "
        "and post_nu (correlations are not stored), parallax-conditioned; "
        + (
            "trained with balanced [Fe/H] sampling, so it is a posterior under the "
            "sampling prior."
            if "post_balanced" in pred.files and bool(pred["post_balanced"])
            else "natural sampling."
        ),
        "Point estimates: per-parameter median (closed form for mixtures, "
        f"{a.summary_samples} draws for flows) and the mode (densest draw or "
        f"component location refined by {a.mode_steps} Adam steps on log p). "
        f"Bias/NMAD criterion per [Fe/H]ref x G cell (n >= 20): |bias| <= "
        f"|bias_post| + max({BIAS_TOL}, 2 SE), NMAD <= NMAD_post + max({NMAD_TOL}, 2 SE).",
        "Secondary mode: k-nearest-draw mode seeking on the theta draws, a basin "
        "with >= 10 % of the draws whose path to the primary peak dips below 0.8 of "
        "the lower peak (distributional_heads.secondary_mode).",
    ]


# ── report rendering ────────────────────────────────────────────────────────


def _fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render_report(m):
    from masked_stellar_autoencoder.training import likelihood_validation as V

    L = [
        "# S2: distributional heads on frozen latents (product B)\n",
        m["summary"] + "\n",
    ]
    if m["decision"]:
        L.append("## Decision\n")
        L.append("A head is competitive when it beats the posterior head on the log and "
                 "energy scores, its Mondrian (G x predicted [Fe/H]) conformal sets cover "
                 f"within +-{COV_TOL} in every G and label-[Fe/H] stratum at 68 and 95 %, "
                 "and no frozen-core [Fe/H]ref x G cell is worse than the posterior "
                 "(median estimator).\n")  # fmt: skip
        L.append("| head | log score (vs post.) | energy (vs post.) | conformal coverage "
                 "| raw coverage | bias/NMAD per cell | competitive |")  # fmt: skip
        L.append("|---|---|---|---|---|---|---|")
        for h, d in m["decision"].items():
            c = d["criteria"]

            def ok(k, c=c):
                return "pass" if c[k]["passed"] else "FAIL"

            L.append(
                f"| {h} | {ok('log_score')} {_fmt(c['log_score']['value'])} "
                f"({_fmt(c['log_score']['posterior'])}){' best' if c['log_score']['best'] else ''}"
                f" | {ok('energy')} {_fmt(c['energy']['value'])} ({_fmt(c['energy']['posterior'])})"
                f"{' best' if c['energy']['best'] else ''} | {ok('coverage_mondrian_gfeh')} "
                f"({len(c['coverage_mondrian_gfeh']['failing'])} failing) | "
                f"{ok('coverage_raw')} ({len(c['coverage_raw']['failing'])} failing) | "
                f"{ok('bias_nmad')} ({len(c['bias_nmad']['failing'])} failing) | "
                f"{'YES' if d['competitive'] else 'no'} |"
            )
        L.append("")
        for h, d in m["decision"].items():
            for k in ("coverage_mondrian_gfeh", "coverage_raw", "bias_nmad"):
                f = d["criteria"][k]["failing"]
                if f:
                    L.append(
                        f"- `{h}` {k} failing: {f[:12]}{' ...' if len(f) > 12 else ''}"
                    )
        L.append("")
    L.append("## Scores (test split)\n")
    L.append("| predictor | n test | log score | median -log p | clean NLL | n scored | "
             "energy | variogram | CRPS Teff [K] | CRPS log g | CRPS [Fe/H] | PIT KS "
             "Teff/logg/[Fe/H] |")  # fmt: skip
    L.append("|---" * 12 + "|")
    for nm, s in m["scores"].items():
        ks = "/".join(_fmt(s["pit_ks"].get(p), 3) for p in V.PARAMS)
        L.append(f"| {nm} | {s['n_test']} | {_fmt(s['nll_label'])} | "
                 f"{_fmt(s['nll_label_median'])} | {_fmt(s['nll_clean'])} | {s['n_scored']} "
                 f"| {_fmt(s['energy'])} | {_fmt(s['variogram'])} | "
                 f"{_fmt(s['crps']['teff'], 1)} | {_fmt(s['crps']['logg'])} | "
                 f"{_fmt(s['crps']['fe_h'])} | {ks} |")  # fmt: skip
    L.append("\nPer G bin: log score / energy score (n)\n")
    L.append("| predictor | " + " | ".join(V.G_NAMES) + " |")
    L.append("|---" * (len(V.G_NAMES) + 1) + "|")
    for nm, s in m["scores"].items():
        L.append(f"| {nm} | " + " | ".join(
            f"{_fmt(s['by_G'][g]['nll_label'])} / {_fmt(s['by_G'][g]['energy'])} "
            f"({s['by_G'][g]['n']})" for g in V.G_NAMES) + " |")  # fmt: skip
    L.append("")
    L.append("## Coverage of the test labels\n")
    L.append("Columns: raw HPD (scored subsample), raw clean HPD (EIV heads, theta "
             "density), conformal marginal, Mondrian G, Mondrian G x predicted [Fe/H] "
             "(all test labels). Cells: coverage at 68 / 95 %.\n")  # fmt: skip
    kinds = [("raw", "raw"), ("raw_clean", "raw clean"), ("marginal", "conf. marginal"),
             ("mondrian_g", "Mondrian G"), ("mondrian_gfeh", "Mondrian G x [Fe/H]")]  # fmt: skip
    for nm, cov in m["coverage"].items():
        have = [(k, t) for k, t in kinds if f"{k}_0.68" in cov]
        if not have:
            continue
        L.append(f"**{nm}**\n")
        L.append("| stratum | n | " + " | ".join(t for _, t in have) + " |")
        L.append("|---" * (len(have) + 2) + "|")

        def cell(k, fam, i, cov=cov):
            vals = []
            for level in KEY_LEVELS:
                tab = cov[f"{k}_{level:g}"]
                r = tab["all"] if fam == "all" else tab.get(fam, [{}] * (i + 1))[i]
                vals.append(_fmt(r.get("coverage"), 3))
            return " / ".join(vals)

        ref = cov[f"{have[-1][0]}_0.68"]
        L.append("| all | " + str(ref["all"]["n"]) + " | "
                 + " | ".join(cell(k, "all", 0) for k, _ in have) + " |")  # fmt: skip
        for fam, title in (("G", "G"), ("feh_label", "label [Fe/H]"),
                           ("feh_pred", "pred. [Fe/H]")):  # fmt: skip
            for i, r in enumerate(ref.get(fam, [])):
                if not r["n"]:
                    continue
                L.append(f"| {title} {r['bin']} | {r['n']} | "
                         + " | ".join(cell(k, fam, i) for k, _ in have) + " |")  # fmt: skip
        L.append("")
    L.append(
        "## Frozen held-out core: median offset / NMAD (n) per reference [Fe/H] bin\n"
    )
    cols = list(V.G_NAMES) + ["G<=16", "all"]
    for e_name, tab in m["frozen_core"].items():
        L.append(f"**{e_name}**\n")
        L.append("| [Fe/H]ref | " + " | ".join(cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for lo, hi in zip(V.FEH_BINS[:-1], V.FEH_BINS[1:], strict=True):
            cells = []
            for g in cols:
                r = next(r for r in tab if r["feh_lo"] == lo and r["g"] == g)
                cells.append(f"{_fmt(r['bias'], 2, True)}/{_fmt(r['nmad'], 2)} ({r['n']})"
                             if r["n"] else "-")  # fmt: skip
            L.append(f"| [{lo:g}, {hi:g}) | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## Test split vs labels: median offset / NMAD (n)\n")
    L.append("| estimator | " + " | ".join(f"{p} {g}" for p in V.PARAMS
                                         for g in ("G<=16", "G>16")) + " |")  # fmt: skip
    L.append("|---" * 7 + "|")
    for e_name, tab in m["test_labels"].items():
        L.append(f"| {e_name} | " + " | ".join(
            f"{_fmt(tab[p][g]['bias'], 0 if p == 'teff' else 3, True)}/"
            f"{_fmt(tab[p][g]['nmad'], 0 if p == 'teff' else 3)} ({tab[p][g]['n']})"
            for p in V.PARAMS for g in ("G<=16", "G>16")) + " |")  # fmt: skip
    L.append("")
    if m["anchors"]:
        L.append("## Anchors (test stars): median offset / NMAD (n)\n")
        ests = list(next(iter(m["anchors"].values())))
        L.append("| anchor | " + " | ".join(ests) + " |")
        L.append("|---" * (len(ests) + 1) + "|")
        for name, v in m["anchors"].items():
            nd = 0 if name.startswith("teff") else 3
            L.append(f"| {name} | " + " | ".join(
                f"{_fmt(v[e]['bias'], nd, True)}/{_fmt(v[e]['nmad'], nd)} ({v[e]['n']})"
                for e in ests) + " |")  # fmt: skip
        L.append("")
    if m["metal_poor"]:
        L.append("## Metal-poor selection ([Fe/H]ref < -2.5, held-out core)\n")
        for g, tab in m["metal_poor"].items():
            L.append(f"**{g}**: " + "; ".join(
                f"{k}: {r['n_selected']} selected, purity {_fmt(r['purity'], 2)}, "
                f"completeness {_fmt(r['completeness'], 2)}" for k, r in tab.items()))  # fmt: skip
        L.append("")
    if m["bimodality"]:
        L.append("## Non-Gaussianity: fraction of stars with a separated secondary mode "
                 "holding >= 10 % of the mass (scored test stars)\n")  # fmt: skip
        L.append("| head | all | " + " | ".join(V.G_NAMES) + " |")
        L.append("|---" * (len(V.G_NAMES) + 2) + "|")
        for h, d in m["bimodality"].items():
            L.append(f"| {h} | {_fmt(d['all']['fraction'])} ({d['all']['n']}) | "
                     + " | ".join(f"{_fmt(d[g]['fraction'])} ({d[g]['n']})"
                                  for g in V.G_NAMES) + " |")  # fmt: skip
        L.append("")
    if m["speed"]:
        L.append("## Speed (stars/s)\n")
        L.append("| predictor | device | rows | summaries (median, 16/84 %) | HPD "
                 "membership test |\n|---|---|---|---|---|")  # fmt: skip
        for nm, d in m["speed"].items():
            for dev, r in d.items():
                L.append(f"| {nm} | {dev} | {r['rows']} | {r['summaries_per_s']:.0f} | "
                         f"{r['hpd_test_per_s']:.0f} |")  # fmt: skip
        L.append("")
    if m["training"]:
        L.append("## Training (per member: best stop NLL, epochs, minutes)\n")
        for h, rows in m["training"].items():
            L.append(f"- `{h}`: " + "; ".join(
                f"{_fmt(r.get('best_valid'))}, {r.get('epochs')} ep, "
                f"{r.get('train_seconds', 0) / 60:.1f} min" for r in rows))  # fmt: skip
        L.append("")
    if m["files"]:
        L.append("## QA plots\n")
        L.extend(f"- `{f}`" for f in m["files"])
        L.append("")
    L.append("## Notes\n")
    L.extend(f"- {x}" for x in m["notes"])
    return "\n".join(L) + "\n"


# ── plots ───────────────────────────────────────────────────────────────────


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e6e2", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b9b3")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.xaxis.label.set_color("#2b2b28")
    ax.yaxis.label.set_color("#2b2b28")
    ax.title.set_color("#2b2b28")


def _figure(nrows, ncols, size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        _axes_style(ax)
    return fig, axes


def _save(fig, path):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _plots(a, out, m, ev, conf, theta, G, names):
    from masked_stellar_autoencoder.training import distributional_heads as dh
    from masked_stellar_autoencoder.training import likelihood_validation as V

    files = []
    order = [nm for nm in ("flow", "tmix", "t", POSTERIOR, "flow_noeiv") if nm in names]
    order += [nm for nm in names if nm not in order]
    levels = np.array(CURVE_LEVELS)
    # (1) coverage - nominal vs nominal, per stratum, raw and conformal.
    strata = (
        [("all", None)]
        + [(g, b) for b, g in enumerate(V.G_NAMES)]
        + [("[Fe/H] < -2", -1)]
    )
    fig, axes = _figure(2, len(strata), (2.6 * len(strata), 5.4))
    for row, kind in enumerate(("raw", "mondrian_gfeh")):
        for col, (title, b) in enumerate(strata):
            ax = axes[row, col]
            ax.axhspan(-COV_TOL, COV_TOL, color="#ecece8", zorder=0)
            ax.axhline(0, color="#8a8a84", linewidth=0.8)
            for nm in order:
                e = ev[nm]
                rows = e["table_row"]
                if kind == "raw":
                    r = rows[e["score_idx"]]
                    lv = e["score_hpd_label"]
                    covered = lv[:, None] <= levels[None]
                else:
                    if nm not in conf:
                        continue
                    r = conf[nm]["test_rows"]
                    covered = conf[nm]["covered_mondrian_gfeh"]
                gb = dh.bin_index(G[r], V.G_BINS)
                msk = (np.ones(len(r), bool) if b is None else
                       (theta[r, 2] < -2) if b == -1 else gb == b)  # fmt: skip
                if msk.sum() < V.MIN_N:
                    continue
                c, ls = _style(nm)
                ax.plot(levels, covered[msk].mean(0) - levels, color=c, linestyle=ls,
                        linewidth=1.6, marker="o", markersize=3, label=_label(nm))  # fmt: skip
            ax.set_ylim(-0.25, 0.25)
            if row == 1:
                ax.set_xlabel("nominal HPD level")
            if col == 0:
                ax.set_ylabel(("raw" if kind == "raw" else "conformal (Mondrian)")
                              + "\ncoverage - nominal")  # fmt: skip
            if row == 0:
                ax.set_title(title, fontsize=9)
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="upper center", ncol=len(order), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.04))  # fmt: skip
    _save(fig, out / "qa_coverage.png")
    files.append("qa_coverage.png")
    # (2) bias and NMAD vs [Fe/H]ref bin (G <= 16), heads vs baselines.
    fig, axes = _figure(1, 2, (10, 3.8))
    bins = list(zip(V.FEH_BINS[:-1], V.FEH_BINS[1:], strict=True))
    x = np.arange(len(bins))
    series = [POSTERIOR] + [k for k in m["frozen_core"] if k.startswith("s1_")]
    series += [f"{nm}_median" for nm in order if nm != POSTERIOR]
    for e_name in series:
        tab = m["frozen_core"][e_name]
        base = e_name[: -len("_median")] if e_name.endswith("_median") else e_name
        c, ls = _style(base)
        cells = [
            next(r for r in tab if r["feh_lo"] == lo and r["g"] == "G<=16")
            for lo, _ in bins
        ]
        ok_ = np.array([r["n"] >= V.MIN_N for r in cells])
        for ax, key in zip(axes[0], ("bias", "nmad"), strict=True):
            y = np.array(
                [r[key] if o else np.nan for r, o in zip(cells, ok_, strict=True)]
            )
            ax.plot(x, y, color=c, linestyle=ls, marker="o", markersize=4, linewidth=1.6,
                    label=_label(base))  # fmt: skip
    for ax, ylabel in zip(axes[0], ("median offset vs [Fe/H]ref [dex]", "NMAD [dex]"),
                          strict=True):  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"[{lo:g},{hi:g})" for lo, hi in bins], rotation=30, fontsize=7
        )
        ax.set_xlabel("[Fe/H]ref bin [dex] (held-out core, G <= 16)")
        ax.set_ylabel(ylabel)
    axes[0, 0].axhline(0, color="#8a8a84", linewidth=0.8)
    axes[0, 1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1)
    )
    _save(fig, out / "qa_bias_nmad.png")
    files.append("qa_bias_nmad.png")
    # (3) PIT histograms per parameter (label predictive, scored test stars).
    fig, axes = _figure(1, 3, (11, 3.4))
    edges = np.linspace(0, 1, 21)
    titles = ("Teff", "log g", "[Fe/H]")
    for j, ax in enumerate(axes[0]):
        ax.axhline(1.0, color="#8a8a84", linewidth=0.8)
        for nm in order:
            pit = ev[nm]["score_pit"][:, j]
            pit = pit[np.isfinite(pit)]
            if not len(pit):
                continue
            hst, _ = np.histogram(pit, edges, density=True)
            c, ls = _style(nm)
            ax.step(edges, np.r_[hst, hst[-1]], where="post", color=c, linestyle=ls,
                    linewidth=1.6, label=_label(nm))  # fmt: skip
        ax.set_xlabel(f"PIT of the {titles[j]} label")
        ax.set_ylabel("density")
    axes[0, 0].legend(frameon=False, fontsize=8)
    _save(fig, out / "qa_pit.png")
    files.append("qa_pit.png")
    # (4) example faint stars: log g vs Teff draws.
    exp = out / "examples.npz"
    if exp.exists():
        ex = np.load(exp)
        k = len(ex["table_row"])
        if k:
            from matplotlib.lines import Line2D

            fig, axes = _figure(1, k, (3.2 * k, 3.3))
            # Drawn bottom to top: posterior, t-mixture, flow.
            show = [
                nm for nm in (POSTERIOR, "tmix", "flow") if f"samples_{nm}" in ex.files
            ]
            for i, ax in enumerate(axes[0]):
                pts = np.concatenate([ex[f"samples_{nm}"][i][:, :2] for nm in show])
                lo, hi = np.nanpercentile(pts, 1, 0), np.nanpercentile(pts, 99, 0)
                pad = 0.15 * (hi - lo)
                for nm in show:
                    s = ex[f"samples_{nm}"][i]
                    ax.scatter(s[:, 0], s[:, 1], s=3, color=_style(nm)[0], alpha=0.3,
                               linewidths=0, rasterized=True)  # fmt: skip
                lb = ex["label"][i]
                ax.scatter(
                    [lb[0]], [lb[1]], marker="x", s=45, color="#2b2b28", zorder=5
                )
                ax.set_xlim(
                    hi[0] + pad[0], lo[0] - pad[0]
                )  # Teff increases to the left
                ax.set_ylim(hi[1] + pad[1], lo[1] - pad[1])  # log g increases downwards
                ax.set_xlabel("Teff [K]")
                if i == 0:
                    ax.set_ylabel("log g [dex]")
                ax.set_title(f"G = {ex['G'][i]:.2f}, secondary mass {ex['second'][i]:.2f}",
                             fontsize=8)  # fmt: skip
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=_style(nm)[0],
                    markersize=5,
                    label=_label(nm),
                )
                for nm in show[::-1]
            ]
            handles.append(Line2D([], [], marker="x", linestyle="", color="#2b2b28",
                                  markersize=6, label="label"))  # fmt: skip
            fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                       fontsize=8, bbox_to_anchor=(0.5, 1.08))  # fmt: skip
            _save(fig, out / "qa_examples.png")
            files.append("qa_examples.png")
    return files


def _write_predictions(a, out, lab, ev, conf, names):
    """Per-star products of each head: test + core rows."""
    sel = np.load(out / "selection.npz")
    keep_rows = np.union1d(sel["test"], sel["core"])
    for nm in names:
        if nm == POSTERIOR:
            continue
        e = ev[nm]
        rows = e["table_row"]
        k = np.isin(rows, keep_rows)
        sec = np.full(len(rows), np.nan, "f4")
        sec[e["score_idx"]] = e["score_second"]
        blob = dict(source_id=lab["source_id"][rows[k]], table_row=rows[k],
                    split=lab["split"][rows[k]], median=e["quant"][k, :, 1].astype("f4"),
                    q16=e["quant"][k, :, 0].astype("f4"), q84=e["quant"][k, :, 2].astype("f4"),
                    mode=e["mode"][k].astype("f4"), lp_label=e["lp_label"][k].astype("f4"),
                    p_mp=e["p_mp"][k].astype("f4"), second_mass=sec[k])  # fmt: skip
        if nm in conf and "groups_mondrian_gfeh" in conf[nm]:
            thr = json.loads((out / f"conformal_{nm}.json").read_text())["thresholds"]
            g = conf[nm]["groups_mondrian_gfeh"][k]
            blob["conformal_group"] = g
            for level in KEY_LEVELS:
                t = thr["mondrian_gfeh"][f"{level:g}"]
                blob[f"conformal{round(level * 100)}_nlp"] = np.array(
                    [t.get(str(int(x)), np.nan) for x in g], "f4"
                )
        np.savez(out / f"s2_predictions_{nm}.npz", **blob)


STAGES = dict(
    data=(stage_data, ("selection.npz",)),
    train=(stage_train, ()),
    evaluate=(stage_evaluate, ()),
    calibrate=(stage_calibrate, ()),
    report=(stage_report, ("report.md", "metrics.json")),
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--inputs", type=Path, required=True,
                   help="S1 run: latent.npy, predictions.npz, labels.npz")  # fmt: skip
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--records", type=Path, help="records.npz (anchor tables)")
    p.add_argument("--core-rows", type=Path, help="frozen core rows (rows.npy)")
    p.add_argument("--baseline", type=Path, action="append",
                   help="S1 run whose infer_<arm>.npz MLEs join the tables (repeatable)")  # fmt: skip
    p.add_argument("--baseline-arms", default="all")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument(
        "--heads", default=",".join(ARMS), help=f"comma list of {list(ARMS)}"
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--max-train", type=int, default=2_000_000)
    p.add_argument("--max-stop", type=int, default=100_000)
    p.add_argument("--max-calib", type=int, default=300_000)
    p.add_argument("--max-test", type=int, default=0, help="0: every split-2 row")
    p.add_argument("--max-stats", type=int, default=200_000)
    p.add_argument("--min-train", type=int, default=50)
    p.add_argument(
        "--weights", type=Path, help="npz with per-table-row importance weights"
    )
    p.add_argument("--weights-key", default="weight")
    p.add_argument("--hidden", type=_ints, default=[256, 256])
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--flow-transforms", type=int, default=3)
    p.add_argument("--flow-hidden", type=_ints, default=[128, 128])
    p.add_argument("--flow-bins", type=int, default=16)
    p.add_argument("--components", type=int, default=4)
    p.add_argument("--df", type=float, default=5.0)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eiv-samples", type=int, default=8)
    p.add_argument("--stop-mc", type=int, default=16)
    p.add_argument("--eval-mc", type=int, default=32)
    p.add_argument("--hpd-mc", type=int, default=8)
    p.add_argument("--summary-samples", type=int, default=128)
    p.add_argument("--score-samples", type=int, default=256)
    p.add_argument("--n-score", type=int, default=100_000)
    p.add_argument("--mode-steps", type=int, default=25)
    p.add_argument("--eval-chunk", type=int, default=4096)
    p.add_argument("--score-chunk", type=int, default=256)
    p.add_argument("--mondrian-min", type=int, default=300)
    p.add_argument("--n-examples", type=int, default=4)
    p.add_argument("--example-samples", type=int, default=2000)
    p.add_argument("--speed-rows", type=int, default=50_000)
    p.add_argument("--speed-rows-cpu", type=int, default=5_000)
    p.add_argument("--chunk-rows", type=int, default=262_144)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.heads = [h for h in a.heads.split(",") if h]
    unknown = [h for h in a.heads if h not in ARMS]
    if unknown:
        raise SystemExit(f"unknown head(s) {unknown}; choose from {list(ARMS)}")
    a.baseline_arms = [x for x in a.baseline_arms.split(",") if x]
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
        fn(a, out)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    log(f"done in {time.time() - t0:.0f} s; peak RSS {rss:.1f} GB")
    if torch.cuda.is_available() and a.device.startswith("cuda"):
        log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
