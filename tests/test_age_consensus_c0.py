"""Synthetic recovery tests for the C0 age consensus (``training/age_consensus``)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from masked_stellar_autoencoder.training.age_consensus import (  # noqa: E402
    IDENTITY,
    LINEAR,
    OFFSET,
    T0,
    AgeMaps,
    Groups,
    MapSet,
    MonotoneMaps,
    Records,
    fit_consensus,
    map_values,
    solve_stars,
)

# true maps: (offset at T0, slope - 1, [Fe/H] slope)
TRUE = {"anchor": (0.0, 0.0, 0.0), "B": (0.10, -0.15, 0.0), "C": (-0.05, 0.10, 0.10),
        "D": (0.20, 0.0, 0.0)}  # fmt: skip
NAMES = ["anchor", "B", "C", "D"]
CAP_C = 0.8


def _synthetic(seed=0, n=4000, censor=True, groups=False, eta=0.0):
    """Four age catalogues of known maps/noise on overlapping stars. ``eta``:
    a shared error common to B and C (same inputs)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-0.3, 1.1, n)
    feh = rng.normal(-0.2, 0.3, n)
    n_cl = 150
    if groups:  # clusters of 8 stars sharing an age (the rest are field stars)
        t[: 8 * n_cl] = np.repeat(t[: 8 * n_cl : 8], 8)
    shared = rng.normal(0, eta, n) if eta else np.zeros(n)
    frac = {"anchor": 0.35, "B": 0.6, "C": 0.5, "D": 0.4}
    quoted = {"anchor": 0.05, "B": 0.04, "C": 0.05, "D": 0.0}
    kappa = {"anchor": 1.0, "B": 2.0, "C": 1.5, "D": 1.0}
    omega = {"anchor": 0.01, "B": 0.01, "C": 0.01, "D": 0.10}
    rows = []
    for k, name in enumerate(NAMES):
        idx = np.flatnonzero(rng.random(n) < frac[name])
        b, A, beta = TRUE[name]
        mu = t[idx] + b + A * (t[idx] - T0) + beta * feh[idx]
        if name in ("B", "C"):
            mu = mu + shared[idx]
        q = quoted[name] * rng.uniform(0.5, 2.0, len(idx))  # quoted errors vary
        sd = np.sqrt((kappa[name] * q) ** 2 + omega[name] ** 2)
        y = mu + rng.normal(0, 1, len(idx)) * sd
        cens = np.zeros(len(idx), "i1")
        if name == "C" and censor:
            cens[y >= CAP_C] = 1
            y = np.minimum(y, CAP_C)
        eff = np.full(len(idx), 0 if name in ("B", "C") else -1)
        rows.append((idx, np.full(len(idx), k), y, q, feh[idx], cens, eff))
    star, mid, y, sig, fh, cens, eff = (
        np.concatenate([r[j] for r in rows]) for j in range(7)
    )
    rec = Records(star.astype("i8"), mid.astype("i8"), np.zeros(len(y), "i8"),
                  mid.astype("i8"), y, sig, fh, cens, n, effect=eff.astype("i8"))  # fmt: skip
    g = Groups.none(n)
    if groups:
        gid = np.full(n, -1, "i8")
        gid[: 8 * n_cl] = np.repeat(np.arange(n_cl), 8)
        g = Groups(gid, np.where(gid >= 0, 0.9, 0.0), np.full(n_cl, 0.01))
    return rec, g, t, kappa, omega


def _maps(anchor="anchor", tiers=None):
    tiers = tiers or {}
    tier = [IDENTITY if s == anchor else tiers.get(s, LINEAR) for s in NAMES]
    return MapSet(
        [AgeMaps(NAMES, tier, np.zeros((4, 1), bool), [s == "C" for s in NAMES])]
    )


def _strip_effect(rec):
    return Records(rec.star, rec.map_id, rec.cls, rec.noise_id, rec.y, rec.sig, rec.feh,
                   rec.censor, rec.n_star)  # fmt: skip


def test_recovers_maps_noise_and_latents():
    rec, g, t, kappa, omega = _synthetic()
    rec = _strip_effect(rec)
    # Gaussian noise here (nu -> inf), so kappa and omega are directly comparable.
    fit = fit_consensus(rec, g, _maps(), 4, nu=1e6, n_outer=40)
    coef = fit.coefs[0]
    for k, name in enumerate(NAMES[1:], start=1):
        b, A, _ = TRUE[name]
        assert abs(coef[k, 0] - b) < 0.02, (name, coef[k])
        assert abs(coef[k, 1] - A) < 0.04, (name, coef[k])
    assert abs(coef[2, 2] - TRUE["C"][2]) < 0.04
    se = np.sqrt(np.diag(fit.theta_cov))
    assert np.all(np.isfinite(se)) and se.max() < 0.05
    # the anchor is fitted too: identity map but learned noise (not a fixed value)
    assert abs(np.exp(fit.log_kappa[0]) - kappa["anchor"]) < 0.25
    assert abs(np.exp(fit.log_kappa[1]) - kappa["B"]) < 0.2
    assert abs(np.exp(fit.log_omega[3]) - omega["D"]) < 0.015
    sol = solve_stars(rec, g, fit, 4)
    ok = np.isfinite(sol["t"]) & (sol["limit"] == 0)
    z = (sol["t"][ok] - t[ok]) / sol["sd"][ok]
    assert abs(np.median(sol["t"][ok] - t[ok])) < 0.01
    assert 0.85 < 1.4826 * np.median(np.abs(z - np.median(z))) < 1.2
    # map values and their standard errors
    grid = np.array([0.0, 0.6, 1.0])
    f, _, sef = map_values(fit, "B", grid)
    assert np.allclose(f - grid, coef[1, 0] + coef[1, 1] * (grid - T0))
    assert np.all(sef > 0) and np.all(sef < 0.03)


def test_censoring_beats_naive_treatment():
    rec, g, t, _, _ = _synthetic(seed=1)
    rec = _strip_effect(rec)
    fit = fit_consensus(rec, g, _maps(), 4, n_outer=40, covariance=False)
    naive = Records(rec.star, rec.map_id, rec.cls, rec.noise_id, rec.y, rec.sig, rec.feh,
                    np.zeros_like(rec.censor), rec.n_star)  # fmt: skip
    fit_n = fit_consensus(naive, g, _maps(), 4, n_outer=40, covariance=False)
    err = abs(fit.coefs[0][2, 1] - TRUE["C"][1])
    err_n = abs(fit_n.coefs[0][2, 1] - TRUE["C"][1])
    assert err < 0.04 and err < 0.5 * err_n, (fit.coefs[0][2], fit_n.coefs[0][2])
    # stars measured only by C at the cap get a lower limit, not a value
    sol = solve_stars(rec, g, fit, 4)
    only_one = np.bincount(rec.star, minlength=rec.n_star) == 1
    capped = np.zeros(rec.n_star, bool)
    capped[rec.star[rec.censor == 1]] = True
    sel = only_one & capped
    assert (
        sel.any()
        and np.all(sol["limit"][sel] == 1)
        and np.all(np.isnan(sol["sd"][sel]))
    )


def test_anchor_change_shifts_scale_by_the_map():
    rec, g, t, _, _ = _synthetic(seed=2, censor=False)
    rec = _strip_effect(rec)
    fit_a = fit_consensus(rec, g, _maps(), 4, n_outer=40, covariance=False)
    fit_b = fit_consensus(rec, g, _maps(anchor="B"), 4, n_outer=40, covariance=False)
    sa, sb = solve_stars(rec, g, fit_a, 4), solve_stars(rec, g, fit_b, 4)
    ok = np.isfinite(sa["t"]) & np.isfinite(sb["t"])
    # On the B scale every latent is B's map of the anchor-scale latent ...
    fb, _, _ = map_values(fit_a, "B", sa["t"][ok])
    assert np.median(np.abs(sb["t"][ok] - fb)) < 0.01
    b, A, _ = TRUE["B"]
    assert (
        abs(np.median(sb["t"][ok] - (sa["t"][ok] + b + A * (sa["t"][ok] - T0)))) < 0.02
    )
    # ... and the anchor's own map on the B scale is the inverse of B's.
    assert abs(fit_b.coefs[0][0, 1] - (1.0 / (1.0 + A) - 1.0)) < 0.05


def test_conatal_groups_tighten_members():
    rec, g, t, _, _ = _synthetic(seed=3, groups=True, censor=False)
    rec = _strip_effect(rec)
    fit = fit_consensus(
        rec, g, _maps(tiers={"D": OFFSET}), 4, n_outer=30, covariance=False
    )
    sol = solve_stars(rec, g, fit, 4, rho=fit.rho)
    mem = g.gid >= 0
    ok = mem & np.isfinite(sol["t"]) & np.isfinite(sol["t_rec"])
    err_con = np.median(np.abs(sol["t"][ok] - t[ok]))
    err_rec = np.median(np.abs(sol["t_rec"][ok] - t[ok]))
    assert err_con < 0.6 * err_rec
    assert np.median(fit.rho[mem]) > 0.8


def test_shared_effect_is_learned_and_not_double_counted():
    eta = 0.08
    rec, g, t, _, _ = _synthetic(seed=4, censor=False, eta=eta, groups=True)
    fit = fit_consensus(
        rec, g, _maps(), 4, n_effect=1, nu=1e6, n_outer=40, covariance=False
    )
    assert abs(np.exp(fit.log_eta[0]) - eta) < 0.025, np.exp(fit.log_eta)
    # stars measured only by B and C: their shared error is not averaged away
    sol = solve_stars(rec, g, fit, 4, n_effect=1, rho=fit.rho)
    by = np.zeros((rec.n_star, 4), bool)
    by[rec.star, rec.map_id] = True
    sel = by[:, 1] & by[:, 2] & ~by[:, 0] & ~by[:, 3] & (g.gid < 0)
    z = (sol["t"][sel] - t[sel]) / sol["sd"][sel]
    assert 0.8 < 1.4826 * np.median(np.abs(z - np.median(z))) < 1.25
    assert np.median(sol["sd"][sel]) > 0.7 * eta


def test_non_age_observable_with_monotone_map():
    """A non-age source (e.g. ln P_rot) enters through its own monotone map."""
    rng = np.random.default_rng(5)
    n = 3000
    t = rng.uniform(-0.5, 1.0, n)
    feh = rng.normal(-0.1, 0.2, n)
    a_idx = np.flatnonzero(rng.random(n) < 0.5)
    p_idx = np.flatnonzero(rng.random(n) < 0.7)
    y_age = t[a_idx] + rng.normal(0, 0.05, len(a_idx))
    tp = t[p_idx]
    y_rot = (
        2.0
        + 0.8 * tp
        + 0.3 * tp**2
        + 0.2 * feh[p_idx]
        + rng.normal(0, 0.06, len(p_idx))
    )
    star = np.r_[a_idx, p_idx].astype("i8")
    map_id = np.r_[np.zeros(len(a_idx)), np.ones(len(p_idx))].astype("i8")
    rec = Records(star, map_id, np.zeros(len(star), "i8"), map_id.copy(), np.r_[y_age, y_rot],
                  np.r_[np.full(len(a_idx), 0.05), np.zeros(len(p_idx))],
                  np.r_[feh[a_idx], feh[p_idx]], np.zeros(len(star), "i1"), n)  # fmt: skip
    maps = MapSet([AgeMaps(["anchor"], [IDENTITY], np.zeros((1, 1), bool), [False]),
                   MonotoneMaps(["ln_prot"], direction=[1.0])])  # fmt: skip
    fit = fit_consensus(rec, Groups.none(n), maps, 2, n_outer=40, covariance=False)
    grid = np.array([-0.4, 0.0, 0.4, 0.8])
    f, d, _ = map_values(fit, "ln_prot", grid, feh=0.0)
    assert np.max(np.abs(f - (2.0 + 0.8 * grid + 0.3 * grid**2))) < 0.05 and np.all(
        d > 0
    )
    # stars with a rotation record only get an age from the inverted map
    sol = solve_stars(rec, Groups.none(n), fit, 2)
    only_rot = np.setdiff1d(p_idx, a_idx)
    err = sol["t"][only_rot] - t[only_rot]
    assert abs(np.median(err)) < 0.03 and np.median(np.abs(err)) < 0.08


# --------------------------------------------------------------------------- #
# Script-level checks (scripts/age_consensus_c0.py)
# --------------------------------------------------------------------------- #
def _script():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "age_consensus_c0.py"
    spec = importlib.util.spec_from_file_location("age_consensus_c0_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def test_predict_heldout_recovers_known_map():
    import pandas as pd

    c0 = _script()
    rng = np.random.default_rng(7)
    n = 3000
    t = rng.uniform(0.0, 1.1, n)
    sd = np.full(n, 0.03)
    t_obs = t + rng.normal(0, 0.03, n)  # consensus with its own uncertainty
    y = t + 0.08 + 0.2 * (t - T0) + rng.normal(0, 0.06, n)  # held-out catalogue
    recs = pd.DataFrame({"source_id": np.arange(n), "lage": y, "sig": 0.05, "feh": 0.0,
                         "cls": "SGB", "censor": np.zeros(n, "i1")})  # fmt: skip
    rows = c0.predict_heldout(
        "cat", recs, np.arange(n), {"t": t_obs, "sd": sd}, tier_hint=LINEAR
    )
    row = next(r for r in rows if r["class"] == "all")
    assert row["n_test"] > 1000
    assert abs(row["bias_raw"] - 0.08) < 0.03  # raw residual shows the offset
    assert abs(row["bias_mapped"]) < 0.01 and row["nmad_mapped"] < row["nmad_raw"]
    assert 0.8 < row["z_nmad"] < 1.25


def test_run_quick_end_to_end(tmp_path):
    """The CLI ``run --quick`` on a small synthetic record set (all stages but
    the variants and folds)."""
    import pandas as pd

    c0 = _script()
    rng = np.random.default_rng(11)
    n = 900
    t = rng.uniform(0.0, 1.1, n)
    giant = np.arange(n) < 450
    n_cl = 30  # clusters: 4 giants + 4 subgiants with one age
    cid = np.full(n, -1)
    for c in range(n_cl):
        mem = np.r_[np.arange(4 * c, 4 * c + 4), 450 + np.arange(4 * c, 4 * c + 4)]
        t[mem] = t[mem[0]]
        cid[mem] = 100 + c
    feh = rng.normal(-0.2, 0.25, n)

    def cat(source, idx, offset, sig, stage, stage_class, logg):
        y = t[idx] + offset + rng.normal(0, 0.06, len(idx))
        age = 10**y
        return pd.DataFrame({"source": source, "source_id": 1000 + idx, "age": age,
                             "lage": y, "sig": sig, "stage": stage, "stage_class": stage_class,
                             "good": True, "feh": feh[idx], "logg": logg})  # fmt: skip

    gi, sg = np.flatnonzero(giant), np.flatnonzero(~giant)
    frames = [
        cat("apokasc3", gi[:330], 0.0, 0.05, "RGB", "giant", 2.5),
        cat("willett26_kepler", gi[100:], 0.02, 0.06, "RC", "giant", 2.4),
        cat("apokasc2", gi[:150], -0.05, 0.07, "RGB", "giant", 2.5),
        cat("leung23", gi[:250], 0.03, 0.1, "unknown", "unknown", np.nan),
        cat("xiang24", sg, -0.1, 0.03, "SGB", "subgiant", 3.8),
        cat("starhorse23_lamostl7", sg[100:], -0.15, 0.04, "SGB", "subgiant", 3.9),
        cat("galah_dr4", np.r_[gi[300:], sg[:300]], 0.0, np.nan, "x", "giant", 3.0),
    ]
    rec = pd.concat(frames, ignore_index=True)
    rec.loc[(rec.source == "galah_dr4") & (rec.source_id >= 1450), "stage_class"] = (
        "subgiant"
    )
    for c in ("source", "stage", "stage_class"):
        rec[c] = rec[c].astype("category")
    rec_path, tgt_path = tmp_path / "records.parquet", tmp_path / "targets.npz"
    rec.to_parquet(rec_path, index=False)
    m = cid >= 0
    pairs = 1000 + sg[-40:].reshape(20, 2)
    np.savez(tgt_path, member=1000 + np.flatnonzero(m), member_cluster=cid[m].astype("i4"),
             member_prob=np.full(m.sum(), 0.9, "f4"), pair1=pairs[:, 0], pair2=pairs[:, 1],
             source_id=1000 + np.arange(n))  # fmt: skip
    work, docs = tmp_path / "work", tmp_path / "docs"
    c0.main(["run", "--records", str(rec_path), "--targets", str(tgt_path), "--work", str(work),
             "--docs", str(docs), "--quick", "--n-outer", "15", "--threads", "2"])  # fmt: skip
    stars = pd.read_parquet(work / "consensus_stars.parquet")
    assert np.isfinite(stars.t_A).sum() > 800 and np.isfinite(stars.t_C).sum() > 800
    maps = pd.read_csv(docs / "maps_A.csv").set_index("key")
    assert maps.loc["apokasc3", "tier"] == "identity"
    val = pd.read_csv(docs / "validation_records.csv")
    assert {"apokasc2", "leung23"} <= set(val.catalogue)
    for name in ("noise_A.csv", "tiers_C.csv", "bridge_variants.csv", "summary.json"):
        assert (docs / name).exists()


def test_profile_gradient_matches_finite_differences():
    """The envelope gradient of the profiled map objective (latents, shared
    errors and group ages re-solved at every evaluation) equals its finite-
    difference gradient, so L-BFGS gets a true descent direction."""
    from masked_stellar_autoencoder.training import age_consensus as ac

    rec, g, t, _, _ = _synthetic(seed=8, censor=False, eta=0.08, groups=True)
    maps = _maps()
    st = ac._State(rec, g, maps, 4, 1)
    rng = np.random.default_rng(0)
    theta = maps.theta0() + torch.as_tensor(rng.normal(0, 0.05, maps.theta0().numel()))
    v = ac._noise_var(
        st, torch.zeros(4, dtype=ac.DT), torch.full((4,), np.log(0.05), dtype=ac.DT)
    )
    w = torch.ones(st.R, dtype=ac.DT)
    t_lin = torch.as_tensor(t + rng.normal(0, 0.05, len(t)), dtype=ac.DT)
    rho = st.prior * st.member
    e2 = st.eta2(torch.tensor([np.log(0.08)], dtype=ac.DT))

    def F(th):
        return ac._map_objective(st, th, w, v, st.y_obs, t_lin, rho, e2)

    th = theta.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(F(th), th)
    h = 1e-6
    fd = torch.stack([(F(theta + h * e) - F(theta - h * e)) / (2 * h)
                      for e in torch.eye(theta.numel(), dtype=ac.DT)])  # fmt: skip
    assert torch.allclose(grad, fd, rtol=1e-4, atol=1e-4 * float(fd.abs().max())), (
        grad,
        fd,
    )


# --------------------------------------------------------------------------- #
# C0b mechanisms
# --------------------------------------------------------------------------- #
def test_nested_shared_errors_are_learned():
    """An outer family (all isochrone pipelines) and a nested sub-family (two
    pipelines on the same spectra) each get their eta; stars measured only by
    the two same-spectra pipelines keep both shared errors in their width."""
    rng = np.random.default_rng(21)
    n, n_cl = 5000, 300
    t = rng.uniform(-0.2, 1.1, n)
    t[: 8 * n_cl] = np.repeat(t[: 8 * n_cl : 8], 8)
    eta_o, eta_k = 0.08, 0.06
    u_o, u_k = rng.normal(0, eta_o, n), rng.normal(0, eta_k, n)
    spec = {"anchor": (0.4, 0.05, -1, -1), "B": (0.7, 0.04, 0, 0), "C": (0.6, 0.04, 0, 0),
            "D": (0.5, 0.05, 0, -1)}  # fmt: skip
    rows = []
    for k, (name, (frac, s, eff, sub)) in enumerate(spec.items()):
        idx = np.flatnonzero(rng.random(n) < frac)
        shared = (u_o[idx] if eff >= 0 else 0.0) + (u_k[idx] if sub >= 0 else 0.0)
        y = t[idx] + shared + rng.normal(0, s, len(idx))
        rows.append((idx, np.full(len(idx), k), y, np.full(len(idx), s),
                     np.full(len(idx), eff), np.full(len(idx), sub)))  # fmt: skip
    star, mid, y, sig, eff, sub = (
        np.concatenate([r[j] for r in rows]) for j in range(6)
    )
    rec = Records(star.astype("i8"), mid.astype("i8"), np.zeros(len(y), "i8"), mid.astype("i8"),
                  y, sig, np.zeros(len(y)), np.zeros(len(y), "i1"), n,
                  effect=eff.astype("i8"), subeffect=sub.astype("i8"))  # fmt: skip
    gid = np.full(n, -1, "i8")
    gid[: 8 * n_cl] = np.repeat(np.arange(n_cl), 8)
    g = Groups(gid, np.where(gid >= 0, 1.0, 0.0), np.full(n_cl, 0.01))
    names = list(spec)
    maps = MapSet([AgeMaps(names, [IDENTITY, OFFSET, OFFSET, OFFSET], np.zeros((4, 1), bool),
                           [False] * 4)])  # fmt: skip
    fit = fit_consensus(
        rec, g, maps, 4, 1, n_sub=1, nu=1e6, n_outer=30, covariance=False
    )
    eo, ek = np.exp(fit.log_eta)
    assert abs(eo - eta_o) < 0.03 and abs(ek - eta_k) < 0.03, (eo, ek)
    sol = solve_stars(rec, g, fit, 4, rho=fit.rho)
    by = np.zeros((n, 4), bool)
    by[rec.star, rec.map_id] = True
    sel = by[:, 1] & by[:, 2] & ~by[:, 0] & ~by[:, 3] & (gid < 0)
    z = (sol["t"][sel] - t[sel]) / sol["sd"][sel]
    assert 0.8 < 1.4826 * np.median(np.abs(z - np.median(z))) < 1.25


def test_interval_records_use_a_flat_top_likelihood():
    """A source reporting intervals: a narrow interval behaves like its value
    (offset recovered); a wide one is a flat top, not an exact value: its
    offset lies anywhere inside the feasible range and its noise is not
    driven to the floor (the imputed values' conditional variance enters
    the restricted likelihood)."""
    rng = np.random.default_rng(22)
    n = 3000
    t = rng.uniform(0.0, 1.0, n)
    a_idx = np.flatnonzero(rng.random(n) < 0.6)
    b_idx = np.flatnonzero(rng.random(n) < 0.6)
    ya = t[a_idx] + rng.normal(0, 0.05, len(a_idx))
    yb = t[b_idx] + 0.1 + rng.normal(0, 0.03, len(b_idx))
    star = np.r_[a_idx, b_idx].astype("i8")
    mid = np.r_[np.zeros(len(a_idx)), np.ones(len(b_idx))].astype("i8")
    maps = MapSet(
        [
            AgeMaps(
                ["anchor", "B"],
                [IDENTITY, OFFSET],
                np.zeros((2, 1), bool),
                [False, False],
            )
        ]
    )
    cens = np.r_[np.zeros(len(a_idx)), np.full(len(b_idx), 2)].astype("i1")

    def run(lo, hi):
        rec = Records(star, mid, np.zeros(len(star), "i8"), mid.copy(), np.r_[ya, lo],
                      np.r_[np.full(len(a_idx), 0.05), np.zeros(len(b_idx))],
                      np.zeros(len(star)), cens, n, y2=np.r_[ya, hi])  # fmt: skip
        return fit_consensus(rec, Groups.none(n), maps, 2, n_outer=30, covariance=False)

    narrow = run(yb - 0.001, yb + 0.001)
    assert abs(narrow.coef_of("B")[0] - 0.1) < 0.01
    wide = run(yb - 0.05, yb + 0.25)  # every value lies 0.05 above its interval's floor
    b_w = wide.coef_of("B")[0]
    assert 0.05 < b_w < 0.35  # anywhere on the flat top
    assert np.exp(wide.log_omega[1]) > 0.03  # not an exact value


def test_gauge_change_is_an_exact_reparametrisation():
    """Fit on anchor X; re-express the solution on catalogue Y's scale (an
    offset-only reference) and refit there from it: the solution must not move
    (the co-natal term has fixed memberships and a Student-t member weight, so
    the likelihood is covariant under a common shift of the latents)."""
    rng = np.random.default_rng(23)
    n, n_cl = 4000, 200
    t = rng.uniform(-0.2, 1.1, n)
    t[: 6 * n_cl] = np.repeat(t[: 6 * n_cl : 6], 6)
    regime = np.arange(n) % 2  # two regimes linked only through clusters
    rows = []
    spec = [("X", 0, 0.0, 0.0, 0.05), ("Z", 0, 0.05, 0.1, 0.06), ("Y", 1, -0.1, 0.0, 0.05),
            ("W", 1, 0.08, -0.1, 0.07)]  # fmt: skip
    for k, (name, reg, b, A, s) in enumerate(spec):
        idx = np.flatnonzero((regime == reg) & (rng.random(n) < 0.8))
        y = t[idx] + b + A * (t[idx] - T0) + rng.normal(0, s, len(idx))
        rows.append((idx, np.full(len(idx), k), y, np.full(len(idx), s)))
    star, mid, y, sig = (np.concatenate([r[j] for r in rows]) for j in range(4))
    rec = Records(star.astype("i8"), mid.astype("i8"), np.zeros(len(y), "i8"), mid.astype("i8"),
                  y, sig, np.zeros(len(y)), np.zeros(len(y), "i1"), n)  # fmt: skip
    gid = np.full(n, -1, "i8")
    gid[: 6 * n_cl] = np.repeat(np.arange(n_cl), 6)
    g = Groups(gid, np.where(gid >= 0, 0.9, 0.0), np.full(n_cl, 0.01))
    names = [s[0] for s in spec]
    ct = np.zeros((4, 1), bool)
    mapsA = MapSet(
        [AgeMaps(names, [IDENTITY, LINEAR, OFFSET, LINEAR], ct, [False] * 4)]
    )
    fa = fit_consensus(rec, g, mapsA, 4, n_outer=40, tol=1e-5, covariance=False)
    d = fa.coefs[0][2, 0]  # Y's offset: t_Y = t_X + d
    cB = fa.coefs[0].copy()
    cB[:, 0] = cB[:, 0] - d * (1.0 + cB[:, 1])
    cB[2] = 0.0
    cB[0, 0], cB[0, 1] = -d, 0.0
    mapsB = MapSet(
        [AgeMaps(names, [OFFSET, LINEAR, IDENTITY, LINEAR], ct, [False] * 4)]
    )
    fb = fit_consensus(rec, g, mapsB, 4, n_outer=40, tol=1e-5, covariance=False, coefs0=[cB],
                       lk0=fa.log_kappa, lw0=fa.log_omega, t0=fa.t + d, w0=fa.weight, rho0=fa.rho)  # fmt: skip
    assert np.max(np.abs(fb.coefs[0] - cB)) < 2e-3, (fb.coefs[0], cB)
    ok = np.isfinite(fa.t)
    assert np.max(np.abs(fb.t[ok] - (fa.t[ok] + d))) < 2e-3


def test_noise_identifiability_needs_triads_or_an_identified_partner():
    from masked_stellar_autoencoder.training.age_consensus import noise_identified

    n = 400
    # units 0 and 1 overlap only pairwise: their noise split is not identified
    star = np.r_[np.arange(n), np.arange(n)]
    group = np.r_[np.zeros(n), np.ones(n)].astype("i8")
    unit = group.copy()
    ident, triad, _ = noise_identified(star, group, unit, 2)
    assert not ident.any() and triad.sum() == 0
    # a third independent measurement on the same stars identifies all three
    ident3, triad3, _ = noise_identified(np.r_[star, np.arange(n)], np.r_[group, np.full(n, 2)],
                                         np.r_[unit, np.full(n, 2)], 3)  # fmt: skip
    assert ident3.all() and (triad3 == n).all()
    # unit 3 paired only with an identified unit becomes identified (propagation);
    # units 4 and 5 paired only with each other stay unidentified
    s2 = np.r_[np.arange(n), np.arange(n), np.arange(n), 1000 + np.arange(n), 1000 + np.arange(n),
               2000 + np.arange(n), 2000 + np.arange(n)]  # fmt: skip
    g2 = np.r_[np.zeros(n), np.ones(n), np.full(n, 2), np.zeros(n), np.full(n, 3), np.full(n, 4),
               np.full(n, 5)].astype("i8")  # fmt: skip
    u2 = np.r_[np.zeros(n), np.ones(n), np.full(n, 2), np.zeros(n), np.full(n, 3), np.full(n, 4),
               np.full(n, 5)].astype("i8")  # fmt: skip
    ident2, _, pair2 = noise_identified(s2, g2, u2, 6)
    assert ident2[:4].all() and not ident2[4:].any()


# --------------------------------------------------------------------------- #
# C0b script mechanisms
# --------------------------------------------------------------------------- #
def _fit_frame(rows):
    """Fit-record frame in the layout of ``prepare``: rows of (source, ids,
    lage, cls, group)."""
    import pandas as pd

    out = []
    for source, ids, lage, cls, group in rows:
        out.append(pd.DataFrame({"source": source, "source_id": np.asarray(ids, "i8"), "lage": lage,
                                 "sig": 0.05, "cls": cls, "feh": 0.0, "censor": np.int8(0),
                                 "lage2": np.nan, "group": group, "prio": 0}))  # fmt: skip
    fit = pd.concat(out, ignore_index=True)
    return fit.assign(representative=True)


def test_identifiability_gate_and_domain_censoring():
    """A catalogue that overlaps only pairwise with another unidentified one
    gets an offset map; a map shared with identified units keeps its slope;
    pipelines on the same spectra count as one independent measurement;
    unconstrained records below a map's data range become upper limits."""
    import pandas as pd

    c0 = _script()
    rng = np.random.default_rng(3)
    n = 1500
    tg = rng.uniform(-0.3, 1.0, 2 * n)
    ts = rng.uniform(-0.1, 1.0, n)
    a, b = np.arange(n), n + np.arange(n)
    sg = 10_000 + np.arange(n)
    lone = 20_000 + np.arange(5)  # single-record apok2 stars far below its range
    fit = _fit_frame([
        ("apokasc3", a, tg[a], "RGB", "apokasc"),
        ("willett26_kepler", a, tg[a] + 0.02, "RGB", "willett26"),
        ("chaplin14", a, tg[a] - 0.01, "RGB", "chaplin14"),  # third independent group: triads
        ("apok2", np.r_[b, lone], np.r_[tg[b] + 0.05, np.full(5, -1.5)], "giant", "apok2"),
        ("willett26_k2_bhm", b, tg[b], "giant", "willett26"),  # pairwise only with apok2
        ("xiang24", sg, ts, "SGB", "xiang24"),
        ("starhorse23_lamostl7", sg, ts + 0.05, "SGB", "starhorse23"),  # same LAMOST spectra
        ("galah_dr4", sg, ts - 0.05, "SGB", "galah_dr4"),
    ])  # fmt: skip
    data = {"fit": fit, "members": pd.DataFrame({"source_id": [], "cid": [], "prob": []}),
            "pairs": pd.DataFrame({"s1": [], "s2": [], "prior": []})}  # fmt: skip
    prob = c0.build_problem(data, c0.Config("gate"), log=lambda m: None)
    tiers = prob["tiers"].set_index("key")
    assert tiers.tier["apok2"] == OFFSET and not tiers.noise_identified["apok2"]
    assert (
        tiers.tier["willett26_dnu"] == LINEAR
        and tiers.noise_identified["willett26_dnu"]
    )
    assert tiers.triad_stars["xiang24"] == 0 and not tiers.noise_identified["xiang24"]
    off = c0.build_problem(
        data, c0.Config("nogate", identifiability=False), log=lambda m: None
    )
    assert off["tiers"].set_index("key").tier["apok2"] == LINEAR
    df = prob["fit_df"]
    low = df[df.source_id.isin(lone)]
    assert len(low) == 5 and (low.censor == -1).all()
    assert np.allclose(low.lage, tiers.domain_lo["apok2"], atol=1e-3)
    assert (
        tiers.n_below_domain["apok2"] >= 5
    )  # (+ the 0.1 % tail of its constrained records)


def test_bridge_width_does_not_depend_on_the_anchor():
    """Components join along a maximum spanning tree: every edge of the tree,
    and so the reference tier it sets, is the same whichever component holds
    the anchor (chain 0 -40- 1 -10- 2 with one co-natal group linking 0 and 2)."""
    import pandas as pd

    c0 = _script()
    s = np.r_[np.arange(100), np.arange(60, 160), np.arange(150, 400)]
    c = np.r_[np.zeros(100), np.ones(100), np.full(250, 2)].astype(int)
    sc = pd.DataFrame({"s": s, "c": c})
    gid = np.full(400, -1)
    gid[[10, 300]] = 7

    def need(k):
        return 5

    w0 = dict(c0._attach_order(sc, gid, 0, need))
    w2 = dict(c0._attach_order(sc, gid, 2, need))
    assert w0 == {1: 40, 2: 10} and w2 == {1: 10, 0: 40}
    # an edge below ``need`` leaves the component out
    w_strict = c0._attach_order(sc, gid, 0, lambda k: 20)
    assert w_strict == [(1, 40), (2, 10)] and dict(
        c0._attach_order(sc, gid, 2, lambda k: 20)
    ) == {0: 1, 1: 10}


def test_arbitration_offset_recovers_a_known_offset():
    """A validation-only source with a constant offset from the consensus
    (including flat-top interval records): offset recovered within its error."""
    import pandas as pd

    c0 = _script()
    rng = np.random.default_rng(5)
    n = 60
    t = rng.uniform(-1.5, 1.0, n)
    sd = np.full(n, 0.03)
    t_obs = t + rng.normal(0, 0.03, n)
    y = t + 0.12 + rng.normal(0, 0.06, n)
    cens = np.zeros(n, "i1")
    cens[:5] = 2
    recs = pd.DataFrame({"source_id": np.arange(n), "lage": np.where(cens == 2, y - 0.05, y),
                         "lage2": np.where(cens == 2, y + 0.05, np.nan),
                         "sig": np.where(cens == 2, 0.0, 0.06), "censor": cens})  # fmt: skip
    row = c0.offset_vs_consensus("bench", recs, np.arange(n), {"t": t_obs, "sd": sd})
    assert row["n"] == n and row["n_censored"] == 5
    assert abs(row["offset"] - 0.12) < 3 * row["offset_se"] and row["offset_se"] < 0.02
    assert abs(row["offset_quoted"] - 0.12) < 0.03


def _hr23_fits(path, ids, names, logage):
    from astropy.table import Table

    t = Table({"Name": np.array(names), "ID": np.asarray(ids, "i8"),
               "Type": np.array(["o"] * len(ids)), "logAge16": logage - 0.05,
               "logAge50": logage, "logAge84": logage + 0.05, "CMDCl50": np.full(len(ids), 0.9)})  # fmt: skip
    t.write(path, overwrite=True)


def test_run_with_bridge_labels_end_to_end(tmp_path):
    """``run --quick --labels``: benchmark clusters, cluster catalogues, extra
    seismology, WD+MS and wide pairs enter as mapped sources; benchmark
    families get offset maps, nodes carry the cluster records, the gauge-C fit
    started from gauge A stays put, and the arbitration table recovers the
    LDB offset."""
    import json

    import pandas as pd

    c0 = _script()
    rng = np.random.default_rng(17)
    n = 900
    t = rng.uniform(0.0, 1.1, n)
    giant = np.arange(n) < 450
    n_cl = 30
    cid = np.full(n, -1)
    for c in range(n_cl):
        mem = np.r_[np.arange(4 * c, 4 * c + 4), 450 + np.arange(4 * c, 4 * c + 4)]
        t[mem] = t[mem[0]]
        cid[mem] = 100 + c
    t_cl = np.array([t[4 * c] for c in range(n_cl)])
    feh = rng.normal(-0.2, 0.25, n)

    def cat(source, idx, offset, sig, stage, stage_class, logg):
        y = t[idx] + offset + rng.normal(0, 0.06, len(idx))
        return pd.DataFrame({"source": source, "source_id": 1000 + idx, "age": 10**y, "lage": y,
                             "sig": sig, "stage": stage, "stage_class": stage_class, "good": True,
                             "feh": feh[idx], "logg": logg})  # fmt: skip

    gi, sg = np.flatnonzero(giant), np.flatnonzero(~giant)
    rec = pd.concat([
        cat("apokasc3", gi[:330], 0.0, 0.05, "RGB", "giant", 2.5),
        cat("willett26_kepler", gi[100:], 0.02, 0.06, "RC", "giant", 2.4),
        cat("xiang24", sg, -0.1, 0.03, "SGB", "subgiant", 3.8),
        cat("starhorse23_lamostl7", sg[100:], -0.15, 0.04, "SGB", "subgiant", 3.9),
        cat("galah_dr4", sg[:300], 0.0, 0.05, "x", "subgiant", 3.8),
    ], ignore_index=True)  # fmt: skip
    for c in ("source", "stage", "stage_class"):
        rec[c] = rec[c].astype("category")
    rec_path, tgt_path = tmp_path / "records.parquet", tmp_path / "targets.npz"
    rec.to_parquet(rec_path, index=False)
    m = cid >= 0
    np.savez(tgt_path, member=1000 + np.flatnonzero(m), member_cluster=cid[m].astype("i4"),
             member_prob=np.full(m.sum(), 0.9, "f4"), pair1=np.zeros(0, "i8"), pair2=np.zeros(0, "i8"),
             source_id=1000 + np.arange(n))  # fmt: skip
    lab = tmp_path / "labels"
    lab.mkdir()
    ids = 100 + np.arange(n_cl)
    names = [f"CL_{i}" for i in ids]
    names[0] = "Melotte_22"
    clusters = tmp_path / "clusters.fits"
    _hr23_fits(clusters, ids, names, t_cl + 9 + 0.03 + rng.normal(0, 0.05, n_cl))
    pd.DataFrame({"catalogue": np.repeat(["cantatgaudin20", "dias21"], n_cl), "hr23_id": np.r_[ids, ids],
                  "value": np.r_[t_cl + 9 - 0.05, t_cl + 9 + 0.02] + rng.normal(0, 0.05, 2 * n_cl),
                  "err_lo": np.r_[np.full(n_cl, np.nan), np.full(n_cl, 0.05)], "feh": 0.0}).to_parquet(
        lab / "cluster_msto_catalogues.parquet")  # fmt: skip
    # benchmarks: LDB (+0.08 dex) on 12 clusters, EB on 8 (one published as a range), GC none
    ldb, eb = np.arange(12), np.arange(12, 20)
    age_l = 10 ** (t_cl[ldb] + 0.08 + rng.normal(0, 0.02, len(ldb)))
    age_e = 10 ** (t_cl[eb] + rng.normal(0, 0.02, len(eb)))
    pri = pd.DataFrame({
        "cluster": [names[i] for i in np.r_[ldb, eb]], "hr23_name": [names[i] for i in np.r_[ldb, eb]],
        "hr23_id": ids[np.r_[ldb, eb]], "cluster_type": "OC", "age_gyr": np.r_[age_l, age_e],
        "err_lo_gyr": np.r_[age_l, age_e] * 0.03, "err_hi_gyr": np.r_[age_l, age_e] * 0.03,
        "sys_err_gyr": np.nan, "model_range_lo_gyr": np.nan, "model_range_hi_gyr": np.nan,
        "method": ["LDB"] * len(ldb) + ["EB"] * len(eb), "feh": 0.0, "primary": True,
        "notes": [""] * (len(ldb) + len(eb) - 1) + ["published as a range between two values"]})  # fmt: skip
    pri.to_csv(lab / "cluster_priors.csv", index=False)
    pd.DataFrame({"source_id": [5_000_001, 5_000_002], "cluster": "NGC 104", "hr23_name": [None, None],
                  "prob": [0.9, 0.9]}).to_parquet(lab / "cluster_members_extra.parquet")  # fmt: skip
    for name in ("seis_zinn22_k2gap", "seis_mackereth21_tess"):
        g = gi[300:340]
        pd.DataFrame({"source_id": 1000 + g, "kind": "age_gyr", "value": 10 ** t[g], "err_lo": 0.2 * 10 ** t[g],
                      "err_hi": 0.2 * 10 ** t[g], "feh": feh[g]}).to_parquet(lab / f"{name}.parquet")  # fmt: skip
    comp = sg[300:320]
    wd_ids = 7_000_000 + np.arange(len(comp))
    pd.DataFrame({"source": "qiu21_base9", "source_id": wd_ids, "value": 10 ** t[comp],
                  "err_lo": 0.1 * 10 ** t[comp], "err_hi": 0.1 * 10 ** t[comp],
                  "companion_source_id": (1000 + comp).astype(float), "limit": 0}).to_parquet(
        lab / "wd_ages_published.parquet")  # fmt: skip
    pd.DataFrame({"wd_source_id": wd_ids, "ms_source_id": 1000 + comp, "usable_H": True,
                  "atm_hint": "H"}).to_parquet(lab / "wdms_pairs.parquet")  # fmt: skip
    pp = 1000 + sg[-40:].reshape(20, 2)
    pd.DataFrame({"source_id1": pp[:, 0], "source_id2": pp[:, 1], "R_chance_align": 0.01,
                  "binary_type": "MSMS"}).to_parquet(lab / "wide_pairs_eb21.parquet")  # fmt: skip
    work, docs = tmp_path / "work", tmp_path / "docs"
    c0.main(["run", "--records", str(rec_path), "--targets", str(tgt_path), "--work", str(work),
             "--docs", str(docs), "--labels", str(lab), "--clusters", str(clusters), "--quick",
             "--n-outer", "20", "--threads", "2"])  # fmt: skip
    tiers = pd.read_csv(docs / "tiers_A.csv").set_index("key")
    assert tiers.tier["bench_ldb"] == OFFSET and tiers.tier["bench_eb"] == OFFSET
    assert {"cl_hr23", "cl_cg20", "cl_dias21", "zinn22_k2gap"} <= set(tiers.index)
    summ = json.loads((docs / "summary.json").read_text())
    gc = summ["gauge_check"]
    assert (
        gc["C_fromA_max_coef_move"] < 0.01
        and gc["C_scratch_median_abs_latent_diff"] < 0.01
    )
    arb = pd.read_csv(docs / "arbitration_offsets.csv").set_index(["variant", "source"])
    ldb_a = arb.loc[("ARB_A", "bench_ldb")]
    assert ldb_a.n == 12 and abs(ldb_a.offset - 0.08) < 0.05
    assert arb.loc[("ARB_A", "bench_eb")].n_censored == 1
    assert arb.loc[("main_A", "wd_qiu21")].n == len(comp)
    ctab = pd.read_csv(docs / "clusters_A.csv")
    assert {"info_share_benchmarks", "T_without_benchmarks"} <= set(ctab.columns)
    assert (ctab.young_benchmark == "Pleiades").sum() == 1
    stars = pd.read_parquet(work / "consensus_stars.parquet")
    assert (stars.source_id > 0).all() and "t_C_fromA" in stars


def test_maps_flatten_below_their_data_range():
    """Below the latent where a map reaches its data edge ``y_lo`` the map is
    flat (no information); the edge is fixed in measurement units, so it moves
    with the map's offset (gauge-covariant) and the map stays continuous."""
    from masked_stellar_autoencoder.training.age_consensus import FLAT_SLOPE

    ct = np.zeros((2, 1), bool)
    fam = AgeMaps(
        ["S", "S2"], [LINEAR, LINEAR], ct, [False, False], y_lo=[0.1, -np.inf]
    )
    coef = torch.zeros((2, fam.n_coef()), dtype=torch.float64)
    coef[:, 0], coef[:, 1] = 0.2, 0.1  # f = t + 0.2 + 0.1 (t - T0) above the edge
    t = torch.linspace(-1.5, 1.0, 251, dtype=torch.float64)
    m = torch.zeros(251, dtype=torch.long)
    zeros = torch.zeros(251, dtype=torch.float64)
    f, a = fam.evaluate(coef, m, t, zeros, torch.zeros((251, 0)), m)
    t_star = T0 + (0.1 - T0 - 0.2) / 1.1
    below = t < t_star
    assert torch.allclose(a[below], torch.full_like(a[below], FLAT_SLOPE))
    assert torch.allclose(a[~below], torch.full_like(a[~below], 1.1))
    assert (
        abs(float(f[below].max()) - 0.1) < 1e-3 and float(f[~below].min()) >= 0.1 - 1e-9
    )
    f2, a2 = fam.evaluate(
        coef, torch.ones(251, dtype=torch.long), t, zeros, torch.zeros((251, 0)), m
    )
    assert torch.allclose(a2, torch.full_like(a2, 1.1))  # no edge: linear everywhere
    # an offset change of the map moves the edge latent with it (same y_lo)
    coef[0, 0] = 0.5
    f3, _ = fam.evaluate(coef, m, t, zeros, torch.zeros((251, 0)), m)
    assert abs(float(f3[t < T0 + (0.1 - T0 - 0.5) / 1.1].max()) - 0.1) < 1e-3
    assert float(fam.penalty(a)) == 0.0


def test_young_clusters_follow_the_cluster_level_records():
    """Young clusters whose few member stars carry coherently wrong ages (an
    isochrone catalogue reads 2 Gyr for 100 Myr stars, inside its own data
    range) against a cluster-level record. With the node a fixed member
    (weight = its prior) and the catalogue's map flat below its data edge
    (member records there are missing), the cluster-level record sets the
    young latents and its offset is recovered. Without both, the members win:
    the likelihood counts records, and the node is the minority."""
    rng = np.random.default_rng(29)
    n_f, n_cl = 3000, 120
    tf = rng.uniform(0.0, 1.0, n_f)
    T = rng.uniform(-1.2, 1.0, n_cl)
    young = T < -0.5
    n_m = np.where(young, 2, 5)  # few (hot, prior-dominated) members in young clusters
    n = n_f + int((n_m + 1).sum())
    gid = np.r_[np.full(n_f, -1), np.repeat(np.arange(n_cl), n_m + 1)]
    t = np.r_[tf, np.repeat(T, n_m + 1)]
    node = np.zeros(n, bool)
    node[np.cumsum(n_m + 1) + n_f - 1] = True
    field, mem = np.arange(n_f), np.flatnonzero((gid >= 0) & ~node)
    nodes = np.flatnonzero(node)
    # the catalogue reads ~1 Gyr (t = 0, the young end of its field data) for
    # every younger member
    yS = np.r_[tf, np.maximum(t[mem], 0.0)] + rng.normal(0, 0.1, n_f + len(mem))
    star = np.r_[field, field, mem, nodes].astype("i8")
    mid = np.r_[np.zeros(n_f), np.ones(n_f + len(mem)), np.full(n_cl, 2)].astype("i8")
    y = np.r_[tf + rng.normal(0, 0.05, n_f), yS, T + 0.1 + rng.normal(0, 0.08, n_cl)]
    sig = np.r_[np.full(n_f, 0.05), np.full(n_f + len(mem), 0.1), np.full(n_cl, 0.08)]
    rec = Records(star, mid, np.zeros(len(y), "i8"), mid, y, sig, np.zeros(len(y)),
                  np.zeros(len(y), "i1"), n)  # fmt: skip
    ct = np.zeros((3, 1), bool)
    edge = [-np.inf, float(np.percentile(yS[:n_f], 0.1)), -np.inf]
    prior = np.where(node, 1.0, np.where(gid >= 0, 0.9, 0.0))
    err, koff = {}, {}
    for tag, fixed, y_lo in (("both", node, edge), ("neither", None, None)):
        maps = MapSet(
            [
                AgeMaps(
                    ["X", "S", "K"],
                    [IDENTITY, LINEAR, OFFSET],
                    ct,
                    [False] * 3,
                    y_lo=y_lo,
                )
            ]
        )
        g = Groups(gid, prior, np.full(n_cl, 0.01), fixed)
        fit = fit_consensus(rec, g, maps, 3, n_outer=30, covariance=False)
        sol = solve_stars(rec, g, fit, 3, rho=fit.rho)
        err[tag] = float(np.median(np.abs(sol["t"][nodes][young] - T[young])))
        koff[tag] = float(fit.coef_of("K")[0])
        if tag == "both":
            ym = np.isin(gid[mem], np.flatnonzero(young))
            s_rec = (mid == 1) & np.isin(star, mem[ym])
            assert (fit.weight[s_rec] == 0).mean() > 0.9  # missing below the edge
            assert abs(fit.coef_of("S")[0]) < 0.03  # S's map is not dragged by them
    assert err["both"] < 0.1 and abs(koff["both"] - 0.1) < 0.05, (err, koff)
    assert err["neither"] > 0.3, (err, koff)


def test_co_natal_pairs_below_their_maps_edge_stay_finite():
    """Regression (C0b check fit went NaN at outer iteration 2): wide pairs
    whose two stars carry only upper limits at a map's data edge drift below
    it; their records must not all be dropped (no member holds the latent),
    and a star without record information must keep a finite latent (a NaN
    latent times a zero weight made the map objective NaN)."""
    from masked_stellar_autoencoder.training.age_consensus import _require_finite

    rng = np.random.default_rng(1)
    n_f, n_p = 600, 20
    tf = rng.uniform(0.0, 1.0, n_f)
    n = n_f + 2 * n_p
    pair = n_f + np.arange(2 * n_p)
    yS = tf + rng.normal(0, 0.1, n_f)
    lo = float(np.percentile(yS, 0.1))
    star = np.r_[np.arange(n_f), np.arange(n_f), pair].astype("i8")
    mid = np.r_[np.zeros(n_f), np.ones(n_f), np.ones(2 * n_p)].astype("i8")
    y = np.r_[tf + rng.normal(0, 0.05, n_f), yS, np.full(2 * n_p, lo)]
    sig = np.r_[np.full(n_f, 0.05), np.full(n_f, 0.1), np.full(2 * n_p, 0.1)]
    cens = np.r_[np.zeros(2 * n_f), -np.ones(2 * n_p)].astype("i1")
    rec = Records(
        star, mid, np.zeros(len(y), "i8"), mid, y, sig, np.zeros(len(y)), cens, n
    )
    gid = np.r_[np.full(n_f, -1), np.repeat(np.arange(n_p), 2)]
    g = Groups(gid, np.where(gid >= 0, 0.9, 0.0), np.full(n_p, 0.01))
    maps = MapSet([AgeMaps(["X", "S"], [IDENTITY, OFFSET], np.zeros((2, 1), bool), [False, False],
                           y_lo=[-np.inf, lo])])  # fmt: skip
    fit = fit_consensus(rec, g, maps, 2, n_outer=8, covariance=False)
    assert np.isfinite(fit.coefs[0]).all() and np.isfinite(fit.t).all()
    assert np.isfinite(fit.log_kappa).all() and np.isfinite(fit.log_omega).all()
    assert abs(fit.coef_of("S")[0]) < 0.03
    with pytest.raises(FloatingPointError, match="map coefficients"):
        _require_finite("map coefficients", torch.tensor([0.0, float("nan")]), 2)


# --------------------------------------------------------------------------- #
# C0c: non-age observables (rotation, lithium) through monotone maps
# --------------------------------------------------------------------------- #
def test_monotone_maps_are_monotone_by_construction():
    """Any coefficients (raw ramp weights of either sign), either direction and
    the slope floor: f is monotone in t and df/dt has the map's sign."""
    rng = np.random.default_rng(11)
    fam = MonotoneMaps(["up", "down"], direction=[1.0, -1.0],
                       knots=(-1.6, -1.1, -0.6, -0.2, 0.2, 0.6, 1.0), smooth=0.2,
                       use_z=[True, True], slope_floor=1e-3)  # fmt: skip
    t = torch.linspace(-3.0, 2.0, 501, dtype=torch.float64)
    for _ in range(20):
        coef = torch.as_tensor(rng.normal(0, 3, (2, fam.n_coef())))
        for m, sign in ((0, 1.0), (1, -1.0)):
            loc = torch.full((len(t),), m, dtype=torch.long)
            z = torch.full((len(t), 1), float(rng.normal()), dtype=torch.float64)
            f, d = fam.evaluate(coef, loc, t, torch.zeros_like(t), z, loc)
            assert torch.all(sign * d >= 1e-3 - 1e-12)
            assert torch.all(sign * torch.diff(f) > 0)


def _li_synthetic(seed=3, n=4000, limit=1.2):
    """Ages for half the stars (anchor), A(Li) for 70 %: decreasing in t,
    reported as an upper limit at ``limit`` where the line is not detected."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1.5, 1.0, n)
    a_idx = np.flatnonzero(rng.random(n) < 0.5)
    l_idx = np.flatnonzero(rng.random(n) < 0.7)

    def true_li(tt):
        return 3.2 - 0.72 * np.log1p(np.exp((tt + 0.8) / 0.25))

    li = true_li(t[l_idx]) + rng.normal(0, 0.12, len(l_idx))
    cens = np.where(li < limit, -1, 0).astype("i1")
    li = np.where(cens == -1, limit, li)
    y_age = t[a_idx] + rng.normal(0, 0.05, len(a_idx))
    star = np.r_[a_idx, l_idx].astype("i8")
    map_id = np.r_[np.zeros(len(a_idx)), np.ones(len(l_idx))].astype("i8")
    rec = Records(star, map_id, np.zeros(len(star), "i8"), map_id.copy(), np.r_[y_age, li],
                  np.r_[np.full(len(a_idx), 0.05), np.zeros(len(l_idx))], np.zeros(len(star)),
                  np.r_[np.zeros(len(a_idx), "i1"), cens], n)  # fmt: skip
    return rec, t, true_li, a_idx, l_idx


def test_lithium_upper_limits_enter_censored():
    """A decreasing A(Li) map with non-detections reported as upper limits: the
    censored likelihood recovers the map below the limit's level where the
    naive fit (limits as values) is biased high, and stars with A(Li) only get
    ages from the inverted map."""
    rec, t, true_li, a_idx, l_idx = _li_synthetic()
    knots = (-1.6, -1.1, -0.6, -0.2, 0.2, 0.6, 1.0)

    def maps():
        return MapSet([AgeMaps(["anchor"], [IDENTITY], np.zeros((1, 1), bool), [False]),
                       MonotoneMaps(["a_li"], direction=[-1.0], knots=knots, smooth=0.2,
                                    use_x=[False], y_center=[2.0], slope_floor=1e-3)])  # fmt: skip

    fit = fit_consensus(
        rec, Groups.none(rec.n_star), maps(), 2, n_outer=40, covariance=False
    )
    naive = _strip_effect(rec)
    naive.censor[:] = 0
    fit_n = fit_consensus(naive, Groups.none(rec.n_star), maps(), 2, n_outer=40,
                          covariance=False)  # fmt: skip
    grid = np.array([-1.2, -0.6, -0.3, -0.1])  # the last: ~70 % of records are limits
    f, d, _ = map_values(fit, "a_li", grid)
    fn, _, _ = map_values(fit_n, "a_li", grid)
    err, err_n = np.abs(f - true_li(grid)), np.abs(fn - true_li(grid))
    assert np.all(d < 0)
    assert err.max() < 0.12, err
    assert err_n[-1] > 2 * err[-1]  # below the limit's level the naive map is biased
    sol = solve_stars(rec, Groups.none(rec.n_star), fit, 2)
    det = np.setdiff1d(l_idx, a_idx)
    det = det[(true_li(t[det]) > 1.6) & (t[det] > -1.0)]  # on the informative ramp
    e = sol["t"][det] - t[det]
    assert abs(np.median(e)) < 0.05 and np.median(np.abs(e)) < 0.15


def test_rotation_map_learned_through_cluster_members_only():
    """Rotation stars carry no age record; clusters (a node record on the
    anchor scale plus members with periods) teach the monotone ln P map,
    including its colour term."""
    rng = np.random.default_rng(21)
    n_cl, n_mem, n_field = 150, 10, 2000
    T = rng.uniform(-1.3, 0.8, n_cl)
    n = n_cl * (n_mem + 1) + n_field
    t = np.r_[T, np.repeat(T, n_mem), rng.uniform(-1.3, 0.8, n_field)]
    gid = np.r_[
        np.arange(n_cl), np.repeat(np.arange(n_cl), n_mem), np.full(n_field, -1)
    ]
    node = np.arange(n_cl)
    rot = np.arange(n_cl, n)
    zc = rng.uniform(-1, 1, len(rot))

    def true_lnp(tt, z):
        return 1.0 + 0.4 * z + 0.9 * 0.2 * np.log1p(np.exp((tt + 0.9) / 0.2))

    y_rot = true_lnp(t[rot], zc) + rng.normal(0, 0.1, len(rot))
    y_node = T + rng.normal(0, 0.05, n_cl)
    star = np.r_[node, rot].astype("i8")
    map_id = np.r_[np.zeros(n_cl), np.ones(len(rot))].astype("i8")
    extra = np.r_[np.zeros(n_cl), zc][:, None]
    rec = Records(star, map_id, np.zeros(len(star), "i8"), map_id.copy(), np.r_[y_node, y_rot],
                  np.r_[np.full(n_cl, 0.05), np.full(len(rot), 0.02)], np.zeros(len(star)),
                  np.zeros(len(star), "i1"), n, extra=extra)  # fmt: skip
    fixed = np.zeros(n, bool)
    fixed[node] = True
    g = Groups(
        gid.astype("i8"), np.where(gid >= 0, 1.0, 0.0), np.full(n_cl, 0.01), fixed
    )
    maps = MapSet([AgeMaps(["anchor"], [IDENTITY], np.zeros((1, 1), bool), [False]),
                   MonotoneMaps(["rot"], direction=[1.0], knots=(-1.6, -1.1, -0.6, -0.2, 0.2, 0.6, 1.0),
                                smooth=0.2, use_x=[False], use_z=[True], y_center=[2.0],
                                slope_floor=1e-3)])  # fmt: skip
    in_fit = gid >= 0
    remap = np.full(n, -1, "i8")
    remap[in_fit] = np.arange(in_fit.sum())
    r = rec.subset(in_fit[rec.star])
    r = Records(remap[r.star], r.map_id, r.cls, r.noise_id, r.y, r.sig, r.feh, r.censor,
                int(in_fit.sum()), extra=r.extra)  # fmt: skip
    gs = Groups(gid[in_fit], g.prior[in_fit], g.spread, fixed[in_fit])
    # the node (age) noise is identified, as for real clusters (triads of
    # catalogue ages, member records, benchmarks); with it free and only
    # rotation as the second measurement, map slope and noise trade off (the
    # errors-in-variables slope needs the variance ratio, cf. the age maps)
    fit = fit_consensus(r, gs, maps, 2, n_outer=40, covariance=False,
                        fix_noise=np.array([True, False]), lk0=np.zeros(2),
                        lw0=np.log([0.01, 0.2]))  # fmt: skip
    grid = np.array([-1.0, -0.6, -0.2, 0.2, 0.6])
    for z in (-0.5, 0.5):
        f, d, _ = map_values(fit, "rot", grid, z=z)
        assert np.all(d > 0)
        assert np.max(np.abs(f - true_lnp(grid, z))) < 0.08, (z, f - true_lnp(grid, z))
    # field stars with a period only: ages from the inverted map
    sol = solve_stars(rec, Groups.none(n), fit, 2)
    fld = np.arange(n_cl * (n_mem + 1), n)
    fld = fld[t[fld] > -0.7]  # where the map has slope
    e = sol["t"][fld] - t[fld]
    assert abs(np.median(e)) < 0.05 and np.median(np.abs(e)) < 0.2
