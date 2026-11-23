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
    e2 = st.eta2_blk(torch.tensor([np.log(0.08)], dtype=ac.DT))

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
