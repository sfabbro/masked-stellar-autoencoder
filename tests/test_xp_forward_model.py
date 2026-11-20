"""Tests for the data-driven XP forward model (synthetic emulator, known covariances)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training import xp_forward_model as xf

CPU = torch.device("cpu")
BOUNDS = np.array([[4500.0, 6500.0], [1.5, 4.5], [-2.5, 0.3]])
BANDS = ("G", "J", "KS")
SMOKE = Path(
    "/tmp/claude-0/-home-user/d60e9c7a-7b13-5bb2-ade1-fb8da7adbc26/scratchpad/v9_smoke/finetune.fits"
)


def _true_model(nc: int = 6, seed: int = 0) -> xf.XPForwardModel:
    """A fixed random emulator with known scatter (2 % fractional, 0.005 floor)."""
    torch.manual_seed(seed)
    model = xf.XPForwardModel(
        BANDS,
        theta_loc=[5500.0, 3.0, -1.0],
        theta_scale=[600.0, 1.0, 0.8],
        xp_scale=np.ones(2 * nc),
        mag_loc=np.array([4.0, 2.5, 2.0]),
        mag_scale=np.array([1.5, 1.2, 1.2]),
        n_coeff=nc,
        config=xf.ModelConfig(hidden=32, depth=2, rank=1),
    )
    with torch.no_grad():
        for loss in model.xp_losses:
            loss.log_fraction.fill_(math.log(0.02))
            loss.log_floor.fill_(math.log(0.005))
            loss.factor_weight.zero_()
        model.log_mag_scatter.fill_(math.log(0.01))
        model.intrinsic[-1].bias.add_(
            torch.cat([torch.full((2 * nc,), 1.5), torch.zeros(3)])
        )
    return model.double().eval()


def _simulate(model, n, seed=1, dense=True, noise=(0.5, 3.0)):
    """Stars drawn from ``model`` with per-star known covariances (S/N varies by 6x)."""
    rng = np.random.default_rng(seed)
    nc = model.n_coeff
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    theta = rng.uniform(
        BOUNDS[:, 0] + 0.1 * span, BOUNDS[:, 1] - 0.1 * span, size=(n, 3)
    )
    ebv = rng.uniform(0.0, 0.5, n)
    plx = rng.uniform(0.5, 5.0, n)
    plx_err = 0.08 * plx
    with torch.no_grad():
        xp, mag = model(torch.as_tensor(theta), torch.as_tensor(ebv))
    xp, mag = xp.numpy(), mag.numpy()
    level = rng.uniform(*noise, n)
    covs = np.zeros((n, 2, nc, nc))
    obs = np.zeros((n, 2 * nc))
    for i in range(n):
        for k in range(2):
            c = xp[i, k * nc : (k + 1) * nc]
            sd = level[i] * (0.01 + 0.01 * np.abs(c))
            r = np.eye(nc)
            if dense:
                a = rng.normal(size=(nc, nc))
                r = a @ a.T + nc * np.eye(nc)
                d = np.sqrt(np.diag(r))
                r = r / d[:, None] / d[None]
            covs[i, k] = sd[:, None] * sd[None] * r
            model_sd = np.sqrt((0.02 * c) ** 2 + 0.005**2)
            obs[i, k * nc : (k + 1) * nc] = (
                c
                + np.linalg.cholesky(covs[i, k]) @ rng.normal(size=nc)
                + model_sd * rng.normal(size=nc)
            )
    mag_err = rng.uniform(0.01, 0.05, (n, 3))
    m_obs = (
        mag
        + 5 * np.log10(100 / plx)[:, None]
        + np.sqrt(mag_err**2 + 0.01**2) * rng.normal(size=(n, 3))
    )
    diag = np.concatenate(
        [np.diagonal(covs[:, k], axis1=1, axis2=2) for k in range(2)], 1
    )
    data = xf.StarData(
        rows=np.arange(n),
        source_id=np.arange(n) + 1000,
        xp=obs.astype(np.float32),
        xp_mask=np.ones((n, 2 * nc), bool),
        xp_cov=xf.pack_covariance(covs).astype(np.float32) if dense else None,
        xp_var=None if dense else diag.astype(np.float32),
        mag=m_obs.astype(np.float32),
        mag_err=mag_err.astype(np.float32),
        mag_mask=np.ones((n, 3), bool),
        parallax=plx + plx_err * rng.normal(size=n),
        parallax_err=plx_err,
        parallax_zp=np.zeros(n),
        g_mag=m_obs[:, 0].astype(np.float64),
        ebv_sfd=ebv / 0.8,
        theta=theta,
        theta_err=np.tile([50.0, 0.1, 0.05], (n, 1)),
        band_names=BANDS,
    )
    return data, {"theta": theta, "ebv": ebv, "parallax": plx, "xp": xp, "mag": mag}


def _coverage(res, theta):
    in68 = (theta >= res["lo68"]) & (theta <= res["hi68"])
    in95 = (theta >= res["lo95"]) & (theta <= res["hi95"])
    return float(in68.mean()), float(in95.mean())


def _small_inference(n):
    return xf.InferenceConfig(
        chunk_rows=n,
        n_candidates=32,
        bank_theta=256,
        steps=30,
        profile_scan=4,
        profile_steps=10,
    )


def test_nuisance_box_and_unit_round_trip() -> None:
    sfd = np.array([0.0, 0.1, -0.2, np.nan])
    plx = np.array([2.0, 0.01, -0.5, np.nan])
    err = np.array([0.02, 0.5, 0.3, np.nan])
    lo, hi, e_max = xf.nuisance_box(sfd, plx, err)
    np.testing.assert_allclose(e_max, [0.05, 0.17, 0.05, 0.05])
    np.testing.assert_allclose(10 ** hi[0], 2.0 + 6 * 0.02)
    np.testing.assert_allclose(10 ** lo[0], 2.0 - 6 * 0.02)
    assert np.all(10**lo >= xf.PARALLAX_MIN_MAS - 1e-12) and np.all(hi > lo)
    np.testing.assert_allclose(10 ** hi[3], xf.PARALLAX_MAX_MAS)
    u = xf.nuisance_to_unit(
        np.array([0.02, 0.1, 0.0, 0.03]), np.array([2.0, 0.3, 0.1, 1.0]), lo, hi, e_max
    )
    batch = {
        k: torch.as_tensor(v)
        for k, v in (("e_max", e_max), ("log_plx_lo", lo), ("log_plx_hi", hi))
    }
    ebv, parallax = xf.unit_to_nuisance(torch.as_tensor(u), batch)
    np.testing.assert_allclose(ebv.numpy(), [0.02, 0.1, 0.0, 0.03], atol=1e-12)
    np.testing.assert_allclose(parallax.numpy()[[0, 1, 3]], [2.0, 0.3, 1.0], rtol=1e-10)


def test_diagonal_fallback_equals_dense_diagonal_covariance() -> None:
    model = _true_model()
    dense, truth = _simulate(model, 20, dense=False)
    covs = np.zeros((20, 2, 6, 6))
    var = dense.xp_var.astype(np.float64)
    for k in range(2):
        covs[:, k] = np.einsum("ni,ij->nij", var[:, k * 6 : (k + 1) * 6], np.eye(6))
    dense_cov = dense.subset(np.arange(20))
    dense_cov.xp_cov, dense_cov.xp_var = (
        xf.pack_covariance(covs).astype(np.float32),
        None,
    )
    theta = torch.as_tensor(truth["theta"])
    ebv = torch.as_tensor(truth["ebv"])
    plx = torch.as_tensor(truth["parallax"])
    a = xf.star_nll(
        model, theta, ebv, plx, dense.batch(np.arange(20), CPU, torch.float64)
    )
    b = xf.star_nll(
        model, theta, ebv, plx, dense_cov.batch(np.arange(20), CPU, torch.float64)
    )
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_masked_bands_and_coefficients_are_marginalised() -> None:
    model = _true_model()
    data, truth = _simulate(model, 10)
    theta = torch.as_tensor(truth["theta"])
    ebv = torch.as_tensor(truth["ebv"])
    plx = torch.as_tensor(truth["parallax"])
    full = xf.star_nll(
        model,
        theta,
        ebv,
        plx,
        data.batch(np.arange(10), CPU, torch.float64),
        parts=True,
    )
    masked = data.subset(np.arange(10))
    masked.mag_mask = masked.mag_mask.copy()
    masked.mag_mask[:, 1] = False  # J missing everywhere
    masked.mag = masked.mag.copy()
    masked.mag[:, 1] = np.nan
    batch = masked.batch(np.arange(10), CPU, torch.float64)
    part = xf.star_nll(model, theta, ebv, plx, batch, parts=True)
    # Diagonal photometry: removing a band subtracts exactly its Gaussian term.
    with torch.no_grad():
        _, mag_hat = model(theta, ebv)
    resid = torch.as_tensor(data.mag[:, 1], dtype=torch.float64) - (
        mag_hat[:, 1] + 5 * torch.log10(100 / plx)
    )
    var = torch.as_tensor(data.mag_err[:, 1], dtype=torch.float64) ** 2 + 0.01**2
    term = 0.5 * (resid**2 / var + torch.log(var) + math.log(2 * math.pi))
    torch.testing.assert_close(
        part["nll_phot"], full["nll_phot"] - term, atol=1e-6, rtol=1e-6
    )
    assert torch.equal(part["dof_phot"], full["dof_phot"] - 1)
    # Missing XP coefficients with NaN values: finite NLL and gradients, dof reduced.
    masked.xp_mask = masked.xp_mask.copy()
    masked.xp = masked.xp.copy()
    masked.xp_mask[:3, 2] = False
    masked.xp[:3, 2] = np.nan
    masked.xp_mask[5, 6:] = False  # whole RP band missing
    masked.xp[5, 6:] = np.nan
    batch = masked.batch(np.arange(10), CPU, torch.float64)
    th = theta.clone().requires_grad_(True)
    out = xf.star_nll(model, th, ebv, plx, batch, parts=True)
    out["nll"].sum().backward()
    assert torch.isfinite(out["nll"]).all() and torch.isfinite(th.grad).all()
    assert out["dof_xp"][0] == 11 and out["dof_xp"][5] == 6 and out["dof_xp"][9] == 12


def test_label_noise_hook() -> None:
    model = _true_model().float()
    data, _ = _simulate(model.double(), 16)
    model = model.float()
    batch = data.batch(np.arange(16), CPU, torch.float32)
    u = torch.full((16, 2), 0.5)
    gen = torch.Generator().manual_seed(0)
    plug = xf.batch_loss(model, batch, u, xf.TrainConfig(), gen)
    marg = xf.batch_loss(
        model,
        batch,
        u,
        xf.TrainConfig(label_noise="marginal", label_noise_samples=4),
        gen,
    )
    assert plug.shape == marg.shape == (16,) and torch.isfinite(marg).all()
    batch["theta_err"] = torch.zeros_like(batch["theta_err"])
    exact = xf.batch_loss(
        model,
        batch,
        u,
        xf.TrainConfig(label_noise="marginal", label_noise_samples=4),
        gen,
    )
    torch.testing.assert_close(exact, plug)
    with pytest.raises(ValueError, match="label_noise"):
        xf.batch_loss(model, batch, u, xf.TrainConfig(label_noise="bogus"), gen)


def test_inference_with_the_true_emulator_covers_at_nominal_rate() -> None:
    model = _true_model()
    data, truth = _simulate(model, 48)
    pool = np.random.default_rng(5).uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(2000, 3))
    res = xf.infer_stars(
        model, data, BOUNDS, pool, _small_inference(48), device=CPU, log=lambda s: None
    )
    cov68, cov95 = _coverage(res, truth["theta"])
    assert 0.55 <= cov68 <= 0.82, cov68  # 144 intervals: binomial sd 0.04
    assert 0.90 <= cov95 <= 1.0, cov95
    assert np.all(res["lo68"] <= res["mle"] + 1e-9) and np.all(
        res["mle"] <= res["hi68"] + 1e-9
    )
    assert np.all(res["lo95"] <= res["lo68"] + 1e-9) and np.all(
        res["hi68"] <= res["hi95"] + 1e-9
    )
    assert np.median(np.abs(res["parallax_mle"] / truth["parallax"] - 1)) < 0.08
    assert np.all(res["E_mle"] <= 1.2 * data.ebv_sfd + 0.05 + 1e-9)
    assert res["dof_xp"].min() == 12 and np.isfinite(res["chi2_xp"]).all()


def test_training_recovers_the_emulator_and_inference_covers() -> None:
    truth_model = _true_model()
    train, _ = _simulate(truth_model, 3000, seed=2, noise=(0.3, 1.5))
    valid, _ = _simulate(truth_model, 100, seed=3, noise=(0.3, 1.5))
    test, truth = _simulate(truth_model, 40, seed=4, noise=(0.3, 1.5))
    model = xf.build_model(train, xf.ModelConfig(hidden=64, depth=2, rank=1))
    cfg = xf.TrainConfig(
        epochs=20, batch_size=256, lr=3e-3, scatter_lr=1e-2, valid_rows=100, log_every=0
    )
    hist = xf.train_forward_model(
        model, train, valid, cfg, device=CPU, log=lambda s: None
    )
    epochs = hist["epochs"]
    assert epochs[0]["scatter_frozen"] and not epochs[-1]["scatter_frozen"]
    assert epochs[-1]["train_nll"] < epochs[0]["train_nll"] - 100
    assert 0.6 < epochs[-1]["valid_chi2_xp_per_dof"] < 1.3
    frac = model.describe()["bp_fraction_median"]
    assert 0.012 < frac < 0.03, frac  # true fractional scatter 0.02
    assert hist["train_nuisance"]["ebv"].shape == (3000,)
    md = model.double().eval()
    with torch.no_grad():
        xp_hat, _ = md(torch.as_tensor(truth["theta"]), torch.as_tensor(truth["ebv"]))
    err = np.abs(xp_hat.numpy() * md.xp_scale.numpy() - truth["xp"])
    var = np.concatenate(
        [
            np.diagonal(
                xf.unpack_covariance(torch.as_tensor(test.xp_cov))[:, k].numpy(),
                axis1=1,
                axis2=2,
            )
            for k in range(2)
        ],
        1,
    )
    noise = np.sqrt(
        var + (0.02 * truth["xp"]) ** 2 + 0.005**2
    )  # known + true model scatter
    assert np.median(err / noise) < 2.0, np.median(err / noise)
    res = xf.infer_stars(
        model, test, xf.theta_bounds_from_labels(train.theta), train.theta, _small_inference(40),
        device=CPU, log=lambda s: None,
    )  # fmt: skip
    cov68, cov95 = _coverage(res, truth["theta"])
    assert 0.5 <= cov68 <= 0.88, cov68  # 120 intervals
    assert cov95 >= 0.88, cov95


def test_checkpoint_round_trip(tmp_path) -> None:
    model = _true_model().float()
    xf.save_model(model, tmp_path / "m.pt", extra={"theta_bounds": BOUNDS})
    loaded, extra = xf.load_model(tmp_path / "m.pt")
    th = torch.as_tensor([[5000.0, 3.0, -1.0]])
    e = torch.as_tensor([0.2])
    for a, b in zip(model(th, e), loaded(th, e), strict=True):
        torch.testing.assert_close(a, b)
    np.testing.assert_array_equal(extra["theta_bounds"], BOUNDS)


# ─── table and sidecar I/O ──────────────────────────────────────────────────


def _write_table(path: Path, n: int = 150, seed: int = 0) -> dict:
    """A small finetune.fits look-alike with physically plausible XP and photometry."""
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    model = _true_model(nc=xf.N_COEFF, seed=seed)
    data, truth = _simulate(model, n, seed=seed, dense=False, noise=(0.5, 2.0))
    g = rng.uniform(10.0, 17.0, n)
    f = 10 ** ((xf.G_REF - g) / 2.5)
    cols = {
        "source_id": np.arange(n, dtype=np.int64) + 10**15,
        "G": g.astype(np.float32),
    }
    for k, name in enumerate(xf.XP_COLUMNS):
        cols[name] = data.xp[:, k]
    raw_err = np.sqrt(data.xp_var.astype(np.float64)) * f[:, None]  # raw archive units
    for k, name in enumerate(xf.XP_ERROR_COLUMNS):
        cols[name] = raw_err[:, k].astype(np.float32)
    for b in xf.DEFAULT_BANDS:
        if b.name != "G":  # G is the XP normalisation magnitude written above
            cols[b.name] = (data.mag[:, 1] + rng.normal(0, 0.3, n)).astype(np.float32)
        if b.error:
            cols[b.error] = np.full(n, 0.03, np.float32)
    cols["J"][:5] = np.nan
    cols["PARALLAX"] = data.parallax.astype(np.float32)
    cols["e_parallax"] = data.parallax_err.astype(np.float32)
    cols["EBV"] = data.ebv_sfd.astype(np.float32)
    for k, p in enumerate(xf.PARAMS):
        values = truth["theta"][:, k].copy()
        values[-10:] = np.nan  # unlabelled rows
        cols[p] = values.astype(np.float32)
        cols[f"e_{p}"] = np.full(n, [50.0, 0.1, 0.05][k], np.float32)
    Table(cols).write(path, overwrite=True)
    return {"g": g, "raw_err": raw_err, "data": data}


def test_table_io_and_diagonal_raw_error_conversion(tmp_path) -> None:
    info = _write_table(tmp_path / "t.fits")
    rows = np.array([7, 3, 120, 3])
    out = xf.read_table_rows(tmp_path / "t.fits", rows, ["source_id", "G", "bp_1"])
    np.testing.assert_array_equal(out["source_id"], rows + 10**15)
    labels = xf.labels_from_table(tmp_path / "t.fits", core_rows=np.array([1, 2]))
    assert set(xf.LABEL_FIELDS) <= set(labels)
    assert set(np.unique(labels["split"])) <= {0, 1, 2}
    assert labels["has_core_feh"][[1, 2]].all() and labels["has_core_feh"].sum() == 2
    data, diag = xf.load_star_data(
        tmp_path / "t.fits",
        rows,
        labels=labels,
        zero_point="constant",
        log=lambda s: None,
    )
    expected = (
        info["raw_err"][rows] / 10 ** ((xf.G_REF - info["g"][rows]) / 2.5)[:, None]
    ) ** 2
    np.testing.assert_allclose(data.xp_var, expected, rtol=2e-6)
    np.testing.assert_allclose(data.parallax_zp, xf.L21_CONSTANT_ZP)
    expected_theta = [labels[p][7] for p in xf.PARAMS]
    np.testing.assert_allclose(data.theta[0], expected_theta, rtol=1e-6)
    ps1 = xf.read_table_rows(tmp_path / "t.fits", rows, ["G_PS1"])["G_PS1"]
    k = data.band_names.index("G_PS1")
    np.testing.assert_array_equal(data.mag_mask[:, k], ps1 >= 14.0)  # saturation limit
    j = data.band_names.index("J")
    assert not data.mag_mask[3, j]  # row 3 has NaN J
    assert diag["xp_covariance"].startswith("diagonal")
    with pytest.raises(ValueError, match="sidecar"):
        xf.load_star_data(
            tmp_path / "t.fits", rows, zero_point="sidecar", log=lambda s: None
        )


def _write_sidecar(path: Path, n: int, g: np.ndarray, seed: int = 0) -> dict:
    import h5py

    from masked_stellar_autoencoder.data import covariance_sidecar as cs

    rng = np.random.default_rng(seed)
    nc = cs.N_COEFF
    out = {}
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = cs.FORMAT_VERSION
        f["source_id"] = np.arange(n, dtype=np.int64) + 10**15
        f["row_index"] = np.arange(n, dtype=np.int64)
        f["found_xp"] = np.ones(n, bool)
        f["found_astrometry"] = np.ones(n, bool)
        for band in ("bp", "rp"):
            a = rng.normal(size=(n, nc, 3))
            corr = np.einsum("nik,njk->nij", a, a) + 3 * np.eye(nc)
            d = np.sqrt(np.einsum("nii->ni", corr))
            corr = corr / d[:, :, None] / d[:, None, :]
            f[f"{band}_coefficient_correlations"] = cs.pack_correlations(corr).astype(
                np.float32
            )
            err = rng.uniform(1.0, 5.0, (n, nc)).astype(np.float32)
            f[f"{band}_coefficient_errors"] = err
            f[f"{band}_standard_deviation"] = np.full(n, 1.2, np.float32)
            f[f"{band}_coefficients"] = rng.normal(size=(n, nc))
            out[band] = (err, corr)
        f["phot_g_mean_mag"] = g.astype(np.float32)
        f["phot_g_mean_flux_over_error"] = np.full(n, 1000.0, np.float32)
        f["parallax"] = np.linspace(0.5, 2.0, n)
        f["parallax_error"] = np.full(n, 0.05, np.float32)
        f["parallax_zpt_l21"] = np.full(n, -0.03, np.float32)
    return out


def test_sidecar_covariances_and_zero_point(tmp_path) -> None:
    info = _write_table(tmp_path / "t.fits", n=40)
    side = _write_sidecar(tmp_path / "s.h5", 40, info["g"])
    sidecar = xf.SidecarXPCovariance(tmp_path / "s.h5")
    rows = np.array([5, 1, 30])
    data, diag = xf.load_star_data(
        tmp_path / "t.fits", rows, sidecar=sidecar, log=lambda s: None
    )
    batch = data.batch(np.arange(3), CPU, torch.float64)
    f = 10 ** ((xf.G_REF - info["g"][rows].astype(np.float32).astype(np.float64)) / 2.5)
    err, corr = side["rp"]
    expected = (
        err[rows, :, None] * err[rows, None, :] * corr[rows] / (f**2)[:, None, None]
    )
    np.testing.assert_allclose(batch["xp_cov"][:, 1].numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(data.parallax, np.linspace(0.5, 2.0, 40)[rows])
    np.testing.assert_allclose(data.parallax_zp, -0.03, rtol=1e-6)
    assert diag["zero_point"] == "sidecar" and diag["sidecar_parallax_found"] == 3
    sidecar.close()


def test_cli_end_to_end(tmp_path) -> None:
    import importlib.util

    _write_table(tmp_path / "t.fits", n=150)
    np.save(tmp_path / "core.npy", np.array([0, 1, 2]))
    spec = importlib.util.spec_from_file_location(
        "run_xp_forward_model",
        Path(__file__).resolve().parents[1] / "scripts" / "run_xp_forward_model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    manifest = mod.main([
        "--table", str(tmp_path / "t.fits"), "--labels-from-table", "--core-rows", str(tmp_path / "core.npy"),
        "--output", str(out), "--epochs", "2", "--batch-size", "32", "--hidden", "16", "--depth", "1",
        "--rank", "1", "--max-heldout-stars", "3", "--valid-rows", "8", "--n-candidates", "8",
        "--bank-theta", "16", "--infer-chunk-rows", "4", "--bands", "G,J,KS,W1",
    ])  # fmt: skip
    pred = np.load(out / "xp_forward_predictions.npz")
    for key in ("source_id", "table_row", "mle", "lo68", "hi68", "lo95", "hi95", "E_mle",
                "parallax_mle", "chi2_xp", "dof_xp", "at_bound"):  # fmt: skip
        assert key in pred.files, key
    n = len(pred["table_row"])
    assert pred["mle"].shape == (n, 3) and pred["at_bound"].shape == (n, 5)
    assert (
        set(pred["subset"]) == {"heldout", "core"}
        and (pred["subset"] == "core").sum() == 3
    )
    assert np.isfinite(pred["mle"]).all() and (pred["dof_xp"] == 110).all()
    for name in (
        "model.pt",
        "history.json",
        "report.md",
        "report.json",
        "manifest.json",
        "train_nuisance.npz",
    ):
        assert (out / name).exists(), name
    assert manifest["rows"]["core"] == 3
    train = np.load(out / "train_nuisance.npz")
    assert not np.isin(train["table_row"], [0, 1, 2]).any()  # core rows never train


@pytest.mark.skipif(not SMOKE.exists(), reason="local smoke table not available")
def test_smoke_rows_cpu(tmp_path) -> None:
    labels = xf.labels_from_table(SMOKE)
    lab = np.isfinite(np.stack([labels[p] for p in xf.PARAMS], 1)).all(1)
    rows = np.flatnonzero(lab & (labels["split"] == 0))[:200]
    data, _ = xf.load_star_data(
        SMOKE, rows, labels=labels, zero_point="constant", log=lambda s: None
    )
    model = xf.build_model(data, xf.ModelConfig(hidden=16, depth=1, rank=1))
    hist = xf.train_forward_model(
        model,
        data,
        None,
        xf.TrainConfig(epochs=1, batch_size=64, log_every=0),
        device=CPU,
        log=lambda s: None,
    )
    assert math.isfinite(hist["epochs"][0]["train_nll"])
    test, _ = xf.load_star_data(SMOKE, np.flatnonzero(lab & (labels["split"] == 2))[:3], labels=labels,
                                zero_point="constant", log=lambda s: None)  # fmt: skip
    cfg = xf.InferenceConfig(
        chunk_rows=3,
        n_candidates=8,
        bank_theta=16,
        steps=10,
        profile_scan=2,
        profile_steps=5,
    )
    res = xf.infer_stars(
        model,
        test,
        xf.theta_bounds_from_labels(data.theta),
        data.theta,
        cfg,
        device=CPU,
        log=lambda s: None,
    )
    assert np.isfinite(res["mle"]).all() and np.isfinite(res["chi2_xp"]).all()
