"""End-to-end S1 pipeline on a synthetic table, tiny model and records (CPU)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler

from masked_stellar_autoencoder.models.model import make_model
from masked_stellar_autoencoder.training.anchored_survey_map import (
    FULL,
    IDENTITY,
    OFFSET,
    AnchoredSurveyMaps,
    StudentTHead,
)
from masked_stellar_autoencoder.training.common_scale_labels import LABEL_FIELDS

ROOT = Path(__file__).resolve().parents[1]
COLS = ["G", "BP", "RP", "bp_1", "PARALLAX", "EBV"]
ERRS = [None, None, None, "bpe_1", "e_parallax", None]
SURVEYS = ["anchor_seis", "big", "saga_mw"]


def _write_inputs(tmp, n=600, seed=0):
    from astropy.io import fits

    rng = np.random.default_rng(seed)
    teff = rng.uniform(4000, 6500, n)
    logg = rng.uniform(1.0, 4.8, n)
    feh = rng.uniform(-3.0, 0.3, n)
    G = rng.uniform(10, 17.5, n)
    plx = rng.uniform(0.2, 3.0, n)
    table = dict(
        source_id=np.arange(n, dtype="i8") + 10**15, G=G, BP=G + 0.5, RP=G - 0.6,
        bp_1=(teff - 5000) / 500 + 0.3 * feh + rng.normal(0, 0.1, n),
        bpe_1=np.full(n, 0.1), PARALLAX=plx, e_parallax=np.full(n, 0.1),
        EBV=rng.uniform(0, 0.1, n), teff=teff, logg=logg, fe_h=feh,
        e_fe_h=np.full(n, 0.1),
    )  # fmt: skip
    cols = [fits.Column(name=k, format="K" if k == "source_id" else "D", array=v)
            for k, v in table.items()]  # fmt: skip
    fits.BinTableHDU.from_columns(cols).writeto(tmp / "finetune.fits")
    rows, surveys, values = [], [], []
    seis = rng.choice(n, n // 3, replace=False)
    rows.append(seis)
    surveys.append(np.zeros(len(seis), int))
    values.append(np.column_stack([np.full(len(seis), np.nan), logg[seis],
                                   np.full(len(seis), np.nan)]))  # fmt: skip
    rows.append(np.arange(n))
    surveys.append(np.ones(n, int))
    values.append(np.column_stack([teff + 30, logg + 0.05, feh + 0.05]))
    core = rng.choice(n, n // 3, replace=False)
    rows.append(core)
    surveys.append(np.full(len(core), 2))
    values.append(np.column_stack([np.full(len(core), np.nan),
                                   np.full(len(core), np.nan), feh[core]]))  # fmt: skip
    val = np.concatenate(values)
    rec = dict(
        row=np.concatenate(rows), survey=np.concatenate(surveys).astype("i4"),
        value=np.column_stack([val, np.full(len(val), np.nan)]).astype("f4"),
        error=np.tile([50.0, 0.05, 0.05, np.nan], (len(val), 1)).astype("f4"),
        survey_names=np.array(SURVEYS), survey_is_anchor=np.array([True, False, False]),
        parameters=np.array(["teff", "logg", "fe_h", "alpha"]),
    )  # fmt: skip
    np.savez(tmp / "records.npz", **rec)
    config = dict(
        data=dict(feature_cols=COLS, error_cols=ERRS, recon_cols=COLS[:5],
                  ft_datafile=str(tmp / "finetune.fits")),
        model=dict(layer_dims=[32, 16], pt_activ_func="elu", rtdl_embed=4,
                   norm="layer", decoder_dims=[16, 32]),
        training={}, finetuning=dict(active="relu", metal_poor={}),
        preprocessing=dict(astrometry_input_policy="snr_clipped",
                           astrometry_snr_cap=10.0, parallax_target_space="log10_mas"),
        multisurvey=dict(records=str(tmp / "records.npz"), fe_h_core=["saga_"],
                         censor={"big": {"fe_h": -2.8}}),
    )  # fmt: skip
    (tmp / "config.yaml").write_text(yaml.safe_dump(config))
    torch.manual_seed(seed)
    model = make_model(len(COLS), [32, 16], 5, "elu", 4, "layer", decoder_dims=[16, 32])
    names = ["teff", "logg", "fe_h", "parallax"]
    head = StudentTHead(16, len(names), torch.nn.ReLU(), 5.0)
    scalers = [StandardScaler().fit(v[:, None])
               for v in (teff, logg, feh, np.log10(plx))]  # fmt: skip
    tier = np.array([[IDENTITY] * 3, [FULL] * 3, [OFFSET] * 3])
    core = np.zeros((3, 3), bool)
    core[2, 2] = True
    maps = AnchoredSurveyMaps(tier, core, np.ones((3, 3)))
    with torch.no_grad():
        maps.linear.uniform_(-0.05, 0.05)
        maps.bias.uniform_(-0.1, 0.1)
    X = np.column_stack([table[c] for c in COLS])
    state = dict(
        autoencoder_state_dict=model.state_dict(),
        prediction_head_state_dict=head.state_dict(), head="student_t", nu=5.0,
        label_names=names, label_scalers=scalers,
        featurescaler=RobustScaler().fit(X), survey_maps_state_dict=maps.state_dict(),
        survey_names=SURVEYS, map_tier=tier, map_core=core, map_counts=np.ones((3, 3)),
        map_kind="anchored", error_fill=None, drop_input_features=[],
        n_components=1, full_covariance=True, min_scale=1e-3,
    )  # fmt: skip
    torch.save(state, tmp / "ckpt.pth")


def test_s1_pipeline_end_to_end(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "s1_likelihood_head", ROOT / "scripts" / "s1_likelihood_head.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "s1", "--config", str(tmp_path / "config.yaml"),
        "--checkpoint", str(tmp_path / "ckpt.pth"), "--output", str(out),
        "--device", "cpu", "--epochs", "3", "--n-candidates", "16",
        "--min-train", "10", "--row-block", "128",
    ])  # fmt: skip
    mod.main()
    lab = np.load(out / "labels.npz")
    assert set(LABEL_FIELDS) <= set(lab.files)
    assert lab["split"].tolist().count(-1) == 0 and len(lab["teff"]) == 600
    # The censored big-survey [Fe/H] (< -2.8) leaves only the core record.
    low = lab["fe_h_ref"] < -2.85
    assert (lab["n_feh"][low] == 1).all()
    pred = np.load(out / "predictions.npz")
    assert np.load(out / "latent.npy").shape == (600, 16)
    assert np.isfinite(pred["post_median"]).all()
    s1 = np.load(out / "s1_predictions.npz")
    for k in ("source_id", "table_row", "mle", "lo68", "hi68", "lo95", "hi95",
              "neyman68_lo", "neyman68_hi", "neyman95_lo", "neyman95_hi", "smax",
              "at_bound", "post_median", "post_q16", "post_q84"):  # fmt: skip
        assert k in s1.files, k
    assert s1["mle"].shape[1] == 3 and set(s1["split"]) <= {0, 1, 2}
    # Rows with no score inside the label support are left out (NaN).
    prof = (s1["split"] == 2) & np.isfinite(s1["lo68"]).all(1)
    assert prof.sum() >= 0.5 * (s1["split"] == 2).sum()
    assert np.all(s1["lo95"][prof] <= s1["lo68"][prof] + 1e-6)
    assert np.isfinite(s1["neyman68_lo"][prof]).all()  # 'all' arm: calibrated
    assert s1["in_neyman68_param"].shape == s1["mle"].shape
    assert np.all(s1["neyman95_hi"][prof] >= s1["neyman68_hi"][prof] - 1e-6)
    core = np.load(out / "s1_predictions_core.npz")
    assert np.isnan(core["neyman68_lo"]).all()  # too few valid core stars
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["decision"]) == {"core", "all"}
    assert "mle_all" in m["frozen_core"] and "logg_anchor" in m["anchors"]
    report = (out / "report.md").read_text()
    assert "## Decision" in report and "ADOPT" in report
    assert math.isfinite(m["runs"]["all"]["t_mle_s"])


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "s1_likelihood_head", ROOT / "scripts" / "s1_likelihood_head.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_sidecar(path, source_id, rng):
    import h5py

    zpt = rng.uniform(-0.04, 0.0, len(source_id)).astype("f4")
    zpt[::7] = np.nan  # e.g. 2-parameter solutions: constant zero point
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "1"
        f["source_id"] = source_id
        f["parallax_zpt_l21"] = zpt
    return zpt


def test_s1_parallax_nuisance_ensemble_variant(tmp_path, monkeypatch):
    """Variant: latents without PARALLAX, theta' with log10 parallax, its
    measurement term, a 2-head ensemble; labels and baseline arms reused."""
    from astropy.io import fits

    from masked_stellar_autoencoder.training.likelihood_head import load_head

    _write_inputs(tmp_path)
    mod = _load_script()
    common = ["--config", str(tmp_path / "config.yaml"), "--checkpoint",
              str(tmp_path / "ckpt.pth"), "--device", "cpu", "--epochs", "3",
              "--n-candidates", "64", "--min-train", "10", "--row-block", "128",
              "--min-calib", "20", "--profile-rows", "none"]  # fmt: skip
    base = tmp_path / "base"
    monkeypatch.setattr(sys, "argv", ["s1", *common, "--output", str(base),
                                      "--arms", "all"])  # fmt: skip
    mod.main()
    with fits.open(tmp_path / "finetune.fits") as h:
        sid = np.asarray(h[1].data["source_id"])
        plx = np.asarray(h[1].data["PARALLAX"], "f8")
    zpt = _write_sidecar(tmp_path / "sidecar.h5", sid, np.random.default_rng(1))
    out = tmp_path / "plx"
    monkeypatch.setattr(sys, "argv", [
        "s1", *common, "--output", str(out), "--parallax-nuisance", "--ensemble", "2",
        "--labels", str(base / "labels.npz"), "--baseline", str(base),
        "--sidecar", str(tmp_path / "sidecar.h5"),
    ])  # fmt: skip
    mod.main()
    # Latents: PARALLAX hidden; the posterior head keeps the full input.
    assert json.loads((out / "latent_meta.json").read_text())["hidden_inputs"] == [
        "PARALLAX"
    ]
    assert not np.allclose(np.load(out / "latent.npy"), np.load(base / "latent.npy"))
    np.testing.assert_allclose(
        np.load(out / "predictions.npz")["post_median"],
        np.load(base / "predictions.npz")["post_median"], rtol=1e-5,
    )  # fmt: skip
    par = np.load(out / "parallax.npz")
    expect = np.where(np.isfinite(zpt), zpt, -0.017)
    np.testing.assert_allclose(par["zero_point"], expect, rtol=1e-6)
    np.testing.assert_allclose(par["parallax_corrected"], plx - expect, rtol=1e-6)
    heads = [load_head(out / n)[0] for n in ("head_plx.pt", "head_plx_1.pt")]
    assert all(h.config["n_params"] == 4 for h in heads)
    assert not torch.equal(heads[0].theta_scale, torch.ones(4))
    inf = np.load(out / "infer_plx.npz")
    assert inf["mle"].shape[1] == 3 and inf["mle_search"].shape[1] == 4
    assert int(inf["n_members"]) == 2
    found = np.isfinite(inf["smax"])
    assert found.mean() > 0.5
    v = inf["mle_parallax"][found]
    assert np.all((v > 10**-1.5 * 0.999) & (v < 10**2.5 * 1.001))
    # No profile intervals here (--profile-rows none; the 4-D profiles are
    # tested on the analytic toy); coverage comes from the label statistics.
    assert not inf["profiled"].any() and np.isnan(inf["lo68"]).all()
    test = (np.load(out / "labels.npz")["split"][inf["table_row"]] == 2) & found
    assert np.isfinite(inf["lam_label_param"][test]).mean() > 0.5
    single = np.load(out / "infer_plx_single.npz")
    assert single["mle"].shape == inf["mle"].shape and single["stat_rows"].any()
    s1 = np.load(out / "s1_predictions_plx.npz")
    assert "mle_parallax" in s1.files and "parallax_pull" in s1.files
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["decision"]) == {"all", "plx"}  # baseline arm + variant
    assert {"posterior", "mle_all", "mle_plx", "mle_plx_single"} <= set(m["precision"])
    assert {"all", "plx", "plx_single"} <= set(m["edges"])
    assert "parallax_window_edge" in m["edges"]["plx"]["held-out"]
    assert "support_edge" in m["edges"]["all"]["held-out"]
    for k in ("neyman68", "wilks68_joint", "wilks68_fe_h", "neyman68_fe_h"):
        assert k in m["coverage"]["plx"], k
    assert "wilks68_joint" in m["coverage"]["plx_single"]
    assert "neyman68" not in m["coverage"]["plx_single"]
    assert m["runs"]["plx"]["members"] == 2
    assert len(m["runs"]["plx"]["training_members"]) == 2
    assert m["runs"]["all"]["source"] == str(base)
    report = (out / "report.md").read_text()
    for s in ("## Precision summary", "## MLE at the edges", "Arm `plx`", "Wilks 68"):
        assert s in report, s


def test_hidden_parallax_latents_ignore_the_parallax(tmp_path):
    from astropy.io import fits

    from masked_stellar_autoencoder.training.frozen_latents import encode_table

    _write_inputs(tmp_path, n=64)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    state = torch.load(tmp_path / "ckpt.pth", weights_only=False)
    with fits.open(tmp_path / "finetune.fits") as h:
        data = h[1].data.copy()
    data["PARALLAX"] = 3.0 * data["PARALLAX"] + 0.5
    fits.BinTableHDU(data=data).writeto(tmp_path / "moved.fits")
    for name in ("finetune", "moved"):
        encode_table(config, state, tmp_path / f"{name}.fits", tmp_path / name,
                     hide=("PARALLAX",), log=lambda m: None)  # fmt: skip
    a, b = tmp_path / "finetune", tmp_path / "moved"
    np.testing.assert_array_equal(np.load(a / "latent.npy"), np.load(b / "latent.npy"))
    assert json.loads((a / "latent_meta.json").read_text())["hidden_inputs"] == [
        "PARALLAX"
    ]
    # The posterior head (the comparator) still sees and conditions on parallax.
    pa, pb = np.load(a / "predictions.npz"), np.load(b / "predictions.npz")
    assert not np.allclose(pa["post_median"], pb["post_median"])
