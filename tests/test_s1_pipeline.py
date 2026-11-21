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
