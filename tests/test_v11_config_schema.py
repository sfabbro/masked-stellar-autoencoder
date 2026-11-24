"""v11 config validation and input-contract schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from masked_stellar_autoencoder.v11.config import ConfigError, load_config
from masked_stellar_autoencoder.v11.filters import provenance
from masked_stellar_autoencoder.v11.schema import ContractReader, load_registry

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ config


def test_repo_configs_load():
    for name in ("pretrain-v11.yaml", "pretrain-v11-smoke.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg.model.latent_dim == 256
    cfg = load_config(ROOT / "configs/pretrain-v11.yaml")
    assert cfg.masking.xp_dropout == 0.35
    assert cfg.data.train_partitions == ["FIT"]


def test_unknown_key_fails_at_any_depth():
    with pytest.raises(ConfigError, match="model.n_layer"):
        load_config({"model": {"n_layer": 4}})
    with pytest.raises(ConfigError, match="data.synthetic.rows"):
        load_config({"data": {"source": "synthetic", "synthetic": {"rows": 3}}})
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config({"finetuning": {}})


def test_masking_ratios_only_in_masking_section():
    # the v10 bug: masking ratios under `training:` were silently ignored
    with pytest.raises(ConfigError, match="training"):
        load_config({"training": {"xp_masking_ratio": 0.9}})
    with pytest.raises(ConfigError, match="train.xp_masking_ratio"):
        load_config({"train": {"xp_masking_ratio": 0.9}})
    with pytest.raises(ConfigError, match="model.token_ratio"):
        load_config({"model": {"token_ratio": 0.3}})
    cfg = load_config(
        {
            "data": {"source": "synthetic"},
            "masking": {"xp_dropout": 0.9, "token_ratio": 0.6},
        }
    )
    assert cfg.masking.xp_dropout == 0.9 and cfg.masking.token_ratio == 0.6


def test_ranges_and_types():
    with pytest.raises(ConfigError, match="masking.token_ratio"):
        load_config({"data": {"source": "synthetic"}, "masking": {"token_ratio": 1.0}})
    with pytest.raises(ConfigError, match="must be an integer"):
        load_config({"data": {"source": "synthetic"}, "model": {"d_model": 256.0}})
    with pytest.raises(ConfigError, match="divisible"):
        load_config(
            {"data": {"source": "synthetic"}, "model": {"d_model": 100, "n_heads": 8}}
        )
    with pytest.raises(ConfigError, match="overlap"):
        load_config({"data": {"source": "synthetic", "eval_partitions": ["FIT"]}})
    with pytest.raises(ConfigError, match="trees"):
        load_config({"data": {"source": "hats"}})
    with pytest.raises(ConfigError, match="modes"):
        load_config(
            {"data": {"source": "synthetic"}, "filter_integral": {"modes": ["sedd"]}}
        )


# ------------------------------------------------------------------ registry


def test_registry_structure():
    reg = load_registry()
    assert len(set(reg.names)) == len(reg)
    assert reg.xp_indices.size == 110
    assert reg["xp_bp_00"].xp_band == "bp" and reg["xp_rp_54"].xp_coefficient == 54
    kinds = [b.kind for b in reg.bands]
    assert kinds[-4:] == ["cond"] * 4  # conditioning last
    assert not any(b.reconstruct for b in reg.bands if b.kind == "cond")
    # UV -> mid-IR coverage of the full-wavelength master
    for name in (
        "galex_fuv",
        "ls_s_z",
        "ls_n_g",
        "vhs_ks",
        "catwise_w2",
        "allwise_w4",
        "splus_j0660",
    ):
        assert reg[name].kind == "mag"


def test_wavelengths_match_svo_pivots():
    reg = load_registry()
    prov = provenance()["filters"]
    for b in reg.bands:
        if b.svo:
            assert b.svo in prov, b.svo
            assert abs(b.wavelength_nm / prov[b.svo]["pivot_nm"] - 1) < 0.01, b.name


def test_svo_provenance_complete():
    prov = provenance()
    assert prov["filters"]
    for fid, meta in prov["filters"].items():
        assert len(meta["sha256_votable"]) == 64 and meta["url"].endswith(fid)
        assert (
            ROOT / "src/masked_stellar_autoencoder/v11/filters" / meta["file"]
        ).is_file()


# ------------------------------------------------------------------ contract reading


def _v1a_table():
    return pa.table(
        {
            "source_id": pa.array([1, 2, 3, 4], pa.int64()),
            "phot_g_mean_mag": [15.0, 16.0, None, 12.0],
            "phot_g_mean_flux_over_error": [1000.0, 500.0, None, 0.0],
            "phot_bp_mean_mag": [15.5, None, 17.0, 12.4],
            "phot_bp_mean_flux_over_error": [300.0, None, 50.0, 2000.0],
            "j_m": [13.0, 14.0, 14.5, 10.0],
            "j_msigcom": [0.02, 0.03, 0.9, 0.02],
            "h_m": [12.6, 13.6, 14.0, 9.6],
            "h_msigcom": [0.02, 0.03, 0.05, 0.02],
            "ks_m": [12.5, 13.5, 13.9, 9.5],
            "ks_msigcom": [0.02, 0.03, 0.05, 0.02],
            "ph_qual": ["AAA", "AUA", None, "CCC"],
            "g_mean_psf_mag": [15.2, 16.1, 13.5, 12.0],
            "g_mean_psf_mag_error": [0.01, 0.02, 0.01, 0.01],
            "quality_flag": pa.array([4, 1, 8, 16], pa.int64()),
        }
    )


def test_v1a_reader_values_quality_and_absence():
    reg = load_registry()
    r = ContractReader(reg, "gaia_keyed_v1a")
    out = r.read(_v1a_table())
    g = out.values[:, reg.index("gaia_g")]
    ge = out.errors[:, reg.index("gaia_g")]
    np.testing.assert_allclose(g[:2], [15.0, 16.0])
    np.testing.assert_allclose(ge[0], 1.0857362 / 1000, rtol=1e-5)
    assert np.isnan(g[2])  # absent value
    assert np.isnan(ge[3]) and g[3] == 12.0  # flux_over_error 0 -> unknown error
    j = out.values[:, reg.index("tmass_j")]
    assert np.isfinite(j[0]) and np.isfinite(j[1])
    assert np.isnan(j[2])  # error 0.9 > max_error_mag and null ph_qual
    h = out.values[:, reg.index("tmass_h")]
    assert np.isnan(h[1])  # ph_qual 'U' in H
    assert np.isnan(h[3]) is np.False_  # 'C' accepted
    ps1 = out.values[:, reg.index("ps1_g")]
    assert np.isfinite(ps1[0]) and np.isnan(ps1[1])  # quality_flag bits 4|8 required
    assert np.isnan(ps1[2])  # 13.5 brighter than the 14.0 limit
    assert np.isnan(ps1[3])  # flag 16 only
    # bands the table does not have stay absent for every row
    assert np.isnan(out.values[:, reg.index("catwise_w1")]).all()
    assert out.xp_coefficients is None


def test_master_reader_flux_tags_where_and_coalesce():
    reg = load_registry()
    r = ContractReader(reg, "gaia_master_v2", survey_tags={"ps1": "ps1_dr2_hats"})
    t = pa.table(
        {
            "source_id": pa.array([1, 2, 3], pa.int64()),
            "FLUX_G_ls_dr11_sweep": [100.0, 50.0, -1.0],
            "FLUX_IVAR_G_ls_dr11_sweep": [1.0, 4.0, 1.0],
            "PHOTSYS_ls_dr11_sweep": ["S", "N", "S"],
            "ls_dr11_sweep_quality": ["good", "good", "good"],
            "yapermag3_ukidss_dr11_las": [None, 15.0, None],
            "yapermag3err_ukidss_dr11_las": [None, 0.02, None],
            "ukidss_dr11_las_quality": ["no_match", "good", "no_match"],
            "yapermag3_ukidss_dr11_gcs": [14.0, 16.0, 14.0],
            "yapermag3err_ukidss_dr11_gcs": [0.03, 0.03, 0.03],
            "ukidss_dr11_gcs_quality": ["good", "good", "ambiguous"],
            "gMeanPSFMag_ps1_dr2_hats": [15.0, 15.0, 15.0],
            "gMeanPSFMagErr_ps1_dr2_hats": [0.01, 0.01, 0.01],
            "ps1_dr2_hats_quality": ["good", "ambiguous", "good"],
            "qualityFlag_ps1_dr2_hats": pa.array([4, 4, 4], pa.int64()),
            "nDetections_ps1_dr2_hats": pa.array([5, 5, 1], pa.int64()),
        }
    )
    out = r.read(t)
    s_g = out.values[:, reg.index("ls_s_g")]
    n_g = out.values[:, reg.index("ls_n_g")]
    np.testing.assert_allclose(s_g[0], 22.5 - 2.5 * np.log10(100.0), rtol=1e-6)
    np.testing.assert_allclose(
        out.errors[0, reg.index("ls_s_g")], 1.0857362 / 100.0, rtol=1e-5
    )
    assert np.isnan(s_g[1]) and np.isfinite(n_g[1])  # PHOTSYS routes the band
    assert np.isnan(s_g[2])  # negative flux
    y = out.values[:, reg.index("ukidss_y")]
    np.testing.assert_allclose(y[:2], [14.0, 15.0])  # GCS fills LAS gaps; LAS first
    assert np.isnan(y[2])  # GCS ambiguous
    ps1 = out.values[:, reg.index("ps1_g")]
    assert np.isfinite(ps1[0]) and np.isnan(ps1[1]) and np.isnan(ps1[2])


def test_unverified_mapping_refused():
    reg = load_registry()
    t = pa.table(
        {
            "source_id": [1],
            "MAG_AUTO_jplus_dr3": [[15.0] * 12],
            "MAG_ERR_AUTO_jplus_dr3": [[0.01] * 12],
            "jplus_dr3_quality": ["good"],
        }
    )
    with pytest.raises(ValueError, match="unverified"):
        ContractReader(reg, "gaia_master_v2").read(t)
    out = ContractReader(reg, "gaia_master_v2", allow_unverified=True).read(t)
    assert out.values[0, reg.index("jplus_j0660")] == 15.0


def test_missing_quality_column_fails_closed():
    reg = load_registry()
    t = _v1a_table().drop_columns(["quality_flag"])
    with pytest.raises(KeyError, match="quality_flag"):
        ContractReader(reg, "gaia_keyed_v1a").read(t)


def test_contract_yaml_rejects_unknown_keys(tmp_path):
    src = (ROOT / "src/masked_stellar_autoencoder/v11/contract.yaml").read_text()
    bad = src.replace(
        "{name: galex_fuv, survey: galex,",
        "{name: galex_fuv, surveyy: x, survey: galex,",
    )
    p = tmp_path / "contract.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="unknown keys"):
        load_registry(p)
