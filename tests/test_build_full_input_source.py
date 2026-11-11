import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_full_input_source.py"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _names():
    xp = [f"{a}_{j}" for a in ("bp", "rp") for j in range(1, 56)]
    extra = ["U_SMSS", "V_SMSS", "G_SMSS", "R_SMSS", "I_SMSS", "Z_SMSS"]
    extra += ["U_SDSS", "G_SDSS", "R_SDSS", "I_SDSS", "Z_SDSS"]
    extra += ["G_PS1", "R_PS1", "I_PS1", "Z_PS1", "Y_PS1", "J", "H", "KS"]
    return ["W1", "W2", "G", "BP", "RP", *xp, *extra] + [
        "PARALLAX",
        "EBV",
        "pmra",
        "pmdec",
        "RA",
        "DEC",
    ]


@pytest.fixture
def staged(tmp_path):
    names = _names()
    root = tmp_path / "root"
    ids = np.array([101, 202, 303], dtype="i8")  # 101 XP+all, 202 no-XP gaia, 303 none
    parent = root / "parent"
    parent.mkdir(parents=True)
    old_input = np.full((3, len(names)), -9999.0, dtype="f4")
    old_input[0, :5] = 1.0  # five mags only, as in v5
    old_input[0, 5:115] = 0.5
    with h5py.File(parent / "prepared.h5", "w") as f:
        f.attrs["schema"] = "expanded-sparse-six-target-source-v1"
        f.attrs["complete"] = True
        f["source_id"] = ids
        f["feature_names"] = np.array(names, dtype="S16")
        f["scaler_center"] = np.zeros(len(names))
        f["scaler_scale"] = np.ones(len(names))
        f["role"] = np.array([b"fit", b"fit", b"tune"], dtype="S8")
        f["primary_input_eligible"] = np.ones(3, bool)
        f["input"] = old_input
        g = f.create_group("measured_parallax")
        g["parallax_mas"] = np.array([1.5, np.nan, np.nan])
        g["parallax_error_mas"] = np.array([0.05, np.nan, np.nan])
        g["observed"] = np.array([True, False, False])
        g["input_value"] = np.array([1.5, -9999, -9999], dtype="f4")
        f.create_group("native_observations/apogee")["star_row"] = np.array([0])
    (parent / "manifest.json").write_text(
        json.dumps(
            dict(
                status="expanded_six_source_complete",
                training_source_ready=True,
                prepared_path=str(parent / "prepared.h5"),
                prepared_sha256=_sha(parent / "prepared.h5"),
                source_registry=["apogee"],
                feature_profiles={},
            )
        )
    )
    phot = root / "phot" / "raw-observations"
    phot.mkdir(parents=True)
    row = {n: 2.0 for n in names}
    row.update(source_id=101, J=12.0, PARALLAX=1.4, EBV=0.05, e_pmra=0.02, e_pmdec=0.03)
    pd.DataFrame([row]).to_parquet(phot / "0000-0000000000.parquet")
    gaia = root / "gaia" / "labelled-gaia"
    gaia.mkdir(parents=True)
    pd.DataFrame(
        dict(
            source_id=[101, 202],
            phot_g_mean_mag=[9.0, 18.0],
            phot_bp_mean_mag=[9.5, 18.5],
            phot_rp_mean_mag=[8.5, 17.5],
            parallax=[1.4, 0.2],
            parallax_error=[0.05, 0.3],
            pmra=[2.0, -1.0],
            pmdec=[2.0, 4.0],
            pmra_error=[0.02, 0.2],
            pmdec_error=[0.03, 0.2],
            ra=[10.0, 200.0],
            dec=[-5.0, 30.0],
        )
    ).to_parquet(gaia / "00000-0000000000.parquet")
    return root, names


def test_full_inputs_fill_all_slots_keep_labels_and_store_nan(staged, tmp_path):
    root, names = staged
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--parent-run",
            "parent",
            "--photometry-stage",
            "phot",
            "--gaia-stage",
            "gaia",
            "--output",
            str(out),
        ],
        check=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "expanded_six_source_complete"
    profile = manifest["feature_profiles"]["full_inputs_measured_parallax"]
    assert profile["missing_encoding"] == "nan"
    agreement = manifest["full_inputs"]["gaia_vs_original_standardized"]
    assert agreement["G"]["compared"] == 1 and agreement["G"]["differing_gt_1e4"] == 1
    assert agreement["PARALLAX"]["differing_gt_1e4"] == 0  # 1.4 in both
    assert manifest["full_inputs"]["input_origin"] == dict(
        parent_only=1, original_rows=1, gaia_only=1
    )
    with h5py.File(out / "prepared.h5") as f:
        x, vis = f["input"][:], f["visible"][:]
        np.testing.assert_array_equal(vis, np.isfinite(x))
        assert not (x == -9999).any()  # no sentinel in storage
        j = names.index("J")
        assert x[0, j] == 12.0 and vis[0].all()  # original row fills extra bands
        g, bp = names.index("G"), names.index("BP")
        assert x[1, g] == 18.0 and x[1, bp] == 18.5  # no-XP star from Gaia
        assert not vis[1, 5:115].any()  # its XP stays missing
        assert not vis[2].any()  # star absent everywhere keeps nothing
        assert "native_observations/apogee" in f  # labels copied
        ctx, cvis = f["context/values"][:], f["context/visible"][:]
        names_ctx = list(f["context"].attrs["names"])
        assert ctx.shape == (3, 10)
        assert cvis[0, names_ctx.index("ebv")] and not cvis[1, names_ctx.index("ebv")]
        assert cvis[1, names_ctx.index("pmra")] and not cvis[2].any()
