"""Add labelled stars outside the XP pretraining sample, from Gaia DR3 columns.

Feature recovery (``recover_remaining_survey_inputs``) reads the pretraining
files, which contain only XP stars, so labelled stars without XP are never
recovered there. This step takes the candidates a feature run left
unrecovered, fetches their Gaia DR3 columns by exact ``source_id`` from the
ESA Gaia archive, and writes a new feature run with those stars made
recoverable as no-XP stars:

* inputs: G, BP, RP (encoder-scaled, -9999 elsewhere, as in the parent
  feature run); XP stays missing;
* measured parallax from Gaia (signed value and formal error);
* kept when G and at least one of BP/RP are present (the photometry rule of
  the feature lookup);
* ``input_has_xp=False``; ``real_no_xp`` = Gaia ``has_xp_continuous`` False.

The raw Gaia rows are kept as parquet (``gaia-only/``) so the full-input
build can fill proper motions, positions and errors from them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masked_stellar_autoencoder.training import master_labels as ml  # noqa: E402

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
GAIA_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "pmra_error",
    "pmdec_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "has_xp_continuous",
    "ruwe",
)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def gaia_only_inputs(frame, names, center, scale):
    """Encoder-scaled G/BP/RP inputs (-9999 elsewhere) and photometry rule."""
    idx = {n: names.index(n) for n in ("G", "BP", "RP")}
    x = np.full((len(frame), len(names)), -9999.0, dtype="f4")
    visible = np.zeros(x.shape, bool)
    for column, feature in (
        ("phot_g_mean_mag", "G"),
        ("phot_bp_mean_mag", "BP"),
        ("phot_rp_mean_mag", "RP"),
    ):
        v = pd.to_numeric(frame[column], errors="coerce").to_numpy("f8")
        ok = np.isfinite(v)
        j = idx[feature]
        x[ok, j] = ((v[ok] - center[j]) / scale[j]).astype("f4")
        visible[ok, j] = True
    keep = visible[:, idx["G"]] & (visible[:, idx["BP"]] | visible[:, idx["RP"]])
    return x, visible, keep


def fetch_gaia(ids, cache: Path, log, batch=50000):
    import pyvo
    from astropy.table import Table

    cache.mkdir(parents=True, exist_ok=True)
    service = pyvo.dal.TAPService(ml.GAIA_TAP)
    query = (
        "SELECT "
        + ", ".join("g." + c for c in GAIA_COLUMNS)
        + " FROM TAP_UPLOAD.t AS t JOIN gaiadr3.gaia_source AS g"
        " ON g.source_id = t.source_id"
    )
    parts = []
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        path = cache / f"{start:09d}.parquet"
        if not path.exists():
            result = ml._run_upload_job(
                service,
                query,
                Table(dict(source_id=chunk.astype("i8"))),
                log=log,
                label=start,
            )
            frame = pd.DataFrame(
                {
                    c: np.ma.filled(np.ma.asarray(result[c]), np.nan)
                    if result[c].dtype.kind == "f"
                    else np.asarray(result[c])
                    for c in result.colnames
                }
            )
            frame["source_id"] = frame["source_id"].astype("i8")
            frame.to_parquet(path.with_suffix(".part"), index=False)
            path.with_suffix(".part").replace(path)
            log(f"gaia batch {start}: {len(chunk)} ids, {len(frame)} rows")
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-run", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--gaia-parquet", type=Path, help="use these rows (tests/offline)")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    m = dict(status="adding_gaia_only_stars", training_performed=False)

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(m, indent=2, default=str) + "\n")
        tmp.replace(a.output / "manifest.json")

    def log(msg):
        print(msg, flush=True)

    save()
    try:
        fdir = a.root / a.feature_run
        fm = json.loads((fdir / "manifest.json").read_text())
        source = fdir / "additional-star-features.h5"
        if (
            fm.get("status") != "additional_full_raw_features_complete"
            or sha(source) != fm["output_sha256"]
        ):
            raise ValueError("Parent feature run is not complete or changed")
        with h5py.File(source) as f:
            ids = f["source_id"][:].astype("i8")
            recovered = f["feature_recoverable"][:].astype(bool)
            names = f["feature_names"][:].astype("U").tolist()
            center = f["scaler_center"][:].astype("f8")
            scale = f["scaler_scale"][:].astype("f8")
        todo = ids[~recovered]
        m.update(
            parent_feature_run=a.feature_run,
            candidates=len(ids),
            parent_recovered=int(recovered.sum()),
            gaia_lookup_ids=len(todo),
        )
        save()
        gaia_dir = a.output / "gaia-only"
        if a.gaia_parquet:
            frame = pd.read_parquet(a.gaia_parquet)
            frame = frame[frame.source_id.isin(todo)].reset_index(drop=True)
            gaia_dir.mkdir(exist_ok=True)
            frame.to_parquet(gaia_dir / "000000000.parquet", index=False)
        else:
            frame = fetch_gaia(todo, gaia_dir, log)
        frame = frame.drop_duplicates("source_id").reset_index(drop=True)
        x, visible, keep = gaia_only_inputs(frame, names, center, scale)
        plx = pd.to_numeric(frame["parallax"], errors="coerce").to_numpy("f8")
        eplx = pd.to_numeric(frame["parallax_error"], errors="coerce").to_numpy("f8")
        plx_ok = np.isfinite(plx) & np.isfinite(eplx) & (eplx > 0)
        has_xp = frame["has_xp_continuous"].astype(bool).to_numpy()
        pos = np.searchsorted(ids, frame.source_id.to_numpy("i8"))
        if np.any(np.diff(ids) <= 0) or not np.array_equal(ids[pos], frame.source_id):
            raise ValueError("Gaia rows must join candidates by exact sorted ID")
        rows = pos[keep]
        m.update(
            gaia_rows=len(frame),
            gaia_not_found=int(len(todo) - len(frame)),
            gaia_only_recoverable=int(keep.sum()),
            gaia_only_with_parallax=int((keep & plx_ok).sum()),
            gaia_only_official_has_xp=int((keep & has_xp).sum()),
        )
        save()
        output = a.output / "additional-star-features.h5"
        tmp = output.with_suffix(".part.h5")
        shutil.copyfile(source, tmp)
        with h5py.File(tmp, "a") as f:
            for key, values in (("input", x[keep]), ("visible", visible[keep])):
                data = f[key][:]
                data[rows] = values
                f[key][...] = data
            for key, value in (
                ("feature_recoverable", True),
                ("input_has_xp", False),
                ("real_xp_present", False),
            ):
                data = f[key][:]
                data[rows] = value
                f[key][...] = data
            if "real_no_xp" in f:
                data = f["real_no_xp"][:]
                data[rows] = ~has_xp[keep]
                f["real_no_xp"][...] = data
            mp = f["measured_parallax"]
            for key, values in (
                ("parallax_mas", np.where(plx_ok, plx, np.nan)[keep]),
                ("parallax_error_mas", np.where(plx_ok, eplx, np.nan)[keep]),
                ("observed", plx_ok[keep]),
            ):
                data = mp[key][:]
                data[rows] = values
                mp[key][...] = data
            gaia_only = np.zeros(len(ids), bool)
            gaia_only[rows] = True
            f["gaia_only_star"] = gaia_only
            f.attrs["gaia_only_scope"] = __doc__.splitlines()[0]
        tmp.replace(output)
        m.update(
            status="additional_full_raw_features_complete",
            output_sha256=sha(output),
            input_recovered=int(recovered.sum() + keep.sum()),
            input_with_xp=int(recovered.sum()),
            gaia_only_dir=str(gaia_dir),
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
