"""Rebuild a six-target source with full 140-slot encoder inputs and head context.

Takes a complete expanded source run (e.g. ``expanded_six_source_20261111_v6``)
and writes a new run with the same stars, roles, labels and native
observation groups. Only the encoder inputs change: every slot is filled from
the staged catalogues instead of only XP and five magnitudes. It also adds a
per-star metadata context for the prediction heads.

Input precedence per star:
1. original pretraining rows (all 140 slots), from the photometry-stage parquet
   or, for stars outside that stage, from feature-recovery row pointers;
2. Gaia DR3 columns (G, BP, RP, parallax, pm, RA/Dec) from the Gaia stage,
   filling slots that are still missing (no-XP stars).

Stored inputs use NaN for missing plus ``visible``; the encoder token is
applied at load time (profile ``missing_encoding: nan``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masked_stellar_autoencoder.training import full_input_assembly as fia  # noqa: E402

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
ORIGINAL = Path(
    "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"
)
CONTEXT_RAW = ("PARALLAX", "EBV", "pmra", "pmdec", "RA", "DEC")
CONTEXT_ERRORS = ("e_parallax", "e_pmra", "e_pmdec")
GAIA_ERRORS = {
    "parallax_error": "e_parallax",
    "pmra_error": "e_pmra",
    "pmdec_error": "e_pmdec",
}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows_for(ids, query):
    query = np.asarray(query)
    if query.dtype.kind not in "iu":
        raise ValueError("Exact integer Gaia identifiers required")
    pos = np.searchsorted(ids, query)
    hit = (pos < len(ids)) & (query > 0)
    hit[hit] &= ids[pos[hit]] == query[hit]
    return np.flatnonzero(hit), pos[hit]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", required=True)
    p.add_argument(
        "--photometry-stage", default="full_gaia_photometry_stage_20261108_v1"
    )
    p.add_argument("--gaia-stage", default="full_gaia_gaia_stage_20261108_v2")
    p.add_argument(
        "--feature-run", help="feature-recovery run with original row pointers"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--original", type=Path, default=ORIGINAL)
    a = p.parse_args()
    root, original_path = a.root, a.original
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="building_full_inputs",
        training_source_ready=False,
        training_performed=False,
    )

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(m, indent=2, default=str) + "\n")
        tmp.replace(a.output / "manifest.json")

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    save()
    try:
        pm = json.loads((root / a.parent_run / "manifest.json").read_text())
        if pm.get("status") != "expanded_six_source_complete" or not pm.get(
            "training_source_ready"
        ):
            raise ValueError("Parent source is not complete")
        parent = Path(pm["prepared_path"])
        if sha(parent) != pm["prepared_sha256"]:
            raise ValueError("Parent prepared.h5 changed")
        with h5py.File(parent) as old:
            ids = old["source_id"][:].astype("i8")
            names = old["feature_names"][:].astype("U").tolist()
            center = old["scaler_center"][:].astype("f8")
            scale = old["scaler_scale"][:].astype("f8")
            roles = old["role"][:].astype("S8")
            eligible = old["primary_input_eligible"][:].astype(bool)
            if np.any(np.diff(ids) <= 0) or len(names) != 140:
                raise ValueError("Sorted unique IDs and the 140-slot layout required")
            n = len(ids)
            log(f"parent {a.parent_run}: {n} stars")
            inputs = np.empty((n, len(names)), dtype="f4")
            for start in range(0, n, 262144):
                block = old["input"][start : start + 262144].astype("f4")
                block[~np.isfinite(block) | (block == fia.MISSING_TOKEN)] = np.nan
                inputs[start : start + len(block)] = block
            mp = old["measured_parallax"]
            meas_plx = mp["parallax_mas"][:].astype("f8")
            meas_err = mp["parallax_error_mas"][:].astype("f8")
            meas_ok = mp["observed"][:].astype(bool)
        before = np.isfinite(inputs)
        ctx = {k: np.full(n, np.nan) for k in CONTEXT_RAW + CONTEXT_ERRORS}
        origin = np.zeros(n, dtype="i1")  # 0 parent only, 1 original row, 2 gaia only
        want = list(dict.fromkeys(["source_id", *names, *CONTEXT_ERRORS]))

        def absorb_original(rows, frame):
            raw = fia.raw_frame_to_matrix(frame, names)
            inputs[rows] = fia.merge_rows(
                inputs[rows], fia.standardize(raw, center, scale), overwrite=True
            )
            for k in CONTEXT_RAW + CONTEXT_ERRORS:
                if k in frame:
                    ctx[k][rows] = fia.clean(frame[k].to_numpy())
            origin[rows] = 1

        stage = root / a.photometry_stage / "raw-observations"
        files = sorted(stage.glob("*.parquet"))
        m.update(photometry_stage_files=len(files))
        for i, path in enumerate(files):
            schema = pq.read_schema(path).names
            frame = pq.read_table(
                path, columns=[c for c in want if c in schema]
            ).to_pandas()
            hit, rows = rows_for(ids, frame["source_id"].to_numpy())
            if len(hit):
                absorb_original(rows, frame.iloc[hit].reset_index(drop=True))
            if i % 200 == 0:
                m.update(
                    photometry_files_done=i + 1, original_rows=int((origin == 1).sum())
                )
                save()
                log(f"photometry stage {i + 1}/{len(files)}")
        if a.feature_run:
            fpath = root / a.feature_run / "additional-star-features.h5"
            with h5py.File(fpath) as ff, h5py.File(original_path) as orig:
                fid = ff["source_id"][:].astype("i8")
                shard = ff["original_shard"][:]
                orow = ff["original_row"][:]
                keys = json.loads(ff.attrs["raw_shards_json"])
                hit, rows = rows_for(ids, fid)
                todo = hit[(origin[rows] == 0) & (orow[hit] >= 0)]
                for k, key in enumerate(keys):
                    sel = todo[shard[todo] == k]
                    if not len(sel):
                        continue
                    order = np.argsort(orow[sel])
                    sel = sel[order]
                    d = orig[key]
                    fields = [c for c in want if c in (d.dtype.names or ())]
                    records = d.fields(fields)[orow[sel]]
                    frame = pd.DataFrame({c: records[c] for c in fields})
                    _, rr = rows_for(ids, fid[sel])
                    absorb_original(rr, frame)
                m.update(feature_run_original_rows=int(len(todo)))
                save()
        gaia = root / a.gaia_stage / "labelled-gaia"
        gfiles = sorted(gaia.glob("*.parquet"))
        gaia_cols = list(fia.GAIA_TO_FEATURE) + list(GAIA_ERRORS) + ["source_id"]
        for i, path in enumerate(gfiles):
            schema = pq.read_schema(path).names
            frame = pq.read_table(
                path, columns=[c for c in gaia_cols if c in schema]
            ).to_pandas()
            hit, rows = rows_for(ids, frame["source_id"].to_numpy())
            if not len(hit):
                continue
            frame = frame.iloc[hit].reset_index(drop=True)
            raw = fia.gaia_frame_to_matrix(frame, names)
            inputs[rows] = fia.merge_rows(
                inputs[rows], fia.standardize(raw, center, scale), overwrite=False
            )
            for gcol, feat in [
                ("parallax", "PARALLAX"),
                ("pmra", "pmra"),
                ("pmdec", "pmdec"),
                ("ra", "RA"),
                ("dec", "DEC"),
                *GAIA_ERRORS.items(),
            ]:
                if gcol in frame:
                    values = fia.clean(frame[gcol].to_numpy())
                    keep = np.isnan(ctx[feat][rows])
                    ctx[feat][rows[keep]] = values[keep]
            origin[rows[origin[rows] == 0]] = 2
            if i % 500 == 0:
                log(f"gaia stage {i + 1}/{len(gfiles)}")
        # The admitted measured parallax is the head's parallax context too.
        ctx["PARALLAX"][meas_ok] = meas_plx[meas_ok]
        ctx["e_parallax"][meas_ok] = meas_err[meas_ok]
        context = fia.context_raw(
            parallax=ctx["PARALLAX"],
            parallax_error=ctx["e_parallax"],
            ebv=ctx["EBV"],
            pmra=ctx["pmra"],
            pmdec=ctx["pmdec"],
            pmra_error=ctx["e_pmra"],
            pmdec_error=ctx["e_pmdec"],
            ra=ctx["RA"],
            dec=ctx["DEC"],
        )
        fit_rows = np.flatnonzero((roles == b"fit") & eligible)
        c_center, c_scale = fia.context_scaler(context, fit_rows)
        context_scaled = ((context - c_center) / c_scale).astype("f4")
        visible = np.isfinite(inputs)
        gained = visible & ~before
        m.update(
            stars=n,
            input_origin=dict(
                parent_only=int((origin == 0).sum()),
                original_rows=int((origin == 1).sum()),
                gaia_only=int((origin == 2).sum()),
            ),
            visible_fit_counts={
                name: int(visible[fit_rows, j].sum()) for j, name in enumerate(names)
            },
            newly_visible_slots_fit={
                name: int(gained[fit_rows, j].sum()) for j, name in enumerate(names)
            },
            context_names=list(fia.CONTEXT_NAMES),
            context_visible_fit=np.isfinite(context[fit_rows]).sum(0).tolist(),
            context_center=c_center.tolist(),
            context_scale=c_scale.tolist(),
        )
        save()
        output = a.output / "prepared.h5"
        with h5py.File(parent) as old, h5py.File(output, "x") as f:
            for key, value in old.attrs.items():
                f.attrs[key] = value
            f.attrs["complete"] = False
            for key in old:
                if key != "input":
                    old.copy(old[key], f, name=key)
            f.create_dataset(
                "input",
                data=inputs,
                chunks=(min(8192, n), len(names)),
                compression="lzf",
            )
            f.create_dataset(
                "visible",
                data=visible,
                chunks=(min(8192, n), len(names)),
                compression="lzf",
            )
            f["input_origin"] = origin
            g = f.create_group("context")
            g.create_dataset("values", data=context_scaled, compression="lzf")
            g.create_dataset(
                "visible", data=np.isfinite(context_scaled), compression="lzf"
            )
            g.attrs["names"] = list(fia.CONTEXT_NAMES)
            g.attrs["center"] = c_center
            g.attrs["scale"] = c_scale
            f.attrs["input_missing_encoding"] = "nan"
            f.attrs["complete"] = True
        j = names.index("PARALLAX")
        result = dict(pm)
        result.update(
            status="expanded_six_source_complete",
            training_source_ready=True,
            prepared_path=str(output),
            prepared_sha256=sha(output),
            parent_run=a.parent_run,
            full_inputs=dict(
                photometry_stage=a.photometry_stage,
                gaia_stage=a.gaia_stage,
                feature_run=a.feature_run,
                **{
                    k: m[k]
                    for k in (
                        "input_origin",
                        "context_names",
                        "context_center",
                        "context_scale",
                    )
                },
            ),
        )
        profiles = dict(result.get("feature_profiles", {}))
        profiles["full_inputs_measured_parallax"] = dict(
            ready=True,
            feature_index=j,
            override_dataset="measured_parallax/input_value",
            availability_dataset="measured_parallax/observed",
            missing_encoding="nan",
            context_dataset="context",
        )
        result["feature_profiles"] = profiles
        m.clear()
        m.update(result)
        save()
        log("complete")
    except Exception as exc:
        m.update(status="failed", error=repr(exc), training_source_ready=False)
        save()
        raise


if __name__ == "__main__":
    main()
