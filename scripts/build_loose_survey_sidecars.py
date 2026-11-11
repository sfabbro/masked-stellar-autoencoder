"""Extended label groups for the large surveys with failed-fit cuts only.

The staged groups used conservative, explicitly provisional cuts (S/N >= 30
for LAMOST LRS/MRS and BOSSNet; for DESI: S/N_R >= 10, [Fe/H] error < 0.5 dex
and the published [Fe/H]-correction box 4200 <= Teff < 6600 K, 1.8 <= logg <
5.5). This builder reads the same native catalogues and keeps every converged
fit:

* native failure flags (DESI SUCCESS, RVS_WARN, RR_SPECTYPE=STAR, PRIMARY;
  BOSSNet flag_bad) and exact positive Gaia DR3 IDs (DESI also <= 1");
* finite values with positive quoted errors, the same broad domain limits;
* S/N >= ``--snr-min`` (default 10) instead of 30;
* one row per star: highest S/N, then lowest row.

Each output group holds ONLY stars that have no record in the corresponding
staged group (read from ``--parent-run``), so no star is labelled twice by the
same pipeline and the stricter group keeps its map. DESI stars inside the
correction box get the published correction; stars outside it go to a
separate group with raw [Fe/H] (not extrapolated), so each has its own map.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from masked_stellar_autoencoder.training.multisurvey_overlay import (  # noqa: E402
    write_observation_sidecar,
)

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
TABLES = [
    dict(
        name="lamost_dr12v2_lrs_lasp_ext",
        parent_group="lamost_dr12v2_lrs_lasp",
        path=ROOT
        + "multisurvey_lamost_dr12_lrs_source_direct_v2/dr12_lrs_stellar.fits.gz",
        columns=["teff", "logg", "feh"],
        errors=["teff_err", "logg_err", "feh_err"],
        id="gaia_source_id",
        snr="snrg",
        release="LAMOST DR12 v2.0 LRS",
        pipeline="LASP LRS",
        dependency_id="lamost-dr12v2-lrs-lasp",
    ),
    dict(
        name="lamost_dr12v2_mrs_lasp_ext",
        parent_group="lamost_dr12v2_mrs_lasp",
        path=ROOT
        + "multisurvey_lamost_dr12_mrs_source_direct_v2/dr12_mrs_stellar.fits.gz",
        columns=["teff_lasp", "logg_lasp", "feh_lasp"],
        errors=["teff_lasp_err", "logg_lasp_err", "feh_lasp_err"],
        id="gaia_source_id",
        snr="snr",
        release="LAMOST DR12 v2.0 MRS",
        pipeline="LASP MRS",
        dependency_id="lamost-dr12v2-mrs-lasp",
    ),
    dict(
        name="sdss_dr20_bossnet_081_ext",
        parent_group="sdss_dr20_bossnet_081",
        path=ROOT
        + "decision_dr20_legacy_source_audit_20261106_v1/dr20_bossnet.fits.gz",
        columns=["teff", "logg", "fe_h"],
        errors=["e_teff", "e_logg", "e_fe_h"],
        id="gaia_dr3_source_id",
        snr="snr",
        flag="flag_bad",
        release="SDSS DR20 Astra 0.8.1 BOSSNet",
        pipeline="BOSSNet SDSS-V BOSS",
        dependency_id="sdss-dr20-bossnet-081",
    ),
]
DESI = dict(
    path=ROOT + "desi_collection_v1/mwsall-pix-iron.fits",
    parent_group="desi",
    release="DESI DR1 MWS iron v1.0",
    pipeline="RVSpecFit (RVTAB)",
    dependency_id="desi-dr1-rvspecfit",
)
DOMAIN = ((2500.0, 10000.0), (-1.0, 6.5), (-5.0, 1.0))


def exact_ids(raw):
    raw = np.asarray(raw)
    if raw.dtype.kind in "iu":
        return raw.astype("i8")
    if raw.dtype.kind not in "SU":
        raise ValueError("Native Gaia IDs must be integers or decimal strings")
    text = np.char.strip(raw.astype("U"))
    ok = np.char.isdecimal(text) & (np.char.str_len(text) <= 19)
    return np.array(
        [
            int(t) if g and int(t) <= np.iinfo("i8").max else 0
            for t, g in zip(text, ok, strict=True)
        ],
        dtype="i8",
    )


def observed_mask(val, err):
    obs = np.isfinite(val) & np.isfinite(err) & (err > 0) & (val != -9999)
    for j, (lo, hi) in enumerate(DOMAIN):
        obs[:, j] &= (val[:, j] >= lo) & (val[:, j] <= hi)
    return obs


def one_per_star(block):
    """Highest S/N, then lowest native row, per Gaia ID."""
    order = np.lexsort((block["source_row"], -block["snr"], block["source_id"]))
    sid = block["source_id"][order]
    order = order[np.r_[True, np.diff(sid) != 0]] if len(order) else order
    return {k: v[order] for k, v in block.items()}


def drop_existing(block, existing):
    keep = ~np.isin(block["source_id"], existing)
    return {k: v[keep] for k, v in block.items()}, int((~keep).sum())


def table_block(spec, snr_min, rows_per_chunk=100000):
    pieces = []
    with fits.open(spec["path"], memmap=True) as f:
        table = f[1].data
        names = {n.lower(): n for n in f[1].columns.names}
        for start in range(0, len(table), rows_per_chunk):
            b = table[start : start + rows_per_chunk]
            ids = exact_ids(b[names[spec["id"]]])
            val = np.column_stack([b[names[k]] for k in spec["columns"]]).astype("f8")
            err = np.column_stack([b[names[k]] for k in spec["errors"]]).astype("f8")
            snr = np.asarray(b[names[spec["snr"]]], dtype="f8")
            obs = observed_mask(val, err)
            keep = (ids > 0) & np.isfinite(snr) & (snr >= snr_min) & obs.any(1)
            if spec.get("flag"):
                keep &= ~np.asarray(b[names[spec["flag"]]], dtype=bool)
            pieces.append(
                dict(
                    source_id=ids[keep],
                    source_row=np.arange(start, start + len(b), dtype="i8")[keep],
                    value=val[keep],
                    error=err[keep],
                    observed=obs[keep],
                    snr=snr[keep],
                )
            )
        rows = len(table)
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}, rows


def desi_blocks(path, rows_per_chunk=65536):
    """(inside-correction-box, outside-box) blocks with failed-fit cuts only."""
    from desi_stellar_labels import (
        check_target_alignment,
        corrected_feh,
        separation_arcsec,
    )

    inside, outside = [], []
    with fits.open(path, memmap=True) as hdul:
        rv, gaia, fiber, sp = (
            hdul[n].data for n in ["RVTAB", "GAIA", "FIBERMAP", "SPTAB"]
        )
        for start in range(0, len(rv), rows_per_chunk):
            end = min(start + rows_per_chunk, len(rv))
            r, g = rv[start:end], gaia[start:end]
            check_target_alignment(r, fiber[start:end])
            check_target_alignment(r, sp[start:end])
            ids = np.asarray(g["SOURCE_ID"], dtype="i8")
            corrected, valid, _ = corrected_feh(r["FEH"], r["TEFF"], r["LOGG"])
            sep = separation_arcsec(r["TARGET_RA"], r["TARGET_DEC"], g["RA"], g["DEC"])
            base = (ids > 0) & np.isfinite(sep) & (sep <= 1.0)
            base &= np.asarray(r["SUCCESS"], bool) & np.asarray(r["PRIMARY"], bool)
            base &= np.asarray(r["RVS_WARN"]) == 0
            base &= np.char.strip(np.asarray(r["RR_SPECTYPE"]).astype(str)) == "STAR"
            snr = np.asarray(r["SN_R"], dtype="f8")
            err = np.column_stack([r["TEFF_ERR"], r["LOGG_ERR"], r["FEH_ERR"]]).astype(
                "f8"
            )
            for box, out in ((True, inside), (False, outside)):
                feh = (
                    np.where(valid, corrected, np.asarray(r["FEH"], "f8"))
                    if box
                    else np.asarray(r["FEH"], "f8")
                )
                val = np.column_stack([r["TEFF"], r["LOGG"], feh]).astype("f8")
                obs = observed_mask(val, err)
                keep = base & (valid == box) & obs.any(1)
                out.append(
                    dict(
                        source_id=ids[keep],
                        source_row=np.arange(start, end, dtype="i8")[keep],
                        value=val[keep],
                        error=err[keep],
                        observed=obs[keep],
                        snr=np.nan_to_num(snr[keep], nan=-1.0),
                    )
                )
    cat = lambda ps: {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}  # noqa: E731
    return cat(inside), cat(outside), len(rv)


def registry_entry(release, pipeline, dependency_id, fields, policy):
    return dict(
        release=release,
        pipeline=pipeline,
        dependency_id=dependency_id,
        targets={
            t: dict(quantity=t, unit=u, field=f)
            for t, u, f in zip(
                ("teff", "logg", "fe_h"),
                ("K", "dex(cm/s2)", "dex"),
                fields,
                strict=True,
            )
        },
        quality_policy=policy,
        disjoint_from_staged_group=True,
        training_enabled=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent-run", required=True, help="source run holding the staged groups"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snr-min", type=float, default=10.0)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--only", nargs="*", help="subset of output group names")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="building_loose_groups",
        snr_min=a.snr_min,
        groups={},
        training_performed=False,
    )

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(m, indent=2, default=str) + "\n")
        tmp.replace(a.output / "manifest.json")

    save()
    try:
        pm = json.loads((Path(a.root) / a.parent_run / "manifest.json").read_text())
        existing = {}
        with h5py.File(pm["prepared_path"]) as f:
            ids = f["source_id"][:].astype("i8")
            for g in [t["parent_group"] for t in TABLES] + [DESI["parent_group"]]:
                existing[g] = np.sort(ids[f["native_observations"][g]["star_row"][:]])
        blocks, registry = {}, {}
        policy = f"failed-fit cuts only; S/N >= {a.snr_min}; one row per star (max S/N)"
        for spec in TABLES:
            if a.only and spec["name"] not in a.only:
                continue
            block, rows = table_block(spec, a.snr_min)
            block = one_per_star(block)
            block, overlap = drop_existing(block, existing[spec["parent_group"]])
            blocks[spec["name"]] = block
            registry[spec["name"]] = registry_entry(
                spec["release"],
                spec["pipeline"],
                spec["dependency_id"],
                spec["columns"],
                policy + (" ; flag_bad false" if spec.get("flag") else ""),
            )
            m["groups"][spec["name"]] = dict(
                native_rows=rows,
                stars=len(block["source_id"]),
                already_in_staged_group=overlap,
            )
            save()
        if not a.only or any(n.startswith("desi") for n in a.only):
            inside, outside, rows = desi_blocks(DESI["path"])
            for name, block, fields, note in (
                (
                    "desi_rvs_ext_corrected",
                    inside,
                    ("TEFF", "LOGG", "FEH_corrected"),
                    "published [Fe/H] correction (inside its validity box)",
                ),
                (
                    "desi_rvs_ext_uncorrected",
                    outside,
                    ("TEFF", "LOGG", "FEH"),
                    "raw [Fe/H]; outside the correction box, never extrapolated",
                ),
            ):
                block = one_per_star(block)
                block, overlap = drop_existing(block, existing[DESI["parent_group"]])
                if name == "desi_rvs_ext_uncorrected":
                    block, more = drop_existing(
                        block, blocks["desi_rvs_ext_corrected"]["source_id"]
                    )
                    overlap += more
                blocks[name] = block
                registry[name] = registry_entry(
                    DESI["release"],
                    DESI["pipeline"],
                    DESI["dependency_id"],
                    fields,
                    "SUCCESS, PRIMARY, RVS_WARN=0, STAR, <=1 arcsec; " + note,
                )
                m["groups"][name] = dict(
                    native_rows=rows,
                    stars=len(block["source_id"]),
                    already_in_staged_group=overlap,
                )
                save()
        for name, block in blocks.items():
            block["error_known"] = (
                block["observed"] & np.isfinite(block["error"]) & (block["error"] > 0)
            )
            m["groups"][name].update(
                observed_per_target=block["observed"].sum(0).tolist(),
                fe_h_below_minus2=int(
                    (block["observed"][:, 2] & (block["value"][:, 2] < -2)).sum()
                ),
            )
        sidecar = a.output / "loose-survey-sidecar.h5"
        receipt = write_observation_sidecar(
            sidecar,
            blocks,
            registry,
            provenance=dict(
                builder="scripts/build_loose_survey_sidecars.py",
                parent_run=a.parent_run,
                snr_min=a.snr_min,
            ),
        )
        m.update(
            status="complete",
            sidecar=receipt,
            additional_sources=[
                dict(name=n, path=receipt["path"], sha256=receipt["sha256"])
                for n in blocks
            ],
            new_stars_total=int(
                len(
                    np.unique(np.concatenate([b["source_id"] for b in blocks.values()]))
                )
            ),
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
