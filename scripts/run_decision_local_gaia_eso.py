"""Stream our full raw Gaia input coordinates locally; collect Gaia-ESO candidates."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from xmatcher import CrossMatch, MatchRequest, MatchSpec, SideOverrides

RAW = "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"
GES = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/rave_dr6_cnn_recovery_sources_20261028_v1/gaia_eso_dr51_raw_source_20261029_v1/gaia_eso_dr51_raw.h5"


def directions(ra, dec, pmra=None, pmdec=None, dt=0.0):
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    r = np.column_stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    if dt and pmra is not None:
        east = np.column_stack([-np.sin(a), np.cos(a), np.zeros(len(a))])
        north = np.column_stack(
            [-np.cos(a) * np.sin(d), -np.sin(a) * np.sin(d), np.cos(d)]
        )
        r += (
            dt
            * np.deg2rad(1 / 3600000)
            * (pmra[:, None] * east + pmdec[:, None] * north)
        )
        r /= np.linalg.norm(r, axis=1)[:, None]
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with open(GES, "rb") as f:
        if (
            hashlib.file_digest(f, "sha256").hexdigest()
            != "9932d63fc89278425ace2ee9f44507990d0efc7ece0750028c27c28eaf0992e4"
        ):
            raise ValueError("Gaia-ESO source differs")
    with h5py.File(GES) as f:
        gra = f["raw/RA"][:]
        gdec = f["raw/DECLINATION"][:]
    tree = cKDTree(directions(gra, gdec))
    radius = 2 * np.sin(np.deg2rad(10 / 3600) / 2)
    before = Path(RAW).stat()
    m = dict(
        status="auditing_raw_coordinates",
        raw_path=RAW,
        raw_bytes=before.st_size,
        raw_mtime_ns=before.st_mtime_ns,
        training_performed=False,
        identity_promoted=False,
        external_data_upload=False,
        rows_scanned=0,
        raw_rows=0,
        valid_position_rows=0,
        candidate_rows=0,
        shards={},
        assumptions="RA/DEC degrees ICRS and pmra=mu_alpha*cos(delta), pmdec mas/yr; Gaia reference epoch assumed 2016, to validate against ESA; Gaia-ESO source epochs unknown; candidates only",
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(m, indent=2) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    pieces = []
    try:
        with h5py.File(RAW) as f:
            datasets = [k for k in f if isinstance(f[k], h5py.Dataset)]
            m["raw_rows"] = sum(len(f[k]) for k in datasets)
            for key in datasets:
                d = f[key]
                required = ["source_id", "RA", "DEC", "pmra", "pmdec"]
                m["shards"][key] = dict(
                    rows=len(d),
                    fields=list(d.dtype.names or ()),
                    attributes={k: str(v)[:1000] for k, v in d.attrs.items()},
                )
                if not set(required).issubset(d.dtype.names or ()):
                    raise ValueError("Raw coordinate/motion fields absent: " + key)
            m["status"] = "scanning_local_full_input_set"
            save()
            for key in datasets:
                d = f[key]
                for start in range(0, len(d), 250000):
                    b = d.fields(required)[start : start + 250000]
                    sid = np.asarray(b["source_id"])
                    ra = np.asarray(b["RA"], dtype="f8")
                    dec = np.asarray(b["DEC"], dtype="f8")
                    pmra = np.asarray(b["pmra"], dtype="f8")
                    pmdec = np.asarray(b["pmdec"], dtype="f8")
                    if sid.dtype.kind not in "iu":
                        raise ValueError("IDs are not exact integers")
                    good = (
                        (sid > 0)
                        & np.isfinite(ra)
                        & np.isfinite(dec)
                        & (ra >= 0)
                        & (ra < 360)
                        & (dec >= -90)
                        & (dec <= 90)
                    )
                    known = (
                        np.isfinite(pmra)
                        & np.isfinite(pmdec)
                        & (pmra != -9999)
                        & (pmdec != -9999)
                    )
                    take = np.flatnonzero(good)
                    selected = np.zeros(len(b), bool)
                    for epoch in [2000.0, 2010.0, 2016.0]:
                        use = take if epoch == 2016 else take[known[take]]
                        if not len(use):
                            continue
                        xyz = directions(
                            ra[use], dec[use], pmra[use], pmdec[use], epoch - 2016.0
                        )
                        distance, _ = tree.query(
                            xyz, k=1, distance_upper_bound=radius, workers=4
                        )
                        selected[use[np.isfinite(distance)]] = True
                    ii = np.flatnonzero(selected)
                    if len(ii):
                        part = pl.DataFrame(
                            dict(
                                source_id=sid[ii].astype("i8"),
                                ra=ra[ii],
                                dec=dec[ii],
                                pmra=np.where(known[ii], pmra[ii], np.nan),
                                pmdec=np.where(known[ii], pmdec[ii], np.nan),
                                raw_row=start + ii,
                                raw_shard=[key] * len(ii),
                            )
                        )
                        part.write_parquet(
                            a.output / (key + "-" + str(start) + "-candidates.parquet")
                        )
                        pieces.append(part)
                    m["rows_scanned"] += len(b)
                    m["valid_position_rows"] += int(good.sum())
                    m["candidate_rows"] += len(ii)
                    save()
        after = Path(RAW).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Raw source changed during scan")
        if not pieces:
            raise ValueError("No candidates; inspect coordinate units/coverage")
        raw = pl.concat(pieces)
        unique = raw.select(["source_id", "ra", "dec", "pmra", "pmdec"]).unique()
        conflicts = unique.group_by("source_id").len().filter(pl.col("len") > 1)
        finite = unique.join(
            conflicts.select("source_id"), on="source_id", how="anti"
        ).filter(pl.col("pmra").is_finite() & pl.col("pmdec").is_finite())
        finite.write_parquet(a.output / "local-gaia-unique-motion-candidates.parquet")
        targets = pl.DataFrame(
            dict(row_id=np.arange(len(gra), dtype="i8"), ra=gra, dec=gdec)
        )
        m["epoch_matches"] = {}
        for epoch in [2000.0, 2010.0, 2016.0]:
            pairs = CrossMatch().crossmatch_request(
                MatchRequest(
                    cat1=targets,
                    cat2=finite,
                    engine="fast",
                    spec=MatchSpec(
                        radius_arcsec=1.0,
                        find="all",
                        target_epoch=epoch,
                        fallback_policy="error",
                    ),
                    side1=SideOverrides(id_column="row_id", epoch=epoch, frame="icrs"),
                    side2=SideOverrides(
                        id_column="source_id",
                        epoch=2016.0,
                        pm_ra_column="pmra",
                        pm_dec_column="pmdec",
                        frame="icrs",
                    ),
                )
            )
            pairs.write_parquet(
                a.output / ("local-epoch-" + str(int(epoch)) + "-pairs.parquet")
            )
            m["epoch_matches"][str(epoch)] = dict(
                pairs=pairs.height, matched_source_rows=pairs["row_id"].n_unique()
            )
        m.update(
            status="local_full_input_candidate_audit_complete",
            distinct_candidate_ids=raw["source_id"].n_unique(),
            conflicting_astrometry_ids=conflicts.height,
            scope="Scanned all shards of our raw input set, not a claim to contain every Gaia catalogue source; unit/epoch assumptions and native unknown epochs require comparison with ESA before identity promotion",
        )
    except Exception as e:
        m.update(
            status="local_candidate_audit_failed",
            error_type=type(e).__name__,
            error=str(e),
        )
    save()
    print(json.dumps({k: v for k, v in m.items() if k != "shards"}), flush=True)


if __name__ == "__main__":
    main()
