"""Approved Gaia-ESO→ESA candidate upload and xmatcher epoch sensitivity."""

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pyvo
import requests
from astropy.table import Table
from xmatcher import CrossMatch, MatchRequest, MatchSpec, SideOverrides

SOURCE = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/rave_dr6_cnn_recovery_sources_20261028_v1/gaia_eso_dr51_raw_source_20261029_v1/gaia_eso_dr51_raw.h5"
SOURCE_SHA = "9932d63fc89278425ace2ee9f44507990d0efc7ece0750028c27c28eaf0992e4"


class BoundedSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", (20, 180))
        return super().request(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with open(SOURCE, "rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SOURCE_SHA:
            raise ValueError("Native source differs")
    with h5py.File(SOURCE) as f:
        ra = f["raw/RA"][:]
        dec = f["raw/DECLINATION"][:]
    if len(ra) != 114916:
        raise ValueError("Source row count differs")
    rows = np.arange(len(ra), dtype="i8")
    source = Table(dict(row_id=rows, ra=ra, dec=dec))
    m = dict(
        status="submitting_approved_query",
        training_performed=False,
        identity_promoted=False,
        source_rows=len(rows),
        source_sha256=SOURCE_SHA,
        search_radius_arcsec=10,
        match_radius_arcsec=1,
        source_epoch="unknown_per_row",
        assumed_epochs=[2000.0, 2010.0, 2016.0],
        authorization="User explicitly approved sending these 114916 positions and row identifiers to ESA on 2026-11-06",
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(m, indent=2) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    query = "SELECT s.row_id,g.source_id,g.ra,g.dec,g.ref_epoch,g.pmra,g.pmdec,g.phot_g_mean_mag FROM TAP_UPLOAD.ges AS s JOIN gaiadr3.gaia_source AS g ON 1=CONTAINS(POINT('ICRS',g.ra,g.dec),CIRCLE('ICRS',s.ra,s.dec,0.002777777777777778))"
    (a.output / "query.adql").write_text(query + "\n")
    try:
        service = pyvo.dal.TAPService(
            "https://gea.esac.esa.int/tap-server/tap", session=BoundedSession()
        )
        tables = []
        m["archive_jobs"] = []
        for start in range(0, len(source), 5000):
            stop = min(start + 5000, len(source))
            entry = dict(
                source_start=start, source_stop=stop, status="submission_pending"
            )
            m["archive_jobs"].append(entry)
            save()
            if start == 0:
                prior = json.loads(Path("resume-job.json").read_text())
                if (
                    prior["source_sha256"] != SOURCE_SHA
                    or prior["source_start"] != 0
                    or prior["source_stop"] != stop
                ):
                    raise ValueError("Existing archive upload binding differs")
                job = pyvo.dal.AsyncTAPJob(
                    prior["url"], session=service._session, delete=False
                )
            else:
                job = service.submit_job(
                    query, uploads={"ges": source[start:stop]}, maxrec=1000000
                )
            entry.update(url=job.url, status="submitted")
            save()
            if job.phase == "PENDING":
                job.run()
            elif job.phase not in ["QUEUED", "EXECUTING", "COMPLETED"]:
                raise RuntimeError("Existing archive job terminal phase " + job.phase)
            deadline = time.monotonic() + 900
            while job.phase not in ["COMPLETED", "ERROR", "ABORTED"]:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Known archive job exceeded wait budget; reconcile its URL, do not resubmit"
                    )
                time.sleep(3)
            entry["phase"] = job.phase
            save()
            if job.phase != "COMPLETED":
                job.raise_if_error()
                raise RuntimeError("Known batch archive phase " + job.phase)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                part = job.fetch_result().to_table()
            if len(part) >= 1000000 or any(
                "overflow" in str(x.message).lower() for x in messages
            ):
                raise ValueError("Possibly truncated batch")
            part.write(
                a.output / ("esa-batch-" + str(start) + ".parquet"), format="parquet"
            )
            tables.append(part)
            entry.update(status="complete", candidate_pairs=len(part))
            save()
        from astropy.table import vstack

        table = vstack(tables, metadata_conflicts="silent")
        if np.asarray(table["source_id"]).dtype.kind not in "iu":
            raise ValueError("Noninteger Gaia IDs")
        raw = pl.DataFrame(
            {
                name: np.asarray(table[name].filled(np.nan))
                if table[name].dtype.kind == "f" and hasattr(table[name], "filled")
                else np.asarray(table[name])
                for name in table.colnames
            }
        )
        raw.write_parquet(a.output / "esa-candidate-pairs.parquet")
        gaia = raw.select(
            ["source_id", "ra", "dec", "ref_epoch", "pmra", "pmdec"]
        ).unique(subset=["source_id"])
        finite = gaia.filter(
            pl.col("pmra").is_finite()
            & pl.col("pmdec").is_finite()
            & pl.col("ref_epoch").is_finite()
        )
        native = pl.DataFrame(dict(row_id=rows, ra=ra, dec=dec))
        unique = []
        m["epoch_counts"] = {}
        for epoch in m["assumed_epochs"]:
            pairs = CrossMatch().crossmatch_request(
                MatchRequest(
                    cat1=native,
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
                        epoch_column="ref_epoch",
                        pm_ra_column="pmra",
                        pm_dec_column="pmdec",
                        frame="icrs",
                    ),
                )
            )
            pairs.write_parquet(
                a.output / ("epoch-" + str(int(epoch)) + "-pairs.parquet")
            )
            counts = pairs.group_by("row_id").agg(
                pl.col("source_id").n_unique().alias("count")
            )
            good = (
                pairs.join(
                    counts.filter(pl.col("count") == 1).select("row_id"), on="row_id"
                )
                .select(["row_id", "source_id"])
                .unique()
            )
            unique.append(good.rename({"source_id": "gaia_" + str(int(epoch))}))
            m["epoch_counts"][str(epoch)] = dict(
                pairs=pairs.height, unique_rows=good.height
            )
            save()
        stable = unique[0]
        for frame in unique[1:]:
            stable = stable.join(frame, on="row_id", how="inner")
        stable = stable.filter(
            (pl.col("gaia_2000") == pl.col("gaia_2010"))
            & (pl.col("gaia_2000") == pl.col("gaia_2016"))
        )
        stable.write_parquet(
            a.output / "epoch-stable-candidates-not-training-identities.parquet"
        )
        m.update(
            status="full_source_candidate_epoch_audit_complete",
            candidate_pairs=len(table),
            unique_gaia_candidates=gaia.height,
            finite_motion_candidates=finite.height,
            epoch_stable_rows=stable.height,
            scope="All source positions attempted; finite search radius and unknown source epochs limit completeness and identity certainty",
        )
    except Exception as e:
        m.update(
            status="candidate_query_or_epoch_audit_failed",
            error_type=type(e).__name__,
            error=str(e),
            resubmit_allowed=False,
        )
    save()
    print(json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
