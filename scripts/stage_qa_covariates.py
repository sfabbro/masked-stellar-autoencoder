"""Gaia DR3 covariates (and GSP-Phot) of the stars an evaluation will score.

usage: stage_qa_covariates.py --ids-from FILE.npz [...] --out covariates.parquet
                              [--gspphot gspphot.parquet] [--partition test]

Collects the ``source_id`` of every given file (predictions in any layout the
evaluator reads, or any npz with ``source_id``, ``pair1``/``pair2``,
``member``), keeps those in the evaluated ``msa-split-v1`` partition (plus
QA-reserved objects), and queries the Gaia archive for ``phot_g_mean_mag``,
``bp_rp``, ``l``, ``b``, ``ruwe``, ``has_xp_continuous`` and the GSP-Phot
Teff, log g, [M/H] and distance (async TAP jobs with uploaded ids). Writes
the covariates for ``evaluate_model.py --covariates`` and, with
``--gspphot``, the GSP-Phot rows in the competitor layout. Where the Gaia
HATS tree is mounted the same columns can be read from it instead.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from masked_stellar_autoencoder.data.msa_split import FREEZE, MsaSplit
from masked_stellar_autoencoder.data.twomass_gaia import TAP_ASYNC, _votable_ids
from masked_stellar_autoencoder.qa.competitors import _gspphot_frame

COLUMNS = ("phot_g_mean_mag", "bp_rp", "l", "b", "ruwe", "has_xp_continuous",
           "teff_gspphot", "logg_gspphot", "mh_gspphot", "distance_gspphot")  # fmt: skip
PART = {"test": 3, "dev": 1}


def ids_of(path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        keys = [
            k
            for k in ("source_id", "pair1", "pair2", "member", "primary_id")
            if k in z.files
        ]
        return np.concatenate([np.asarray(z[k], np.int64).ravel() for k in keys])


def query(ids, poll=5.0) -> pd.DataFrame:
    import requests

    q = ("SELECT g.source_id, " + ", ".join(f"g.{c}" for c in COLUMNS)
         + " FROM tap_upload.ids AS u JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id")  # fmt: skip
    r = requests.post(
        TAP_ASYNC,
        data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", PHASE="RUN", UPLOAD="ids,param:t1", QUERY=q),
        files=dict(t1=("ids.vot", _votable_ids(ids))), allow_redirects=False, timeout=600,
    )  # fmt: skip
    job = r.headers["Location"]
    while (phase := requests.get(job + "/phase", timeout=60).text.strip()) not in (
        "COMPLETED", "ERROR", "ABORTED"):  # fmt: skip
        time.sleep(poll)
    if phase != "COMPLETED":
        raise RuntimeError(f"{job}: {phase}")
    return pd.read_csv(
        io.StringIO(requests.get(job + "/results/result", timeout=900).text)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ids-from", action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gspphot", type=Path)
    p.add_argument("--split-dir", default=str(FREEZE))
    p.add_argument("--partition", choices=sorted(PART), default="test")
    p.add_argument("--chunk", type=int, default=50_000)
    a = p.parse_args()
    ids = np.unique(np.concatenate([ids_of(f) for f in a.ids_from]))
    ids = ids[ids > 0]
    part = MsaSplit.from_freeze(a.split_dir).partition(ids)
    ids = ids[(part == PART[a.partition]) | (part == 4)]
    print(f"{len(ids)} ids in {a.partition.upper()} or QA-reserved", flush=True)
    parts = []
    k, chunk = 0, a.chunk
    while k < len(ids):
        try:
            parts.append(query(ids[k : k + chunk]))
        except (RuntimeError, OSError) as exc:  # archive limits: retry smaller
            if chunk <= 1000:
                raise
            chunk //= 2
            print(f"  retry with {chunk} ids ({exc})", flush=True)
            continue
        k += chunk
        print(f"  {min(k, len(ids))}/{len(ids)}", flush=True)
    t = pd.concat(parts, ignore_index=True)
    cov = t[
        ["source_id", "phot_g_mean_mag", "bp_rp", "l", "b", "ruwe", "has_xp_continuous"]
    ]
    cov = cov.rename(columns=dict(phot_g_mean_mag="G", has_xp_continuous="has_xp"))
    cov["has_xp"] = cov["has_xp"].astype(str).str.lower().isin(("true", "1", "t"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(a.out, index=False)
    if a.gspphot:
        _gspphot_frame(t).to_parquet(a.gspphot, index=False)
    print(f"wrote {a.out} ({len(cov)} rows)")


if __name__ == "__main__":
    main()
