"""Corrected 2MASS sidecars for tables built from the Andrae file.

usage: build_twomass_sidecar.py --table finetune.fits [--table unlabelled.npz ...]
                                --output-dir DIR --twomass archive|mask|TABLE

The J/H/KS (and E_*) columns copied from ``sslset-realmags-full-052725.h5``
belong to other stars (``data/twomass_gaia.py``: the "Gaia-keyed"
``adql_matches/twomass.fits`` they came from is mis-keyed and is refused).
For each table (a FITS table with ``source_id``, ``G`` and ``KS``, or an npz
in the unlabelled layout: ``source_id``, ``fields``, ``features``) this looks
up the union of their ids once (``archive``: Gaia archive designation join,
exact; ``mask``: all missing; or a designation-keyed table) and writes, per table,
``<stem>_twomass_gaia.npz`` in the table's row order (``source_id`` + J, H,
KS, E_J, E_H, E_KS; NaN without a usable match). Pass it as
``data.extra_columns`` (fine-tune table: finetune_data replaces the columns)
or ``multisurvey.unlabelled.extra_columns`` (unlabelled pool). Also writes
``twomass_check.json``: the Ks-G check before and after per table, the scan
counts, and fails (exit 1) when a corrected table does not pass.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from masked_stellar_autoencoder.data.twomass_gaia import (
    ks_g_check,
    resolve,
    write_sidecar,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_table(path: Path) -> dict:
    """source_id, G and the current KS of a FITS table or unlabelled npz."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            fields = [str(f) for f in z["fields"]]
            feats = z["features"]
            return dict(
                source_id=np.asarray(z["source_id"], np.int64),
                G=np.asarray(feats[:, fields.index("G")], np.float64),
                KS=np.asarray(feats[:, fields.index("KS")], np.float64),
            )
    from astropy.io import fits

    with fits.open(path, memmap=True) as h:
        d = h[1].data
        return {c: np.asarray(d[c]) for c in ("source_id", "G", "KS")}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--twomass", required=True, help="archive, mask or a table path")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {t: read_table(t) for t in a.table}
    for t, d in tables.items():
        log(f"{t}: {len(d['source_id'])} rows")
    ids = np.unique(np.concatenate([d["source_id"] for d in tables.values()]))
    cols, info = resolve(ids, a.twomass, log=log)
    report = dict(twomass=info, tables={})
    failed = []
    for t, d in tables.items():
        pos = np.searchsorted(ids, d["source_id"])
        mine = {c: v[pos] for c, v in cols.items()}
        out = a.output_dir / f"{t.stem}_twomass_gaia.npz"
        write_sidecar(out, d["source_id"], mine)
        check = ks_g_check(d["G"], d["KS"], mine["KS"])
        report["tables"][str(t)] = dict(
            sidecar=str(out),
            rows=int(len(d["source_id"])),
            with_usable_ks=int(np.isfinite(mine["KS"]).sum()),
            check=check,
        )
        log(f"{t}: Ks-G corr {check['old']['corr']:.3f} -> {check['new']['corr']:.3f}")
        if a.twomass != "mask" and not check["passed"]:
            failed.append(str(t))
    (a.output_dir / "twomass_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if failed:
        log(f"Ks-G check failed for {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
