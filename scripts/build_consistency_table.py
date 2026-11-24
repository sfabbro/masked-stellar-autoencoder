"""Inputs for stars of a target list, in the validation.fits layout (no labels).

usage: build_consistency_table.py --targets T.npz --config CFG --output DIR

For the label-free consistency diagnostic: members of wide binaries
(El-Badry+2021) and clusters (Hunt & Reffert 2023, incl. globulars) should
share [Fe/H]. ``T.npz`` holds ``source_id``; the star inputs are copied from
the raw pretraining HDF5 exactly as build_stopgap_finetune_table does (same
fields, same cleaning), and written to ``DIR/stars.fits`` with NaN labels, so
predict_fits_rows predicts them like the benchmark stars.
``--labelled-table`` also lists the targets that are in a fine-tune table
(``labelled.npz``), whose labels the model has seen. The raw 2MASS columns
are replaced as in build_stopgap_finetune_table (``--twomass``: mask by
default, archive, a designation-keyed table, or none to keep them).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stopgap_finetune_table import (  # noqa: E402
    RAW,
    TWOMASS_DEFAULT,
    replace_twomass,
    scan_shard,
)

PARAMETERS = ("teff", "logg", "fe_h", "alpha")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--raw", default=RAW)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--twomass", default=TWOMASS_DEFAULT)
    p.add_argument(
        "--labelled-table",
        type=Path,
        help="fine-tune table; writes labelled.npz (its targets, by source_id)",
    )
    a = p.parse_args()
    out = a.output / "stars.fits"
    if a.labelled_table and not (a.output / "labelled.npz").exists():
        from astropy.io import fits

        with fits.open(a.labelled_table, memmap=True) as h:
            sid = np.asarray(h[1].data["source_id"], dtype="i8")
        wanted = np.load(a.targets)["source_id"].astype("i8")
        a.output.mkdir(parents=True, exist_ok=True)
        np.savez(a.output / "labelled.npz", source_id=np.intersect1d(sid, wanted))
    if out.exists():
        print(out, "exists")
        return
    a.output.mkdir(parents=True, exist_ok=True)
    wanted = np.unique(np.load(a.targets)["source_id"].astype("i8"))
    cfg = yaml.safe_load(a.config.read_text())["data"]
    errors = [e for e in cfg["error_cols"] if e is not None]
    fields = list(dict.fromkeys([*cfg["feature_cols"], *errors]))
    import h5py

    with h5py.File(a.raw, "r") as f:
        keys = sorted(f.keys())
    jobs = [(a.raw, k, wanted, fields, a.chunk, 0.0, wanted[:0], i)
            for i, k in enumerate(keys)]  # fmt: skip
    ids, blocks = [], []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (_, sid, block, _, _) in enumerate(pool.map(scan_shard, jobs), 1):
            ids.append(sid)
            blocks.append(block)
            if i % 10 == 0 or i == len(keys):
                n = sum(len(x) for x in ids)
                print(f"{i}/{len(keys)} shards, {n} matched, {time.time() - t0:.0f}s",
                      flush=True)  # fmt: skip
    sid, first = np.unique(np.concatenate(ids), return_index=True)
    matrix = np.concatenate(blocks)[first]
    twomass = replace_twomass(a.twomass, fields, dict(stars=(sid, matrix)))
    from astropy.table import Table

    t = Table()
    t["source_id"] = sid
    nan = np.full(len(sid), np.nan, "f4")
    for name in (*PARAMETERS, "age"):
        t[name], t["e_" + name] = nan, nan
    for j, name in enumerate(fields):
        t[name] = matrix[:, j]
    t.write(out)
    (a.output / "manifest.json").write_text(json.dumps(dict(
        targets=len(wanted), stars_with_inputs=int(len(sid)), raw=a.raw,
        config=str(a.config), twomass=twomass), indent=2, default=str) + "\n")  # fmt: skip
    print("wrote", out, len(sid))


if __name__ == "__main__":
    main()
