"""Compare first-checkpoint predictions with full-release Andrae/Gaia Net lookups."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def digest(path, kind="sha256"):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, kind).hexdigest()


def metrics(x, y):
    residual = np.asarray(x) - np.asarray(y)
    residual = residual[np.isfinite(residual)]
    return (
        dict(
            n=len(residual),
            mae=float(abs(residual).mean()),
            bias=float(np.median(residual)),
            nmad=float(1.4826 * np.median(abs(residual - np.median(residual)))),
        )
        if len(residual)
        else dict(n=0)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qa-run", default="expanded_first_checkpoint_qa_20261107_v1")
    p.add_argument("--reuse-waiting-output", action="store_true")
    a = p.parse_args()
    from saved_array_source_binding import load_bound_source, prepare_report_output

    prepare_report_output(
        a.output, "waiting_for_first_checkpoint_arrays", a.reuse_waiting_output
    )
    qa = ROOT / a.qa_run
    deadline = time.monotonic() + 48 * 3600
    m = dict(
        status="waiting_for_first_checkpoint_arrays",
        training_performed=False,
        model_loaded=False,
        scope="The QA footprint is predictions exported for a fixed panel, not the available raw Gaia input. Full releases are scanned/joined. Andrae M/H and MSA Fe/H are distinct; common-vs-catalogue agreement is not physical truth. Gaia Net training ancestry is unknown.",
    )

    def save():
        p = a.output / "manifest.tmp"
        p.write_text(json.dumps(m, indent=2) + "\n")
        p.replace(a.output / "manifest.json")

    save()
    while time.monotonic() < deadline:
        try:
            producer = json.loads((qa / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            producer = {}
        if producer.get("status") == "upstream_failed":
            m.update(status="upstream_failed", error=producer.get("error"))
            save()
            return
        if producer.get("status") == "first_checkpoint_common_qa_complete":
            break
        time.sleep(30)
    else:
        raise TimeoutError("No fixed snapshot QA arrays in 48h")
    with np.load(qa / "panel.npz") as f:
        panel = {k: f[k] for k in f.files}
    ids = panel["source_id"]
    models = {}
    for name, item in producer["models"].items():
        path = qa / (name + "-common.npz")
        if digest(path) != item["artifacts"][path.name]["sha256"]:
            raise ValueError("Prediction array differs")
        with np.load(path) as f:
            models[name] = f["median"][:, :3]
    xgb = ROOT / "andrae_full_footprint_census_20261106_v2"
    index = xgb / "andrae-full-inference-index.h5"
    im = json.loads((xgb / "manifest.json").read_text())
    if im.get("status") != "complete" or digest(index) != im["index_sha256"]:
        raise ValueError("Full Andrae index differs")
    with h5py.File(index) as f:
        allids = f["source_id"][:]
        pos = np.searchsorted(allids, ids)
        hit = pos < len(allids)
        hit[hit] &= allids[pos[hit]] == ids[hit]
        ii = np.flatnonzero(hit)
        rr = pos[hit]
        andrae = np.full((len(ids), 3), np.nan)
        andrae[ii] = np.column_stack(
            [f[k][rr] for k in ["teff_xgboost", "logg_xgboost", "mh_xgboost"]]
        )
        in_training = np.zeros(len(ids), bool)
        in_training[ii] = f["in_training_sample"][rr]
    del allids
    gaia_path = ROOT / "decision_xp_catalogue_qa_20261106_v1/gaia-xp.parquet"
    if digest(gaia_path, "md5") != "4cc09524fc1b4fe774dbeeb9dc0d50c2":
        raise ValueError("Gaia Net publisher checksum differs")
    gaia = np.full((len(ids), 3), np.nan)
    seen = np.zeros(len(ids), bool)
    scanned = 0
    source = pq.ParquetFile(gaia_path)
    for batch in source.iter_batches(
        batch_size=500000, columns=["source_id", "logTeff", "logg", "FeH"]
    ):
        sid = batch.column(0).to_numpy(zero_copy_only=False)
        if sid.dtype.kind not in "iu":
            raise ValueError("Comparator IDs cannot pass through floats")
        if sid.dtype.kind == "u" and (sid > np.iinfo("i8").max).any():
            raise ValueError("Unsigned Gaia ID exceeds signed exact-ID range")
        sid = sid.astype("i8", copy=False)
        pos = np.searchsorted(ids, sid)
        good = pos < len(ids)
        good[good] &= ids[pos[good]] == sid[good]
        q = np.flatnonzero(good)
        pp = pos[good]
        if seen[pp].any() or len(np.unique(pp)) != len(pp):
            raise ValueError("Comparator duplicate exact IDs")
        gaia[pp] = np.column_stack(
            [
                10 ** batch.column(1).to_numpy(zero_copy_only=False)[q],
                batch.column(2).to_numpy(zero_copy_only=False)[q],
                batch.column(3).to_numpy(zero_copy_only=False)[q],
            ]
        )
        seen[pp] = True
        scanned += len(sid)
    if scanned != 219196764:
        raise ValueError("Full Gaia Net row count differs")
    m.update(
        status="scoring",
        full_release_counts=dict(andrae=174922161, gaia_net=scanned),
        predicted_qa_stars=len(ids),
        matched=dict(
            andrae=int(hit.sum()),
            gaia_net=int(seen.sum()),
            andrae_known_outside_training=int((hit & ~in_training).sum()),
        ),
        comparators={},
    )
    for name, values in [("andrae", andrae), ("gaia_net", gaia)]:
        m["comparators"][name] = {
            model: {
                ["teff", "logg", "mh_vs_fe_h_proxy" if name == "andrae" else "fe_h"][
                    j
                ]: metrics(pred[:, j], values[:, j])
                for j in range(3)
            }
            for model, pred in models.items()
        }
    refs = pd.read_parquet(
        ROOT / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
    )
    refidx = {int(v): i for i, v in enumerate(refs.source_id)}
    m["seismic_devtest"] = {}
    for subset, take in [
        ("all", panel["seismic_devtest"]),
        (
            "andrae_known_outside_training",
            panel["seismic_devtest"] & hit & ~in_training,
        ),
    ]:
        ii = np.flatnonzero(take)
        truth = refs.loggSeis.to_numpy()[[refidx[int(v)] for v in ids[ii]]]
        m["seismic_devtest"][subset] = {
            name: metrics(values[ii, 1], truth)
            for name, values in {**models, "andrae": andrae, "gaia_net": gaia}.items()
        }
    m["native_devtest"] = {}
    src = load_bound_source(producer, digest)
    with h5py.File(src["prepared_path"]) as f:
        for name in src["source_registry"]:
            g = f["native_observations/" + name]
            rows = g["star_row"][:]
            pos = np.searchsorted(panel["row"], rows)
            valid = pos < len(ids)
            valid[valid] &= panel["row"][pos[valid]] == rows[valid]
            gr = np.flatnonzero(valid)
            pp = pos[valid]
            y = g["value"][:][gr, :3]
            obs = g["observed"][:][gr, :3].astype(bool)
            m["native_devtest"][name] = {}
            for subset, take in [
                (
                    "all",
                    (panel["role"][pp] == b"devtest") & panel["input_eligible"][pp],
                ),
                (
                    "andrae_known_outside_training",
                    (panel["role"][pp] == b"devtest")
                    & hit[pp]
                    & ~in_training[pp]
                    & panel["input_eligible"][pp],
                ),
            ]:
                m["native_devtest"][name][subset] = {
                    label: {
                        model: metrics(
                            values[pp[take & obs[:, j]], j], y[take & obs[:, j], j]
                        )
                        for model, values in {
                            **models,
                            "andrae": andrae,
                            "gaia_net": gaia,
                        }.items()
                    }
                    for j, label in enumerate(["teff", "logg", "fe_h_Andrae_MH_proxy"])
                    if obs[:, j].any()
                }
    np.savez_compressed(
        a.output / "matched-xp-catalogues.npz",
        source_id=ids,
        andrae=andrae,
        andrae_present=hit,
        andrae_in_training_sample=in_training,
        gaia_net=gaia,
    )
    m.update(
        status="first_checkpoint_xp_comparison_complete",
        array_sha256=digest(a.output / "matched-xp-catalogues.npz"),
    )
    save()


if __name__ == "__main__":
    main()
