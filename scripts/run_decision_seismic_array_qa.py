"""Compare all decision arms to identical seismic references from saved arrays only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
REFERENCE = (
    ROOT / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
)
REF_MANIFEST = ROOT / "seismic_gravity_qa_20261106_v1/manifest.json"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(a.config.read_text())
    expected = json.loads(REF_MANIFEST.read_text())["sources"]["apokasc3"][
        "reference_sha256"
    ]
    if sha(REFERENCE) != expected:
        raise ValueError("Seismic reference differs")
    ref = pd.read_parquet(REFERENCE)
    ids = ref.source_id.to_numpy()
    pending = {v["run_id"] for v in cfg["arms"]}
    results = {}
    deadline = time.monotonic() + 24 * 3600

    def save():
        d = dict(
            status="waiting_or_scoring" if pending else "complete",
            pending=sorted(pending),
            results=results,
            reference_sha256=expected,
            training_performed=False,
            model_loaded=False,
            scope="Saved devtest pilot panels, not all seismic stars. Same exact IDs required across arms; current historical teacher/parent exposure unverified. Point accuracy versus seismic logg; not physical PDF calibration.",
        )
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    panel_ids = None
    while pending and time.monotonic() < deadline:
        for rid in sorted(pending):
            path = ROOT / rid / "manifest.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            if m.get("status") == "failed":
                results[rid] = dict(status="upstream_failed")
                pending.remove(rid)
                save()
                continue
            if m.get("status") != "remediation_pilot_complete":
                continue
            if m.get("parent_sha256") not in [None, cfg["checkpoint_sha256"]]:
                raise ValueError("Parent binding differs")
            pred = ROOT / rid / "development-predictions.npz"
            with np.load(pred, allow_pickle=False) as f:
                sid = f["source_id"]
                if sid.dtype.kind not in "iu" or len(np.unique(sid)) != len(sid):
                    raise ValueError("Invalid exact prediction IDs")
                ci = np.searchsorted(ids, sid)
                hit = ci < len(ids)
                hit[hit] &= ids[ci[hit]] == sid[hit]
                rows = np.flatnonzero(hit)
                source_ids = sid[rows]
                if panel_ids is None:
                    panel_ids = np.sort(source_ids)
                np.testing.assert_array_equal(np.sort(source_ids), panel_ids)
                y = ref.loggSeis.to_numpy()[ci[hit]]
                prediction = f["native_median"][rows, 1]
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite gravity prediction")
                delta = prediction - y
                bias = float(np.median(delta))
                result = dict(
                    status="complete",
                    stars=len(y),
                    selected_update=m["selected_update"],
                    selected_checkpoint_sha256=m["selected_checkpoint_sha256"],
                    prediction_sha256=sha(pred),
                    matched_ids_sha256=hashlib.sha256(
                        np.sort(source_ids).astype("<i8").tobytes()
                    ).hexdigest(),
                    median_bias=bias,
                    mae=float(abs(delta).mean()),
                    nmad=float(1.4826 * np.median(abs(delta - bias))),
                    rmse=float(np.sqrt(np.mean(delta**2))),
                )
                np.savez_compressed(
                    a.output / (rid + "-seismic-paired.npz"),
                    source_id=source_ids,
                    logg_seismic=y,
                    logg_prediction=prediction,
                    reference_error=ref.e_loggSeis.to_numpy()[ci[hit]],
                )
                results[rid] = result
            pending.remove(rid)
            save()
        if pending:
            time.sleep(45)
    if pending:
        raise TimeoutError(
            "Decision seismic QA upstream wait expired; manifest preserves pending arms"
        )


if __name__ == "__main__":
    main()
