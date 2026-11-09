"""Read-only active-input and XP-availability census; no model loading."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="reading_supervised_inputs", model_loaded=False, training_performed=False
    )

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(m, indent=2) + "\n")
        tmp.replace(a.output / "manifest.json")

    save()
    source = json.loads(
        (ROOT / "expanded_six_source_20261107_v5/manifest.json").read_text()
    )
    with h5py.File(source["prepared_path"]) as f:
        names = f["feature_names"][:].astype("U").tolist()
        role = f["role"][:].astype("S8")
        eligible = f["primary_input_eligible"][:].astype(bool)
        xp = np.array([i for i, n in enumerate(names) if n.startswith(("bp_", "rp_"))])
        m["source_sha256"] = source["prepared_sha256"]
        m["source_root_datasets"] = list(f)
        counts = {
            r.decode(): dict(
                eligible=0,
                any_xp=0,
                all110_xp=0,
                no_xp=0,
                visible_features=np.zeros(len(names), dtype="i8"),
            )
            for r in np.unique(role)
        }
        for start in range(0, len(role), 32768):
            stop = min(start + 32768, len(role))
            v = f["input"][start:stop]
            visible = np.isfinite(v) & (v != -9999)
            for r in counts:
                take = (role[start:stop] == r.encode()) & eligible[start:stop]
                d = counts[r]
                z = visible[take]
                n = z[:, xp].sum(1)
                d["eligible"] += int(take.sum())
                d["any_xp"] += int((n > 0).sum())
                d["all110_xp"] += int((n == 110).sum())
                d["no_xp"] += int((n == 0).sum())
                d["visible_features"] += z.sum(0)
        for r, d in counts.items():
            d["visible_features"] = {
                n: int(c)
                for n, c in zip(names, d["visible_features"], strict=True)
                if c
            }
        m["supervised_roles"] = counts
        m["measured_parallax_available_by_role"] = {
            r.decode(): int(
                (
                    (role == r)
                    & eligible
                    & f["measured_parallax/observed"][:].astype(bool)
                ).sum()
            )
            for r in np.unique(role)
        }
        flags = {}
        for key in [
            "input_has_xp",
            "real_no_xp",
            "real_xp_present",
            "official_has_xp_known",
            "official_has_xp_continuous",
        ]:
            if key in f:
                v = f[key][:].astype(bool)
                flags[key] = {
                    r.decode(): int(((role == r) & eligible & v).sum())
                    for r in np.unique(role)
                }
        m["stored_modality_flags"] = flags
    m["status"] = "reading_150m_reconstruction_pool"
    save()
    cache = ROOT / "residual-huber-e86-150m-cache-v1"
    cm = json.loads((cache / "manifest.json").read_text())
    cf = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    cn = cm["request"]["features"]
    c_xp = np.array([i for i, n in enumerate(cn) if n.startswith(("bp_", "rp_"))])
    result = dict(rows=len(cf), any_xp=0, all110_xp=0, no_xp=0)
    visible_features = np.zeros(len(cn), dtype="i8")
    begin = time.monotonic()
    for start in range(0, len(cf), 65536):
        v = cf[start : start + 65536]
        good = np.isfinite(v) & (v != -9999)
        n = good[:, c_xp].sum(1)
        result["any_xp"] += int((n > 0).sum())
        result["all110_xp"] += int((n == 110).sum())
        result["no_xp"] += int((n == 0).sum())
        visible_features += good.sum(0)
        if start % (65536 * 64) == 0:
            m["reconstruction_progress_rows"] = start + len(v)
            m["reconstruction_pool"] = result
            save()
    result["visible_features"] = {
        n: int(c) for n, c in zip(cn, visible_features, strict=True) if c
    }
    result["seconds"] = time.monotonic() - begin
    m.update(
        status="current_run_contract_audit_complete",
        reconstruction_pool=result,
        reconstruction_progress_rows=len(cf),
    )
    save()


if __name__ == "__main__":
    main()
