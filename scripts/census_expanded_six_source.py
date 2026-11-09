"""Count unique stars, labels and shared target support in the expanded source."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sm = json.loads(
        (ROOT / "expanded_six_source_20261107_v2/manifest.json").read_text()
    )
    if sm.get("status") != "expanded_six_source_complete":
        raise ValueError("Expanded source not ready")
    with open(sm["prepared_path"], "rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != sm["prepared_sha256"]:
            raise ValueError("Prepared digest differs")
    with h5py.File(sm["prepared_path"]) as f:
        ids = f["source_id"][:]
        roles = f["role"][:].astype("S8")
        eligible = f["primary_input_eligible"][:].astype(bool)
        union = np.zeros((len(ids), 6), bool)
        known = union.copy()
        records = np.zeros(len(ids), dtype="i2")
        source_counts = {}
        for source in sm["source_registry"]:
            g = f["native_observations/" + source]
            rows = g["star_row"][:]
            obs = g["observed"][:].astype(bool)
            k = g["error_known"][:].astype(bool)
            if np.any(np.diff(rows) <= 0):
                raise ValueError("A source has duplicate or unordered star rows")
            union[rows] |= obs
            known[rows] |= k
            records[rows] += obs.any(1)
            source_counts[source] = dict(
                records=len(rows),
                fit_records=int((roles[rows] == b"fit").sum()),
                fit_labels=obs[roles[rows] == b"fit"].sum(0).tolist(),
            )
        result = dict(
            status="complete",
            training_performed=False,
            source_sha256=sm["prepared_sha256"],
            target_order=["teff", "logg", "fe_h", "alpha", "age", "distance"],
            all_feature_stars=len(ids),
            roles={},
            sources=source_counts,
            scope="Distinct source_id counts differ from source records, scalar labels and repeated training presentations. Source-native quantities and posterior-teacher age/distance definitions retained.",
        )
        for role in ["fit", "tune", "cal", "devtest", "external", "excluded"]:
            take = (roles == role.encode()) & eligible
            result["roles"][role] = dict(
                input_stars=int(take.sum()),
                labelled_unique_stars=int((take & union.any(1)).sum()),
                unique_stars_per_target=union[take].sum(0).tolist(),
                source_records=int(records[take].sum()),
                complete_six_unique_stars=int((take & union.all(1)).sum()),
                known_quoted_error_unique_stars=known[take].sum(0).tolist(),
                multiple_source_stars=int((take & (records > 1)).sum()),
                measured_parallax_stars=int(
                    (take & f["measured_parallax/observed"][:]).sum()
                ),
            )
        (a.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["roles"]["fit"]), flush=True)


if __name__ == "__main__":
    main()
