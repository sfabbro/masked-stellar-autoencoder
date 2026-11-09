"""Source-only identity, input visibility and role audit for the predeclared corners."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(a.config.read_text())
    source = json.loads(Path(cfg["source_manifest"]).read_text())
    with h5py.File(source["prepared_path"]) as f:
        ids = f["source_id"][:]
        requested = np.sort(np.asarray(cfg["corner_source_ids"], dtype="i8"))
        rows = np.searchsorted(ids, requested)
        np.testing.assert_array_equal(ids[rows], requested)
        names = f["feature_names"][:].astype("U")
        xp = np.array([n.startswith(("bp_", "rp_")) for n in names])
        x = f["input"][rows]
        parallax = f["measured_parallax/observed"][rows]
        records = []
        for i, row in enumerate(rows):
            r = dict(
                source_id=int(ids[row]),
                visible_xp_coefficients=int(
                    (np.isfinite(x[i, xp]) & (x[i, xp] != -9999)).sum()
                ),
                measured_parallax_observed=bool(parallax[i]),
            )
            for key in [
                "role",
                "primary_input_eligible",
                "real_xp_present",
                "real_no_xp",
                "xp_status",
                "reserved_reference",
                "known_supervised_ancestor_exposed",
            ]:
                value = f[key][row]
                r[key] = value.decode() if isinstance(value, bytes) else value.item()
            records.append(r)
    result = dict(
        status="complete",
        model_loaded=False,
        training_performed=False,
        source_expected_sha256=source["prepared_sha256"],
        full_source_rehashed=False,
        records=records,
    )
    (a.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
