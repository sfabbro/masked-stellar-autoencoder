"""RAVE DR6 GAUGUIN/MADERA values and published Gaia EDR3 best-neighbour IDs.

CDS III/283/gaugmad fixed-width native observations, independently of CNN/BDASP.
This is a source join, not an independently validated positional match or training set.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
from rave_cnn_source import _checked_digest, _join, _read_xmatch, _stat, sha256_file

# CDS ReadMe bytes are one-based inclusive; Python slices are zero-based half-open.
FIELDS = (
    ("fe_h", 22, 37),
    ("fe_h_error", 38, 55),
    ("fe_h_chisq", 56, 70),
    ("fe_h_nlines", 71, 73),
    ("al_h", 74, 89),
    ("al_h_error", 90, 108),
    ("al_h_chisq", 109, 123),
    ("al_h_nlines", 124, 125),
    ("ni_h", 126, 141),
    ("ni_h_error", 142, 158),
    ("ni_h_chisq", 159, 173),
    ("ni_h_nlines", 174, 175),
    ("alpha_fe", 176, 191),
    ("alpha_fe_error", 192, 208),
    ("alpha_fe_chisq", 209, 223),
    ("n_elements", 224, 225),
)


def read_gauguin(path, n):
    obs = np.empty(n, dtype="S32")
    values = np.full((n, len(FIELDS)), np.nan, dtype=np.float64)
    seen = 0
    with gzip.open(path, "rb") as f:
        for i, line in enumerate(f):
            seen += 1
            if i >= n or len(line.rstrip(b"\r\n")) != 225:
                raise ValueError("GAUGUIN row count or fixed-width schema differs")
            obs[i] = line[:21].rstrip(b" ")
            for j, (_, a, b) in enumerate(FIELDS):
                raw = line[a:b].strip()
                if raw:
                    values[i, j] = float(raw)
        if seen != n:
            raise ValueError("GAUGUIN row count differs")
    return obs, values


def produce(
    gauguin,
    crossmatch,
    output,
    gauguin_sha,
    cross_sha,
    expected_rows=453490,
    expected_cross_rows=517357,
):
    gauguin, crossmatch, output = map(Path, (gauguin, crossmatch, output))
    _, gstat = _checked_digest(gauguin, gauguin_sha)
    _, xstat = _checked_digest(crossmatch, cross_sha)
    obs, values = read_gauguin(gauguin, expected_rows)
    xmatch = _read_xmatch(crossmatch, expected_cross_rows)
    joined = _join({"rave_obs_id": obs}, xmatch)
    known = joined["identity_candidate"]
    counts = {
        "gauguin_rows": len(obs),
        "identity_candidate_observations": int(known.sum()),
        "identity_candidate_stars": int(
            np.unique(joined["gaia_edr3_dr3_source_id"][known]).size
        ),
        "ambiguous_observation_key_rows": int((joined["crossmatch_count"] > 1).sum()),
        "per_measurement": {},
    }
    for name in ("fe_h", "alpha_fe", "al_h", "ni_h"):
        ok = np.isfinite(values[:, [x[0] for x in FIELDS].index(name)])
        counts["per_measurement"][name] = {
            "finite_rows": int(ok.sum()),
            "identity_candidate_rows": int((ok & known).sum()),
        }
    if _stat(gauguin) != gstat or _stat(crossmatch) != xstat:
        raise ValueError("source changed during read")
    with h5py.File(output, "x") as f:
        f.attrs["scope"] = (
            "GAUGUIN with MADERA input; published EDR3 best-neighbour association not independently verified; no roles/features/training. Correlated with MADERA."
        )
        f.attrs["gauguin_sha256"] = gauguin_sha
        f.attrs["crossmatch_sha256"] = cross_sha
        f.create_dataset(
            "source_row",
            data=np.arange(len(obs), dtype=np.int64),
            compression="gzip",
            shuffle=True,
        )
        f.create_dataset("rave_obs_id", data=obs, compression="gzip")
        g = f.create_group("gauguin_raw")
        for j, (name, _, _) in enumerate(FIELDS):
            g.create_dataset(name, data=values[:, j], compression="gzip", shuffle=True)
        g = f.create_group("published_gaia_edr3_crossmatch")
        for k, v in joined.items():
            g.create_dataset(
                k.replace("cnn_observation_count", "gauguin_observation_count"),
                data=v,
                compression="gzip",
                shuffle=True,
            )
        f.attrs["counts_json"] = json.dumps(counts, sort_keys=True)
    return {
        "status": "gauguin_source_complete",
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "gauguin_sha256": gauguin_sha,
        "crossmatch_sha256": cross_sha,
        "counts": counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gauguin", required=True)
    p.add_argument("--crossmatch", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gauguin-sha256", required=True)
    p.add_argument("--crossmatch-sha256", required=True)
    p.add_argument("--receipt", required=True)
    a = p.parse_args()
    if Path(a.receipt).exists():
        raise FileExistsError(a.receipt)
    result = produce(
        a.gauguin, a.crossmatch, a.output, a.gauguin_sha256, a.crossmatch_sha256
    )
    with open(a.receipt, "x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
