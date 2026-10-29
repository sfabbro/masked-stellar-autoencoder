"""Source-only RAVE DR6 MADERA plus published Gaia EDR3 identity map.

Retains every observation and raw measurement. The published map is best-neighbour,
not an all-candidate, epoch-validated crossmatch. GAUGUIN is a separate pending source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rave_cnn_source import (
    _checked_digest,
    _join,
    _obs_column,
    _read_xmatch,
    _stat,
    sha256_file,
)

COLUMNS = (
    "teff_madera",
    "teff_cal_madera",
    "teff_error_madera",
    "logg_madera",
    "logg_cal_madera",
    "logg_error_madera",
    "m_h_madera",
    "m_h_cal_madera",
    "m_h_error_madera",
    "chisq_madera",
    "snr_madera",
    "algo_conv_madera",
)


def produce(
    madera,
    crossmatch,
    output,
    madera_sha,
    cross_sha,
    expected_rows=518387,
    expected_cross_rows=517357,
):
    madera, crossmatch, output = map(Path, (madera, crossmatch, output))
    _, mstat = _checked_digest(madera, madera_sha)
    _, xstat = _checked_digest(crossmatch, cross_sha)
    with fits.open(madera, memmap=True) as hdus:
        hdu = hdus[1]
        if len(hdu.data) != expected_rows or not set(COLUMNS).issubset(
            hdu.columns.names
        ):
            raise ValueError("MADERA row count or schema differs")
        obs = _obs_column(hdu)
        raw = {k: np.array(hdu.data[k], copy=True) for k in COLUMNS}
    xmatch = _read_xmatch(crossmatch, expected_cross_rows)
    joined = _join({"rave_obs_id": obs}, xmatch)
    known = joined["identity_candidate"]
    counts = {
        "madera_rows": len(obs),
        "crossmatch_rows": len(xmatch["rave_obs_id"]),
        "identity_candidate_observations": int(known.sum()),
        "identity_candidate_stars": int(
            np.unique(joined["gaia_edr3_dr3_source_id"][known]).size
        ),
        "ambiguous_observation_key_rows": int((joined["crossmatch_count"] > 1).sum()),
        "missing_observation_key_rows": int((joined["crossmatch_count"] == 0).sum()),
        "per_measurement": {},
    }
    for name in (
        "teff_madera",
        "teff_cal_madera",
        "logg_madera",
        "logg_cal_madera",
        "m_h_madera",
        "m_h_cal_madera",
    ):
        ok = np.isfinite(raw[name]) & (raw[name] != -9999)
        counts["per_measurement"][name] = {
            "finite_non_sentinel_rows": int(ok.sum()),
            "identity_candidate_rows": int((ok & known).sum()),
        }
    if _stat(madera) != mstat or _stat(crossmatch) != xstat:
        raise ValueError("Input changed during read")
    with h5py.File(output, "x") as f:
        f.attrs["scope"] = (
            "MADERA spectroscopy, published EDR3 best-neighbour association, not independently verified; no GAUGUIN, role assignment, training or features."
        )
        f.attrs["madera_sha256"] = madera_sha
        f.attrs["crossmatch_sha256"] = cross_sha
        f.attrs["source_release"] = "RAVE DR6"
        f.create_dataset(
            "source_row",
            data=np.arange(len(obs), dtype=np.int64),
            compression="gzip",
            shuffle=True,
        )
        f.create_dataset("rave_obs_id", data=obs, compression="gzip")
        g = f.create_group("madera_raw")
        for k, v in raw.items():
            g.create_dataset(k, data=v, compression="gzip", shuffle=True)
        g = f.create_group("published_gaia_edr3_crossmatch")
        for k, v in joined.items():
            g.create_dataset(
                k.replace("cnn_observation_count", "madera_observation_count"),
                data=v,
                compression="gzip",
                shuffle=True,
            )
        f.attrs["counts_json"] = json.dumps(counts, sort_keys=True)
    return {
        "status": "madera_source_complete",
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "madera_sha256": madera_sha,
        "crossmatch_sha256": cross_sha,
        "counts": counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--madera", required=True)
    p.add_argument("--crossmatch", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--madera-sha256", required=True)
    p.add_argument("--crossmatch-sha256", required=True)
    p.add_argument("--receipt", required=True)
    a = p.parse_args()
    if Path(a.receipt).exists():
        raise FileExistsError(a.receipt)
    result = produce(
        a.madera, a.crossmatch, a.output, a.madera_sha256, a.crossmatch_sha256
    )
    with open(a.receipt, "x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
