"""Independent Gaia DR3 value/unit checks for stored parallax inputs.

Queries contain public Gaia source IDs only; no survey labels or positions are
uploaded. Input availability and physical unit consistency are separate checks.
"""

import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(
        (ROOT / "expanded_six_source_20261107_v2/manifest.json").read_text()
    )
    result = dict(
        status="querying_public_gaia",
        pass_gate=False,
        training_performed=False,
        external_payload="Public Gaia DR3 source_id values only; no native spectroscopic labels or source coordinates",
        source_sha256=source["prepared_sha256"],
        samples={},
        availability={},
    )

    def save():
        p = a.output / "manifest.tmp"
        p.write_text(json.dumps(result, indent=2) + "\n")
        p.replace(a.output / "manifest.json")

    save()
    try:
        with h5py.File(source["prepared_path"]) as f:
            ids = f["source_id"][:]
            observed = f["measured_parallax/observed"][:].astype(bool)
            value = f["measured_parallax/parallax_mas"][:]
            error = f["measured_parallax/parallax_error_mas"][:]
            roles = f["role"][:].astype("S8")
        with h5py.File(ROOT / "remediation_source_v3/prepared.h5") as f:
            old = f["source_id"][:]
        existing = np.isin(ids, old)
        samples = []
        for name, take in [
            ("base_known", existing & observed),
            ("new_known", ~existing & observed),
            ("new_negative", ~existing & observed & (value < 0)),
            ("base_missing", existing & ~observed & (roles == b"fit")),
        ]:
            rows = np.flatnonzero(take)
            chosen = np.sort(
                np.random.default_rng(2026110777).choice(
                    rows, min(128, len(rows)), replace=False
                )
            )
            samples.extend(chosen.tolist())
            result["samples"][name] = dict(
                population=len(rows), sample_rows=chosen.tolist()
            )
        for name, take in [
            ("base_fit", existing & (roles == b"fit")),
            ("new_fit", ~existing & (roles == b"fit")),
            ("all_fit", roles == b"fit"),
        ]:
            result["availability"][name] = dict(
                stars=int(take.sum()), known_parallax=int((take & observed).sum())
            )
        rows = np.unique(samples)
        chosen_ids = ids[rows]
        parts = []
        for start in range(0, len(rows), 64):
            sid = chosen_ids[start : start + 64]
            query = (
                "SELECT source_id,parallax,parallax_error,ra,dec,pmra,pmdec FROM gaiadr3.gaia_source WHERE source_id IN ("
                + ",".join(str(int(v)) for v in sid)
                + ")"
            )
            payload = urllib.parse.urlencode(
                dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=query)
            ).encode()
            for retry in range(3):
                try:
                    req = urllib.request.Request(
                        "https://gea.esac.esa.int/tap-server/tap/sync",
                        data=payload,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=120) as response:
                        body = response.read()
                    frame = pd.read_csv(io.BytesIO(body), dtype={"source_id": "Int64"})
                    if "source_id" not in frame or frame.source_id.isna().any():
                        raise ValueError("Invalid Gaia response ID schema")
                    parts.append(frame)
                    (a.output / f"esa-{start:04d}.csv").write_bytes(body)
                    break
                except Exception:
                    if retry == 2:
                        raise
                    time.sleep(5)
            result["queried_ids"] = min(start + 64, len(rows))
            save()
        reference = pd.concat(parts, ignore_index=True)
        if reference.source_id.duplicated().any():
            raise ValueError("Repeated reference IDs")
        reference = reference.set_index("source_id")
        matched = np.array([int(v) in reference.index for v in chosen_ids])
        r = reference.reindex(chosen_ids)
        rp = r.parallax.to_numpy(dtype="f8")
        re = r.parallax_error.to_numpy(dtype="f8")
        result["comparison"] = {}
        passed = True
        for name, item in result["samples"].items():
            group = np.isin(rows, item["sample_rows"])
            valid = group & matched & np.isfinite(rp) & np.isfinite(re) & (re > 0)
            known = valid & observed[rows]
            detail = dict(n=int(valid.sum()), stored_known=int(known.sum()))
            if known.any():
                dp = value[rows[known]] - rp[known]
                de = error[rows[known]] - re[known]
                ppass = bool(
                    np.allclose(value[rows[known]], rp[known], rtol=2e-6, atol=2e-6)
                )
                epass = bool(
                    np.allclose(error[rows[known]], re[known], rtol=2e-6, atol=2e-6)
                )
                detail.update(
                    parallax_matches_dr3_mas=ppass,
                    error_matches_dr3_mas=epass,
                    parallax_max_absolute_difference_mas=float(abs(dp).max()),
                    error_max_absolute_difference_mas=float(abs(de).max()),
                    median_parallax_difference_mas=float(np.median(dp)),
                    median_error_ratio=float(np.median(error[rows[known]] / re[known])),
                    snr_hypothesis_median_absolute_error=float(
                        np.median(abs(value[rows[known]] - rp[known] / re[known]))
                    ),
                )
                passed &= ppass and epass
            if name == "base_missing":
                detail["available_in_esa"] = int(valid.sum())
            result["comparison"][name] = detail
        paired = pd.DataFrame(
            dict(
                source_id=chosen_ids,
                stored_observed=observed[rows],
                stored_value=value[rows],
                stored_error=error[rows],
                dr3_parallax=rp,
                dr3_parallax_error=re,
            )
        )
        paired.to_parquet(a.output / "paired-parallax-check.parquet", index=False)
        result.update(
            status="complete",
            pass_gate=bool(
                passed
                and result["comparison"]["base_known"]["stored_known"] >= 32
                and result["comparison"]["new_known"]["stored_known"] >= 32
            ),
            tolerance="2e-6 relative + 2e-6 mas absolute, for float32 source/serialization. No additive zeropoint or release offset silently accepted.",
            scope="Independent catalogue check, not a zeropoint/astrometric-quality certificate. Missing stored values may be recoverable from raw or ESA data; unavailable inputs are not fabricated.",
        )
        save()
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
