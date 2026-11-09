"""Source-only six-target support, expanded identity/feature and inference coverage audit."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
BASE = ROOT + "remediation_source_v3/prepared.h5"
FEATURE = ROOT + "multisurvey_feature_union_dr12_20261106_v1/new-star-features.h5"
FEATURE_SHA = "d4d8ea8e972625cb040c0fdcf4cc60675c03823b43826cef4b57d6d725e14f23"
MRS = ROOT + "multisurvey_lamost_dr12_mrs_source_direct_v2/source.h5"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    c = json.loads(a.config.read_text())
    result = dict(
        status="running",
        training_performed=False,
        model_loaded=False,
        role_assignment_performed=False,
        targets=["teff", "logg", "fe_h", "alpha", "age", "distance"],
        sources={},
    )

    def save():
        temp = a.output / "manifest.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(a.output / "manifest.json")

    save()
    with h5py.File(BASE) as f:
        ids = f["source_id"][:]
        roles = f["role"][:].astype("U")
        mask = f["observed"][:].astype(bool) & f["quality"][:].astype(bool)
        eligible = f["primary_input_eligible"][:].astype(bool)
        result["base_roles"] = {}
        for role in np.unique(roles):
            keep = (roles == role) & eligible
            m = mask[keep]
            result["base_roles"][role] = dict(
                stars=int(keep.sum()),
                per_target=m.sum(0).tolist(),
                coobserved=(m.astype("i8").T @ m.astype("i8")).tolist(),
                complete_six=int(m.all(1).sum()),
                no_labels=int((~m.any(1)).sum()),
            )
        result["base_extra_fields"] = [
            k
            for k in f
            if any(x in k.lower() for x in ["parallax", "reserved", "ancestor"])
        ]
        if "measured_parallax" in f:
            result["measured_parallax_schema"] = {
                k: list(v.shape) for k, v in f["measured_parallax"].items()
            }
        result["base_feature_names"] = f["feature_names"][:].astype("U").tolist()
    if sha(FEATURE) != FEATURE_SHA:
        raise ValueError("Expanded input table checksum differs")
    with h5py.File(FEATURE) as f:
        new = f["source_id"][:]
        recovered = f["feature_recoverable"][:].astype(bool)
        names = f["feature_names"][:].astype("U").tolist()
        par = [i for i, n in enumerate(names) if "parallax" in n.lower()]
        result["expanded_features"] = dict(
            stars=len(new),
            recoverable=int(recovered.sum()),
            sha256=FEATURE_SHA,
            parallax_columns=par,
            visible_measured_parallax=0,
            scope="Recovered table masks non-XP/non-five-band fields; measured parallax must be separately admitted for a parallax-assisted final arm",
        )
        for j in par:
            count = 0
            for start in range(0, len(new), 100000):
                count += int(f["visible"][start : start + 100000, j].sum())
            result["expanded_features"]["visible_measured_parallax"] += count
        good_ids = new[recovered]
    for name, source in [
        ("lamost_dr12_lrs", c["lamost_sidecar"]),
        ("gaia_gspspec", c["gaia_sidecar"]),
    ]:
        with h5py.File(source) as f:
            key = (
                c["lamost_source_name"]
                if name.startswith("lamost")
                else "gaia_dr3_gspspec"
            )
            g = f["native_observations/" + key]
            sid = g["source_id"][:]
            obs = g["observed"][:].astype(bool)
        idx = np.searchsorted(ids, sid)
        hit = idx < len(ids)
        hit[hit] &= ids[idx[hit]] == sid[hit]
        result["sources"][name] = dict(
            candidate_stars=len(sid),
            in_old_base=int(hit.sum()),
            outside_base_with_recovered_features=int(np.isin(sid, good_ids).sum()),
            per_target=obs.sum(0).tolist(),
            role_counts={
                r: int(((roles[idx[hit]] == r) & eligible[idx[hit]]).sum())
                for r in ["fit", "tune", "cal", "devtest", "external", "excluded"]
            },
        )
        save()
    with h5py.File(MRS) as f:
        result["mrs_schema"] = {}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                result["mrs_schema"][name] = dict(
                    shape=list(obj.shape), dtype=str(obj.dtype)
                )

        f.visititems(visit)
    result["inference_release_requirements"] = [
        "Gaia-ID partition manifest with attempted/succeeded/missing-input/failed/OOD counts; no all-Gaia completeness claim from the 150M cache",
        "six joint-vector PDF representation plus units/transforms/quality flags; marginal uncertainties alone are not a joint PDF",
        "age/distance teacher provenance and chemical alpha conventions retained",
        "benchmark/cluster/link-group exclusions and prior supervised exposure bound before final FIT",
        "measured-parallax recovery and its errors admitted separately for expanded stars",
    ]
    result["status"] = "source_census_complete"
    save()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
