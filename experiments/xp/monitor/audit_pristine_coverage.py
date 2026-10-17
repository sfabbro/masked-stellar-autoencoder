import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

r = Path("experiments/xp/results-stellar-improvement")
p = r / "pristine_full_fetch_v1_r2"
out = {}
with fits.open(p / "pristine-dr1-full.fits", memmap=True) as f:
    d = f[1].data
    ids = np.asarray(d["GaiaDR3"], dtype="i8")
    order = np.argsort(ids)
    sortedids = ids[order]
    observed = (
        np.isfinite(d["CaHKmag0"])
        & np.isfinite(d["e_CaHKmag0"])
        & (d["e_CaHKmag0"] > 0)
    )
    precise = observed & (d["e_CaHKmag0"] <= 0.1)
    with h5py.File(r / "label_registry_v2/labels.h5") as g:
        labels = g["selected/labels"][:]
        pos = np.searchsorted(sortedids, labels["source_id"])
        safe = np.minimum(pos, len(ids) - 1)
        match = (pos < len(ids)) & (sortedids[safe] == labels["source_id"])
        idx = order[safe]
        valid = match & observed[idx]
        fine = match & precise[idx]
        out["native_labels"] = {
            "all_quality_selected_FeH_stars": len(labels),
            "overlap_raw_Prst": int(match.sum()),
            "overlap_measured_CaHK_positive_error": int(valid.sum()),
            "overlap_CaHK_error_le_0p1mag": int(fine.sum()),
            "native_spectroscopic_FeH_lt_minus2_with_precise_CaHK": int(
                (fine & (labels["value"] < -2)).sum()
            ),
            "native_spectroscopic_FeH_lt_minus3_with_precise_CaHK": int(
                (fine & (labels["value"] < -3)).sum()
            ),
        }
    with h5py.File(r / "dataset_view_v2/dataset-view.h5") as g:
        for split in ["train", "validation"]:
            sid = g[split + "/source_id"][:]
            pos = np.searchsorted(sortedids, sid)
            safe = np.minimum(pos, len(ids) - 1)
            match = (pos < len(ids)) & (sortedids[safe] == sid)
            idx = order[safe]
            fine = match & precise[idx]
            lr = g[split + "/selected_label_row"][:]
            part = g[split + "/label_holdout_split"][:]
            fit = fine & (lr >= 0) & (part == 0)
            out[split] = {
                "benchmark_stars": len(sid),
                "raw_Prst_overlap": int(match.sum()),
                "CaHK_error_le_0p1mag_overlap": int(fine.sum()),
                "native_FeH_labels_with_precise_CaHK": int((fine & (lr >= 0)).sum()),
                "native_label_split0_with_precise_CaHK": int(fit.sum()),
            }
    out["caveats"] = [
        "Exact GaiaDR3 intersection; CaHK <=0.1mag reported error is a diagnostic availability threshold only.",
        "Native Pristine morphology/quality flags, variability, extinction and colour conventions require audit before import; no final eligibility asserted.",
        "CaHKmag0 is already dereddened; do not apply extinction twice.",
        "Raw Pristine photometric FeH is not spectroscopic truth. Native metal-poor counts here use the independent native registry selection.",
        "Only benchmark train AND native label split0 are potential fitting rows; validation must remain held out.",
        "Counts quantify coverage, not downstream accuracy benefit.",
    ]
(p / "coverage-audit.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
