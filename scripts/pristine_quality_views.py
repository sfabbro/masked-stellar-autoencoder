"""Versioned Pristine QC and exact-ID indexes; inputs remain immutable."""

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
RAW_SHA = "0001ee6b67ddf0073d39e32fc7d809c0f6cbab7079586facc3b75e678853f713"
BENCH_SHA = "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
REFERENCE = "https://arxiv.org/abs/2308.01344; sections6.3,7.3,7.5; CDS J/A+A/692/A115 ReadMe noteG1"


def quality_masks(data):
    """Input QC never cuts on inferred metallicity; weak candidates do."""

    def finite(name):
        return np.isfinite(data[name])

    checks = {
        "point_source": np.isin(data["Flag"], [-1, -2]),
        "CaHK_measured": finite("CaHKmag0")
        & finite("e_CaHKmag0")
        & (data["e_CaHKmag0"] > 0)
        & (data["e_CaHKmag0"] <= 0.1),
        "nonvariable": finite("Pvar") & (data["Pvar"] >= 0) & (data["Pvar"] < 0.3),
        "astrometry": finite("RUWE") & (data["RUWE"] > 0) & (data["RUWE"] < 1.4),
        "flux_excess": finite("Cstar")
        & finite("s_Cstar")
        & (data["s_Cstar"] > 0)
        & (abs(data["Cstar"]) < data["s_Cstar"]),
        "reddening": finite("E(B-V)") & (data["E(B-V)"] >= 0) & (data["E(B-V)"] < 0.3),
    }
    input_ok = np.logical_and.reduce(list(checks.values()))
    point = data["[Fe/H]phPgm"]
    lo, median, hi = [data[k] for k in ["b_[Fe/H]phP", "[Fe/H]phP", "B_[Fe/H]phP"]]
    # The explicit rule in sections6.3/7.3 is WITHIN-grid fraction>0.8.
    # The table/ReadMe prose says outside; do not invert the executable rule.
    candidate_ok = input_ok & np.isfinite(point) & (point > -4) & (point < 0)
    candidate_ok &= np.isfinite(lo) & np.isfinite(median) & np.isfinite(hi)
    candidate_ok &= (lo <= median) & (median <= hi) & ((hi - lo) / 2 < 0.3)
    candidate_ok &= (
        finite("MCfCaHKP") & (data["MCfCaHKP"] > 0.8) & (data["MCfCaHKP"] <= 1)
    )
    bits = np.zeros(len(input_ok), dtype="u1")
    for index, mask in enumerate(checks.values()):
        bits |= (~mask).astype("u1") << index
    return input_ok, candidate_ok, bits, list(checks)


def exact_rows(reference_ids, query_ids):
    """Return original reference row, or-1; reject ambiguous reference IDs."""
    reference_ids, query_ids = np.asarray(reference_ids), np.asarray(query_ids)
    for array in [reference_ids, query_ids]:
        if array.size and array.dtype.kind not in "iu":
            raise ValueError("Gaia IDs must remain integer arrays")
        if array.dtype.kind == "u" and np.any(array > np.iinfo("i8").max):
            raise ValueError("Gaia ID exceeds signed64 range")
    reference_ids = reference_ids.astype("i8", copy=False)
    query_ids = query_ids.astype("i8", copy=False)
    if np.any(reference_ids <= 0) or len(np.unique(reference_ids)) != len(
        reference_ids
    ):
        raise ValueError("Reference IDs must be unique and positive")
    result = np.full(len(query_ids), -1, dtype="i8")
    if not len(reference_ids):
        return result
    order = np.argsort(reference_ids)
    pos = np.searchsorted(reference_ids[order], query_ids)
    safe = np.minimum(pos, len(reference_ids) - 1)
    match = (
        (query_ids > 0)
        & (pos < len(reference_ids))
        & (reference_ids[order[safe]] == query_ids)
    )
    result[match] = order[safe[match]]
    return result


def build_view(output, scratch, root=ROOT):
    from pristine_async_fetch import digest
    from stellar_dataset_views import publish

    raw = root / "pristine_full_fetch_v1_r2/pristine-dr1-full.fits"
    local = scratch / raw.name
    shutil.copyfile(raw, local)
    if digest(local) != RAW_SHA:
        raise ValueError("Pristine raw checksum changed")
    view = root / "dataset_view_v2/dataset-view.h5"
    registry = root / "label_registry_v2/labels.h5"
    staged = scratch / "pristine-quality-view.h5.building"
    result = {
        "reference": REFERENCE,
        "pristine_sha256": RAW_SHA,
        "benchmark_source_id_sha256": BENCH_SHA,
        "cohorts": {},
    }
    result["quality_profiles"] = {
        "input": {
            "Flag": [-1, -2],
            "CaHK_error_mag": "finite,0<error<=0.1",
            "Pvar": "finite,0<=Pvar<0.3",
            "RUWE": "finite,0<RUWE<1.4",
            "flux_excess": "finite,s_Cstar>0,abs(Cstar)<s_Cstar",
            "E(B-V)": "finite,0<=E(B-V)<0.3",
            "metallicity_cut": "none",
        },
        "weak_candidate_extra": {
            "point_giant_model_FeH": "finite,-4<value<0",
            "percentiles": "finite,lower<=median<=upper,(upper-lower)/2<0.3dex",
            "MCfCaHKP": "finite,0.8<value<=1; explicit paper section7.3 rule",
        },
    }
    with fits.open(local, memmap=True) as hdul, h5py.File(staged, "x") as out:
        data = hdul[1].data
        ids = np.asarray(data["GaiaDR3"], dtype="i8")
        if len(ids) != 6380652 or len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
            raise ValueError("Unexpected raw catalogue IDs/count")
        phot, weak, bits, names = quality_masks(data)
        out.attrs.update(
            raw_path=str(raw),
            raw_sha256=RAW_SHA,
            base_view_path=str(view),
            registry_path=str(registry),
            reference=REFERENCE,
            failed_input_bit_names=json.dumps(names),
            complete=False,
        )
        out["base_benchmark_view"] = h5py.ExternalLink(str(view), "/")
        out["native_registry"] = h5py.ExternalLink(str(registry), "/")
        for name, values in [
            ("source_id", ids),
            ("input_eligible", phot),
            ("weak_candidate_eligible", weak),
            ("failed_input_bits", bits),
        ]:
            out.create_dataset(
                "pristine/" + name, data=values, compression="gzip", compression_opts=1
            )
        source_hash = hashlib.sha256()
        benchmark_ids = {}
        with h5py.File(view) as original:
            for split, expected in [("train", 9000000), ("validation", 1000000)]:
                group = original[split]
                sid = group["source_id"][:]
                benchmark_ids[split] = sid
                if len(sid) != expected:
                    raise ValueError("Benchmark split count changed")
                source_hash.update(sid.astype("<i8").tobytes())
                row = exact_rows(ids, sid)
                matched = row >= 0
                safe = np.maximum(row, 0)
                eligible = matched & phot[safe]
                candidate = matched & weak[safe]
                label = group["selected_label_row"][:]
                label_split = group["label_holdout_split"][:]
                fit = eligible & (label >= 0) & (label_split == 0)
                if split != "train":
                    fit[:] = False
                for name, values in [
                    ("source_id", sid),
                    ("pristine_row", row),
                    ("input_eligible", eligible),
                    ("weak_candidate_eligible", candidate),
                    ("strict_supervised_fit_eligible", fit),
                ]:
                    out.create_dataset(
                        split + "/" + name,
                        data=values,
                        compression="gzip",
                        compression_opts=1,
                    )
                result["cohorts"][split] = dict(
                    rows=len(sid),
                    raw_matches=int(matched.sum()),
                    input_eligible=int(eligible.sum()),
                    weak_candidate_eligible=int(candidate.sum()),
                    native_label_matches=int((eligible & (label >= 0)).sum()),
                    strict_supervised_fit_eligible=int(fit.sum()),
                )
        if source_hash.hexdigest() != BENCH_SHA:
            raise ValueError("Ordered benchmark source-ID hash changed")
        with h5py.File(registry) as reg:
            labels = reg["selected/labels"].fields(["source_id", "value", "split"])[:]
            row = exact_rows(ids, labels["source_id"])
            matched = row >= 0
            eligible = matched & phot[np.maximum(row, 0)]
            partition = np.full(len(labels), 255, dtype="u1")
            partition[exact_rows(benchmark_ids["train"], labels["source_id"]) >= 0] = 0
            partition[
                exact_rows(benchmark_ids["validation"], labels["source_id"]) >= 0
            ] = 1
            fit = eligible & (labels["split"] == 0) & (partition == 0)
            if (
                int(fit.sum())
                != result["cohorts"]["train"]["strict_supervised_fit_eligible"]
            ):
                raise ValueError("Native and benchmark fitting masks disagree")
            for name, values in [
                ("source_id", labels["source_id"]),
                ("pristine_row", row),
                ("input_eligible", eligible),
                ("label_split", labels["split"]),
                ("benchmark_partition", partition),
                ("strict_supervised_fit_eligible", fit),
            ]:
                out.create_dataset(
                    "native/" + name,
                    data=values,
                    compression="gzip",
                    compression_opts=1,
                )
            result["cohorts"]["native"] = dict(
                rows=len(labels),
                input_eligible=int(eligible.sum()),
                strict_supervised_fit_eligible=int(fit.sum()),
                FeH_lt_minus2=int((eligible & (labels["value"] < -2)).sum()),
                FeH_lt_minus3=int((eligible & (labels["value"] < -3)).sum()),
            )
        result.update(
            raw_rows=len(ids),
            input_eligible=int(phot.sum()),
            weak_candidate_eligible=int(weak.sum()),
            failed_input_bit_names=names,
        )
        out.attrs.update(
            complete=True,
            content_id=hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest(),
        )
    destination = output / "pristine-quality-view.h5"
    publish(staged, destination)
    result.update(
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=digest(destination),
        caveats=[
            "Indexed view requires original Pristine FITS, benchmark view and registry; not a standalone copied10Mstar dataset.",
            "Input QC uses measured-photometry flags, not inferredFeH. Weak-candidate selection is separate and is not spectroscopic truth.",
            "Strict supervised fitting requires benchmark train AND native label split0; validation fit mask is identicallyfalse.",
            "Profile is an explicit conservative adaptation, not a claim to exactly reproduce paperTable4 counts.",
            "CaHKmag0 is already dereddened with the favoured giant solution; dwarf solution and target-dependent extinction assumptions require controlled modelling.",
            "MC fraction description conflicts between table prose and section6.3; adopted explicit section7.3 predicate>0.8, preservingrawvalues.",
            "Grid-edge/UMP candidates may fail metallicity-model cuts. RawdataandinputQC remain available; no true extreme star is silently erased.",
        ],
    )
    return result
