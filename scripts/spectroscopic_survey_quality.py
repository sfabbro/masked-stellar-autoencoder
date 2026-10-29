"""Versioned, source-specific quality masks; never modifies source measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from rave_cnn_source import _checked_digest, _stat, sha256_file

GES_FLAGS = frozenset(
    ("SNR", "SRP", "SDS", "IPA", "SSP", "NIA", "SSA", "PSC", "SRV", "SRO", "BIN", "EML")
)


def ges_tokens(values):
    tokens = [set(x.decode("ascii").strip().split("|")) - {""} for x in values]
    return {
        flag: np.array([flag in row for row in tokens], dtype=bool)
        for flag in GES_FLAGS
    }, np.array([bool(row - GES_FLAGS) for row in tokens], dtype=bool)


def valid(values, lower, upper):
    return np.isfinite(values) & (values >= lower) & (values <= upper)


def known_error(error):
    return np.isfinite(error) & (error > 0)


def quality_ges(raw):
    flags, unknown = ges_tokens(raw["SFLAGS"])
    base = ~flags["SNR"] & ~flags["SRP"] & ~unknown
    result = {}
    spec = {
        "TEFF": ("E_TEFF", 1000.0, 50000.0, 1000.0),
        "LOGG": ("E_LOGG", -2.0, 7.0, 1.0),
        "FEH": ("E_FEH", -6.0, 2.0, 1.0),
        "MG1": ("E_MG1", -5.0, 15.0, 1.0),
        "SI1": ("E_SI1", -5.0, 15.0, 1.0),
        "CA1": ("E_CA1", -5.0, 15.0, 1.0),
        "TI1": ("E_TI1", -5.0, 15.0, 1.0),
    }
    for name, (err, lo, hi, emax) in spec.items():
        obs = valid(raw[name], lo, hi)
        err_known = known_error(raw[err])
        is_element = name in ("MG1", "SI1", "CA1", "TI1")
        loose = obs & base & (~flags["NIA"] if is_element else True)
        strict = (
            loose
            & err_known
            & (raw[err] <= emax)
            & ~flags["BIN"]
            & (~flags["SSA"] if is_element else ~flags["SSP"] & ~flags["PSC"])
        )
        result[name] = {
            "available": obs,
            "error_known": err_known,
            "quality_loose": loose,
            "quality_strict": strict,
        }
    result["_flags"] = flags
    result["_unknown_flags"] = unknown
    return result


def quality_madera(raw):
    conv = np.isin(raw["algo_conv_madera"], [0, 2, 3, 4])
    snr = np.isfinite(raw["snr_madera"]) & (raw["snr_madera"] >= 20.0)
    spec = {
        "teff_cal_madera": ("teff_error_madera", 1000.0, 50000.0, 1000.0),
        "logg_cal_madera": ("logg_error_madera", -2.0, 7.0, 1.0),
        "m_h_cal_madera": ("m_h_error_madera", -6.0, 2.0, 1.0),
    }
    result = {}
    for name, (err, lo, hi, emax) in spec.items():
        obs = valid(raw[name], lo, hi)
        known = known_error(raw[err])
        loose = obs & conv
        result[name] = {
            "available": obs,
            "error_known": known,
            "quality_loose": loose,
            "quality_strict": loose & snr & known & (raw[err] <= emax),
        }
    return result


def map_unique(left, right):
    order = np.argsort(right, kind="stable")
    ordered = right[order]
    lo = np.searchsorted(ordered, left, "left")
    hi = np.searchsorted(ordered, left, "right")
    unique = (hi - lo == 1) & (left != b"")
    idx = np.full(len(left), -1, dtype=np.int64)
    idx[unique] = order[lo[unique]]
    return idx


def quality_gauguin(raw, madera, index):
    result = {}
    matched = index >= 0
    conv = np.zeros(len(index), dtype=bool)
    snr = np.zeros(len(index), dtype=bool)
    conv[matched] = np.isin(madera["algo_conv_madera"][index[matched]], [0, 2, 3, 4])
    snr[matched] = np.isfinite(madera["snr_madera"][index[matched]]) & (
        madera["snr_madera"][index[matched]] >= 20.0
    )
    for name, lo, hi in (("fe_h", -6.0, 2.0), ("alpha_fe", -2.0, 2.0)):
        obs = valid(raw[name], lo, hi)
        err = raw[name + "_error"]
        known = known_error(err)
        chi = raw[name + "_chisq"]
        lines = raw["fe_h_nlines" if name == "fe_h" else "n_elements"]
        base = obs & matched & conv & np.isfinite(lines) & (lines > 0)
        strict = base & snr & known & (err <= 1.0) & np.isfinite(chi) & (chi >= 0)
        result[name] = {
            "available": obs,
            "error_known": known,
            "quality_loose": base,
            "quality_strict": strict,
        }
    result["_unique_madera_observation"] = matched
    return result


def produce(ges, madera, gauguin, out, hashes):
    paths = list(map(Path, (ges, madera, gauguin)))
    out = Path(out)
    before = []
    for p, h in zip(paths, hashes, strict=True):
        before.append(_checked_digest(p, h)[1])
    with (
        h5py.File(paths[0]) as gf,
        h5py.File(paths[1]) as mf,
        h5py.File(paths[2]) as af,
    ):
        gr = {
            k: v[:]
            for k, v in gf["raw"].items()
            if k
            in (
                "SFLAGS",
                "TEFF",
                "E_TEFF",
                "LOGG",
                "E_LOGG",
                "FEH",
                "E_FEH",
                "MG1",
                "E_MG1",
                "SI1",
                "E_SI1",
                "CA1",
                "E_CA1",
                "TI1",
                "E_TI1",
            )
        }
        mr = {k: v[:] for k, v in mf["madera_raw"].items()}
        ar = {k: v[:] for k, v in af["gauguin_raw"].items()}
        mi = map_unique(af["rave_obs_id"][:], mf["rave_obs_id"][:])
        results = {
            "gaia_eso": quality_ges(gr),
            "rave_madera": quality_madera(mr),
            "rave_gauguin": quality_gauguin(ar, mr, mi),
        }
        idflags = {
            "gaia_eso": None,
            "rave_madera": mf["published_gaia_edr3_crossmatch/identity_candidate"][:],
            "rave_gauguin": af["published_gaia_edr3_crossmatch/identity_candidate"][:],
        }
        counts = {}
        with h5py.File(out, "x") as target:
            target.attrs["policy"] = "spectroscopic-quality-v1"
            target.attrs["input_sha256_json"] = json.dumps(hashes)
            target.attrs["scope"] = (
                "Per-measurement diagnostics only; no training role, source association certification, unknown-error imputation, or pooled abundance conversion."
            )
            for survey, contents in results.items():
                group = target.create_group(survey)
                group.attrs["identity_evaluated"] = idflags[survey] is not None
                if idflags[survey] is not None:
                    group.create_dataset(
                        "identity_candidate", data=idflags[survey], compression="gzip"
                    )
                counts[survey] = {}
                for label, masks in contents.items():
                    if label == "_flags":
                        subgroup = group.create_group("source_flags")
                        for flag, mask in masks.items():
                            subgroup.create_dataset(flag, data=mask, compression="gzip")
                    elif label.startswith("_"):
                        group.create_dataset(label[1:], data=masks, compression="gzip")
                    else:
                        subgroup = group.create_group(label)
                        counts[survey][label] = {}
                        for tier, mask in masks.items():
                            subgroup.create_dataset(tier, data=mask, compression="gzip")
                            counts[survey][label][tier] = int(mask.sum())
                            counts[survey][label][tier + "_with_identity"] = (
                                int((mask & idflags[survey]).sum())
                                if idflags[survey] is not None
                                else None
                            )
            target.attrs["counts_json"] = json.dumps(counts, sort_keys=True)
    if any(_stat(p) != s for p, s in zip(paths, before, strict=True)):
        raise ValueError("Source changed during quality assessment")
    return {
        "status": "quality_masks_complete",
        "output": str(out.resolve()),
        "output_sha256": sha256_file(out),
        "input_sha256": hashes,
        "counts": counts,
        "gaia_eso_identity": "not_yet_crossmatched",
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("ges", "madera", "gauguin", "out", "receipt"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--input-sha256", nargs=3, required=True)
    a = p.parse_args()
    if Path(a.receipt).exists():
        raise FileExistsError(a.receipt)
    result = produce(a.ges, a.madera, a.gauguin, a.out, a.input_sha256)
    with open(a.receipt, "x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
