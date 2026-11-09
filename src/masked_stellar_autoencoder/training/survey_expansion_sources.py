"""Source-only staging for updated benchmarks and RAVE; no model execution."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .multisurvey_overlay import file_sha256, write_observation_sidecar


def read_vizier_tsv(path, *, expected_rows):
    text = Path(path).read_text()
    if "#Column" not in text or "#Title:" not in text:
        raise ValueError("Expected a VizieR ASU TSV response with schema metadata")
    frame = pd.read_csv(
        StringIO(text), sep="\t", comment="#", dtype=str, keep_default_na=False
    )
    frame = frame.map(lambda value: value.strip())
    if "recno" not in frame:
        raise ValueError("Missing source row provenance")
    frame = frame[frame.recno.str.fullmatch(r"\d+")].copy()
    if len(frame) != expected_rows or frame.recno.duplicated().any():
        raise ValueError("Incomplete or duplicate catalogue rows")
    return frame


def exact_optional_ids(values):
    """Nullable int64 IDs from decimal strings, never through floating point."""
    out = []
    for value in values:
        if value == "":
            out.append(pd.NA)
        elif (
            isinstance(value, str)
            and value.isdecimal()
            and 0 < int(value) <= np.iinfo("i8").max
        ):
            out.append(int(value))
        else:
            raise ValueError("Gaia IDs must be positive decimal strings or missing")
    return pd.array(out, dtype="Int64")


def assemble_benchmarks(parameters, abundances, *, expected_rows=201):
    p = read_vizier_tsv(parameters, expected_rows=expected_rows)
    a = read_vizier_tsv(abundances, expected_rows=expected_rows)
    if (
        p.HIP.duplicated().any()
        or a.Star.duplicated().any()
        or set(p.HIP) != set(a.Star)
    ):
        raise ValueError("Benchmark star identities differ across releases")
    a = a.rename(
        columns={
            "Teff": "abundance_Teff",
            "logg": "abundance_logg",
            "recno": "abundance_row",
        }
    )
    out = p.merge(
        a[
            [
                "Star",
                "abundance_row",
                "abundance_Teff",
                "abundance_logg",
                "Flag",
                "[FeI/H]",
                "e_[FeI/H]",
                "[FeII/H]",
                "e_[FeII/H]",
            ]
        ],
        left_on="HIP",
        right_on="Star",
        validate="one_to_one",
    )
    out["source_id"] = exact_optional_ids(out.GaiaDR3)
    known = out.source_id.notna()
    if out.loc[known, "source_id"].duplicated().any():
        raise ValueError("Multiple benchmark stars share a Gaia source")
    for field in [
        "Teff",
        "e_Teff",
        "logg",
        "e_logg",
        "[FeI/H]",
        "e_[FeI/H]",
        "[FeII/H]",
        "e_[FeII/H]",
    ]:
        out[field] = (
            pd.to_numeric(out[field], errors="raise")
            if not out[field].eq("").any()
            else pd.to_numeric(out[field].replace("", np.nan), errors="raise")
        )
    for field in ["Teff", "logg"]:
        other = pd.to_numeric(out["abundance_" + field], errors="raise")
        if not np.allclose(out[field], other, atol=0.011, rtol=0):
            raise ValueError("Atmospheric parameters differ between benchmark releases")
    out["direct_diameter"] = ~out.Indirect.eq("1")
    out["spectral_warning"] = ~out.Flag.isin(["", "-"])
    out["evaluation_reserved"] = True
    return out


RAVE_SOURCE = "rave_dr6_madera_gauguin"
RAVE_REGISTRY = dict(
    release="RAVE DR6; live EDR3 crossmatch retrieved 2026-11-05",
    pipeline="calibrated MADERA Teff/logg and GAUGUIN FeI/H based on MADERA atmospheres",
    dependency_id="rave-dr6-shared-spectra-madera-gauguin",
    targets={
        "teff": dict(quantity="teff", unit="K", field="teff_cal_madera"),
        "logg": dict(quantity="logg", unit="dex(cm/s2)", field="logg_cal_madera"),
        "fe_h": dict(quantity="fe_h", unit="dex", field="fe_h_gauguin"),
    },
    quality_policy="provisional: MADERA convergence0, SNR>=40, EDR3 separation<=1 arcsec; per-target domain; iron nlines>0",
    repeat_policy="highest SNR, then smallest separation, then observation ID; retain original MADERA row",
    uncertainty="quoted positive finite random errors only; unknown stays NaN; cross-target covariance unavailable, not known zero",
    training_enabled=False,
)


def select_rave(madera, iron, crossmatch):
    m = madera.copy()
    if (
        m.rave_obs_id.duplicated().any()
        or iron.rave_obs_id.duplicated().any()
        or crossmatch.rave_obs_id.duplicated().any()
    ):
        raise ValueError("Ambiguous observation identity")
    sid = crossmatch.source_id.to_numpy()
    if sid.dtype.kind not in "iu" or (sid <= 0).any():
        raise ValueError("RAVE crossmatch IDs must be exact positive integers")
    m["source_row"] = np.arange(len(m), dtype="i8")
    a = m.merge(
        crossmatch[["rave_obs_id", "source_id", "dist"]],
        on="rave_obs_id",
        validate="one_to_one",
    )
    a = a.merge(
        iron[["rave_obs_id", "fe_h_gauguin", "fe_h_error_gauguin", "fe_h_nl_gauguin"]],
        on="rave_obs_id",
        how="left",
        validate="one_to_one",
    )
    a = a[
        (a.algo_conv_madera == 0)
        & (a.snr_madera >= 40)
        & np.isfinite(a.dist)
        & (a.dist <= 1)
        & (a.dist >= 0)
    ]
    a = a.sort_values(
        ["source_id", "snr_madera", "dist", "rave_obs_id"],
        ascending=[True, False, True, True],
    ).drop_duplicates("source_id")
    value = a[["teff_cal_madera", "logg_cal_madera", "fe_h_gauguin"]].to_numpy(
        dtype="f8"
    )
    raw_error = a[
        ["teff_error_madera", "logg_error_madera", "fe_h_error_gauguin"]
    ].to_numpy(dtype="f8")
    error = np.where(np.isfinite(raw_error) & (raw_error > 0), raw_error, np.nan)
    observed = np.isfinite(value) & (value != -9999)
    observed[:, 0] &= (value[:, 0] >= 2500) & (value[:, 0] <= 10000)
    observed[:, 1] &= (value[:, 1] >= -1) & (value[:, 1] <= 6.5)
    observed[:, 2] &= (
        (value[:, 2] > -5) & (value[:, 2] < 1) & (a.fe_h_nl_gauguin.to_numpy() > 0)
    )
    block = dict(
        source_id=a.source_id.to_numpy(dtype="i8"),
        source_row=a.source_row.to_numpy(dtype="i8"),
        value=value,
        error=error,
        observed=observed,
        native_error=raw_error,
        error_known=np.isfinite(error),
        snr=a.snr_madera.to_numpy(dtype="f8"),
        crossmatch_arcsec=a.dist.to_numpy(dtype="f8"),
        iron_nlines=a.fe_h_nl_gauguin.to_numpy(dtype="f8"),
    )
    return a, block


def write_rave_candidate(madera, iron, crossmatch, output, benchmark_ids):
    from astropy.table import Table

    frame, block = select_rave(
        pd.read_csv(madera),
        pd.read_csv(iron),
        Table.read(crossmatch, format="votable").to_pandas(),
    )
    block["benchmark_reserved"] = np.isin(block["source_id"], benchmark_ids)
    receipt = write_observation_sidecar(
        output,
        {RAVE_SOURCE: block},
        {RAVE_SOURCE: RAVE_REGISTRY},
        provenance=dict(
            source_files={str(p): file_sha256(p) for p in [madera, iron, crossmatch]},
            selection=RAVE_REGISTRY["quality_policy"],
            training_enabled=False,
            role_assignment="not assigned; benchmark_reserved must be excluded from future fitting",
        ),
    )
    frame[["rave_obs_id", "source_id", "source_row"]].to_csv(
        Path(output).with_suffix(".rows.csv"), index=False
    )
    return block, receipt
