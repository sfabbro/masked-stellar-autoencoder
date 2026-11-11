"""Consolidated master spectroscopic label table with trace-back to native rows.

One record per native catalogue row (no quality cut), for the staged surveys
(LAMOST DR12 LRS/MRS, DESI DR1, APOGEE DR17, GALAH DR4, Gaia DR3 GSP-Spec,
SDSS DR20 BOSSNet, RAVE DR6) and the master-label census sources. Every record
keeps ``catalogue`` (a key in ``catalogues.json``), ``native_hdu`` and
``native_row`` so the full native row (abundances, flags, ...) can be read
back with :func:`read_native_rows`.

Star identity: ``star_id`` is the Gaia DR3 ``source_id`` when one is known
(publisher, SIMBAD or positional), otherwise a negative serial shared by
records within ``NOGAIA_RADIUS_ARCSEC`` of each other (any survey), or unique
to the record when it has no position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RECORD_COLUMNS = {
    "survey": "str",
    "release": "str",
    "pipeline": "str",
    "tier": "str",
    "dependency_group": "str",
    "catalogue": "str",
    "native_hdu": "str",
    "native_row": "i8",
    "native_id": "str",
    "source_id": "i8",
    "id_method": "str",
    "match_sep_arcsec": "f8",
    "match_n_within": "i8",
    "ra": "f8",
    "dec": "f8",
    "teff": "f8",
    "e_teff": "f8",
    "logg": "f8",
    "e_logg": "f8",
    "fe_h": "f8",
    "e_fe_h": "f8",
    "fe_h_kind": "str",
    "fe_h_corrected": "f8",
    "alpha": "f8",
    "e_alpha": "f8",
    "alpha_kind": "str",
    "snr": "f8",
    "fit_ok": "bool",
    "flag_int": "i8",
    "flag_text": "str",
}
NOGAIA_RADIUS_ARCSEC = 1.0
STAR_BAD = 1 << 23  # APOGEE ASPCAPFLAG


def frame(n: int, **constants) -> pd.DataFrame:
    """Empty record frame of length n with the schema defaults."""
    out = {}
    for name, kind in RECORD_COLUMNS.items():
        if name in constants:
            out[name] = np.full(
                n, constants[name], dtype=object if kind == "str" else kind
            )
        elif kind == "str":
            out[name] = np.full(n, "", dtype=object)
        elif kind == "f8":
            out[name] = np.full(n, np.nan)
        elif kind == "bool":
            out[name] = np.zeros(n, bool)
        else:
            out[name] = np.zeros(n, "i8")
    return pd.DataFrame(out)


def finalize(records: pd.DataFrame) -> pd.DataFrame:
    """Schema order and dtypes; -9999 and non-finite numbers become NaN."""
    records = records[list(RECORD_COLUMNS)].copy()
    for name, kind in RECORD_COLUMNS.items():
        if kind == "f8":
            v = pd.to_numeric(records[name], errors="coerce").to_numpy("f8", copy=True)
            v[~np.isfinite(v) | (v == -9999)] = np.nan
            records[name] = v
        elif kind == "i8":
            records[name] = records[name].fillna(0).astype("i8")
        elif kind == "bool":
            records[name] = records[name].fillna(False).astype(bool)
        else:
            records[name] = records[name].fillna("").astype(str)
    records.loc[records.source_id < 0, "source_id"] = 0
    return records


def exact_ids(raw) -> np.ndarray:
    """Positive int64 Gaia IDs from integer or decimal-string columns (else 0)."""
    raw = np.asarray(raw)
    if raw.dtype.kind in "iu":
        out = raw.astype("i8")
    elif raw.dtype.kind == "f":
        raise ValueError("Gaia IDs must not pass through floating point")
    else:
        text = np.char.strip(raw.astype("U"))
        ok = np.char.isdecimal(text) & (np.char.str_len(text) <= 19)
        out = np.array(
            [
                int(t) if g and int(t) < 2**63 else 0
                for t, g in zip(text, ok, strict=True)
            ],
            dtype="i8",
        )
    out[out < 0] = 0
    return out


def unit_vectors(ra, dec) -> np.ndarray:
    ra, dec = np.radians(np.asarray(ra, "f8")), np.radians(np.asarray(dec, "f8"))
    return np.column_stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )


def group_by_position(ra, dec, radius_arcsec=NOGAIA_RADIUS_ARCSEC) -> np.ndarray:
    """Friends-of-friends labels (0..k-1) linking points within the radius."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(ra)
    if n == 0:
        return np.zeros(0, "i8")
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)
    pairs = cKDTree(unit_vectors(ra, dec)).query_pairs(chord, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1]))
        if len(pairs)
        else ([], ([], [])),
        shape=(n, n),
    )
    return connected_components(graph, directed=False)[1].astype("i8")


def assign_star_ids(records: pd.DataFrame, radius_arcsec=NOGAIA_RADIUS_ARCSEC):
    """star_id: Gaia source_id, else a negative id shared by nearby records.

    Records without Gaia identity and with a position are grouped across
    surveys by friends-of-friends within ``radius_arcsec``; records with
    neither get their own id. Ids are deterministic for a given record order.
    """
    star_id = records.source_id.to_numpy("i8").copy()
    nogaia = star_id <= 0
    pos = (
        nogaia
        & np.isfinite(records.ra.to_numpy())
        & np.isfinite(records.dec.to_numpy())
    )
    labels = np.full(len(records), -1, "i8")
    labels[pos] = group_by_position(
        records.ra.to_numpy()[pos], records.dec.to_numpy()[pos], radius_arcsec
    )
    nopos = np.flatnonzero(nogaia & ~pos)
    base = labels.max() + 1 if pos.any() else 0
    labels[nopos] = base + np.arange(len(nopos))
    # Renumber by first appearance so ids follow the record order.
    sel = np.flatnonzero(nogaia)
    _, first, inverse = np.unique(labels[sel], return_index=True, return_inverse=True)
    rank = np.empty(len(first), "i8")
    rank[np.argsort(first)] = np.arange(len(first))
    star_id[sel] = -(rank[inverse] + 1)
    return star_id


def bitmask_names(values: pd.Series, *, split: str | None = None):
    """Encode categorical (or ``split``-joined) labels as int64 bitmasks."""
    raw = values.astype(str).to_numpy()
    uniq = pd.unique(raw)
    labels = sorted({x for u in uniq for x in (u.split(split) if split else [u]) if x})
    if len(labels) > 63:
        raise ValueError("Too many labels for an int64 bitmask")
    bit = {name: 1 << i for i, name in enumerate(labels)}
    code = {
        u: sum(bit[x] for x in set(u.split(split) if split else [u]) if x) for u in uniq
    }
    return pd.Series(raw).map(code).to_numpy("i8"), labels


def mask_to_text(masks: np.ndarray, labels: list[str]) -> np.ndarray:
    uniq, inverse = np.unique(masks, return_inverse=True)
    text = [";".join(n for i, n in enumerate(labels) if m >> i & 1) for m in uniq]
    return np.asarray(text, dtype=object)[inverse]


def star_table(records: pd.DataFrame) -> pd.DataFrame:
    """One row per star_id: identity, survey coverage, robust label summary.

    Medians use records with ``fit_ok``; [Fe/H] summaries use only records
    whose metallicity is [Fe/H] (not [M/H]). ``star_key`` is ``gaia:<id>`` or
    ``nogaia:<survey>:<native_id>`` of the star's first record.
    """
    sid = records.star_id.to_numpy("i8")
    order = np.argsort(sid, kind="stable")
    s = sid[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    star = s[starts]
    first = order[starts]
    ok = records.fit_ok.to_numpy(bool)
    # [Fe/H] on any solar scale (fe_h, fe_h_a09); not [M/H]-type quantities.
    kind = records.fe_h_kind.astype(str).to_numpy()
    feh_ok = ok & np.char.startswith(kind.astype("U"), "fe_h")

    def med(values, use):
        v = np.where(use, records[values].to_numpy("f8"), np.nan)
        return pd.Series(v).groupby(sid).median().to_numpy()

    feh = np.where(feh_ok, records.fe_h.to_numpy("f8"), np.nan)
    survey_mask, surveys = bitmask_names(records.survey)
    out = pd.DataFrame(
        dict(
            star_id=star,
            source_id=np.where(star > 0, star, 0),
            n_records=np.diff(np.r_[starts, len(s)]),
            survey_mask=np.bitwise_or.reduceat(survey_mask[order], starts),
            has_highres=np.logical_or.reduceat(
                (records.tier.astype(str).to_numpy() == "highres_reference")[order],
                starts,
            ),
            fe_h_median=pd.Series(feh).groupby(sid).median().to_numpy(),
            fe_h_median_highres=pd.Series(
                np.where(
                    records.tier.astype(str).to_numpy() == "highres_reference",
                    feh,
                    np.nan,
                )
            )
            .groupby(sid)
            .median()
            .to_numpy(),
            fe_h_min=pd.Series(feh).groupby(sid).min().to_numpy(),
            teff_median=med("teff", ok),
            logg_median=med("logg", ok),
            ra=pd.Series(records.ra.to_numpy("f8")).groupby(sid).median().to_numpy(),
            dec=pd.Series(records.dec.to_numpy("f8")).groupby(sid).median().to_numpy(),
        )
    )
    out["surveys"] = mask_to_text(out.survey_mask.to_numpy(), surveys)
    out["n_surveys"] = [bin(int(x)).count("1") for x in out.survey_mask]
    keys = np.array([f"gaia:{x}" for x in star], dtype=object)
    neg = star < 0
    keys[neg] = [
        f"nogaia:{records.survey.iat[i]}:{records.native_id.iat[i]}" for i in first[neg]
    ]
    out.insert(1, "star_key", keys)
    if "label_groups" in records:
        gm, glabels = bitmask_names(records.label_groups, split=";")
        out["label_groups"] = mask_to_text(
            np.bitwise_or.reduceat(gm[order], starts), glabels
        )
    for col in ("v2_role", "has_xp", "has_xp_source"):
        if col in records:
            out[col] = records[col].to_numpy()[first]
    return out


@dataclass
class Catalogue:
    """Where a record's native row lives; enough to read it back."""

    key: str
    path: str
    format: str  # fits | csv | vizier_tsv | census
    hdu: str = "1"
    release: str = ""
    url: str = ""
    native_id: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


def save_catalogues(catalogues: dict[str, Catalogue], path: Path, sha=None):
    out = {}
    for key, c in catalogues.items():
        entry = dict(c.__dict__)
        if sha is not None and Path(c.path).is_file():
            entry["sha256"] = sha(c.path)
        out[key] = entry
    path.write_text(json.dumps(out, indent=2) + "\n")


def read_native_rows(records: pd.DataFrame, catalogues: dict | str | Path) -> dict:
    """Full native rows for records, as {catalogue: DataFrame indexed by native_row}.

    ``catalogues`` is the master run's ``catalogues.json`` (path or dict).
    FITS rows come from the recorded HDU; DESI also returns the row-aligned
    GAIA/FIBERMAP/SPTAB columns. VizieR/SAGA/JINAbase rows are re-read with
    the census adapters from the cached downloads.
    """
    from astropy.io import fits

    if not isinstance(catalogues, dict):
        catalogues = json.loads(Path(catalogues).read_text())
    out = {}
    for key, block in records.groupby("catalogue"):
        c = catalogues[key]
        rows = np.unique(block.native_row.to_numpy("i8"))
        if c["format"] == "fits":
            with fits.open(c["path"], memmap=True) as f:
                hdus = [c["hdu"], *c.get("extra", {}).get("aligned_hdus", [])]
                parts = []
                for k, h in enumerate(hdus):
                    if h == "auto":  # first non-empty table (Astra summaries)
                        h = next(
                            i
                            for i, x in enumerate(f)
                            if i and x.header.get("NAXIS2", 0)
                        )
                    h = int(h) if str(h).isdigit() else h
                    data = f[h].data[rows]
                    part = pd.DataFrame(
                        {
                            n: list(data[n]) if data[n].ndim > 1 else data[n]
                            for n in data.columns.names
                        },
                        index=rows,
                    )
                    if k:
                        part = part.add_prefix(f"{h}.")
                    parts.append(part)
            out[key] = pd.concat(parts, axis=1)
        elif c["format"] == "csv":
            table = pd.read_csv(c["path"], low_memory=False)
            out[key] = table.iloc[rows].set_index(pd.Index(rows))
        else:
            from masked_stellar_autoencoder.training import master_labels as ml

            cache = Path(c["extra"]["cache"])
            if c["format"] == "vizier_tsv":
                spec = next(s for s in ml.VIZIER_SOURCES if s.source == key)
                raw, _ = ml.fetch_vizier(spec, cache)
            elif key == "saga_mw":
                raw, _ = ml.fetch_saga(cache)
            elif key == "jinabase_2021":
                raw, _ = ml.fetch_jinabase(cache)
            else:
                raise ValueError(f"No native reader for {key}")
            out[key] = raw.iloc[rows].set_index(pd.Index(rows))
    return out
