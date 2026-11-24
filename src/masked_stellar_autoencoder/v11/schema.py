"""v11 input contract: the canonical band list (UV -> mid-IR, XP, conditioning) and
the column contracts that read each band from a table.

The contract lives in ``contract.yaml`` next to this module and is validated on load
(unknown keys, duplicate names, bad kinds and malformed rules raise). Two contracts
are defined:

* ``gaia_keyed_v1a``: today's Gaia-keyed bands (Gaia G/BP/RP, XP, PS1, SDSS,
  SkyMapper, 2MASS, CatWISE, parallax/pm, SFD) in the native column names of each
  source, as partition-aligned trees joined on ``source_id``;
* ``gaia_master_v2``: the xmatcher ``ray-anchor`` full-wavelength master
  (``<column>_<tag>`` per survey, ``<tag>_quality`` match flags).

Both feed the same band list, so a model trained on v1a loads and keeps training on
the master without code changes: bands that a table lacks are absent for every row.

Reading (:class:`ContractReader.read`) turns an Arrow table into dense arrays
``values[n, n_bands]`` and ``errors[n, n_bands]`` in physical units (magnitudes,
mas, mas/yr, mag), NaN where a band is absent or fails its quality rules, plus the raw
XP coefficients, errors and packed correlations (normalisation to G = 8.5 and the
filter-integral quantities are computed by :mod:`masked_stellar_autoencoder.v11.data`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONTRACT_PATH = Path(__file__).with_name("contract.yaml")
KINDS = ("mag", "xp", "cond")
N_XP = 55
N_XP_CORR = N_XP * (N_XP - 1) // 2
MAG_PER_LN = 2.5 / math.log(10.0)  # 1.0857...: sigma_m = MAG_PER_LN * sigma_f / f
NMGY_ZP = 22.5

_BAND_KEYS = {
    "name",
    "survey",
    "kind",
    "system",
    "wavelength_nm",
    "svo",
    "xp_system",
    "dereddened",
}
_SPEC_KEYS = {
    "value",
    "index",
    "error",
    "error_index",
    "error_flux_over_error",
    "flux_nmgy",
    "ivar_nmgy",
    "where",
    "quality",
    "bright_limit",
    "faint_limit",
}
_RULE_KEYS = {"column", "rule", "values", "mask", "min", "max", "index"}
_RULES = {"in", "bits_none", "bits_any", "range", "char_in"}
_SURVEY_CONTRACT_KEYS = {"tag", "tag2", "quality", "bands", "verified"}


def _unknown(where: str, got: dict, allowed: set[str]) -> None:
    extra = sorted(set(got) - allowed)
    if extra:
        raise ValueError(f"{where}: unknown keys {extra} (allowed: {sorted(allowed)})")


@dataclass(frozen=True)
class Band:
    index: int
    name: str
    survey: str
    kind: str
    system: str = ""
    wavelength_nm: float = float("nan")
    svo: str | None = None
    xp_system: str | None = None
    dereddened: bool = False
    xp_band: str | None = None  # "bp" / "rp" for kind xp
    xp_coefficient: int = -1  # 0..54 for kind xp

    @property
    def reconstruct(self) -> bool:
        """Conditioning tokens (parallax, pm, SFD) are inputs only."""
        return self.kind != "cond"


@dataclass(frozen=True)
class BandRegistry:
    version: str
    bands: tuple[Band, ...]
    surveys: tuple[str, ...]
    survey_descriptions: dict[str, str]
    contracts: dict[str, dict]
    defaults: dict[str, float]
    _by_name: dict[str, int] = field(repr=False, default_factory=dict)

    def __len__(self) -> int:
        return len(self.bands)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bands)

    def index(self, name: str) -> int:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown band {name!r}") from None

    def __getitem__(self, name: str) -> Band:
        return self.bands[self.index(name)]

    def indices(self, kind: str | None = None, survey: str | None = None) -> np.ndarray:
        return np.asarray(
            [
                b.index
                for b in self.bands
                if (kind is None or b.kind == kind)
                and (survey is None or b.survey == survey)
            ],
            np.int64,
        )

    @property
    def xp_indices(self) -> np.ndarray:
        """110 band indices in coefficient order: BP 0..54, then RP 0..54."""
        return self.indices("xp")

    def survey_index(self) -> np.ndarray:
        return np.asarray([self.surveys.index(b.survey) for b in self.bands], np.int64)

    def kind_index(self) -> np.ndarray:
        """0 mag, 1 xp BP, 2 xp RP, 3 conditioning."""
        code = {"mag": 0, "cond": 3}
        return np.asarray(
            [
                (1 if b.xp_band == "bp" else 2) if b.kind == "xp" else code[b.kind]
                for b in self.bands
            ],
            np.int64,
        )


@lru_cache(maxsize=4)
def load_registry(path: str | Path | None = None) -> BandRegistry:
    """Load and validate ``contract.yaml``; XP coefficient bands are generated."""
    p = Path(path) if path is not None else CONTRACT_PATH
    raw = yaml.safe_load(p.read_text())
    _unknown(
        str(p), raw, {"version", "kinds", "surveys", "bands", "defaults", "contracts"}
    )
    if tuple(raw["kinds"]) != KINDS:
        raise ValueError(f"contract kinds must be {KINDS}")
    surveys = dict(raw["surveys"])
    for s, meta in surveys.items():
        _unknown(f"surveys.{s}", meta, {"description"})
    bands: list[Band] = []
    for i, b in enumerate(raw["bands"]):
        _unknown(f"bands[{i}]", b, _BAND_KEYS)
        if b["kind"] not in ("mag", "cond"):
            raise ValueError(
                f"band {b['name']}: kind {b['kind']} (xp bands are generated)"
            )
        if b["survey"] not in surveys:
            raise ValueError(f"band {b['name']}: unknown survey {b['survey']}")
        if b["kind"] == "mag" and not (b.get("wavelength_nm", 0) > 0):
            raise ValueError(f"band {b['name']}: magnitude band needs wavelength_nm")
        bands.append(
            Band(
                index=0,
                name=b["name"],
                survey=b["survey"],
                kind=b["kind"],
                system=b.get("system", ""),
                wavelength_nm=float(b.get("wavelength_nm", float("nan"))),
                svo=b.get("svo"),
                xp_system=b.get("xp_system"),
                dereddened=bool(b.get("dereddened", False)),
            )
        )
    # XP coefficients sit between the photometry and the conditioning tokens.
    first_cond = next(i for i, b in enumerate(bands) if b.kind == "cond")
    xp = [
        Band(0, f"xp_{band}_{k:02d}", "gaia_xp", "xp", xp_band=band, xp_coefficient=k)
        for band in ("bp", "rp")
        for k in range(N_XP)
    ]
    bands = bands[:first_cond] + xp + bands[first_cond:]
    if any(b.kind == "cond" for b in bands[: first_cond + len(xp)]):
        raise ValueError("conditioning bands must come last in contract.yaml")
    bands = [Band(**{**b.__dict__, "index": i}) for i, b in enumerate(bands)]
    names = [b.name for b in bands]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate band names {dup}")
    defaults = {"max_error_mag": 0.5, "min_error_mag": 0.0, **raw.get("defaults", {})}
    _unknown("defaults", defaults, {"max_error_mag", "min_error_mag"})
    contracts = raw["contracts"]
    by_name = {n: i for i, n in enumerate(names)}
    for cname, c in contracts.items():
        _validate_contract(cname, c, by_name, surveys)
    return BandRegistry(
        version=str(raw["version"]),
        bands=tuple(bands),
        surveys=tuple(surveys),
        survey_descriptions={k: v.get("description", "") for k, v in surveys.items()},
        contracts=contracts,
        defaults=defaults,
        _by_name=by_name,
    )


def _validate_rule(where: str, r: dict) -> None:
    _unknown(where, r, _RULE_KEYS)
    if r.get("rule") not in _RULES:
        raise ValueError(f"{where}: rule must be one of {sorted(_RULES)}")
    if "column" not in r:
        raise ValueError(f"{where}: rule needs a column")
    if r["rule"] in ("bits_none", "bits_any") and not isinstance(r.get("mask"), int):
        raise ValueError(f"{where}: {r['rule']} needs an integer mask")
    if r["rule"] in ("in", "char_in") and "values" not in r:
        raise ValueError(f"{where}: {r['rule']} needs values")
    if r["rule"] == "char_in" and not isinstance(r.get("index"), int):
        raise ValueError(f"{where}: char_in needs an integer index")
    if r["rule"] == "range" and "min" not in r and "max" not in r:
        raise ValueError(f"{where}: range needs min and/or max")


def _validate_contract(name: str, c: dict, by_name: dict, surveys: dict) -> None:
    _unknown(f"contracts.{name}", c, {"description", "xp", "surveys"})
    xp = c.get("xp")
    if xp is not None:
        _unknown(
            f"contracts.{name}.xp",
            xp,
            {"coefficients", "errors", "correlations", "g_mag"},
        )
        for k in ("coefficients", "errors", "correlations"):
            if set(xp.get(k, {})) != {"bp", "rp"}:
                raise ValueError(f"contracts.{name}.xp.{k} needs bp and rp")
        if "g_mag" not in xp:
            raise ValueError(f"contracts.{name}.xp needs g_mag (XP normalisation)")
    for s, sc in c["surveys"].items():
        where = f"contracts.{name}.surveys.{s}"
        if s not in surveys:
            raise ValueError(f"{where}: unknown survey")
        _unknown(where, sc, _SURVEY_CONTRACT_KEYS)
        for i, r in enumerate(sc.get("quality", [])):
            _validate_rule(f"{where}.quality[{i}]", r)
        for bname, specs in sc["bands"].items():
            if bname not in by_name:
                raise ValueError(f"{where}: unknown band {bname}")
            for j, spec in enumerate(specs if isinstance(specs, list) else [specs]):
                sw = f"{where}.bands.{bname}[{j}]"
                _unknown(sw, spec, _SPEC_KEYS)
                has_mag = "value" in spec
                has_flux = "flux_nmgy" in spec
                if has_mag == has_flux:
                    raise ValueError(f"{sw}: give exactly one of value / flux_nmgy")
                if has_flux and "ivar_nmgy" not in spec:
                    raise ValueError(f"{sw}: flux_nmgy needs ivar_nmgy")
                if "error" in spec and "error_flux_over_error" in spec:
                    raise ValueError(f"{sw}: two error specs")
                if "where" in spec:
                    _unknown(f"{sw}.where", spec["where"], {"column", "equals"})
                for k, r in enumerate(spec.get("quality", [])):
                    _validate_rule(f"{sw}.quality[{k}]", r)


# ----------------------------------------------------------------- table reading


@dataclass
class ContractArrays:
    """Dense per-row arrays in physical units (NaN = absent)."""

    values: np.ndarray  # [n, n_bands] float32 (XP slots left NaN; filled by data.py)
    errors: np.ndarray  # [n, n_bands] float32
    xp_coefficients: np.ndarray | None  # [n, 2, 55] float64 raw e-/s, NaN absent
    xp_errors: np.ndarray | None  # [n, 2, 55] float64
    xp_correlations: np.ndarray | None  # [n, 2, 1485] float32 packed, or None
    present_counts: dict[str, int]  # band name -> rows present


def _col(table, name: str):
    import pyarrow as pa

    arr = table.column(name)
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return arr


def _numeric(table, name: str, index: int | None = None) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    arr = _col(table, name)
    if index is not None:
        if not pa.types.is_list(arr.type) and not pa.types.is_fixed_size_list(arr.type):
            raise TypeError(f"column {name} is not a list column (index {index})")
        arr = pc.list_element(arr, index)
    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode()
    return np.asarray(
        pc.cast(arr, pa.float64()).to_numpy(zero_copy_only=False), np.float64
    )


def _list_matrix(table, name: str, width: int, dtype) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    arr = _col(table, name)
    n = len(arr)
    out = np.full((n, width), np.nan, dtype)
    lengths = pc.fill_null(pc.list_value_length(arr), -1).to_numpy(zero_copy_only=False)
    ok = lengths == width
    if ok.all():
        out[:] = (
            arr.flatten().to_numpy(zero_copy_only=False).reshape(n, width).astype(dtype)
        )
    elif ok.any():
        good = arr.filter(pa.array(ok)).flatten().to_numpy(zero_copy_only=False)
        out[ok] = good.reshape(-1, width).astype(dtype)
    return out


def _rule_mask(table, rule: dict) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    arr = _col(table, rule["column"])
    kind = rule["rule"]
    if kind == "in":
        values = rule["values"]
        if pa.types.is_dictionary(arr.type):
            arr = arr.dictionary_decode()
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            vs = pa.array([str(v) for v in values], arr.type)
        else:
            vs = pa.array(values).cast(arr.type)
        m = pc.is_in(arr, value_set=vs)
        return np.asarray(pc.fill_null(m, False).to_numpy(zero_copy_only=False), bool)
    if kind in ("bits_none", "bits_any"):
        a = arr.cast(pa.int64())
        valid = np.asarray(arr.is_valid().to_numpy(zero_copy_only=False), bool)
        v = np.asarray(pc.fill_null(a, 0).to_numpy(zero_copy_only=False), np.int64)
        hit = (v & np.int64(rule["mask"])) != 0
        return valid & (~hit if kind == "bits_none" else hit)
    if kind == "range":
        v = np.asarray(pc.cast(arr, pa.float64()).to_numpy(zero_copy_only=False))
        ok = np.isfinite(v)
        if "min" in rule:
            ok &= v >= rule["min"]
        if "max" in rule:
            ok &= v <= rule["max"]
        return ok
    if kind == "char_in":
        k = rule["index"]
        ch = pc.utf8_slice_codeunits(arr.cast(pa.string()), k, k + 1)
        m = pc.is_in(ch, value_set=pa.array(list(rule["values"]), pa.string()))
        return np.asarray(pc.fill_null(m, False).to_numpy(zero_copy_only=False), bool)
    raise ValueError(f"unknown rule {kind}")


def _subst(s: str, tags: dict[str, str]) -> str:
    for k, v in tags.items():
        s = s.replace("{" + k + "}", v)
    return s


def _subst_rule(r: dict, tags: dict[str, str]) -> dict:
    return {**r, "column": _subst(r["column"], tags)}


class ContractReader:
    """Read the bands of one column contract from Arrow tables.

    ``survey_tags`` overrides the contract's survey tags (``{survey: tag}`` or
    ``{survey: {tag: .., tag2: ..}}``). ``bands`` restricts the usable band names
    (others stay absent). ``allow_unverified`` admits mappings marked
    ``verified: false``.
    """

    def __init__(
        self,
        registry: BandRegistry,
        contract: str,
        *,
        survey_tags: dict | None = None,
        bands: list[str] | None = None,
        allow_unverified: bool = False,
    ):
        if contract not in registry.contracts:
            raise KeyError(
                f"unknown contract {contract!r}; have {sorted(registry.contracts)}"
            )
        self.registry = registry
        self.name = contract
        self.contract = registry.contracts[contract]
        self.allow_unverified = allow_unverified
        tags = survey_tags or {}
        unknown = sorted(set(tags) - set(self.contract["surveys"]))
        if unknown:
            raise ValueError(f"data.survey_tags: surveys {unknown} not in {contract}")
        keep = None if bands is None else set(bands)
        if keep is not None:
            bad = sorted(keep - set(registry.names))
            if bad:
                raise ValueError(f"unknown bands {bad}")
        self.max_error = float(registry.defaults["max_error_mag"])
        self.min_error = float(registry.defaults["min_error_mag"])
        self.plan: list[tuple[int, list[dict], list[dict], bool]] = []
        for survey, sc in self.contract["surveys"].items():
            t = {"tag": sc.get("tag", ""), "tag2": sc.get("tag2", "")}
            override = tags.get(survey)
            if isinstance(override, str):
                t["tag"] = override
            elif isinstance(override, dict):
                _unknown(f"data.survey_tags.{survey}", override, {"tag", "tag2"})
                t.update(override)
            squal = [_subst_rule(r, t) for r in sc.get("quality", [])]
            verified = bool(sc.get("verified", True))
            for bname, specs in sc["bands"].items():
                if keep is not None and bname not in keep:
                    continue
                specs = specs if isinstance(specs, list) else [specs]
                resolved = []
                for spec in specs:
                    r = dict(spec)
                    for k in ("value", "error", "error_flux_over_error"):
                        if k in r:
                            r[k] = _subst(r[k], t)
                    for k in ("flux_nmgy", "ivar_nmgy"):
                        if k in r:
                            r[k] = _subst(r[k], t)
                    if "where" in r:
                        r["where"] = {
                            **r["where"],
                            "column": _subst(r["where"]["column"], t),
                        }
                    r["quality"] = [_subst_rule(q, t) for q in r.get("quality", [])]
                    resolved.append(r)
                self.plan.append((registry.index(bname), resolved, squal, verified))
        self.use_xp = self.contract.get("xp") is not None and (
            keep is None or any(registry[n].kind == "xp" for n in keep)
        )

    def columns(self, available: set[str] | None = None) -> list[str]:
        """Every column the contract may read (restricted to ``available``)."""
        cols: set[str] = set()
        for _, specs, squal, _ in self.plan:
            for r in squal:
                cols.add(r["column"])
            for s in specs:
                for k in (
                    "value",
                    "error",
                    "error_flux_over_error",
                    "flux_nmgy",
                    "ivar_nmgy",
                ):
                    if k in s:
                        cols.add(s[k])
                if "where" in s:
                    cols.add(s["where"]["column"])
                for r in s["quality"]:
                    cols.add(r["column"])
        if self.use_xp:
            xp = self.contract["xp"]
            for k in ("coefficients", "errors", "correlations"):
                cols.update(xp[k].values())
            cols.add(xp["g_mag"])
        return sorted(cols if available is None else cols & set(available))

    def _spec_values(self, table, s: dict, n: int):
        names = set(table.column_names)
        main = s.get("value", s.get("flux_nmgy"))
        if main not in names:
            return None
        if "flux_nmgy" in s:
            if s["ivar_nmgy"] not in names:
                raise KeyError(
                    f"{self.name}: {main} present but {s['ivar_nmgy']} missing"
                )
            f = _numeric(table, s["flux_nmgy"])
            ivar = _numeric(table, s["ivar_nmgy"])
            ok = np.isfinite(f) & np.isfinite(ivar) & (f > 0) & (ivar > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.where(
                    ok, NMGY_ZP - 2.5 * np.log10(np.where(ok, f, 1.0)), np.nan
                )
                err = np.where(ok, MAG_PER_LN / (f * np.sqrt(ivar)), np.nan)
        else:
            val = _numeric(table, s["value"], s.get("index"))
            if "error" in s:
                if s["error"] not in names:
                    raise KeyError(
                        f"{self.name}: {main} present but {s['error']} missing"
                    )
                err = _numeric(table, s["error"], s.get("error_index"))
            elif "error_flux_over_error" in s:
                col = s["error_flux_over_error"]
                if col not in names:
                    raise KeyError(f"{self.name}: {main} present but {col} missing")
                foe = _numeric(table, col)
                with np.errstate(divide="ignore", invalid="ignore"):
                    err = np.where(foe > 0, MAG_PER_LN / foe, np.nan)
            else:
                err = np.full(n, np.nan)
        ok = np.isfinite(val)
        if "where" in s:
            w = s["where"]
            if w["column"] not in names:
                raise KeyError(f"{self.name}: where-column {w['column']} missing")
            ok &= _rule_mask(
                table, {"column": w["column"], "rule": "in", "values": [w["equals"]]}
            )
        for r in s["quality"]:
            if r["column"] not in names:
                raise KeyError(
                    f"{self.name}: quality column {r['column']} missing for {main}"
                )
            ok &= _rule_mask(table, r)
        if "bright_limit" in s:
            ok &= val >= s["bright_limit"]
        if "faint_limit" in s:
            ok &= val <= s["faint_limit"]
        return np.where(ok, val, np.nan), np.where(ok, err, np.nan)

    def read(self, table) -> ContractArrays:
        n = table.num_rows
        reg = self.registry
        values = np.full((n, len(reg)), np.nan, np.float32)
        errors = np.full((n, len(reg)), np.nan, np.float32)
        names = set(table.column_names)
        counts: dict[str, int] = {}
        squal_cache: dict[tuple, np.ndarray] = {}
        for bi, specs, squal, verified in self.plan:
            band = reg.bands[bi]
            touched = [s for s in specs if s.get("value", s.get("flux_nmgy")) in names]
            if not touched:
                continue
            if not verified and not self.allow_unverified:
                raise ValueError(
                    f"{self.name}: band {band.name} maps to unverified columns "
                    f"({touched[0].get('value')}); check them and set verified, or "
                    "set data.allow_unverified_columns: true"
                )
            key = tuple(sorted(r["column"] + r["rule"] for r in squal))
            if key not in squal_cache:
                ok = np.ones(n, bool)
                for r in squal:
                    if r["column"] not in names:
                        raise KeyError(
                            f"{self.name}: survey quality column {r['column']} missing "
                            f"(band {band.name})"
                        )
                    ok &= _rule_mask(table, r)
                squal_cache[key] = ok
            sok = squal_cache[key]
            val = np.full(n, np.nan)
            err = np.full(n, np.nan)
            for s in specs:
                got = self._spec_values(table, s, n)
                if got is None:
                    continue
                v, e = got
                fill = ~np.isfinite(val) & np.isfinite(v) & sok
                val[fill], err[fill] = v[fill], e[fill]
            if band.kind == "mag":
                bad_err = np.isfinite(err) & (err > self.max_error)
                val[bad_err] = np.nan
                err[np.isfinite(err) & (err <= self.min_error)] = np.nan
            err[~np.isfinite(val)] = np.nan
            values[:, bi] = val
            errors[:, bi] = err
            counts[band.name] = int(np.isfinite(val).sum())
        xc = xe = xr = None
        if self.use_xp:
            xp = self.contract["xp"]
            if xp["coefficients"]["bp"] in names:
                xc = np.stack(
                    [
                        _list_matrix(table, xp["coefficients"][b], N_XP, np.float64)
                        for b in ("bp", "rp")
                    ],
                    axis=1,
                )
                xe = np.stack(
                    [
                        _list_matrix(table, xp["errors"][b], N_XP, np.float64)
                        for b in ("bp", "rp")
                    ],
                    axis=1,
                )
                if xp["correlations"]["bp"] in names:
                    xr = np.stack(
                        [
                            _list_matrix(
                                table, xp["correlations"][b], N_XP_CORR, np.float32
                            )
                            for b in ("bp", "rp")
                        ],
                        axis=1,
                    )
        return ContractArrays(values, errors, xc, xe, xr, counts)

    def g_mag_column(self) -> str | None:
        xp = self.contract.get("xp")
        return None if xp is None else xp["g_mag"]


def xp_normalisation(g_mag: np.ndarray) -> np.ndarray:
    """f_G = 10**((8.5 - G) / 2.5): raw XP / f_G is the table (G = 8.5) scale."""
    return np.power(10.0, (8.5 - np.asarray(g_mag, np.float64)) / 2.5)


def describe(registry: BandRegistry | None = None) -> dict[str, Any]:
    """Band counts per kind and survey (for logs and the doc)."""
    reg = registry or load_registry()
    out: dict[str, Any] = {"version": reg.version, "n_bands": len(reg)}
    for k in KINDS:
        out[f"n_{k}"] = int((reg.indices(k)).size)
    out["surveys"] = {s: int(reg.indices(survey=s).size) for s in reg.surveys}
    return out
