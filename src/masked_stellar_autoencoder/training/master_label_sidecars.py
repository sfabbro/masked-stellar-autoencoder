"""Training sidecars for master-label sources (Teff, logg, [Fe/H] only).

Turns the census observation table (``master_labels``) into
``write_observation_sidecar`` blocks, one group per training source:

* one record per Gaia star per source. Survey rows: best quality, then highest
  S/N. Merged sources: per-star median with unknown (NaN) error;
* a value is observed only when it is quality-passing and finite. A grid-edge
  (censored) [Fe/H] is *not* observed, because the trainer has no censored
  likelihood;
* DR19 ASPCAP contributes only SDSS-V stars without a DR17 row, since DR17 is
  already a staged source with the same spectra and pipeline;
* the high-resolution literature tier (SAGA, JINAbase, PASTEL, RPA, Yong,
  Roederer) is merged into one source, because compilations republish the
  same papers. Model [M/H] (Roederer) is not used as [Fe/H];
* SkyMapper Da Costa tables 2 and 3 are one pipeline, so they become one source.

Roles are assigned later by the source builder (stable Gaia-ID hash, with
benchmark and globular-cluster stars external). Nothing here enables training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS = ("teff", "logg", "fe_h")


@dataclass(frozen=True)
class TrainingSource:
    name: str
    members: tuple[str, ...]
    merge: bool
    fields: tuple[str | None, str | None, str | None]
    note: str = ""
    # Stars already labelled by these census sources are left out of this group.
    exclude_members: tuple[str, ...] = ()


TRAINING_SOURCES: tuple[TrainingSource, ...] = (
    TrainingSource(
        "sdss_dr19_aspcap_sdss5",
        ("sdss_dr19_astra_aspcap",),
        False,
        ("teff", "logg", "fe_h"),
        "SDSS-V stars only; DR17 rows are the staged apogee source",
    ),
    TrainingSource(
        "sdss_dr12_sspp",
        ("sdss_dr12_sspp",),
        False,
        ("TEFF_ADOP", "LOGG_ADOP", "FEH_ADOP"),
    ),
    TrainingSource(
        "gaia_eso_dr5_1", ("gaia_eso_dr5_1",), False, ("TEFF", "LOGG", "FEH")
    ),
    TrainingSource("s5_dr1", ("s5_dr1",), False, ("teff50", "logg50", "feh50")),
    TrainingSource("pigs_aat", ("pigs_aat",), False, ("TeffF", "loggF", "[Fe/H]F")),
    TrainingSource(
        "lamost_vmp_li2018",
        ("lamost_vmp_li2018",),
        False,
        ("Teff", "log(g)", "[Fe/H]"),
        "same spectra as staged LAMOST LASP; separate pipeline",
    ),
    TrainingSource(
        "lamost_cat_li2025",
        ("lamost_cat_li2025",),
        False,
        (None, None, "[Fe/H]"),
        "Ca triplet [Fe/H] only; catalogue Teff/logg are LASP",
    ),
    TrainingSource(
        "pristine_aguado2019",
        ("pristine_aguado2019",),
        False,
        ("Teff", "logg", "[Fe/H]F"),
    ),
    TrainingSource(
        "smss_dacosta2019",
        ("smss_dacosta2019_t2", "smss_dacosta2019_t3"),
        True,
        ("Teff", "logg", "[Fe/H]f"),
    ),
    TrainingSource(
        "bestbrightest_placco2019",
        ("bestbrightest_placco2019",),
        False,
        ("Teff", "logg", "[Fe/H]"),
    ),
    TrainingSource(
        "highres_literature",
        (
            "saga_mw",
            "jinabase_2021",
            "pastel_2020",
            "rpa_hansen2018",
            "rpa_holmbeck2020",
            "smss_yong2021",
            "yong2013_program",
            "yong2013_literature",
            "roederer2014",
        ),
        True,
        ("Teff", "logg", "[Fe/H]"),
        "per-star median over compiled papers; reference tier, not physical truth",
    ),
    # ---- Added for the XP-only v3 run (census v6).
    TrainingSource(
        "highres_literature_extra",
        (
            "ezzeddine2020_rpa",
            "sakari2018_rpa",
            "li2022_lamost_subaru",
            "aoki2013_segue",
            "kirby2010_highres",
            "bensby2014",
        ),
        True,
        ("Teff", "logg", "[Fe/H]"),
        "per-star median; stars already in highres_literature are left out "
        "(SAGA/JINAbase compile many of these papers)",
        exclude_members=(
            "saga_mw",
            "jinabase_2021",
            "pastel_2020",
            "rpa_hansen2018",
            "rpa_holmbeck2020",
            "smss_yong2021",
            "yong2013_program",
            "yong2013_literature",
            "roederer2014",
        ),
    ),
    TrainingSource(
        "kirby2010_dsph",
        ("kirby2010_dsph",),
        False,
        ("Teff", "logg", "[Fe/H]"),
        "DEIMOS spectral synthesis in eight Milky Way dwarf spheroidals",
    ),
    TrainingSource(
        "mastar_params",
        ("sdss_dr17_mastar_params",),
        False,
        ("TEFF_MED", "LOGG_MED", "FEH_CAL_MED"),
        "MaStar VAC v2 median of four methods; calibrated [Fe/H]",
    ),
)

# Overlap-report entries used to choose each training source's map family.
OVERLAP_ALIASES = {
    "sdss_dr19_aspcap_sdss5": "sdss_dr19_astra_aspcap",
    "smss_dacosta2019": "smss_dacosta2019_t2",
    "highres_literature": None,
}


def _select_rows(obs: pd.DataFrame, spec: TrainingSource) -> pd.DataFrame:
    rows = obs[obs.source.isin(spec.members) & (obs.source_id > 0) & obs.quality]
    if spec.exclude_members:
        other = obs[
            obs.source.isin(spec.exclude_members)
            & (obs.source_id > 0)
            & obs.quality
            & obs[list(TARGETS)].notna().any(axis=1)
        ]
        rows = rows[~rows.source_id.isin(other.source_id)]
    if spec.name == "sdss_dr19_aspcap_sdss5":
        release = rows.native_id.str.split(":").str[2]
        dr17_stars = rows.source_id[release == "dr17"].unique()
        rows = rows[(release != "dr17") & ~rows.source_id.isin(dr17_stars)]
    rows = rows.copy()
    rows.loc[rows.fe_h_grid_edge, "fe_h"] = np.nan
    rows.loc[rows.fe_h_kind == "m_h_model", "fe_h"] = np.nan
    for target, field in zip(TARGETS, spec.fields, strict=True):
        if field is None:
            rows[target] = np.nan
            rows["e_" + target] = np.nan
    return rows


def build_block(obs: pd.DataFrame, spec: TrainingSource) -> dict | None:
    """Arrays for ``write_observation_sidecar`` (one record per star)."""
    rows = _select_rows(obs, spec)
    has = rows[list(TARGETS)].notna().any(axis=1)
    rows = rows[has]
    if rows.empty:
        return None
    if spec.merge:
        grouped = rows.groupby("source_id")
        value = grouped[list(TARGETS)].median()
        ids = value.index.to_numpy("i8")
        value = value.to_numpy("f8")
        error = np.full(value.shape, np.nan)
        source_row = np.arange(len(ids), dtype="i8")
        n_rows = grouped.size().to_numpy("i8")
    else:
        rows = rows.assign(
            has_feh=rows.fe_h.notna(), snr_rank=rows.snr.fillna(-np.inf)
        ).sort_values(
            ["source_id", "has_feh", "snr_rank"], ascending=[True, False, False]
        )
        rows = rows.drop_duplicates("source_id")
        ids = rows.source_id.to_numpy("i8")
        value = rows[list(TARGETS)].to_numpy("f8")
        error = rows[["e_" + t for t in TARGETS]].to_numpy("f8")
        source_row = rows.native_row.to_numpy("i8")
        n_rows = np.ones(len(ids), dtype="i8")
        if len(np.unique(source_row)) != len(source_row):
            # Distinct census sources can share native row numbers; renumber.
            source_row = np.arange(len(ids), dtype="i8")
    observed = np.isfinite(value)
    observed[:, 0] &= value[:, 0] > 0
    for j, field in enumerate(spec.fields):
        if field is None:
            observed[:, j] = False
    value = np.where(observed, value, np.nan)
    error = np.where(observed & np.isfinite(error) & (error > 0), error, np.nan)
    return dict(
        source_id=ids,
        source_row=source_row,
        value=value,
        error=error,
        observed=observed,
        error_known=np.isfinite(error),
        native_rows_merged=n_rows,
    )


def registry_entry(obs: pd.DataFrame, spec: TrainingSource) -> dict:
    members = obs[obs.source.isin(spec.members)]
    targets = {}
    for target, unit, field in zip(
        TARGETS, ("K", "dex(cm/s2)", "dex"), spec.fields, strict=True
    ):
        targets[target] = (
            None if field is None else dict(quantity=target, unit=unit, field=field)
        )
    return dict(
        release="; ".join(sorted(members.release.unique())),
        pipeline="; ".join(sorted(members.pipeline.unique())),
        dependency_id=(
            "literature_highres_compilation"
            if spec.name in ("highres_literature", "highres_literature_extra")
            else "; ".join(sorted(members.dependency_group.unique()))
        ),
        targets=targets,
        members=list(spec.members),
        merge="per-star median, unknown error" if spec.merge else "best row per star",
        censoring="grid-edge [Fe/H] not observed",
        note=spec.note,
        training_enabled=False,
    )


def build_all(obs: pd.DataFrame, names=None) -> tuple[dict, dict, dict]:
    """(observations, registry, counts) for the selected training sources."""
    blocks, registry, counts = {}, {}, {}
    for spec in TRAINING_SOURCES:
        if names is not None and spec.name not in names:
            continue
        block = build_block(obs, spec)
        if block is None:
            counts[spec.name] = dict(stars=0)
            continue
        blocks[spec.name] = block
        registry[spec.name] = registry_entry(obs, spec)
        counts[spec.name] = dict(
            stars=int(len(block["source_id"])),
            observed_per_target=block["observed"].sum(0).tolist(),
            known_error_per_target=block["error_known"].sum(0).tolist(),
            fe_h_below_minus2=int(
                (block["observed"][:, 2] & (block["value"][:, 2] < -2)).sum()
            ),
            fe_h_below_minus3=int(
                (block["observed"][:, 2] & (block["value"][:, 2] < -3)).sum()
            ),
        )
    return blocks, registry, counts
