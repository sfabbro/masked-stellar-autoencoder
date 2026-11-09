"""Local, motion-aware TESS/K2 QA candidates from our full raw Gaia input.

No upload, training or identity promotion. TESS positions declare epoch 2000;
K2 positions have unknown epochs and require agreement across three scenarios.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
from astropy.io.votable import parse_single_table
from scipy.spatial import cKDTree

from masked_stellar_autoencoder.training.astrometric_vectors import directions

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
RAW = Path("/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5")


def main():
    from xmatcher import CrossMatch, MatchRequest, MatchSpec, SideOverrides

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sm = json.loads((ROOT / "seismic_gravity_qa_20261106_v1/manifest.json").read_text())
    refs = []
    for name, idfield, nufield, enufield in [
        ("tess_hon_2021", "TIC", "numax", "e_numax"),
        ("k2_gap_dr3", "EPIC", "numax-c", "e_numax-c"),
    ]:
        path = ROOT / "seismic_gravity_qa_20261106_v1" / (name + ".xml")
        with path.open("rb") as f:
            if (
                hashlib.file_digest(f, "sha256").hexdigest()
                != sm["sources"][name]["sha256"]
            ):
                raise ValueError("Seismic catalogue changed")
        tab = parse_single_table(str(path)).to_table(use_names_over_ids=True)
        number = lambda key: np.asarray(
            np.ma.asarray(tab[key], dtype="f8").filled(np.nan)
        )
        native = np.asarray(tab[idfield])
        if native.dtype.kind not in "iu":
            raise ValueError("TIC/EPIC IDs must be exact integers")
        frame = pd.DataFrame(
            dict(
                catalogue=name,
                native_id=native.astype("i8"),
                ra=number("_RA"),
                dec=number("_DE"),
                numax=number(nufield),
                e_numax=number(enufield),
                teff=number("Teff"),
                e_teff=number("e_Teff"),
                mass_flag=number("Flag")
                if "Flag" in tab.colnames
                else np.full(len(tab), np.nan),
                ruwe=number("RUWE")
                if "RUWE" in tab.colnames
                else np.full(len(tab), np.nan),
            )
        )
        refs.append(frame)
    refs = pd.concat(refs, ignore_index=True)
    refs["row_id"] = np.arange(len(refs), dtype="i8")
    valid = (
        np.isfinite(refs.ra)
        & np.isfinite(refs.dec)
        & (refs.ra >= 0)
        & (refs.ra < 360)
        & (abs(refs.dec) <= 90)
    )
    targets = pl.from_pandas(refs.loc[valid, ["row_id", "ra", "dec"]])
    tree = cKDTree(directions(refs.loc[valid, "ra"], refs.loc[valid, "dec"]))
    radius = 2 * np.sin(np.deg2rad(10 / 3600) / 2)
    census = json.loads(
        (ROOT / "andrae_full_footprint_census_20261106_v2/manifest.json").read_text()
    )
    before = RAW.stat()
    identity = census["raw_identity"]
    if (
        before.st_size != identity["bytes"]
        or before.st_mtime_ns != identity["mtime_ns"]
    ):
        raise ValueError("Raw Gaia identity differs")
    m = dict(
        status="scanning_raw_input",
        training_performed=False,
        identity_promoted=False,
        external_data_upload=False,
        rows_scanned=0,
        reference_rows=len(refs),
        valid_position_rows=int(valid.sum()),
        raw_identity=identity,
        scope="Public TESS/K2 source-only local QA matching; subset of Gaia in our raw input, not the entire Gaia catalogue. No use as training labels or final model selection.",
    )

    def save():
        p = a.output / "manifest.tmp"
        p.write_text(json.dumps(m, indent=2) + "\n")
        p.replace(a.output / "manifest.json")

    save()
    parts = []
    with h5py.File(RAW) as raw:
        keys = [k for k in raw if isinstance(raw[k], h5py.Dataset)]
        for key in keys:
            d = raw[key]
            fields = ["source_id", "RA", "DEC", "pmra", "pmdec"]
            if not set(fields).issubset(d.dtype.names or ()):
                raise ValueError("Missing coordinate schema")
            for start in range(0, len(d), 250000):
                b = d.fields(fields)[start : start + 250000]
                sid = b["source_id"]
                if sid.dtype.kind not in "iu" or (
                    sid.dtype.kind == "u" and (sid > np.iinfo("i8").max).any()
                ):
                    raise ValueError("Gaia IDs cannot pass through floats")
                ra, dec, pmra, pmdec = [
                    np.asarray(b[n], dtype="f8") for n in fields[1:]
                ]
                good = (
                    (sid > 0)
                    & np.isfinite(ra)
                    & np.isfinite(dec)
                    & (ra >= 0)
                    & (ra < 360)
                    & (abs(dec) <= 90)
                    & np.isfinite(pmra)
                    & np.isfinite(pmdec)
                    & (pmra != -9999)
                    & (pmdec != -9999)
                )
                take = np.flatnonzero(good)
                selected = np.zeros(len(b), bool)
                for epoch in [2000.0, 2010.0, 2016.0]:
                    if not len(take):
                        continue
                    dist, _ = tree.query(
                        directions(
                            ra[take], dec[take], pmra[take], pmdec[take], epoch - 2016
                        ),
                        k=1,
                        distance_upper_bound=radius,
                        workers=4,
                    )
                    selected[take[np.isfinite(dist)]] = True
                ii = np.flatnonzero(selected)
                if len(ii):
                    parts.append(
                        pl.DataFrame(
                            dict(
                                source_id=sid[ii].astype("i8"),
                                ra=ra[ii],
                                dec=dec[ii],
                                pmra=pmra[ii],
                                pmdec=pmdec[ii],
                            )
                        )
                    )
                m["rows_scanned"] += len(b)
            m["last_shard"] = key
            save()
    after = RAW.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Raw source changed")
    if not parts:
        raise ValueError("No candidate positions")
    candidates = pl.concat(parts).unique()
    conflicts = candidates.group_by("source_id").len().filter(pl.col("len") > 1)
    candidates = candidates.join(
        conflicts.select("source_id"), on="source_id", how="anti"
    ).sort("source_id")
    candidates.write_parquet(a.output / "raw-gaia-candidates.parquet")
    c = candidates.to_pandas().set_index("source_id")
    epoch_results = {}
    for epoch in [2000.0, 2010.0, 2016.0]:
        pairs = CrossMatch().crossmatch_request(
            MatchRequest(
                cat1=targets,
                cat2=candidates,
                engine="fast",
                spec=MatchSpec(
                    radius_arcsec=2.0,
                    find="all",
                    target_epoch=epoch,
                    fallback_policy="error",
                ),
                side1=SideOverrides(id_column="row_id", epoch=epoch, frame="icrs"),
                side2=SideOverrides(
                    id_column="source_id",
                    epoch=2016.0,
                    pm_ra_column="pmra",
                    pm_dec_column="pmdec",
                    frame="icrs",
                ),
            )
        )
        df = pairs.select(["row_id", "source_id"]).unique().to_pandas()
        row = df.row_id.to_numpy(dtype="i8")
        si = df.source_id.to_numpy(dtype="i8")
        b = c.loc[si]
        left = directions(refs.loc[row, "ra"], refs.loc[row, "dec"])
        right = directions(b.ra, b.dec, b.pmra, b.pmdec, epoch - 2016)
        df["separation_arcsec"] = (
            np.rad2deg(
                2 * np.arcsin(np.minimum(1, np.linalg.norm(left - right, axis=1) / 2))
            )
            * 3600
        )
        df = df.sort_values(["row_id", "separation_arcsec"])
        first = df.groupby("row_id", sort=False).head(1).copy()
        # Compute the runner-up by an explicit per-target sequence, independent of pandas index semantics.
        second_map = {
            int(r): float(g.separation_arcsec.iloc[1])
            for r, g in df.groupby("row_id")
            if len(g) > 1
        }
        first["runner_up_arcsec"] = [
            second_map.get(int(r), np.inf) for r in first.row_id
        ]
        first = first[
            (first.separation_arcsec <= 0.5) & (first.runner_up_arcsec >= 1.0)
        ].set_index("row_id")
        epoch_results[epoch] = first
        df.to_parquet(a.output / f"epoch-{int(epoch)}-all-pairs.parquet", index=False)
    selected = []
    for row, ref in refs.iterrows():
        epochs = (
            [2000.0] if ref.catalogue == "tess_hon_2021" else [2000.0, 2010.0, 2016.0]
        )
        if not all(row in epoch_results[e].index for e in epochs):
            continue
        best = [int(epoch_results[e].loc[row, "source_id"]) for e in epochs]
        if len(set(best)) != 1:
            continue
        selected.append(
            dict(
                ref,
                source_id=best[0],
                maximum_separation_arcsec=max(
                    float(epoch_results[e].loc[row, "separation_arcsec"])
                    for e in epochs
                ),
            )
        )
    output = pd.DataFrame(selected)
    output = output[~output.duplicated(["catalogue", "source_id"], keep=False)].copy()
    tess = output.catalogue == "tess_hon_2021"
    quality = (
        tess
        & (output.numax > 0)
        & (output.e_numax > 0)
        & (output.e_numax / output.numax < 0.2)
        & (output.teff > 0)
        & (output.e_teff > 0)
    )
    output["logg_nu_scaling"] = np.nan
    output["e_logg_stat"] = np.nan
    output.loc[quality, "logg_nu_scaling"] = (
        4.438
        + np.log10(output.loc[quality, "numax"] / 3090)
        + 0.5 * np.log10(output.loc[quality, "teff"] / 5772)
    )
    output.loc[quality, "e_logg_stat"] = np.sqrt(
        (output.loc[quality, "e_numax"] / output.loc[quality, "numax"]) ** 2
        + (0.5 * output.loc[quality, "e_teff"] / output.loc[quality, "teff"]) ** 2
    ) / np.log(10)
    output.to_parquet(a.output / "seismic-gaia-qa-candidates.parquet", index=False)
    m.update(
        status="conservative_seismic_gaia_candidates_complete",
        candidate_gaia_stars=candidates["source_id"].n_unique(),
        conflicting_astrometry_ids=conflicts.height,
        matched_reference_rows=len(output),
        matched_by_catalogue=output.groupby("catalogue").size().to_dict(),
        tess_with_nu_scaling_gravity=int(quality.sum()),
        solar_convention=dict(numax_microhz=3090, teff_kelvin=5772, logg_cgs=4.438),
        tess_mass_flag_scope="Retained, not a primary filter: the mass flag may use Gaia geometry. RUWE is retained for slices, not forced normal.",
        k2_scope="Unknown position epochs require a stable unique counterpart across three scenarios. Gravity not derived until pipeline/evolutionary-state solar reference conventions are audited.",
        reference_dependence="nu-max scaling uses external Teff, shared solar calibration and potentially shared photometric inputs. Formal errors omit global scaling systematics; adopted solar nu uncertainty 30/3090 corresponds to about .0042 dex common offset.",
        source_reference="Hon et al 2021, https://doi.org/10.3847/1538-4357/ac14b1; K2 GAP DR3 https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/926/191",
    )
    with (a.output / "seismic-gaia-qa-candidates.parquet").open("rb") as f:
        m["candidate_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output") + 1]) / "manifest.json"
            try:
                state = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                state = {}
            state.update(status="seismic_candidate_staging_failed", error=repr(exc))
            path.write_text(json.dumps(state, indent=2) + "\n")
        raise
