"""Source-bound seismic gravity QA of frozen saved predictions, plus source staging.

No model execution or fitting. APOKASC spectroscopic gravities must never be
confused with its seismic gravity column. FIT exposure is reported separately.
"""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.io.votable import parse
from build_age_anchor_labels import APOKASC3_COLUMNS

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    marginal_scores,
    summarize_residuals,
    summarize_scores,
)

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
PRED = ROOT + "multisurvey_joint_fullfit_20261105_v1/"
MAPPED = ROOT + "frozen_survey_qa_20261105_v2/"
APO = ROOT + "apokasc3_collection_v1/apokasc3-table4.dat.gz"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def summarize(w, mu, var, y, err):
    if not len(y):
        return dict(n=0)
    return dict(
        point=summarize_residuals((w * mu).sum(1), y),
        intrinsic_vs_reference=summarize_scores(marginal_scores(w, mu, var, y)),
        convolved_with_quoted_seismic_error=summarize_scores(
            marginal_scores(w, mu, var, y, err)
        ),
    )


def acquire_vizier(out, catalogue, name):
    url = "https://vizier.cfa.harvard.edu/viz-bin/votable?" + urllib.parse.urlencode(
        {"-source": catalogue, "-out.all": "", "-out.max": "1000000"}
    )
    path = out / (name + ".xml")
    with urllib.request.urlopen(url, timeout=180) as r, path.open("xb") as f:
        while b := r.read(1024 * 1024):
            f.write(b)
    v = parse(path)
    infos = [i for resource in v.resources for i in resource.infos]
    if any(
        i.name == "QUERY_STATUS" and i.value in ("ERROR", "OVERFLOW") for i in infos
    ):
        raise ValueError("Incomplete VizieR catalogue response: " + name)
    tables = []
    for t in v.iter_tables():
        if len(t.array) >= 1000000:
            raise ValueError("Possible VizieR truncation")
        tables.append(
            dict(
                name=t.name,
                rows=len(t.array),
                fields=[
                    dict(
                        name=f.name,
                        unit=str(f.unit),
                        datatype=f.datatype,
                        description=f.description,
                    )
                    for f in t.fields
                ],
            )
        )
    if not tables:
        raise ValueError("No source tables: " + name)
    return dict(
        url=url,
        sha256=sha(path),
        tables=tables,
        admission="Staged source only. Native TIC/EPIC identities require an audited Gaia DR3 bridge; positional or float-ID shortcuts forbidden.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bindings", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    bindings = json.loads(a.bindings.read_text())
    m = dict(
        status="running",
        training_performed=False,
        model_loaded=False,
        sources={},
        roles={},
        scope="Independent oscillation measurements with Teff and survey-calibration dependencies; no certified unseen-parent exposure claim. No source added to FIT.",
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    try:
        if (
            sha(APO)
            != "af151751c7dee1fee007b5703b9c252f68ba0736b3ed46a684e6ebe80ae4031d"
        ):
            raise ValueError("APOKASC source checksum differs")
        df = pd.read_csv(
            APO,
            sep=r"\s+",
            names=APOKASC3_COLUMNS,
            dtype=str,
            engine="python",
            compression="infer",
        )
        if len(df) != 15808:
            raise ValueError("APOKASC row count differs")
        # Parse IDs directly as nullable integers, never pd.to_numeric float paths.
        df["source_id"] = df.GaiaDR3.map(
            lambda s: int(s) if str(s).isdigit() else -1
        ).astype("int64")
        for key in ["loggSeis", "e_loggSeis", "loggSpec", "Teff", "FeH", "Numax"]:
            df[key] = pd.to_numeric(df[key], errors="coerce")
        df = df.loc[
            (df.source_id > 0)
            & df.CatTab.isin(["Gold", "Silver"])
            & df.loggSeis.between(-1, 6)
            & (df.e_loggSeis > 0)
            & np.isfinite(df.e_loggSeis)
        ].sort_values("source_id")
        if df.source_id.duplicated().any():
            raise ValueError("Duplicate APOKASC Gaia identities")
        df.to_parquet(a.output / "apokasc3-gravity-qa-reference.parquet", index=False)
        ids = df.source_id.to_numpy()
        m["sources"]["apokasc3"] = dict(
            raw_rows=15808,
            selected_rows=len(df),
            source_sha256=sha(APO),
            reference_sha256=sha(a.output / "apokasc3-gravity-qa-reference.parquet"),
            selected_categories=["Gold", "Silver"],
            reference_column="loggSeis",
            not_reference_column="loggSpec",
            dependence="Seismic logg uses spectroscopic Teff. APOGEE logg calibration uses Kepler seismology; parent/teacher exposure not certified. Age is not a gravity reference.",
        )
        save()
        for role in ["fit", "tune", "cal", "devtest", "external"]:
            path = PRED + "predictions-" + role + ".h5"
            if sha(path) != bindings["predictions"][role]:
                raise ValueError("Frozen prediction hash differs")
            with h5py.File(path) as f:
                pid = f["source_id"][:]
                pos = np.searchsorted(pid, ids)
                hit = pos < len(pid)
                hit[hit] &= pid[pos[hit]] == ids[hit]
                ref = df.iloc[np.flatnonzero(hit)]
                rows = pos[hit]
                w = f["weights"][:][rows]
                mu = f["component_mean"][:][rows, :, 1]
                lower = f["component_lower"][:][rows]
                var = (lower[:, :, 1, :] ** 2).sum(-1)
                native = f["native_value"][:][rows, :, 1]
                observed = f["native_observed"][:][rows, :, 1].astype(bool)
            y = ref.loggSeis.to_numpy()
            err = ref.e_loggSeis.to_numpy()
            models = {"common": (mu, var)}
            if role in ["devtest", "external"]:
                mp = MAPPED + "mapped-" + role + ".h5"
                if sha(mp) != bindings["mapped"][role]:
                    raise ValueError("Mapped prediction hash differs")
                with h5py.File(mp) as f:
                    np.testing.assert_array_equal(f["source_id"][:], pid)
                    mm = f["component_mean"][:][rows, :, :, 1]
                    vv = f["component_variance"][:][rows, :, :, 1]
                for j, name in enumerate(["APOGEE", "DESI", "GALAH"]):
                    models[name + "_mapped"] = (mm[:, j], vv[:, j])
            subsets = {"all": np.ones(len(ref), bool)}
            for state in ref.EvolSt.unique():
                subsets["state_" + state] = ref.EvolSt.to_numpy() == state
            for lo, hi in zip([-1, 1.5, 2, 2.5, 3, 3.5], [1.5, 2, 2.5, 3, 3.5, 6]):
                subsets[f"logg_{lo}_{hi}"] = (y >= lo) & (y < hi)
            entry = dict(
                matched_stars=len(ref),
                exposure="Direct FIT exposure"
                if role == "fit"
                else "Role-specific holdout from frozen fine tuning; historical parent/teachers may overlap",
                models={},
            )
            for name, (mean, variance) in models.items():
                entry["models"][name] = {
                    key: summarize(
                        w[keep], mean[keep], variance[keep], y[keep], err[keep]
                    )
                    for key, keep in subsets.items()
                }
            entry["native_label_vs_seismic"] = {}
            for j, name in enumerate(["APOGEE", "DESI", "GALAH"]):
                keep = observed[:, j] & np.isfinite(native[:, j])
                entry["native_label_vs_seismic"][name] = (
                    summarize_residuals(native[keep, j], y[keep])
                    if keep.any()
                    else dict(n=0)
                )
            m["roles"][role] = entry
            np.savez_compressed(
                a.output / (role + "-paired.npz"),
                source_id=ref.source_id.to_numpy(),
                logg_seismic=y,
                error=err,
                weights=w,
                component_mean=mu,
                component_variance=var,
                native_logg=native,
                native_observed=observed,
                evolutionary_state=ref.EvolSt.to_numpy(dtype="U"),
            )
            save()
        # These catalogues cover far more than the old 66-star dwarf panel.
        # Keep acquisition failures separate from the completed APOKASC QA.
        for name, cat in [
            ("tess_hon_2021", "J/ApJ/919/131/table1"),
            ("k2_gap_dr3", "J/ApJ/926/191/table2"),
            ("apokasc3_vizier", "J/ApJS/276/69/table4"),
        ]:
            try:
                m["sources"][name] = acquire_vizier(a.output, cat, name)
            except Exception as e:
                m["sources"][name] = dict(status="acquisition_failed", error=repr(e))
            save()
        m["status"] = "apokasc_gravity_qa_complete_other_sources_staged_or_recorded"
        save()
    except Exception as e:
        m.update(status="failed", error=repr(e))
        save()
        raise


if __name__ == "__main__":
    main()
