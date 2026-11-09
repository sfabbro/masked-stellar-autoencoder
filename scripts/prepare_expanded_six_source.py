"""Build a six-target source with exact-ID features, source records and protected roles.

Staged public survey sidecars retain their own quality masks and quantity
semantics. No checkpoint execution or automatic reinterpretation of M/H/alpha/M.
"""

import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def role_buckets(ids):
    z = np.asarray(ids, dtype="u8").copy()
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    buckets = z % 100
    return np.where(
        buckets < 75,
        b"fit",
        np.where(buckets < 85, b"tune", np.where(buckets < 92, b"cal", b"devtest")),
    ).astype("S8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    c = json.loads(a.config.read_text())
    m = dict(
        status="admitting",
        training_performed=False,
        model_loaded=False,
        config_sha256=sha(a.config),
        sources={},
        training_source_ready=False,
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    try:
        for key, expected in [
            ("base", c["base_sha256"]),
            ("new_features", c["new_features_sha256"]),
        ]:
            if sha(c[key]) != expected:
                raise ValueError("Source identity changed: " + key)
        parallax = ROOT / "expanded_raw_parallax_20261107_v1"
        deadline = time.monotonic() + 12 * 3600
        while True:
            try:
                pm = json.loads((parallax / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pm = {}
            if pm.get("status") == "failed":
                raise RuntimeError(
                    "Native expanded parallax recovery failed: " + str(pm.get("error"))
                )
            if pm.get("status") == "complete":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Expanded native parallax recovery did not finish in twelve hours"
                )
            m["status"] = "waiting_for_native_parallax"
            save()
            time.sleep(30)
        parallax_path = parallax / "native-parallax.h5"
        if sha(parallax_path) != pm["output_sha256"]:
            raise ValueError("Expanded native parallax digest changed")
        gcpath = a.output / "clusters.zip"
        with (
            urllib.request.urlopen(
                "https://zenodo.org/records/4891252/files/clusters.zip?download=1",
                timeout=180,
            ) as src,
            gcpath.open("xb") as out,
        ):
            while block := src.read(8 * 1024 * 1024):
                out.write(block)
        if (
            sha(gcpath)
            != "9eafa2e10afc031790cfe36ed8545aa6336cc87c317fab167e9cea9dacbd0748"
        ):
            raise ValueError("Globular membership reference changed")
        clusters = []
        with zipfile.ZipFile(gcpath) as z:
            for name in sorted(z.namelist()):
                if not name.startswith("catalogues/") or not name.endswith(".txt"):
                    continue
                text = z.read(name).decode()
                columns = text.splitlines()[0].lstrip("#").split()
                tab = pd.read_csv(
                    io.StringIO(text),
                    sep=r"\s+",
                    comment="#",
                    names=columns,
                    dtype={"source_id": np.int64},
                )
                take = ((tab.qflag.to_numpy(dtype=int) & 2) != 0) & (
                    tab.memberprob.to_numpy() > 0.9
                )
                clusters.append(tab.source_id.to_numpy()[take])
        protected_gc = np.unique(np.concatenate(clusters))
        benchmark = np.asarray(c["benchmark_fit_exclusion_ids"], dtype="i8")
        with (
            h5py.File(c["base"]) as base,
            h5py.File(c["new_features"]) as new,
            h5py.File(parallax_path) as native,
        ):
            oldids = base["source_id"][:]
            newids = new["source_id"][:]
            np.testing.assert_array_equal(newids, native["source_id"][:])
            selected = new["feature_recoverable"][:].astype(bool) & new["input_has_xp"][
                :
            ].astype(bool)
            newrows = np.flatnonzero(selected)
            extra = newids[newrows]
            if np.intersect1d(oldids, extra).size:
                raise ValueError("Expanded feature IDs overlap base")
            ids = np.sort(np.concatenate([oldids, extra]))
            if (np.diff(ids) <= 0).any():
                raise ValueError("Combined IDs are not unique")
            names = base["feature_names"][:]
            np.testing.assert_array_equal(names, new["feature_names"][:])
            for key in ["scaler_center", "scaler_scale"]:
                np.testing.assert_array_equal(base[key][:], new[key][:])
            oldpos = np.searchsorted(ids, oldids)
            extra_pos = np.searchsorted(ids, extra)
            origin = np.full(len(ids), -1, dtype="i8")
            origin[oldpos] = np.arange(len(oldids))
            new_origin = np.full(len(ids), -1, dtype="i8")
            new_origin[extra_pos] = newrows
            roles = role_buckets(ids)
            roles[oldpos] = base["role"][:].astype("S8")
            bmask = np.isin(ids, benchmark)
            gmask = np.isin(ids, protected_gc)
            original_fit_removed = int(((roles == b"fit") & (bmask | gmask)).sum())
            roles[bmask | gmask] = b"external"
            eligible = np.ones(len(ids), bool)
            eligible[oldpos] = base["primary_input_eligible"][:].astype(bool)
            output = a.output / "prepared.h5"
            m.update(
                status="writing_feature_union",
                base_stars=len(oldids),
                expanded_feature_stars=len(extra),
                total_stars=len(ids),
                benchmark_stars=int(bmask.sum()),
                protected_globular_members=int(gmask.sum()),
                prior_fit_moved_to_external=original_fit_removed,
            )
            save()
            with h5py.File(output, "x") as f:
                f.attrs["schema"] = "expanded-sparse-six-target-source-v1"
                f.attrs["scope"] = (
                    "Common fixed parent feature scaler; signed native parallax with formal errors; source-native observation groups; inherited base roles plus hashed new-star roles; all identified benchmark and high-probability globular members reserved from new fitting. Ancestor exposure remains."
                )
                f["source_id"] = ids
                f["role"] = roles
                f["primary_input_eligible"] = eligible
                f["reserved_benchmark"] = bmask
                f["reserved_globular_member"] = gmask
                f["feature_names"] = names
                f["target_names"] = base["target_names"][:]
                f["scaler_center"] = base["scaler_center"][:]
                f["scaler_scale"] = base["scaler_scale"][:]
                width = base["input"].shape[1]
                datasets = {}
                for name, dtype, shape, fill in [
                    ("input", "f4", (len(ids), width), -9999),
                    ("value", "f8", (len(ids), 6), np.nan),
                    ("error", "f8", (len(ids), 6), np.nan),
                    ("observed", "?", (len(ids), 6), False),
                    ("quality", "?", (len(ids), 6), False),
                    ("error_observed", "?", (len(ids), 6), False),
                    ("label_source", "S64", (len(ids), 6), b""),
                ]:
                    datasets[name] = f.create_dataset(
                        name,
                        shape=shape,
                        dtype=dtype,
                        chunks=(min(8192, len(ids)), shape[1]),
                        compression="lzf",
                        fillvalue=fill,
                    )
                ast = f.create_group("measured_parallax")
                for key, dtype, fill in [
                    ("input_value", "f4", -9999),
                    ("parallax_mas", "f8", np.nan),
                    ("parallax_error_mas", "f8", np.nan),
                    ("observed", "?", False),
                ]:
                    ast.create_dataset(
                        key,
                        shape=(len(ids),),
                        dtype=dtype,
                        chunks=True,
                        compression="lzf",
                        fillvalue=fill,
                    )
                j = names.astype("U").tolist().index("PARALLAX")
                for start in range(0, len(ids), 8192):
                    stop = min(start + 8192, len(ids))
                    ob = origin[start:stop]
                    en = new_origin[start:stop]
                    oldmask = ob >= 0
                    newmask = en >= 0
                    if not np.all(oldmask ^ newmask):
                        raise ValueError("Ambiguous feature origin")
                    for name, dst in datasets.items():
                        v = np.full(
                            (stop - start, dst.shape[1]), dst.fillvalue, dtype=dst.dtype
                        )
                        if oldmask.any():
                            rr = ob[oldmask]
                            lo, hi = int(rr[0]), int(rr[-1]) + 1
                            v[oldmask] = base[name][lo:hi][rr - lo]
                        if name == "input" and newmask.any():
                            rr = en[newmask]
                            lo, hi = int(rr[0]), int(rr[-1]) + 1
                            v[newmask] = new["input"][lo:hi][rr - lo]
                        if name == "input" and not np.isfinite(v).all():
                            raise ValueError(
                                "Nonfinite input in expanded feature union"
                            )
                        dst[start:stop] = v
                    for key in ast:
                        dst = ast[key]
                        v = np.full(stop - start, dst.fillvalue, dtype=dst.dtype)
                        if oldmask.any():
                            rr = ob[oldmask]
                            lo, hi = int(rr[0]), int(rr[-1]) + 1
                            v[oldmask] = base["measured_parallax/" + key][lo:hi][
                                rr - lo
                            ]
                        if newmask.any():
                            rr = en[newmask]
                            lo, hi = int(rr[0]), int(rr[-1]) + 1
                            available = native["observed"][lo:hi][rr - lo]
                            if key == "input_value":
                                raw = native["parallax_mas"][lo:hi][rr - lo]
                                v[newmask] = np.where(
                                    available,
                                    (raw - base["scaler_center"][j])
                                    / base["scaler_scale"][j],
                                    -9999,
                                )
                            else:
                                v[newmask] = native[key][lo:hi][rr - lo]
                        dst[start:stop] = v
                    if start % 262144 == 0:
                        m["feature_rows_written"] = stop
                        save()

                def write_records(name, sid, value, error, observed, known, definition):
                    # Public sidecars preserve publisher row order; join by ID,
                    # sorting every aligned field together in this new bundle.
                    order = np.argsort(sid, kind="stable")
                    sid, value, error, observed, known = [
                        array[order] for array in [sid, value, error, observed, known]
                    ]
                    if (
                        np.any(sid <= 0)
                        or np.any(np.diff(sid) <= 0)
                        or not np.isfinite(value[observed]).all()
                    ):
                        raise ValueError(
                            "Native records require sorted unique exact IDs and finite observed values"
                        )
                    pos = np.searchsorted(ids, sid)
                    hit = pos < len(ids)
                    hit[hit] &= ids[pos[hit]] == sid[hit]
                    hit &= observed.any(1)
                    group = f.create_group("native_observations/" + name)
                    group["star_row"] = pos[hit]
                    for key, array in [
                        ("value", value[hit]),
                        ("error", error[hit]),
                        ("observed", observed[hit]),
                        ("error_known", known[hit] & observed[hit]),
                    ]:
                        group.create_dataset(key, data=array, compression="lzf")
                    group.attrs["definition_json"] = json.dumps(definition)
                    m["sources"][name] = dict(
                        rows=int(hit.sum()),
                        outside_feature_union=int((~hit & observed.any(1)).sum()),
                        observed_per_target=observed[hit].sum(0).tolist(),
                        role_rows={
                            role: int((roles[pos[hit]] == role.encode()).sum())
                            for role in ["fit", "tune", "cal", "devtest", "external"]
                        },
                    )
                    save()

                # Build native atmospheric channels from SHA-bound old exports, never their predictions.
                for si, name in enumerate(["apogee", "desi", "galah"]):
                    chunks = []
                    for role, expected in c["native_prediction_sha256"].items():
                        path = (
                            ROOT
                            / "multisurvey_joint_fullfit_20261105_v1"
                            / ("predictions-" + role + ".h5")
                        )
                        if sha(path) != expected:
                            raise ValueError("Native core export changed: " + role)
                        with h5py.File(path) as p:
                            sid = p["source_id"][:]
                            nv = p["native_value"][:, si, :]
                            ne = p["native_error"][:, si, :]
                            obs = p["native_observed"][:, si, :].astype(bool)
                        v = np.full((len(sid), 6), np.nan)
                        e = v.copy()
                        o = np.zeros(v.shape, bool)
                        k = o.copy()
                        v[:, :3], e[:, :3], o[:, :3] = nv, ne, obs
                        k[:, :3] = obs & np.isfinite(ne) & (ne > 0)
                        if si == 1:
                            rr = np.searchsorted(oldids, sid)
                            np.testing.assert_array_equal(oldids[rr], sid)
                            # Matching source pipeline only, never replace alpha/M with alpha/Fe.
                            source = base["label_source"][:].astype("S64")
                            use = (
                                (source[rr, 3] == b"DESI_RVSpecFit_alpha_fe")
                                & base["observed"][:, 3][rr]
                                & base["quality"][:, 3][rr]
                            )
                            v[use, 3] = base["value"][:, 3][rr[use]]
                            e[use, 3] = base["error"][:, 3][rr[use]]
                            o[use, 3] = True
                            k[use, 3] = base["error_observed"][:, 3][rr[use]]
                            del source
                        chunks.append((sid, v, e, o, k))
                    arrays = [np.concatenate([v[j] for v in chunks]) for j in range(5)]
                    order = np.argsort(arrays[0])
                    arrays = [v[order] for v in arrays]
                    write_records(
                        name,
                        *arrays,
                        dict(
                            target_indices=[0, 1, 2, 3] if si == 1 else [0, 1, 2],
                            source="SHA-bound native-label arrays, not fitted means",
                            coordinate="native",
                        ),
                    )
                base_source = base["label_source"][:].astype("S64")
                for name in [
                    "StarHorse_apogee17_posterior_median",
                    "StarHorse_galahdr3_posterior_median",
                    "StarHorse_catalogue_posterior_median",
                ]:
                    observed = (
                        base["observed"][:].astype(bool)
                        & base["quality"][:].astype(bool)
                        & (base_source == name.encode())
                    )
                    observed[:, :4] = False
                    write_records(
                        name,
                        oldids,
                        base["value"][:],
                        base["error"][:],
                        observed,
                        np.zeros(observed.shape, bool),
                        dict(
                            target_indices=[4, 5],
                            coordinate="native",
                            estimand="posterior_teacher",
                            measurement_error_known=False,
                            shared_Gaia_and_spectroscopic_priors=True,
                        ),
                    )
                del base_source
                for spec in c["additional_sources"]:
                    if sha(spec["path"]) != spec["sha256"]:
                        raise ValueError(
                            "Additional native source changed: " + spec["name"]
                        )
                    with h5py.File(spec["path"]) as sf:
                        g = sf["native_observations/" + spec["name"]]
                        sid = g["source_id"][:]
                        v = g["value"][:]
                        e = g["error"][:]
                        o = g["observed"][:].astype(bool)
                        known = (
                            g["error_known"][:].astype(bool)
                            if "error_known" in g
                            else (o & np.isfinite(e) & (e > 0))
                        )
                        definition = dict(
                            spec,
                            coordinate="native",
                            source_group_attributes={
                                k: str(v) for k, v in g.attrs.items()
                            },
                            no_quantity_aliasing=True,
                        )
                    vv = np.full((len(sid), 6), np.nan)
                    ee = vv.copy()
                    oo = np.zeros(vv.shape, bool)
                    kk = oo.copy()
                    vv[:, :3], ee[:, :3], oo[:, :3], kk[:, :3] = v, e, o, known
                    if spec["name"] == "gaia_dr3_gspspec" and oo[:, 2].any():
                        raise ValueError(
                            "GSP-Spec metallicity must not be relabelled as Fe/H"
                        )
                    if np.any(kk & (~np.isfinite(ee) | (ee <= 0))):
                        raise ValueError("Known errors are invalid")
                    write_records(spec["name"], sid, vv, ee, oo, kk, definition)
                f.attrs["complete"] = True
            with h5py.File(output) as f:
                if not np.array_equal(f["source_id"][:], ids) or not np.array_equal(
                    f["role"][:], roles
                ):
                    raise ValueError("Combined identity/role roundtrip failed")
                if np.any(
                    np.isin(
                        ids[roles == b"fit"], np.concatenate([benchmark, protected_gc])
                    )
                ):
                    raise ValueError("Protected reference entered fitting")
            m.update(
                status="expanded_six_source_complete",
                training_source_ready=True,
                prepared_path=str(output),
                prepared_sha256=sha(output),
                source_registry=list(m["sources"]),
                roles={
                    r: int((roles == r.encode()).sum())
                    for r in ["fit", "tune", "cal", "devtest", "external", "excluded"]
                },
                eligible_role_counts={
                    r: int(((roles == r.encode()) & eligible).sum())
                    for r in ["fit", "tune", "cal", "devtest", "external"]
                },
                feature_profiles=dict(
                    phot_xp=dict(ready=True),
                    phot_xp_measured_parallax=dict(
                        ready=True,
                        feature_index=j,
                        override_dataset="measured_parallax/input_value",
                        availability_dataset="measured_parallax/observed",
                    ),
                ),
                remaining_sources="Gaia-ESO identities and MaSTAR source-quality/dependency admission remain separate; no implicit activation. LAMOST alpha/M and GSP-Spec M/H are excluded from Fe/H or alpha/Fe targets.",
                holdout_scope="New stage protects GBS and all selected globular members as groups. Existing ancestors may contain their labels or inputs. New stars inherit stable ID roles; all products of one star share one role; unlinked binary/open-cluster dependencies remain unknown. Newly staged TESS/K2 identities are not yet admitted.",
                normalization="Feature scaler is identical to parent. Target normalization must remain the selected six-target parent normalization, not be refitted using new target labels.",
            )
            save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
