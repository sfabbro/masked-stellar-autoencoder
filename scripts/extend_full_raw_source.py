"""Append recovered raw XP stars with inherited/protected star-level roles.

Builds an independent larger source. It never changes the pinned running job.
"""

import argparse
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from prepare_expanded_six_source import role_buckets, sha

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def exact_ids(values):
    values = np.asarray(values)
    if (
        values.dtype.kind not in "iu"
        or (values <= 0).any()
        or (values > np.iinfo(np.int64).max).any()
    ):
        raise ValueError("Exact positive signed-range integer Gaia IDs required")
    return values.astype("i8", copy=False)


def union_layout(old_ids, new_ids, old_roles, protected):
    old_ids, new_ids = exact_ids(old_ids), exact_ids(new_ids)
    if np.intersect1d(old_ids, new_ids).size or any(
        np.any(np.diff(v) <= 0) for v in [old_ids, new_ids]
    ):
        raise ValueError("Require disjoint sorted unique integer identities")
    ids = np.sort(np.concatenate([old_ids, new_ids]))
    oldpos = np.searchsorted(ids, old_ids)
    newpos = np.searchsorted(ids, new_ids)
    roles = role_buckets(ids)
    roles[oldpos] = old_roles
    roles[newpos[np.isin(new_ids, protected)]] = b"external"
    return ids, oldpos, newpos, roles


def aligned_copy(
    destination, original, old_positions, new_positions, new_values=None, fill=None
):
    """Merge old rows without changing their values; source indices remain monotone."""
    n = len(old_positions) + len(new_positions)
    old_inverse = np.full(n, -1, dtype="i8")
    old_inverse[old_positions] = np.arange(len(old_positions))
    new_inverse = np.full(n, -1, dtype="i8")
    new_inverse[new_positions] = np.arange(len(new_positions))
    for start in range(0, n, 16384):
        end = min(start + 16384, n)
        oi = old_inverse[start:end]
        ni = new_inverse[start:end]
        values = np.full(
            (end - start, *original.shape[1:]),
            original.fillvalue if fill is None else fill,
            dtype=original.dtype,
        )
        take = oi >= 0
        if take.any():
            rr = oi[take]
            lo, hi = int(rr[0]), int(rr[-1]) + 1
            values[take] = original[lo:hi][rr - lo]
        if new_values is not None:
            take = ni >= 0
            values[take] = new_values[ni[take]]
        destination[start:end] = values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-waiting-output", action="store_true")
    a = p.parse_args()
    if a.output.exists() and a.reuse_waiting_output:
        previous = json.loads((a.output / "manifest.json").read_text())
        if (
            previous.get("status")
            not in ["waiting_for_final_parent_astrometry", "failed"]
            or (a.output / "prepared.h5").exists()
        ):
            raise ValueError(
                "Only a reconciled waiting-only source assembly can restart"
            )
        attempt = 1
        while (a.output / f"pre-assembly-attempt-{attempt}.json").exists():
            attempt += 1
        with (a.output / f"pre-assembly-attempt-{attempt}.json").open("xb") as stream:
            stream.write((a.output / "manifest.json").read_bytes())
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="waiting_for_final_parent_astrometry",
        training_source_ready=False,
        training_performed=False,
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    try:
        deadline = time.monotonic() + 24 * 3600
        while time.monotonic() < deadline:
            path = ROOT / "expanded_six_source_20261107_v4/manifest.json"
            pm = json.loads(path.read_text()) if path.exists() else {}
            if pm.get("status") == "failed":
                raise RuntimeError("Parent astrometry recovery failed")
            if pm.get("status") == "expanded_six_source_complete" and pm.get(
                "training_source_ready"
            ):
                break
            time.sleep(30)
        else:
            raise TimeoutError("Parent source not ready")
        c = json.loads(a.config.read_text())
        feature_dir = ROOT / "remaining_survey_raw_features_20261107_v1"
        fm = json.loads((feature_dir / "manifest.json").read_text())
        features = feature_dir / "additional-star-features.h5"
        if (
            fm.get("status") != "additional_full_raw_features_complete"
            or sha(features) != fm["output_sha256"]
            or sha(pm["prepared_path"]) != pm["prepared_sha256"]
        ):
            raise ValueError("Parent/extra feature binding changed")
        gcpath = ROOT / "expanded_six_source_20261107_v2/clusters.zip"
        if (
            sha(gcpath)
            != "9eafa2e10afc031790cfe36ed8545aa6336cc87c317fab167e9cea9dacbd0748"
        ):
            raise ValueError("Cluster membership source changed")
        protected_gc = []
        with zipfile.ZipFile(gcpath) as archive:
            for name in sorted(archive.namelist()):
                if name.startswith("catalogues/") and name.endswith(".txt"):
                    text = archive.read(name).decode()
                    columns = text.splitlines()[0].lstrip("#").split()
                    tab = pd.read_csv(
                        io.StringIO(text),
                        sep=r"\s+",
                        comment="#",
                        names=columns,
                        dtype={"source_id": np.int64},
                    )
                    good = ((tab.qflag.to_numpy(dtype=int) & 2) != 0) & (
                        tab.memberprob.to_numpy() > 0.9
                    )
                    protected_gc.append(tab.source_id.to_numpy()[good])
        protected_gc = np.unique(np.concatenate(protected_gc))
        bench = np.asarray(c["benchmark_fit_exclusion_ids"], dtype="i8")
        with h5py.File(pm["prepared_path"]) as old, h5py.File(features) as extra:
            old_ids = exact_ids(old["source_id"][:])
            selected = np.flatnonzero(
                extra["feature_recoverable"][:].astype(bool)
                & extra["input_has_xp"][:].astype(bool)
            )
            new_ids = exact_ids(extra["source_id"][:][selected])
            for key in ["feature_names", "scaler_center", "scaler_scale"]:
                np.testing.assert_array_equal(old[key][:], extra[key][:])
            ids, op, np_, roles = union_layout(
                old_ids, new_ids, old["role"][:], np.union1d(bench, protected_gc)
            )
            n = len(ids)
            eligible = np.ones(n, bool)
            eligible[op] = old["primary_input_eligible"][:].astype(bool)
            bmask = np.isin(ids, bench)
            gmask = np.isin(ids, protected_gc)
            if np.any((roles == b"fit") & (bmask | gmask)):
                raise ValueError("A protected physical reference entered FIT")
            m.update(
                status="writing_extended_source",
                parent_source_sha256=pm["prepared_sha256"],
                new_feature_sha256=fm["output_sha256"],
                additional_feature_stars=len(new_ids),
                total_stars=n,
            )
            save()
            output = a.output / "prepared.h5"
            with h5py.File(output, "x") as f:
                for key, value in old.attrs.items():
                    f.attrs[key] = value
                f.attrs["complete"] = False
                for key, value in dict(
                    source_id=ids,
                    role=roles,
                    primary_input_eligible=eligible,
                    reserved_benchmark=bmask,
                    reserved_globular_member=gmask,
                    additional_full_raw_star=np.isin(ids, new_ids),
                ).items():
                    f[key] = value
                for key in [
                    "feature_names",
                    "target_names",
                    "scaler_center",
                    "scaler_scale",
                ]:
                    f[key] = old[key][:]
                for key in [
                    "input",
                    "value",
                    "error",
                    "observed",
                    "quality",
                    "error_observed",
                    "label_source",
                ]:
                    original = old[key]
                    dest = f.create_dataset(
                        key,
                        shape=(n, *original.shape[1:]),
                        dtype=original.dtype,
                        chunks=(8192, *original.shape[1:]),
                        compression="lzf",
                        fillvalue=original.fillvalue,
                    )
                    aligned_copy(
                        dest,
                        original,
                        op,
                        np_,
                        extra["input"][:][selected] if key == "input" else None,
                    )
                    m["last_dataset"] = key
                    save()
                group = f.create_group("measured_parallax")
                j = old["feature_names"][:].astype("U").tolist().index("PARALLAX")
                value = extra["measured_parallax/parallax_mas"][:][selected]
                error = extra["measured_parallax/parallax_error_mas"][:][selected]
                observed = extra["measured_parallax/observed"][:][selected].astype(bool)
                if (
                    not np.isfinite(value[observed]).all()
                    or not np.isfinite(error[observed]).all()
                    or (error[observed] <= 0).any()
                ):
                    raise ValueError("Invalid new signed parallax/error")
                for key, original in old["measured_parallax"].items():
                    dest = group.create_dataset(
                        key,
                        shape=(n,),
                        dtype=original.dtype,
                        chunks=True,
                        compression="lzf",
                        fillvalue=original.fillvalue,
                    )
                    if key == "input_value":
                        nv = np.where(
                            observed,
                            (value - old["scaler_center"][j]) / old["scaler_scale"][j],
                            -9999,
                        )
                    elif key == "parallax_mas":
                        nv = value
                    elif key == "parallax_error_mas":
                        nv = error
                    elif key == "observed":
                        nv = observed
                    elif key == "raw_matched":
                        nv = extra["original_row"][:][selected] >= 0
                    elif key in ["esa_recovered", "gaia_mirror_recovered"]:
                        nv = np.zeros(len(selected), bool)
                    else:
                        raise ValueError("Unknown parent astrometry field " + key)
                    aligned_copy(dest, original, op, np_, nv)
                for key, value in old["measured_parallax"].attrs.items():
                    group.attrs[key] = value
                source_counts = {}
                union = np.zeros((n, 6), bool)
                known_union = union.copy()
                record_count = np.zeros(n, dtype="i2")
                specs = {s["name"]: s for s in c["additional_sources"]}
                for name in pm["source_registry"]:
                    original = old["native_observations/" + name]
                    rows = op[original["star_row"][:]]
                    arrays = {
                        key: original[key][:]
                        for key in ["value", "error", "observed", "error_known"]
                    }
                    if name in specs:
                        spec = specs[name]
                        if sha(spec["path"]) != spec["sha256"]:
                            raise ValueError("Native label sidecar changed " + name)
                        with h5py.File(spec["path"]) as sf:
                            g = sf["native_observations/" + name]
                            sid = exact_ids(g["source_id"][:])
                            # Independently prove the inherited additional-survey labels match native IDs.
                            old_match = np.searchsorted(old_ids, sid)
                            old_hit = old_match < len(old_ids)
                            old_hit[old_hit] &= (
                                old_ids[old_match[old_hit]] == sid[old_hit]
                            )
                            source_observed = g["observed"][:].astype(bool)
                            old_hit &= source_observed.any(1)
                            old_rows = np.flatnonzero(old_hit)
                            old_rows = old_rows[
                                np.argsort(sid[old_rows], kind="stable")
                            ]
                            np.testing.assert_array_equal(
                                old_ids[original["star_row"][:]], sid[old_rows]
                            )
                            for field in ["value", "error", "observed", "error_known"]:
                                native_field = (
                                    g[field][:]
                                    if field in g
                                    else (
                                        source_observed
                                        & np.isfinite(g["error"][:])
                                        & (g["error"][:] > 0)
                                    )
                                )
                                if field == "error_known":
                                    native_field = native_field & source_observed
                                np.testing.assert_array_equal(
                                    original[field][:, :3], native_field[old_rows]
                                )
                            m.setdefault("native_label_roundtrip", {})[name] = dict(
                                parent_records=len(old_rows),
                                source_id_dtype=str(g["source_id"].dtype),
                                exact_signed_integer_join=True,
                            )
                            save()
                            pos = np.searchsorted(new_ids, sid)
                            hit = pos < len(new_ids)
                            hit[hit] &= new_ids[pos[hit]] == sid[hit]
                            obs = g["observed"][:].astype(bool)
                            hit &= obs.any(1)
                            rr = np.flatnonzero(hit)
                            rows = np.r_[rows, np_[pos[hit]]]
                            for key in arrays:
                                source = (
                                    g[key][:]
                                    if key in g
                                    else (
                                        g["observed"][:].astype(bool)
                                        & np.isfinite(g["error"][:])
                                        & (g["error"][:] > 0)
                                    )
                                )
                                values = (
                                    np.zeros((len(rr), 6), dtype=arrays[key].dtype)
                                    if key in ["observed", "error_known"]
                                    else np.full(
                                        (len(rr), 6), np.nan, dtype=arrays[key].dtype
                                    )
                                )
                                values[:, :3] = source[rr]
                                arrays[key] = np.concatenate([arrays[key], values])
                    order = np.argsort(rows)
                    rows = rows[order]
                    arrays = {k: v[order] for k, v in arrays.items()}
                    if (
                        np.any(np.diff(rows) <= 0)
                        or not np.isfinite(arrays["value"][arrays["observed"]]).all()
                        or (arrays["error_known"] & ~arrays["observed"]).any()
                    ):
                        raise ValueError("Native identity/value/mask failure " + name)
                    if np.any(
                        arrays["error_known"]
                        & (~np.isfinite(arrays["error"]) | (arrays["error"] <= 0))
                    ):
                        raise ValueError("Invalid quoted error " + name)
                    if name == "gaia_dr3_gspspec" and arrays["observed"][:, 2].any():
                        raise ValueError("GSP M/H must not become Fe/H")
                    dest = f.create_group("native_observations/" + name)
                    dest["star_row"] = rows
                    for key, values in arrays.items():
                        dest.create_dataset(key, data=values, compression="lzf")
                    for key, value in original.attrs.items():
                        dest.attrs[key] = value
                    union[rows] |= arrays["observed"]
                    known_union[rows] |= arrays["error_known"]
                    record_count[rows] += arrays["observed"].any(1)
                    source_counts[name] = dict(
                        rows=len(rows),
                        fit_records=int(
                            ((roles[rows] == b"fit") & eligible[rows]).sum()
                        ),
                        observed_per_target=arrays["observed"].sum(0).tolist(),
                    )
                    m["last_source"] = name
                    save()
                if not union[np_].any(1).all():
                    raise ValueError("A newly admitted star has no native target")
                f.attrs["complete"] = True
            # Streaming identity-aligned preservation check, including source observation records.
            with h5py.File(output) as f:
                for key in [
                    "role",
                    "primary_input_eligible",
                    "input",
                    "value",
                    "error",
                    "observed",
                    "quality",
                    "error_observed",
                    "label_source",
                ]:
                    for start in range(0, len(old_ids), 16384):
                        rr = op[start : start + 16384]
                        np.testing.assert_array_equal(
                            f[key][int(rr[0]) : int(rr[-1]) + 1][rr - int(rr[0])],
                            old[key][start : start + 16384],
                        )
                for name in pm["source_registry"]:
                    original = old["native_observations/" + name]
                    dest = f["native_observations/" + name]
                    rows = dest["star_row"][:]
                    where = np.searchsorted(rows, op[original["star_row"][:]])
                    for key in ["value", "error", "observed", "error_known"]:
                        np.testing.assert_array_equal(
                            dest[key][:][where], original[key][:]
                        )
            counts = {}
            for role in ["fit", "tune", "cal", "devtest", "external", "excluded"]:
                take = (roles == role.encode()) & eligible
                counts[role] = dict(
                    input_stars=int(take.sum()),
                    labelled_unique_stars=int((take & union.any(1)).sum()),
                    unique_stars_per_target=union[take].sum(0).tolist(),
                    complete_six_unique_stars=int((take & union.all(1)).sum()),
                    source_records=int(record_count[take].sum()),
                    known_quoted_error_unique_stars=known_union[take].sum(0).tolist(),
                )
        shutil.copyfile(gcpath, a.output / "clusters.zip")
        roundtrip = m.get("native_label_roundtrip", {})
        result = dict(pm)
        result.update(
            status="expanded_six_source_complete",
            training_source_ready=True,
            prepared_path=str(output),
            prepared_sha256=sha(output),
            total_stars=n,
            source_registry=pm["source_registry"],
            sources=source_counts,
            roles={r: int((roles == r.encode()).sum()) for r in counts},
            eligible_role_counts={
                r: counts[r]["input_stars"]
                for r in ["fit", "tune", "cal", "devtest", "external"]
            },
            native_label_roundtrip=roundtrip,
            source_extension=dict(
                parent_sha256=pm["prepared_sha256"],
                feature_sha256=fm["output_sha256"],
                additional_stars=len(new_ids),
                old_roles_inputs_labels_preserved=True,
                new_roles="75 FIT / 10 TUNE / 7 CAL / 8 DEV by the same stable Gaia-ID hash; benchmark/globular references external",
                activated_in_running_training=False,
            ),
            census=dict(
                target_order=["teff", "logg", "fe_h", "alpha", "age", "distance"],
                roles=counts,
                sources=source_counts,
            ),
            parallax_availability_scope="Inherited exact parent values plus native signed raw measurements for appended stars; per-role parent recovery counts apply to parent only.",
        )
        m.clear()
        m.update(result)
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc), training_source_ready=False)
        save()
        raise


if __name__ == "__main__":
    main()
