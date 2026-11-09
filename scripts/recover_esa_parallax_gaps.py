"""Recover missing eligible-star parallaxes from ESA using public Gaia IDs only."""

import argparse
import io
import json
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from repair_expanded_parallax import sha, verify_unchanged

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def query(
    ids,
    endpoint="https://gea.esac.esa.int/tap-server/tap/sync",
    table="gaiadr3.gaia_source",
):
    adql = (
        "SELECT source_id,parallax,parallax_error FROM "
        + table
        + " WHERE source_id IN ("
        + ",".join(str(int(v)) for v in ids)
        + ")"
    )
    payload = urllib.parse.urlencode(
        dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=adql)
    ).encode()
    for retry in range(12):
        try:
            req = urllib.request.Request(
                endpoint,
                data=payload,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                body = response.read()
            frame = pd.read_csv(io.BytesIO(body), dtype={"source_id": "Int64"})
            if (
                "source_id" not in frame
                or frame.source_id.isna().any()
                or frame.source_id.duplicated().any()
                or not np.isin(frame.source_id.to_numpy(dtype="i8"), ids).all()
            ):
                raise ValueError("ESA exact identity schema/association failed")
            return body, frame
        except Exception:
            if retry == 11:
                raise
            time.sleep(min(2**retry, 30))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restart-recovery", action="store_true")
    p.add_argument("--resume-failed-recovery", action="store_true")
    p.add_argument("--finalize-validated-cache", action="store_true")
    a = p.parse_args()
    if a.output.exists() and (a.resume_failed_recovery or a.finalize_validated_cache):
        previous = json.loads((a.output / "manifest.json").read_text())
        parent_sha = json.loads(
            (ROOT / "expanded_six_source_20261107_v3/manifest.json").read_text()
        )["prepared_sha256"]
        if (
            previous.get("status") != "failed"
            or previous.get("parent_source_sha256") != parent_sha
            or (a.output / "prepared.h5").exists()
        ):
            raise ValueError(
                "Only failed pre-copy recovery of the same parent may resume"
            )
        generation = 1
        while (a.output / f"failed-recovery-attempt-{generation}.json").exists():
            generation += 1
        with (a.output / f"failed-recovery-attempt-{generation}.json").open(
            "xb"
        ) as stream:
            stream.write((a.output / "manifest.json").read_bytes())
    elif a.output.exists() and a.restart_recovery:
        previous = json.loads((a.output / "manifest.json").read_text())
        parent_sha = json.loads(
            (ROOT / "expanded_six_source_20261107_v3/manifest.json").read_text()
        )["prepared_sha256"]
        if (
            previous.get("status") not in ["recovering_esa_astrometry", "failed"]
            or previous.get("parent_source_sha256") != parent_sha
            or (a.output / "prepared.h5").exists()
        ):
            raise ValueError("Only a reconciled pre-copy recovery may restart")
        attempt = 1
        while (a.output / f"recovery-attempt-{attempt}").exists():
            attempt += 1
        archive = a.output / f"recovery-attempt-{attempt}"
        archive.mkdir(exist_ok=False)
        for path in a.output.iterdir():
            if path.is_file():
                path.replace(archive / path.name)
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="recovering_esa_astrometry",
        training_source_ready=False,
        training_performed=False,
        external_payload="Public Gaia DR3 source_id values only; no positions or survey labels",
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    try:
        pm = json.loads(
            (ROOT / "expanded_six_source_20261107_v3/manifest.json").read_text()
        )
        audit_path = ROOT / "gaia_parallax_release_unit_audit_20261107_v2/manifest.json"
        audit = json.loads(audit_path.read_text())
        if (
            not audit.get("pass_gate")
            or audit["source_sha256"] != pm["prepared_sha256"]
        ):
            raise ValueError("Independent repaired-source ESA audit did not pass")
        parent = Path(pm["prepared_path"])
        if sha(parent) != pm["prepared_sha256"]:
            raise ValueError("Parent source changed")
        with h5py.File(parent) as f:
            ids, roles = f["source_id"][:], f["role"][:]
            eligible = f["primary_input_eligible"][:].astype(bool)
            seen = f["measured_parallax/observed"][:].astype(bool)
            old_seen = seen.copy()
            values = f["measured_parallax/parallax_mas"][:].astype("f8")
            errors = f["measured_parallax/parallax_error_mas"][:].astype("f8")
            j = f["feature_names"][:].astype("U").tolist().index("PARALLAX")
            center, scale = float(f["scaler_center"][j]), float(f["scaler_scale"][j])
        if np.any(np.diff(ids) <= 0):
            raise ValueError("Unique sorted exact IDs required")
        missing = np.flatnonzero(eligible & ~seen)
        if len(missing) > 100000:
            raise ValueError(
                "Unexpected recovery scope; investigate before expanding payload"
            )
        m.update(
            parent_source_sha256=pm["prepared_sha256"],
            requested_stars=len(missing),
            completed_batches=0,
            usable_recovered=0,
            query_batch_size=512,
            query_workers=4,
        )
        save()
        recovered = np.zeros(len(ids), bool)
        esa_recovered = np.zeros(len(ids), bool)
        mirror_recovered = np.zeros(len(ids), bool)
        queried = np.zeros(len(ids), bool)
        response_hashes = {}

        def accept(frame):
            got = frame.source_id.to_numpy(dtype="i8")
            if (
                frame.source_id.isna().any()
                or frame.source_id.duplicated().any()
                or not np.isin(got, ids[missing]).all()
            ):
                raise ValueError(
                    "Cached or retrieved IDs are outside the requested missing cohort"
                )
            pos = np.searchsorted(ids, got)
            np.testing.assert_array_equal(ids[pos], got)
            value = frame.parallax.to_numpy(dtype="f8")
            error = frame.parallax_error.to_numpy(dtype="f8")
            valid = (
                np.isfinite(value) & np.isfinite(error) & (error > 0) & (value != -9999)
            )
            repeated = queried[pos]
            if repeated.any():
                np.testing.assert_array_equal(seen[pos[repeated]], valid[repeated])
                both = repeated & valid
                np.testing.assert_allclose(
                    values[pos[both]], value[both], rtol=2e-6, atol=2e-6
                )
                np.testing.assert_allclose(
                    errors[pos[both]], error[both], rtol=2e-6, atol=2e-6
                )
            pp = pos[valid]
            if old_seen[pp].any():
                raise ValueError(
                    "Recovery would overwrite an originally observed measurement"
                )
            values[pp], errors[pp] = value[valid], error[valid]
            seen[pp] = True
            recovered[pp] = True
            queried[pos] = True
            return pp

        cached = (
            list(a.output.glob("esa-*.csv"))
            + list(a.output.glob("recovery-attempt-*/esa-*.csv"))
            + list(a.output.glob("mirror-*.csv"))
        )
        for path in sorted(cached):
            frame = pd.read_csv(path, dtype={"source_id": "Int64"})
            pp = accept(frame)
            (mirror_recovered if path.name.startswith("mirror-") else esa_recovered)[
                pp
            ] = True
            response_hashes[str(path.relative_to(a.output))] = sha(path)
        m.update(
            usable_recovered=int(recovered.sum()),
            cached_response_files=len(cached),
            remaining_queries=int((~queried[missing]).sum()),
            completed_batches=0,
        )
        save()
        if a.resume_failed_recovery and not a.finalize_validated_cache:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                probe_path = (
                    ROOT / "gaia_mirror_astrometry_probe_20261107_v1/manifest.json"
                )
                probe = (
                    json.loads(probe_path.read_text()) if probe_path.exists() else {}
                )
                if probe.get("status") == "failed":
                    raise RuntimeError("Gaia mirror failed ESA overlap verification")
                if probe.get("pass_gate"):
                    break
                time.sleep(10)
            else:
                raise TimeoutError("Gaia mirror verification unavailable")
            if probe.get("parent_source_sha256") != pm["prepared_sha256"]:
                raise ValueError("Mirror overlap verification has another parent")
            remaining = missing[~queried[missing]]
            batches = [
                remaining[start : start + 512]
                for start in range(0, len(remaining), 512)
            ]
            m.update(
                status="recovering_verified_mirror_astrometry",
                mirror_probe_sha256=sha(probe_path),
                provider="GAVO gaia.dr3lite, verified against ESA DR3",
            )
            save()
            with ThreadPoolExecutor(4) as pool:
                tasks = {
                    pool.submit(
                        query,
                        ids[rows],
                        "https://dc.zah.uni-heidelberg.de/tap/sync",
                        "gaia.dr3lite",
                    ): number
                    for number, rows in enumerate(batches)
                }
                try:
                    for future in as_completed(tasks):
                        number = tasks[future]
                        body, frame = future.result()
                        path = a.output / f"mirror-{generation}-{number:05d}.csv"
                        with path.open("xb") as stream:
                            stream.write(body)
                        response_hashes[path.name] = sha(path)
                        pp = accept(frame)
                        mirror_recovered[pp] = True
                        m.update(
                            completed_batches=m["completed_batches"] + 1,
                            usable_recovered=int(recovered.sum()),
                        )
                        save()
                except Exception:
                    for pending in tasks:
                        pending.cancel()
                    raise
        elif not a.finalize_validated_cache:
            raise ValueError(
                "Use explicit failed-recovery resume or validated-cache finalization"
            )
        m["status"] = "copying_and_verifying_source"
        save()
        output = a.output / "prepared.h5"
        shutil.copyfile(parent, output)
        with h5py.File(output, "r+") as f:
            g = f["measured_parallax"]
            for key, array in dict(
                parallax_mas=values,
                parallax_error_mas=errors,
                observed=seen,
                input_value=np.where(seen, (values - center) / scale, -9999),
            ).items():
                g[key][:] = array
            g.create_dataset("esa_recovered", data=esa_recovered, compression="lzf")
            g.create_dataset(
                "gaia_mirror_recovered", data=mirror_recovered, compression="lzf"
            )
            g.attrs["esa_recovery"] = (
                "Gaia DR3 parallax/error in mas by exact source_id; original ESA responses and verified mirror responses retained; only formerly unavailable eligible inputs changed; no zero-point correction"
            )
        with h5py.File(parent) as first, h5py.File(output) as second:
            checked = verify_unchanged(first, second)
            np.testing.assert_array_equal(
                second["measured_parallax/observed"][:].astype(bool), seen
            )
        counts = {}
        for role in np.unique(roles):
            take = roles == role
            counts[role.decode()] = dict(
                stars=int(take.sum()),
                before=int((take & old_seen).sum()),
                after=int((take & seen).sum()),
                newly_available=int((take & recovered).sum()),
                negative=int((take & seen & (values < 0)).sum()),
            )
        result = dict(pm)
        for role, count in counts.items():
            combined = result["parallax_repair"]["role_availability"][role]
            combined["after"] = count["after"]
            combined["newly_available"] = count["after"] - combined["before"]
            combined["negative"] = count["negative"]
        result.update(
            status="expanded_six_source_complete",
            training_source_ready=True,
            prepared_path=str(output),
            prepared_sha256=sha(output),
            parent_source_sha256=pm["prepared_sha256"],
            esa_parallax_recovery=dict(
                requested_stars=len(missing),
                providers=[
                    "ESA Gaia TAP",
                    "GAVO Gaia DR3 mirror crosschecked against ESA",
                ],
                unresolved_due_to_provider_outage=int((~queried[missing]).sum()),
                source_availability_scope="Unretrieved measurements remain explicitly masked; unavailable in this snapshot is not a claim of Gaia absence.",
                recovered_stars=int(recovered.sum()),
                still_missing_eligible=int((eligible & ~seen).sum()),
                role_availability=counts,
                independent_parent_audit_sha256=sha(audit_path),
                response_sha256=response_hashes,
                unchanged_datasets=checked,
                external_payload=m["external_payload"],
            ),
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
