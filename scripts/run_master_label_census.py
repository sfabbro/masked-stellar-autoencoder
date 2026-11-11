"""Build the master spectroscopic label observation table and its census.

Runs the same way locally and on CANFAR (CPU only, no model):

  1. acquire native sources (SDSS SAS, VizieR) with checksums/receipts;
  2. normalise each into the long observation schema
     (``masked_stellar_autoencoder.training.master_labels``);
  3. resolve missing Gaia DR3 IDs by epoch-stable positional match through the
     ESA Gaia archive (public catalogue positions only);
  4. write ``observations.parquet``, ``stars.parquet`` and ``manifest.json``;
  5. optionally join against already staged survey sidecars on ARC
     (``--existing-root``) to report new unique and new metal-poor stars.

No roles are assigned and nothing is enabled for training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masked_stellar_autoencoder.training import master_labels as ml  # noqa: E402


def download(url: str, path: Path, *, sha1: str | None = None, size: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    if not path.exists():
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
        with urllib.request.urlopen(request, timeout=300) as r, part.open("ab") as f:
            if offset and r.status != 206:
                f.truncate(0)
            while block := r.read(16 * 2**20):
                f.write(block)
        part.replace(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"{path}: size {path.stat().st_size} != {size}")
    if sha1 is not None:
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha1").hexdigest() != sha1:
                raise ValueError(f"{path}: SHA1 differs from publisher list")
    return dict(
        url=url, path=str(path), bytes=path.stat().st_size, sha256=ml.sha256(path)
    )


def existing_sidecar_ids(root: Path, log) -> tuple[dict, list]:
    """Gaia IDs of every native_observations/<source> group found under root.

    Bounded discovery of ``*.h5`` observation sidecars; for each source key the
    largest file is used and every candidate is recorded in the manifest.
    """
    import h5py

    found = {}
    for path in sorted(root.rglob("*.h5")):
        if path.stat().st_size > 50 * 2**30:
            continue
        try:
            with h5py.File(path, "r") as f:
                if "native_observations" not in f:
                    continue
                for key, group in f["native_observations"].items():
                    if "source_id" in group:
                        n = group["source_id"].shape[0]
                        found.setdefault(key, []).append((n, str(path)))
        except OSError:
            continue
    ids, chosen = {}, []
    for key, options in sorted(found.items()):
        n, path = max(options)
        with h5py.File(path, "r") as f:
            values = f["native_observations"][key]["source_id"][:].astype("i8")
        ids[key] = values[values > 0]
        chosen.append(dict(source=key, path=path, rows=n, candidates=options))
        log(f"existing {key}: {n} rows from {path}")
    return ids, chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True, help="download/cache dir")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", nargs="*", help="restrict to these source names")
    p.add_argument("--skip-positional", action="store_true")
    p.add_argument("--gaia-batch", type=int, default=20000)
    p.add_argument("--existing-root", type=Path, action="append", default=[])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        status="running",
        utc=dt.datetime.now(dt.UTC).isoformat(),
        training_performed=False,
        roles_assigned=False,
        script_sha256=ml.sha256(__file__),
        module_sha256=ml.sha256(ml.__file__),
        acquisitions={},
        policies={},
        errors={},
    )

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        tmp.replace(a.output / "manifest.json")

    def log(msg):
        print(f"[{dt.datetime.now(dt.UTC):%H:%M:%S}] {msg}", flush=True)

    wanted = (lambda name: True) if not a.only else (lambda name: name in a.only)
    frames = []
    sdss = a.work / "sdss"
    readers = [
        (ml.ASTRA_ASPCAP, ml.astra_aspcap, dict(sha1=ml.ASTRA_ASPCAP["sha1"])),
        (ml.SSPP, ml.sdss_sspp, dict(size=ml.SSPP["bytes"])),
        (ml.MASTAR, ml.mastar_params, dict(size=ml.MASTAR["bytes"])),
    ]
    readers += [
        (spec, lambda path, spec=spec: ml.astra_summary(path, spec), {})
        for spec in ml.ASTRA_SUMMARIES
    ]
    for spec, reader, kwargs in readers:
        if not wanted(spec["source"]):
            continue
        try:
            path = sdss / spec["url"].rsplit("/", 1)[1]
            log(f"acquire {spec['source']}")
            manifest["acquisitions"][spec["source"]] = download(
                spec["url"], path, **kwargs
            )
            frame = reader(path)
            manifest["policies"][spec["source"]] = {
                k: spec[k]
                for k in ("release", "pipeline", "quality_policy", "grid_policy")
                if k in spec
            }
            epochs = (2015.5,) if spec is ml.MASTAR else (2000.0, 2008.0)
            frames.append((frame, 1.5, epochs, 5.0))
            log(f"{spec['source']}: {len(frame)} rows")
        except Exception as error:
            manifest["errors"][spec["source"]] = f"{type(error).__name__}: {error}"
            log(f"FAILED {spec['source']}: {error}")
        save()
    for spec in ml.VIZIER_SOURCES:
        if not wanted(spec.source):
            continue
        try:
            raw, receipt = ml.fetch_vizier(spec, a.work / "vizier")
            manifest["acquisitions"][spec.source] = receipt
            manifest["policies"][spec.source] = dict(
                release=spec.release,
                pipeline=spec.pipeline,
                tier=spec.tier,
                quality=spec.quality,
                fe_h_kind=spec.fe_h_kind,
                notes=spec.notes,
            )
            frame = ml.vizier_observations(spec, raw)
            epochs = (2000.0,) if spec.epoch == 2000.0 else (spec.epoch,)
            frames.append((frame, spec.radius_arcsec, epochs, spec.search_arcsec))
            log(f"{spec.source}: {len(frame)} rows")
        except Exception as error:
            manifest["errors"][spec.source] = f"{type(error).__name__}: {error}"
            log(f"FAILED {spec.source}: {error}")
        save()
    for spec, fetch, convert in (
        (
            ml.SAGA,
            lambda: ml.fetch_saga(a.work / "saga", log=log),
            ml.saga_observations,
        ),
        (
            ml.JINABASE,
            lambda: ml.fetch_jinabase(a.work / "jinabase"),
            ml.jinabase_observations,
        ),
    ):
        if not wanted(spec["source"]):
            continue
        try:
            raw, receipt = fetch()
            manifest["acquisitions"][spec["source"]] = receipt
            manifest["policies"][spec["source"]] = {
                k: v for k, v in spec.items() if k not in ("fe_h_bins",)
            }
            frame = convert(raw)
            try:
                frame = ml.resolve_by_name(frame, a.work / "simbad", log=log)
            except Exception as error:
                manifest["errors"]["simbad:" + spec["source"]] = (
                    f"{type(error).__name__}: {error}"
                )
                log(f"FAILED simbad {spec['source']}: {error}")
            # Literature positions: J2000 at roughly epoch 2000, often low precision
            frames.append((frame, 3.0, (2000.0,), 30.0))
            log(f"{spec['source']}: {len(frame)} rows")
        except Exception as error:
            manifest["errors"][spec["source"]] = f"{type(error).__name__}: {error}"
            log(f"FAILED {spec['source']}: {error}")
        save()
    resolved = []
    for frame, radius, epochs, search in frames:
        name = frame.source.iloc[0] if len(frame) else "empty"
        if not a.skip_positional and (frame.source_id == 0).any():
            try:
                log(
                    f"positional Gaia match {name}: {(frame.source_id == 0).sum()} rows"
                )
                frame = ml.resolve_positional(
                    frame,
                    a.work / "gaia" / name,
                    radius_arcsec=radius,
                    epochs=epochs,
                    batch=a.gaia_batch,
                    search_arcsec=search,
                    log=log,
                )
            except Exception as error:
                manifest["errors"]["gaia:" + name] = f"{type(error).__name__}: {error}"
                log(f"FAILED gaia {name}: {error}")
        resolved.append(frame)
        save()
    obs = pd.concat(resolved, ignore_index=True)
    obs.to_parquet(a.output / "observations.parquet", index=False)
    existing, chosen = {}, []
    for root in a.existing_root:
        ids, picked = existing_sidecar_ids(root, log)
        existing.update(ids)
        chosen += picked
    result, stars = ml.census(obs, existing or None)
    stars.to_parquet(a.output / "stars.parquet", index=False)
    manifest.update(
        census=result,
        existing_sidecars=chosen,
        outputs={
            name: ml.sha256(a.output / name)
            for name in ("observations.parquet", "stars.parquet")
        },
        status="complete" if not manifest["errors"] else "complete_with_errors",
    )
    save()
    log(json.dumps(result["union"], indent=1))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    main()
