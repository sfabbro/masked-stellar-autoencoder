"""Fetch SVO Filter Profile Service transmission curves for the v11 encoder.

Writes one ``<FilterID with '/' -> '__'>.csv`` per filter (``wavelength_nm,
transmission``; the curve exactly as served, Angstrom converted to nm) into
``src/masked_stellar_autoencoder/v11/filters/`` and a ``provenance.json`` with,
for each filter, the SVO ``filterID``, the request URL, the retrieval time, the
sha256 of the raw VOTable, the detector type (1 = photon counter), the SVO
description and the pivot wavelength recomputed from the curve.

The curves serve two purposes (``masked_stellar_autoencoder.v11.filters``):

* every candidate optical band's *support* (where T > threshold x peak) decides
  whether the band lies inside the XP range and joins the filter-integral term;
* bands that torchgaiaxp 0.3.0 does not package (BASS g/r, MzLS z, OmegaCAM,
  MegaCam u/r, S-PLUS) are integrated from these curves with
  ``XPSpectra.integrate``.

Usage::

    pixi run python scripts/fetch_svo_filters.py            # all filters below
    pixi run python scripts/fetch_svo_filters.py --check    # verify sha256 only
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SVO = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={}"
OUT = (
    Path(__file__).resolve().parent.parent
    / "src/masked_stellar_autoencoder/v11/filters"
)

FILTERS = (
    # SDSS 2.5 m
    *(f"SLOAN/SDSS.{b}" for b in "ugriz"),
    # SkyMapper
    *(f"SkyMapper/SkyMapper.{b}" for b in "uvgriz"),
    # Pan-STARRS1
    *(f"PAN-STARRS/PS1.{b}" for b in "grizy"),
    # DECam: DES, DELVE, Legacy Surveys south
    *(f"CTIO/DECam.{b}" for b in ("g", "r", "i", "z", "Y")),
    # Legacy Surveys north: BASS g, r and MzLS z
    "BOK/BASS.g",
    "BOK/BASS.r",
    "KPNO/MzLS.z",
    # VST OmegaCAM (ATLAS), total system response
    *(f"Paranal/OmegaCAM.{b}_SDSS" for b in "ugriz"),
    # CFHT MegaCam 3rd-generation u and r (installed 2015; UNIONS CFIS)
    "CFHT/MegaCam.u",
    "CFHT/MegaCam.r",
    # S-PLUS (T80Cam-S)
    *(
        f"CTIO/S-PLUS.{b}"
        for b in ("u", "F378", "F395", "F410", "F430", "g", "F515", "r", "F660")
        + ("i", "F861", "z")
    ),
    # J-PLUS (T80Cam)
    *(
        f"OAJ/JPLUS.{b}"
        for b in ("uJAVA", "J0378", "J0395", "J0410", "J0430", "gSDSS", "J0515")
        + ("rSDSS", "J0660", "iSDSS", "J0861", "zSDSS")
    ),
    # Gaia DR3 passbands (support only; the colour uses torchgaiaxp's DR3_Gaia_Vega)
    "GAIA/GAIA3.G",
    "GAIA/GAIA3.Gbp",
    "GAIA/GAIA3.Grp",
)


def file_name(filter_id: str) -> str:
    return filter_id.replace("/", "__") + ".csv"


def parse_votable(raw: bytes) -> tuple[dict[str, str], np.ndarray]:
    root = ET.fromstring(raw)
    params = {}
    rows = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "PARAM":
            params[el.get("name")] = el.get("value")
        elif tag == "TR":
            vals = [td.text for td in el if td.tag.rsplit("}", 1)[-1] == "TD"]
            rows.append([float(v) for v in vals[:2]])
    status = [
        el.get("value")
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1] == "INFO" and el.get("name") == "QUERY_STATUS"
    ]
    if status and status[0] != "OK":
        raise ValueError(f"SVO query status {status[0]}")
    if not rows:
        raise ValueError("no transmission rows")
    return params, np.asarray(rows, np.float64)


def pivot_nm(wl_nm: np.ndarray, t: np.ndarray) -> float:
    return float(
        np.sqrt(np.trapezoid(t * wl_nm, wl_nm) / np.trapezoid(t / wl_nm, wl_nm))
    )


def fetch(filter_id: str) -> tuple[bytes, str]:
    url = SVO.format(filter_id)
    req = urllib.request.Request(url, headers={"User-Agent": "msa-filter-fetch/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read(), url


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="re-fetch and compare sha256")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    prov_path = OUT / "provenance.json"
    old = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov = {
        "source": "SVO Filter Profile Service (Rodrigo & Solano 2020, 2024)",
        "service": "https://svo2.cab.inta-csic.es/theory/fps/",
        "script": "scripts/fetch_svo_filters.py",
        "units": {"wavelength": "nm (SVO Angstrom / 10)", "transmission": "as served"},
        "filters": {},
    }
    bad = 0
    for fid in FILTERS:
        raw, url = fetch(fid)
        sha = hashlib.sha256(raw).hexdigest()
        if args.check:
            ref = old.get("filters", {}).get(fid, {}).get("sha256")
            ok = ref == sha
            bad += not ok
            print(f"{'ok ' if ok else 'DIFF'} {fid}")
            continue
        params, rows = parse_votable(raw)
        unit = params.get("WavelengthUnit", "Angstrom")
        if unit != "Angstrom":
            raise ValueError(f"{fid}: unexpected wavelength unit {unit}")
        wl = rows[:, 0] / 10.0
        t = rows[:, 1]
        order = np.argsort(wl, kind="stable")
        wl, t = wl[order], t[order]
        lines = ["wavelength_nm,transmission"] + [
            f"{w:.4f},{v:.8g}" for w, v in zip(wl, t, strict=True)
        ]
        (OUT / file_name(fid)).write_text("\n".join(lines) + "\n")
        prov["filters"][fid] = {
            "file": file_name(fid),
            "url": url,
            "retrieved_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256_votable": sha,
            "detector_type": params.get("DetectorType"),
            "description": params.get("Description"),
            "facility": params.get("Facility"),
            "instrument": params.get("Instrument"),
            "n_points": int(len(wl)),
            "wavelength_min_nm": float(wl.min()),
            "wavelength_max_nm": float(wl.max()),
            "pivot_nm": pivot_nm(wl, t),
        }
        print(
            f"{fid}: {len(wl)} points, pivot {prov['filters'][fid]['pivot_nm']:.1f} nm"
        )
    if not args.check:
        prov_path.write_text(json.dumps(prov, indent=1, sort_keys=False) + "\n")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
