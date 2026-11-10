"""Bounded, standard-library-only CANFAR inventory for additional survey sources."""

import argparse
import datetime as dt
import gzip
import json
import os
from pathlib import Path

TOKENS = (
    "lamost",
    "mastar",
    "rave",
    "gaiaeso",
    "gaia-eso",
    "gaia_eso",
    "gspspec",
    "gaia_rvs",
    "boss",
    "segue",
    "sppparams",
    "sspp",
    "astra",
    "aspcap",
    "h3survey",
    "s5dr1",
    "pigs",
    "pristine",
    "saga",
    "jina",
    "ambre",
    "pastel",
    "vmp",
    "master_label",
)


def card_value(text):
    text = text.lstrip()
    if not text.startswith("'"):
        return text.split("/")[0].strip()
    result = []
    i = 1
    while i < len(text):
        if text[i] == "'":
            if i + 1 < len(text) and text[i + 1] == "'":
                result.append("'")
                i += 2
                continue
            return "".join(result).strip()
        result.append(text[i])
        i += 1
    raise ValueError("Unterminated FITS string")


def fits_schema(path):
    """Read only headers, with a 1 MiB header limit; never load table rows."""
    with open(path, "rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
    with gzip.open(path, "rb") if compressed else open(path, "rb") as stream:
        for _ in range(3):
            header = {}
            for _ in range(365):
                block = stream.read(2880)
                if len(block) != 2880:
                    raise ValueError("Truncated FITS header")
                cards = [block[i : i + 80].decode("ascii") for i in range(0, 2880, 80)]
                end = False
                for card in cards:
                    key = card[:8].strip()
                    if key == "END":
                        end = True
                        break
                    if card[8:10] == "= ":
                        value = card_value(card[10:])
                        header[key] = value
                if end:
                    break
            else:
                raise ValueError("FITS header exceeds bound")
            if header.get("XTENSION") == "BINTABLE":
                count = int(header["TFIELDS"])
                return dict(
                    rows=int(header["NAXIS2"]),
                    row_bytes=int(header["NAXIS1"]),
                    columns=[header.get(f"TTYPE{i}", "") for i in range(1, count + 1)],
                    units={
                        header.get(f"TTYPE{i}", str(i)): header[f"TUNIT{i}"]
                        for i in range(1, count + 1)
                        if f"TUNIT{i}" in header
                    },
                    header_only=True,
                    content_sha256_verified=False,
                )
            axes = int(header.get("NAXIS", "0"))
            size = 0 if axes == 0 else abs(int(header["BITPIX"])) // 8
            for i in range(1, axes + 1):
                size *= int(header["NAXIS" + str(i)])
            size += int(header.get("PCOUNT", "0"))
            # Do not decompress a large non-table image merely to reach another HDU.
            if compressed and size > 2**20:
                raise ValueError(
                    "Compressed non-table data exceeds header-only scan bound"
                )
            stream.seek(((size + 2879) // 2880) * 2880, 1)
    raise ValueError("No table in first three HDUs")


def inventory(root, max_depth=3, max_files=100000):
    root = Path(root)
    found = []
    examined = 0
    for base, dirs, files in os.walk(root):
        depth = len(Path(base).relative_to(root).parts)
        dirs[:] = (
            sorted(d for d in dirs if d not in {".git", ".pixi", ".cache"})
            if depth < max_depth
            else []
        )
        for name in sorted(files):
            examined += 1
            if examined > max_files:
                return dict(
                    root=str(root),
                    complete=False,
                    reason="file bound reached",
                    examined=examined,
                    sources=found,
                )
            path = Path(base) / name
            relative = str(path.relative_to(root)).lower()
            if not any(token in relative for token in TOKENS):
                continue
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            row = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if name.lower().endswith((".fits", ".fits.gz", ".fit")):
                try:
                    row.update(fits_schema(path))
                except Exception as error:
                    row["schema_error"] = str(error)
            found.append(row)
    return dict(root=str(root), complete=True, examined=examined, sources=found)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not Path("/arc/projects/k-pop").is_dir():
        raise RuntimeError("Execute the source inventory on CANFAR")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="inventory_complete",
        training_performed=False,
        model_loaded=False,
        utc=dt.datetime.now(dt.UTC).isoformat(),
        script_sha256=os.environ.get("MSA_INVENTORY_SCRIPT_SHA256"),
        inventories=[inventory(root) for root in args.root],
    )
    (args.output / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                sources=sum(len(r["sources"]) for r in result["inventories"]),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
