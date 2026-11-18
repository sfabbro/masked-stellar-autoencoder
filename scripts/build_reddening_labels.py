"""Per-star E(B-V) labels for fine-tune tables (training.reddening_labels).

usage: build_reddening_labels.py --fit-table FITS --split-config YAML
           --apply NAME=FITS [NAME=FITS ...] --out DIR

Fits the intrinsic colours on the training split (finetune_data's split of
``--split-config``) of ``--fit-table``, then writes, for every ``--apply``
table, ``DIR/NAME_reddening.npz`` with one entry per table row (source_id,
ebv_star, e_ebv_star), plus ``DIR/intrinsic_colours.npz`` and
``DIR/diagnostics.json``. Fine-tune configs load the npz through
``data.extra_columns``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from masked_stellar_autoencoder.training.finetune_data import (
    _read_selected_fits,
    kept_rows,
    split_row_positions,
)
from masked_stellar_autoencoder.training.reddening_labels import (
    fit_intrinsic_colours,
    observed_colours,
    reddening_labels,
)

COLUMNS = [
    "source_id", "teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h",
    "BP", "RP", "KS", "E_KS", "EBV",
]  # fmt: skip


def read(path):
    t = _read_selected_fits(str(path), COLUMNS)
    return {c: t[c].to_numpy("f8", na_value=np.nan) for c in COLUMNS if c != "source_id"} | {
        "source_id": t["source_id"].to_numpy("i8")
    }  # fmt: skip


def nmad(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-table", required=True)
    p.add_argument("--split-config", required=True, type=Path)
    p.add_argument("--apply", nargs="+", required=True, help="NAME=FITS")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-map-ebv", type=float, default=0.02)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(a.split_config.read_text())
    rows = kept_rows(a.fit_table, cfg)
    train = rows[
        split_row_positions(len(rows), cfg["finetuning"].get("metal_poor") or {})[0]
    ]
    d = read(a.fit_table)
    col = observed_colours(d["BP"], d["RP"], d["KS"], d["E_KS"])
    model = fit_intrinsic_colours(
        d["teff"][train], d["logg"][train], d["fe_h"][train], col[train],
        d["EBV"][train], max_map_ebv=a.max_map_ebv,
    )  # fmt: skip
    np.savez(a.out / "intrinsic_colours.npz", **model.to_dict())
    diag = {"fit_table": a.fit_table, "train_rows": int(len(train)),
            "max_map_ebv": a.max_map_ebv, "scatter_bp_rp": model.scatter[0].tolist(),
            "scatter_rp_ks": model.scatter[1].tolist(),
            "scatter_teff_edges": model.scatter_edges.tolist(), "tables": {}}  # fmt: skip
    for item in a.apply:
        name, path = item.split("=", 1)
        t = d if path == a.fit_table else read(path)
        c = observed_colours(t["BP"], t["RP"], t["KS"], t["E_KS"])
        ebv, err = reddening_labels(
            model, t["teff"], t["logg"], t["fe_h"], c, e_teff=t["e_teff"],
            e_logg=t["e_logg"], e_feh=t["e_fe_h"], e_ks=t["E_KS"],
        )  # fmt: skip
        np.savez(
            a.out / f"{name}_reddening.npz", source_id=t["source_id"],
            ebv_star=ebv.astype("f4"), e_ebv_star=err.astype("f4"),
        )  # fmt: skip
        m = t["EBV"]
        ok = np.isfinite(ebv) & np.isfinite(m)
        low = ok & (m < 0.02)
        diag["tables"][name] = {
            "rows": int(len(ebv)),
            "labelled": int(np.isfinite(ebv).sum()),
            "median_error": float(np.nanmedian(err)),
            "low_map_bias": float(np.median(ebv[low] - m[low])) if low.any() else None,
            "low_map_nmad": nmad(ebv[low] - m[low]),
            "frac_above_map_3sigma": float(np.mean(ebv[ok] - m[ok] > 3 * err[ok])),
            "ratio_to_map_by_map_bin": {
                f"{lo}-{hi}": float(np.median(ebv[k] / m[k]))
                for lo, hi in ((0.05, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 5.0))
                if (k := ok & (m >= lo) & (m < hi)).sum() > 50
            },
        }
        print(name, json.dumps(diag["tables"][name]), flush=True)
    (a.out / "diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")


if __name__ == "__main__":
    main()
