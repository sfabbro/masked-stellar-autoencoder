"""Table of v7 ablation anchor metrics (one row per variant).

usage: summarize_ablation.py DIR

DIR holds ``<variant>/finetune_anchor_metrics.jsonl`` (one JSON line per
epoch, from the anchored trainer). For each variant the epoch with the lowest
``score`` is reported (same rule as the full runs), with its held-out
residuals: Teff vs IRFM, seismic log g (also after conditioning on the
measured parallax, ``_plx``), the [Fe/H] core overall and below -2 / -2.5,
the predicted-bin calibration below -2, E(B-V), 16-84 % coverage (``cov``),
and the dust branch where present.
"""

import argparse
import json
from pathlib import Path


def cell(m, key, field, fmt):
    v = m.get(key, {}).get(field)
    return format(v, fmt) if v is not None else "-"


COLUMNS = [
    ("Teff nmad", "teff_anchor", "nmad", ".1f"),
    ("Teff bias", "teff_anchor", "bias", "+.1f"),
    ("logg nmad", "logg_anchor", "nmad", ".3f"),
    ("logg cov", "logg_anchor", "cov", ".2f"),
    ("logg+plx", "logg_anchor_plx", "nmad", ".3f"),
    ("FeH nmad", "fe_h_core", "nmad", ".3f"),
    ("FeH<-2 bias", "fe_h_core_mp", "bias", "+.3f"),
    ("FeH<-2 nmad", "fe_h_core_mp", "nmad", ".3f"),
    ("FeH<-2 cov", "fe_h_core_mp", "cov", ".2f"),
    ("FeH<-2.5 bias", "fe_h_core_vmp", "bias", "+.3f"),
    ("pred<-2 ref-pred", "fe_h_mp_predbin", "bias", "+.3f"),
    ("EBV nmad", "ebv_star", "nmad", ".4f"),
    ("dust nmad", "dust_branch_ebv", "nmad", ".4f"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir", type=Path)
    a = p.parse_args()
    head = ["variant", "ep", "score", *[c[0] for c in COLUMNS]]
    rows = []
    for f in sorted(a.dir.glob("*/finetune_anchor_metrics.jsonl")):
        ms = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if not ms:
            continue
        best = min(ms, key=lambda m: m["score"])
        rows.append(
            [f.parent.name, f"{best['epoch']}/{len(ms)}", f"{best['score']:.2f}"]
            + [cell(best, k, fld, fmt) for _, k, fld, fmt in COLUMNS]
        )
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(str(x).rjust(w) for x, w in zip(r, widths, strict=True)))


if __name__ == "__main__":
    main()
