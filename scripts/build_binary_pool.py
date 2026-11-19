"""Reference pool of wide-binary companions inheriting their primary's [Fe/H].

usage: build_binary_pool.py --stars stars.fits --teacher pred.npz
                            --targets targets.npz --config CFG --output pool.npz

Components of a wide binary share [Fe/H]. A companion of a bright FGK dwarf
(the "primary": G < ``--max-g``, 5000-6500 K, log g > 3.8, narrow [Fe/H]
interval) gets the primary's predicted [Fe/H] (``--teacher``, e.g. v7 with
parallax conditioning, where the common scale is pinned by the high-res
core) with error sd_primary (+) ``--floor`` dex. Used as a reference pool
(``multisurvey.reference_pool``), it ties cool or faint companions to the
FGK scale. 20 % of the pairs (hash of the smaller source_id) are held out
for evaluation (``heldout``). Writes the pool in the unlabelled layout
(``features``, ``fields``) plus ``value``, ``error``, ``heldout``,
``source_id``, ``primary_id``, ``teff_companion``.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


def heldout(a, b, fraction=5):
    lo = np.minimum(a, b).astype(np.uint64)
    h = (lo * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(33)
    return (h % np.uint64(fraction)) == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stars", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-g", type=float, default=15.0)
    p.add_argument("--max-sd", type=float, default=0.15)
    p.add_argument("--floor", type=float, default=0.05)
    a = p.parse_args()
    if a.output.exists():
        print(a.output, "exists")
        return
    T = np.load(a.teacher)
    sid = T["source_id"]
    o = np.argsort(sid)
    sid, med, q, gmag = sid[o], T["median"][o], T["quantiles"][o], T["G"][o]
    sd = 0.5 * (q[:, 2, 2] - q[:, 2, 0])
    teff, logg, feh = med[:, 0], med[:, 1], med[:, 2]
    primary_ok = (
        (gmag < a.max_g) & (teff > 5000) & (teff < 6500) & (logg > 3.8)
        & np.isfinite(feh) & (sd < a.max_sd)
    )  # fmt: skip

    def look(ids):
        pos = np.searchsorted(sid, ids).clip(max=len(sid) - 1)
        return pos, sid[pos] == ids

    tg = np.load(a.targets)
    p1, o1 = look(tg["pair1"])
    p2, o2 = look(tg["pair2"])
    ok = o1 & o2
    p1, p2 = p1[ok], p2[ok]
    prim = np.concatenate([p1, p2])
    comp = np.concatenate([p2, p1])
    use = primary_ok[prim]
    prim, comp = prim[use], comp[use]
    hold = heldout(sid[prim], sid[comp])
    # One record per companion: the brightest qualifying primary.
    order = np.lexsort((gmag[prim], comp))
    prim, comp, hold = prim[order], comp[order], hold[order]
    first = np.r_[True, comp[1:] != comp[:-1]]
    prim, comp, hold = prim[first], comp[first], hold[first]
    print(f"{len(comp)} companions ({hold.sum()} held out); companion Teff "
          f"<4500 K: {(teff[comp] < 4500).sum()}")  # fmt: skip

    cfg = yaml.safe_load(a.config.read_text())["data"]
    errors = [e for e in cfg["error_cols"] if e is not None]
    fields = list(dict.fromkeys([*cfg["feature_cols"], *errors]))
    from astropy.io import fits

    with fits.open(a.stars, memmap=True) as h:
        d = h[1].data
        star_sid = np.asarray(d["source_id"], dtype="i8")
        if not np.array_equal(star_sid, sid):
            pos = np.searchsorted(star_sid, sid[comp])
            if not np.array_equal(star_sid[pos], sid[comp]):
                raise ValueError("Companions missing from the star table")
        else:
            pos = comp
        features = np.column_stack([np.asarray(d[f][pos], dtype="f4") for f in fields])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.output, features=features, fields=np.array(fields, dtype="U40"),
        value=feh[prim], error=np.hypot(sd[prim], a.floor), heldout=hold,
        source_id=sid[comp], primary_id=sid[prim], teff_companion=teff[comp],
        g_companion=gmag[comp],
    )  # fmt: skip
    print("wrote", a.output)


if __name__ == "__main__":
    main()
