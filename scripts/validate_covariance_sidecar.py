"""Validate covariance sidecars written by ``build_covariance_sidecar.py``.

For a random sample of found stars in each sidecar:

- XP covariance reconstruction: exact symmetry, Cholesky (float64) success
  rate, diagonal = published errors squared, correlation eigenvalues, |R| <= 1;
  the packing is cross-checked against an independent construction from the
  Gaia DR3 datamodel formula (``M[i,j] = C[j(j-1)/2 + i]``, i < j) and against
  the transposed (wrong) packing, whose positive-definite rate should collapse.
- Astrometry: 5x5 covariance Cholesky rate, L21 zero-point coverage.
- Partition lookup: stored HATS partition vs the source_id partition.
- With ``--table-fits`` (for the sidecar whose ``row_index`` indexes that FITS
  table): source_id alignment over all rows, and on the sample the table XP
  coefficients x f (f = 10**((8.5 - G)/2.5)) against the HATS coefficients,
  table errors against HATS errors, table G against phot_g_mean_mag.

Writes a JSON report (``--out``) and prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_covariance_sidecar import assign_partitions, load_partitions  # noqa: E402

from masked_stellar_autoencoder.data import covariance_sidecar as cs  # noqa: E402


def _q(x, qs=(0.5, 0.9, 0.99, 1.0)) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {
        f"q{int(q * 100)}" if q < 1 else "max": float(np.quantile(x, q)) for q in qs
    }


def _cholesky_ok(m: np.ndarray) -> np.ndarray:
    ok = np.zeros(len(m), bool)
    for i, a in enumerate(m):
        if np.all(np.isfinite(a)):
            try:
                np.linalg.cholesky(a)
                ok[i] = True
            except np.linalg.LinAlgError:
                pass
    return ok


def check_xp(sc: cs.CovarianceSidecar, rows: np.ndarray) -> dict:
    out = {}
    for band in cs.BANDS:
        r = sc.read([f"{band}_coefficient_errors", f"{band}_coefficient_correlations",
                     f"{band}_standard_deviation"], rows)  # fmt: skip
        e, c = r[f"{band}_coefficient_errors"], r[f"{band}_coefficient_correlations"]
        cov = cs.xp_covariance(e, c)
        corr = cs.unpack_correlations(c.astype(np.float64))
        n = cs.N_COEFF
        # independent construction (datamodel formula), first rows
        indep = np.zeros((min(5, len(c)), n, n))
        for k in range(len(indep)):
            indep[k] = np.eye(n)
            for j in range(n):
                for i in range(j):
                    indep[k, i, j] = indep[k, j, i] = c[k, j * (j - 1) // 2 + i]
        # the transposed packing (row-major upper triangle) as a negative control
        wrong = np.zeros_like(corr)
        iu, ju = np.triu_indices(n, 1)
        wrong[:, iu, ju] = c
        wrong[:, ju, iu] = c
        wrong[:, range(n), range(n)] = 1.0
        eig = np.linalg.eigvalsh(corr)[:, 0]
        # basis-function correlations are banded: |r| decays with |i - j|. Share of
        # off-diagonal |r| within |i - j| <= 2 (a scrambled packing gives ~107/1485).
        off = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        near, any_off = (off >= 1) & (off <= 2), off >= 1

        def band_mass(m):
            return float(
                np.median(np.abs(m[:, near]).sum(1) / np.abs(m[:, any_off]).sum(1))
            )

        e64 = e.astype(np.float64)
        out[band] = dict(
            rows=int(len(rows)),
            max_asymmetry=float(np.abs(cov - np.swapaxes(cov, -1, -2)).max()),
            cholesky_ok_fraction=float(_cholesky_ok(cov).mean()),
            diag_max_rel_error=float(
                np.nanmax(np.abs(np.diagonal(cov, axis1=1, axis2=2) - e64**2) / e64**2)
            ),
            independent_packing_max_abs_diff=float(
                np.abs(indep - corr[: len(indep)]).max()
            ),
            transposed_packing_cholesky_ok_fraction=float(_cholesky_ok(wrong).mean()),
            near_diagonal_abs_mass=band_mass(corr),
            near_diagonal_abs_mass_transposed=band_mass(wrong),
            near_diagonal_abs_mass_scrambled=float(near.sum() / any_off.sum()),
            correlation_min_eigenvalue=_q(eig, (0.0, 0.01, 0.5)),
            correlation_nonpositive_min_eigenvalue=int(np.sum(eig <= 0)),
            max_abs_correlation=float(np.nanmax(np.abs(c))),
            standard_deviation=_q(r[f"{band}_standard_deviation"]),
        )
    return out


def check_astrometry(sc: cs.CovarianceSidecar, rows: np.ndarray) -> dict:
    aps = sc.read("astrometric_params_solved", rows)["astrometric_params_solved"]
    cov = sc.astrometric_covariance(rows)
    zpt, ok = sc.parallax_zero_point(rows)
    out = dict(rows=int(len(rows)), solutions={int(k): int(v) for k, v in
               zip(*np.unique(aps, return_counts=True), strict=True)})  # fmt: skip
    for code, name in ((31, "5p"), (95, "6p")):
        m = aps == code
        if m.any():
            out[name] = dict(
                cholesky_ok_fraction=float(_cholesky_ok(cov[m]).mean()),
                zpt_mas=_q(zpt[m], (0.01, 0.5, 0.99)),
                zpt_in_range_fraction=float(ok[m].mean()),
            )
    stored = sc.read("parallax_zpt_l21", rows)["parallax_zpt_l21"]
    out["stored_zpt_max_abs_diff_mas"] = float(
        np.nanmax(np.abs(stored - zpt), initial=0.0)
    )
    return out


def check_partitions(sc, rows, partitions) -> dict:
    r = sc.read(["source_id", "hats_norder", "hats_npix"], rows)
    k, p = assign_partitions(r["source_id"], partitions)
    same = (k == r["hats_norder"]) & (p == r["hats_npix"])
    return dict(rows=int(len(rows)), source_id_partition_agrees=int(same.sum()),
                found_in_other_partition=int((~same).sum()))  # fmt: skip


def check_table(sc, sample, fits_path: Path, chunk: int = 1_000_000) -> dict:
    from astropy.io import fits

    out = {}
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        n_fits = len(data)
        sid = np.asarray(sc.read("source_id")["source_id"])
        row_index = np.asarray(sc.read("row_index")["row_index"])
        mismatch = 0
        for s in range(0, len(row_index), chunk):
            rows = row_index[s : s + chunk]
            lo, hi = int(rows.min()), int(rows.max()) + 1
            col = np.asarray(data["source_id"][lo:hi], np.int64)[rows - lo]
            mismatch += int(np.sum(col != sid[s : s + chunk]))
        out["alignment"] = dict(fits_rows=n_fits, sidecar_rows=int(len(sid)),
                                source_id_mismatches=mismatch)  # fmt: skip
        rows = row_index[sample]
        order = np.argsort(rows)
        sub = data[rows[order]]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        names = set(data.columns.names)
        table = {c: np.asarray(sub[c], np.float64)[inv] for c in names
                 if c == "G" or c.startswith(("bp_", "rp_", "bpe_", "rpe_"))}  # fmt: skip
    r = sc.read(["phot_g_mean_mag", "bp_coefficients", "rp_coefficients",
                 "bp_coefficient_errors", "rp_coefficient_errors"], sample)  # fmt: skip
    g = r["phot_g_mean_mag"].astype(np.float64)
    out["g_minus_phot_g_mean_mag"] = _q(np.abs(table["G"] - g))
    f = cs.xp_normalisation_factor(g)
    for band in cs.BANDS:
        x = np.stack([table[f"{band}_{k}"] for k in range(1, cs.N_COEFF + 1)], axis=1)
        c = r[f"{band}_coefficients"]
        e = r[f"{band}_coefficient_errors"].astype(np.float64)
        raw = x * f[:, None]
        rel = np.abs(raw - c) / np.abs(c)
        res = dict(
            rows=int(len(sample)),
            coefficient_rel_diff=_q(rel),
            coefficient_rel_diff_first5=_q(rel[:, :5]),
            coefficient_diff_over_error=_q(np.abs(raw - c) / e),
        )
        if f"{band}e_1" in table:
            te = np.stack(
                [table[f"{band}e_{k}"] for k in range(1, cs.N_COEFF + 1)], axis=1
            )
            res["table_error_rel_diff_raw"] = _q(np.abs(te - e) / e)
            res["table_error_rel_diff_normalised"] = _q(np.abs(te * f[:, None] - e) / e)
        out[band] = res
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--table-fits", type=Path, help="FITS table indexed by row_index")
    p.add_argument(
        "--table-name", default="table", help="sidecar NAME for --table-fits"
    )
    p.add_argument("--xp-hats", default="/arc/projects/hats/gaia/dr3_xp_continuous")
    p.add_argument("--sample", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    partitions = load_partitions(Path(a.xp_hats))
    rng = np.random.default_rng(a.seed)
    report = {}
    for item in a.sidecar:
        name, _, path = item.partition("=")
        with cs.CovarianceSidecar(path) as sc:
            flags = sc.read(["found_xp", "found_astrometry"])
            xp_rows = np.flatnonzero(flags["found_xp"])
            ast_rows = np.flatnonzero(flags["found_astrometry"])
            pick = lambda rows: np.sort(
                rng.choice(rows, min(a.sample, len(rows)), replace=False)
            )  # noqa: E731
            xs, ast = pick(xp_rows), pick(ast_rows)
            rep = dict(
                path=path,
                rows=len(sc),
                found_xp=int(len(xp_rows)),
                found_astrometry=int(len(ast_rows)),
                xp=check_xp(sc, xs),
                astrometry=check_astrometry(sc, ast),
                partitions=check_partitions(sc, np.union1d(xs, ast), partitions),
            )
            if a.table_fits is not None and name == a.table_name:
                rep["table_vs_hats"] = check_table(sc, xs, a.table_fits)
        report[name] = rep
    text = json.dumps(report, indent=2)
    if a.out:
        a.out.write_text(text + "\n")
    print(text, flush=True)
    return report


if __name__ == "__main__":
    main()
