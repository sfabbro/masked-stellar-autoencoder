"""Cross-fitted same-star survey residual covariance; not known per-survey errors."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import gammaln

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
PRED = ROOT / "multisurvey_joint_fullfit_20261105_v1"
SHA = {
    "fit": "ea54924fd609c40de18e2cd6a5324d28bf400ad4d1c297146ec5d13ba98480ff",
    "tune": "98ce312df18022146c588d9b94c68bd5e2ea6319376a9eeed26494d356ac0ec5",
}


def logscore(residual, covariance, student=False):
    chol = np.linalg.cholesky(covariance)
    z = np.linalg.solve(chol, residual.T).T
    quadratic = (z * z).sum(1)
    logdet = 2 * np.log(np.diag(chol)).sum()
    d = residual.shape[1]
    if not student:
        return 0.5 * (quadratic + logdet + d * np.log(2 * np.pi))
    # t_4 with covariance equal to the covariance argument (scale = covariance/2).
    nu = 4.0
    scale = (nu - 2) / nu
    return -(
        gammaln((nu + d) / 2)
        - gammaln(nu / 2)
        - 0.5 * (d * np.log(nu * np.pi) + logdet + d * np.log(scale))
        - (nu + d) / 2 * np.log1p(quadratic / (scale * nu))
    )


def features(value):
    return np.column_stack(
        [
            np.ones(len(value)),
            (value[:, 0] - 5000) / 1000,
            value[:, 1] - 3,
            value[:, 2] + 0.5,
        ]
    )


def crossfit_fold(ids):
    # Gaia source IDs have structured low bits; parity is not a random split.
    z = np.asarray(ids, dtype="u8").copy()
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return (z & np.uint64(1)).astype("i1")


def fit_bias(x, y, error):
    beta = np.zeros((x.shape[1], 3))
    for j in range(3):
        weight = 1 / error[:, j]
        for _ in range(4):
            beta[:, j] = np.linalg.lstsq(
                x * weight[:, None], y[:, j] * weight, rcond=None
            )[0]
            z = (y[:, j] - x @ beta[:, j]) / error[:, j]
            weight = (
                1
                / error[:, j]
                * np.sqrt(np.minimum(1.0, 2.5 / np.maximum(abs(z), 1e-8)))
            )
    return beta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    data = {}
    for role, expected in SHA.items():
        path = PRED / ("predictions-" + role + ".h5")
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError("Frozen source changed")
        with h5py.File(path) as f:
            data[role] = {
                k: f[k][:]
                for k in [
                    "source_id",
                    "native_value",
                    "native_error",
                    "native_observed",
                ]
            }
    result = dict(
        status="running",
        training_performed=False,
        model_loaded=False,
        pairs={},
        scope="Cross-fitted paired catalogue residuals. Pair covariance combines both surveys, shared errors and remaining mapping discrepancy; not an identified individual-survey covariance. No claim of covariance benefit for XP reconstruction.",
        sources=SHA,
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(result, indent=2) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        arrays = {}
        for role, v in data.items():
            mask = v["native_observed"][:, [left, right], :].all((1, 2))
            values = v["native_value"][mask][:, [left, right], :]
            errors = v["native_error"][mask][:, [left, right], :]
            valid = (
                np.isfinite(values).all((1, 2))
                & np.isfinite(errors).all((1, 2))
                & (errors > 0).all((1, 2))
            )
            values, errors = values[valid], errors[valid]
            e = np.sqrt((errors**2).sum(1) + 2 * np.array([100.0, 0.1, 0.1]) ** 2)
            arrays[role] = (
                v["source_id"][mask][valid],
                features(values.mean(1)),
                values[:, 0] - values[:, 1],
                e,
            )
        ids, x, y, e = arrays["fit"]
        name = (
            ["APOGEE", "DESI", "GALAH"][left]
            + "__"
            + ["APOGEE", "DESI", "GALAH"][right]
        )
        if min(len(ids), len(arrays["tune"][0])) < 100:
            result["pairs"][name] = dict(
                status="insufficient_overlap", fit=len(ids), tune=len(arrays["tune"][0])
            )
            save()
            continue
        fold = crossfit_fold(ids)
        if min((fold == 0).sum(), (fold == 1).sum()) < 16:
            raise ValueError("Insufficient cross-fitting support in each fold")
        residual = np.empty_like(y)
        for f in [0, 1]:
            beta = fit_bias(x[fold != f], y[fold != f], e[fold != f])
            residual[fold == f] = (y[fold == f] - x[fold == f] @ beta) / e[fold == f]
        covariance = np.cov(residual, rowvar=False)
        diag = np.diag(np.maximum(np.diag(covariance), 0.01))
        full = 0.9 * covariance + 0.1 * diag
        beta = fit_bias(x, y, e)
        tids, tx, ty, te = arrays["tune"]
        z = (ty - tx @ beta) / te
        comparison = {
            k: float(logscore(z, c, student=t).mean())
            for k, c, t in [
                ("identity", np.eye(3), False),
                ("fitted_diagonal", diag, False),
                ("fitted_full", full, False),
                ("fitted_full_t4", full, True),
            ]
        }
        sd = np.sqrt(np.diag(covariance))
        corr = covariance / (sd[:, None] * sd[None, :])
        result["pairs"][name] = dict(
            status="complete",
            fit=len(ids),
            tune=len(tids),
            crossfit_correlation=corr.tolist(),
            covariance=covariance.tolist(),
            tune_mean_negative_log_score=comparison,
            delta_full_minus_diagonal=comparison["fitted_full"]
            - comparison["fitted_diagonal"],
        )
        np.savez_compressed(
            a.output / (name + "-residuals.npz"),
            fit_source_id=ids,
            fit_crossfit_residual=residual,
            tune_source_id=tids,
            tune_residual=z,
        )
        save()
    result["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
