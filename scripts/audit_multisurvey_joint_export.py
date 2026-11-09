"""Independent saved-prediction integrity and moment checks; no model execution."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(
    description="Audit saved full-FIT devtest prediction parameters without executing a stellar network"
)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--fit-ids", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
run = args.run
out = args.output
out.mkdir(parents=True, exist_ok=True)
m = json.loads((run / "manifest.json").read_text())
receipt = m["result"]["exports"]["devtest"]
p = args.predictions
with p.open("rb") as f:
    actual = hashlib.file_digest(f, "sha256").hexdigest()
assert actual == receipt["sha256"], "Complete-file SHA must match before any analysis"
fit = np.load(args.fit_ids, allow_pickle=False)
assert fit.dtype == np.int64 and len(fit) == len(np.unique(fit)) == m["fit_stars"]
assert (
    hashlib.sha256(np.sort(fit).astype("<i8").tobytes()).hexdigest()
    == m["fit_ids_sha256"]
)
with h5py.File(p) as h:
    ids = h["source_id"][:]
    assert (
        ids.dtype == np.int64
        and len(ids) == receipt["stars"]
        and (np.diff(ids) > 0).all()
    )
    assert h.attrs["role"] == "devtest" and not h.attrs["artificial_masking"]
    assert h.attrs["selected_checkpoint_sha256"] == m["result"]["selected"]["sha256"]
    assert (
        hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
        == receipt["source_ids_sha256"]
    )
    assert not np.intersect1d(fit, ids).size
    counts = np.zeros((3, 3), dtype="i8")
    maxerr = 0.0
    nonpositive = 0
    for start in range(0, len(ids), 8192):
        sl = slice(start, start + 8192)
        w, mu, lower, mean, cov = [
            h[k][sl]
            for k in (
                "weights",
                "component_mean",
                "component_lower",
                "mean",
                "covariance",
            )
        ]
        assert all(np.isfinite(a).all() for a in (w, mu, lower, mean, cov))
        assert (w >= 0).all()
        np.testing.assert_allclose(w.sum(1), 1, atol=1e-12, rtol=1e-12)
        assert (np.diagonal(lower, axis1=-2, axis2=-1) > 0).all() and (
            np.triu(lower, 1) == 0
        ).all()
        expected_mean = (w[:, :, None] * mu).sum(1)
        np.testing.assert_allclose(expected_mean, mean, atol=1e-8, rtol=1e-12)
        delta = mu - mean[:, None]
        expected_cov = (
            w[:, :, None, None]
            * (
                lower @ lower.transpose(0, 1, 3, 2)
                + delta[:, :, :, None] * delta[:, :, None, :]
            )
        ).sum(1)
        np.testing.assert_allclose(expected_cov, cov, atol=1e-8, rtol=1e-11)
        scale = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
        assert (scale > 0).all()
        corr = cov / (scale[:, :, None] * scale[:, None, :])
        assert np.linalg.eigvalsh(corr).min() > -1e-10
        observed = h["native_observed"][sl]
        values = h["native_value"][sl]
        assert np.isfinite(values[observed]).all()
        counts += observed.sum(0)
        nonpositive += int((mean[:, 0] <= 0).sum())
        maxerr = max(maxerr, float(abs(expected_mean - mean).max()))
    summary = dict(
        devtest_stars=len(ids),
        actual_fit_first_pass_stars=len(fit),
        fit_devtest_overlap=0,
        devtest_prediction_sha256=actual,
        source_ids_sha256=receipt["source_ids_sha256"],
        complete_file_sha="pass",
        finite_parameters="pass",
        normalized_mixture_weights="pass",
        positive_component_cholesky="pass",
        analytic_mean_and_covariance="pass",
        covariance_correlation_psd="pass",
        native_observed_labels_finite="pass",
        nonpositive_mean_teff=nonpositive,
        max_absolute_mean_reduction_error=maxerr,
        native_scalar_counts=dict(zip(("APOGEE", "DESI", "GALAH"), counts.tolist())),
        scope="Independent local audit of saved prediction parameters, no native checkpoint execution; full FIT catalogue SHA remains producer-computed. One complete FIT identity stream independently checked against source census.",
    )
    (out / "independent-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
