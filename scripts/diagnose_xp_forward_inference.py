"""Diagnose XP forward-model inference on real held-out stars.

Run 2 of 20261122 trained to XP chi^2/dof ~ 1 at the labels, yet its MLEs had
chi^2/dof 2-5, half of the stars at the lower E bound and Teff uncorrelated with
the labels, while inference on synthetic stars drawn from the same model recovers
theta to ~10 K. This script separates the candidate explanations on ``--n``
held-out stars of a trained ``model.pt``:

* ``exact NLL``: the exact likelihood (learned covariance at each point's own
  prediction, :func:`star_nll`) at the labels (E and parallax profiled) and at the
  standard MLE. If the labels score better, the search failed; if the MLE scores
  better, the likelihood itself prefers the wrong point.
* ``start``: the candidate-bank start against the labels.
* ``oracle``: the standard passes started at the labels.
* ``loose_e``: the standard inference with a looser lower bound on ``E``
  (``--loose-e-min``).

Writes ``diagnosis.json`` and ``diagnosis.md`` to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import masked_stellar_autoencoder.training.xp_forward_model as xf

NAMES = ("teff", "logg", "fe_h")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--covariance-sidecar")
    p.add_argument("--reddening-labels")
    p.add_argument("--core-rows")
    p.add_argument("--bands", default=",".join(xf.LIKELIHOOD_BANDS))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--infer-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loose-e-min", type=float, default=-0.3)
    return p.parse_args(argv)


def _stats(est: np.ndarray, lab: np.ndarray) -> dict:
    out = {}
    for j, n in enumerate(NAMES):
        ok = np.isfinite(est[:, j]) & np.isfinite(lab[:, j])
        d = est[ok, j] - lab[ok, j]
        out[n] = {
            "median_offset": float(np.median(d)),
            "median_abs": float(np.median(np.abs(d))),
            "corr": float(np.corrcoef(est[ok, j], lab[ok, j])[0, 1]),
        }
    return out


def _exact_nll(model, data, theta, ebv, parallax, device) -> np.ndarray:
    """Exact per-star NLL (covariance at each point's own prediction), float64."""
    m = model.to(device=device, dtype=torch.float64).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(data), 1024):
            idx = np.arange(s, min(s + 1024, len(data)))
            batch = data.batch(idx, device, torch.float64)
            t = lambda a: torch.as_tensor(a[idx], device=device, dtype=torch.float64)  # noqa: E731
            nll = xf.star_nll(m, t(theta), t(ebv), t(parallax), batch)
            out.append(nll.cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> None:
    a = parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, a.infer_dtype)
    labels = xf.load_labels(a.labels)
    core = (
        np.load(a.core_rows).astype(np.int64) if a.core_rows else np.zeros(0, np.int64)
    )
    sel = xf.select_rows(
        labels, core, max_train=0, max_valid=0, max_heldout=a.n, seed=a.seed
    )
    sidecar = (
        xf.SidecarXPCovariance(a.covariance_sidecar) if a.covariance_sidecar else None
    )
    reddening = xf.ReddeningLabels(a.reddening_labels) if a.reddening_labels else None
    bands = xf.bands_by_name([b for b in a.bands.split(",") if b])
    data, _ = xf.load_star_data(
        a.table, sel["heldout"], labels=labels, bands=bands, sidecar=sidecar,
        reddening=reddening, zero_point="sidecar" if sidecar else "constant", log=log,
    )  # fmt: skip
    keep = np.flatnonzero(np.isfinite(data.theta).all(axis=1))
    data = data.subset(keep)
    lab = data.theta
    log(f"{len(data)} labelled held-out stars")
    model, extra = xf.load_model(a.checkpoint)
    for p in model.parameters():
        p.requires_grad_(False)
    bounds = np.asarray(extra["theta_bounds"])
    pool = np.asarray(extra["theta_pool"])
    cfg = xf.InferenceConfig()
    report: dict = {"n": len(data), "checkpoint": a.checkpoint, "dtype": a.infer_dtype}

    t0 = time.time()
    std = xf.infer_stars(
        model, data, bounds, pool, cfg, device=device, dtype=dtype, log=log
    )
    report["standard"] = {"mle": _stats(std["mle"], lab), "start": _stats(std["init"][:, :3], lab),
                          "E_at_bound": float(np.mean(std["E_bound"] != 0)),
                          "chi2_per_dof_median": float(np.median(std["chi2_xp"] / std["dof_xp"])),
                          "seconds": round(time.time() - t0, 1)}  # fmt: skip
    log(f"standard: {json.dumps(report['standard'])}")

    prof = xf.profile_nuisance_at_labels(
        model, data, device=device, dtype=torch.float64
    )
    nll_lab = _exact_nll(model, data, lab, prof["ebv"], prof["parallax"], device)
    nll_mle = _exact_nll(
        model, data, std["mle"], std["E_mle"], std["parallax_mle"], device
    )
    gap = nll_mle - nll_lab
    report["exact_nll"] = {
        "label_better_fraction": float(np.mean(gap > 0)),
        "gap_median": float(np.median(gap)),
        "gap_quantiles_10_90": [float(q) for q in np.quantile(gap, [0.1, 0.9])],
        "chi2_per_dof_at_labels_median": float(
            np.median(prof["chi2_xp"] / prof["dof_xp"])
        ),
    }
    log(f"exact NLL: {json.dumps(report['exact_nll'])}")

    batch = data.batch(np.arange(len(data)), torch.device("cpu"), torch.float64)
    u = xf.nuisance_to_unit(prof["ebv"], prof["parallax"], batch["log_plx_lo"].numpy(),
                            batch["log_plx_hi"].numpy(), batch["e_max"].numpy())  # fmt: skip
    oracle_init = np.concatenate([lab, u], axis=1)
    ora = xf.infer_stars(model, data, bounds, pool, cfg, device=device, dtype=dtype, log=log,
                         init=oracle_init)  # fmt: skip
    nll_ora = _exact_nll(
        model, data, ora["mle"], ora["E_mle"], ora["parallax_mle"], device
    )
    report["oracle"] = {"mle": _stats(ora["mle"], lab),
                        "E_at_bound": float(np.mean(ora["E_bound"] != 0)),
                        "nll_minus_label_median": float(np.median(nll_ora - nll_lab)),
                        "nll_minus_standard_median": float(np.median(nll_ora - nll_mle))}  # fmt: skip
    log(f"oracle: {json.dumps(report['oracle'])}")

    # A looser lower bound on E (the emulator's E gauge comes from noisy labels).
    e_min_default = xf.E_MIN
    xf.E_MIN = a.loose_e_min
    try:
        loose = xf.infer_stars(
            model, data, bounds, pool, cfg, device=device, dtype=dtype, log=log
        )
    finally:
        xf.E_MIN = e_min_default
    nll_loose = _exact_nll(
        model, data, loose["mle"], loose["E_mle"], loose["parallax_mle"], device
    )
    report["loose_e"] = {"mle": _stats(loose["mle"], lab), "start": _stats(loose["init"][:, :3], lab),
                      "E_at_bound": float(np.mean(loose["E_bound"] != 0)),
                      "E_min": a.loose_e_min,
                      "nll_minus_label_median": float(np.median(nll_loose - nll_lab))}  # fmt: skip
    log(f"loose E_min: {json.dumps(report['loose_e'])}")

    (out / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# XP forward-model inference diagnosis", "", f"{len(data)} held-out stars, {a.checkpoint}", "",
             "| run | Teff corr / med abs | log g corr / med abs | [Fe/H] corr / med abs | E at bound |",
             "|---|---|---|---|---|"]  # fmt: skip
    for name, key in (("standard start", ("standard", "start")), ("standard MLE", ("standard", "mle")),
                      ("oracle MLE", ("oracle", "mle")), ("loose-E start", ("loose_e", "start")),
                      ("loose-E MLE", ("loose_e", "mle"))):  # fmt: skip
        st = report[key[0]][key[1]]
        eb = (
            report[key[0]].get("E_at_bound", float("nan"))
            if key[1] == "mle"
            else float("nan")
        )
        cells = [f"{st[n]['corr']:.2f} / {st[n]['median_abs']:.3g}" for n in NAMES]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {eb:.2f} |")
    lines += ["", f"Exact NLL: {json.dumps(report['exact_nll'])}", "",
              f"Oracle: NLL - label {report['oracle']['nll_minus_label_median']:.2f}, "
              f"- standard {report['oracle']['nll_minus_standard_median']:.2f} (median)", "",
              f"Loose E_min {a.loose_e_min}: NLL - label {report["loose_e"]['nll_minus_label_median']:.2f} (median)", ""]  # fmt: skip
    (out / "diagnosis.md").write_text("\n".join(lines))
    log("done")


if __name__ == "__main__":
    main()
