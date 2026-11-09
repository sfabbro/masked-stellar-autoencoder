"""Audit all predeclared stress corners; never infer population prevalence from them."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def gravity_modes(draws):
    """Descriptive logg peak diagnostic, checked at two histogram smoothings.

    Requires >0.5 dex peak separation, >=10% mass on each side and a valley
    below 70% of the smaller peak, at both smoothing widths. Not a dip test,
    a general joint-mode count, or a validated astrophysical-mode classifier.
    """
    x = np.asarray(draws)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 100:
        raise ValueError("Require at least 100 finite scalar gravity draws")
    bounds = np.quantile(x, [0.001, 0.999])
    hist, edges = np.histogram(x, bins=100, range=bounds)
    centers = (edges[1:] + edges[:-1]) / 2
    diagnostics = []
    for width in [2.0, 3.0]:
        smooth = gaussian_filter1d(hist.astype(float), width)
        peaks, _ = find_peaks(smooth, prominence=0.1 * smooth.max(), distance=8)
        pairs = []
        for left, right in zip(peaks[:-1], peaks[1:]):
            valley = left + int(np.argmin(smooth[left : right + 1]))
            low = float((x < centers[valley]).mean())
            depth = float(smooth[valley] / min(smooth[left], smooth[right]))
            separation = float(centers[right] - centers[left])
            strong = separation > 0.5 and min(low, 1 - low) >= 0.1 and depth < 0.7
            pairs.append(
                dict(
                    peaks=centers[[left, right]].tolist(),
                    valley=float(centers[valley]),
                    lower_branch_mass=low,
                    upper_branch_mass=1 - low,
                    valley_to_smaller_peak=depth,
                    strong=bool(strong),
                )
            )
        diagnostics.append(
            dict(
                smoothing_bins=width,
                peak_locations=centers[peaks].tolist(),
                pairs=pairs,
            )
        )
    return dict(
        clear_separated_gravity_modes=all(
            any(p["strong"] for p in d["pairs"]) for d in diagnostics
        ),
        smoothing_checks=diagnostics,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    root = a.directory
    arrays = {}
    for family in ["gaussian", "mdn"]:
        with np.load(
            root / f"decision6_{family}_parallax_20261106_v1-whole-vector-corners.npz"
        ) as f:
            arrays[family] = {
                k: f[k] for k in ["source_id", "native_draws", "reference", "observed"]
            }
    np.testing.assert_array_equal(
        arrays["gaussian"]["source_id"], arrays["mdn"]["source_id"]
    )
    m = arrays["mdn"]
    if not m["observed"][:, :3].all():
        raise ValueError("Native atmospheric reference missing")
    result = dict(
        scope="All ten predeclared corner stars; metal-poor stress sample, not random Gaia or random development stars",
        all_native_reference_feh_below_minus2=bool((m["reference"][:, 2] < -2).all()),
        stars=[],
    )
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(5, 2, figsize=(12, 14))
    for i, (sid, ax) in enumerate(zip(m["source_id"], axes.flat)):
        mixture = m["native_draws"][:, i, 1]
        gaussian = arrays["gaussian"]["native_draws"][:, i, 1]
        record = dict(
            source_id=int(sid),
            reference_teff=float(m["reference"][i, 0]),
            reference_logg=float(m["reference"][i, 1]),
            reference_feh=float(m["reference"][i, 2]),
            mixture=gravity_modes(mixture),
            gaussian=gravity_modes(gaussian),
        )
        result["stars"].append(record)
        lo, hi = np.quantile(np.r_[mixture, gaussian], [0.001, 0.999])
        lo = min(lo, m["reference"][i, 1] - 0.15)
        hi = max(hi, m["reference"][i, 1] + 0.15)
        bins = np.linspace(lo, hi, 90)
        ax.hist(
            mixture,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.5,
            color="#247c9b",
            label="Mixture",
        )
        ax.hist(
            gaussian,
            bins=bins,
            density=True,
            histtype="step",
            color="#666666",
            linewidth=1.3,
            label="Gaussian control",
        )
        ax.axvline(m["reference"][i, 1], color="#cf6b36", label="Native reference")
        selected = int(sid) == 4185355646116022528
        suffix = " · previously shown" if selected else ""
        ax.set_title(
            f"{int(sid)}{suffix}\nNative [Fe/H] = {m['reference'][i, 2]:.2f}",
            fontsize=9,
        )
        if selected:
            for spine in ax.spines.values():
                spine.set_color("#cf6b36")
                spine.set_linewidth(2)
        ax.set(xlabel="log g [dex]", ylabel="Predictive density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        "All 10 predeclared metal-poor stress-test stars\n8,192 genuine predictive draws per star and model · not a representative population sample",
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.05, top=0.88, wspace=0.22, hspace=0.65
    )
    fig.savefig(root / "all-ten-stress-logg-gallery.png", dpi=150)
    plt.close(fig)
    result["clear_gravity_bimodality_count"] = sum(
        s["mixture"]["clear_separated_gravity_modes"] for s in result["stars"]
    )
    result["total_stress_stars"] = len(result["stars"])
    (root / "stress-bimodality-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
