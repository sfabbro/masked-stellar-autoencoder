"""Unattended descriptive mode QA and random single-star corners from saved draws."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_corner_bimodality import gravity_modes
from plot_decision_six_corners import corner

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
UPSTREAM = ROOT / "representative_corner_draws_20261106_v1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    failures = {}
    deadline = time.monotonic() + 24 * 3600

    def save(status="waiting_or_scoring"):
        t = a.output / "manifest.tmp"
        t.write_text(
            json.dumps(
                dict(
                    status=status,
                    results=results,
                    failures=failures,
                    training_performed=False,
                    model_loaded=False,
                    scope="Descriptive sample-mode diagnostic on a uniform eligible-development panel; not all-Gaia prevalence, a general joint-mode count or physical branch validation.",
                ),
                indent=2,
            )
            + "\n"
        )
        t.replace(a.output / "manifest.json")

    save()
    while time.monotonic() < deadline:
        path = UPSTREAM / "manifest.json"
        if not path.exists():
            time.sleep(45)
            continue
        try:
            m = json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(45)
            continue
        for rid, state in m.get("results", {}).items():
            if rid in results or rid in failures:
                continue
            if state.get("status") != "complete":
                failures[rid] = state
                save()
                continue
            source = UPSTREAM / (rid + "-representative-draws.npz")
            with source.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != state["artifact_sha256"]
                ):
                    raise ValueError("Native draw SHA differs")
            with np.load(source) as f:
                draws = f["native_draws"]
                ids = f["source_id"]
                random = f["random_panel"].astype(bool)
                stress = f["stress_panel"].astype(bool)
                ref = f["reference"]
                observed = f["observed"].astype(bool)
            flags = np.array(
                [
                    gravity_modes(draws[:, i, 1])["clear_separated_gravity_modes"]
                    for i in range(len(ids))
                ]
            )
            selected = np.random.default_rng(2026110643).choice(
                np.flatnonzero(random), size=12, replace=False
            )
            plots = []
            # These exact same random identities are selected independently of mode flags for every arm.
            for i in selected[:2]:
                out = a.output / (rid + "-" + str(int(ids[i])) + "-corner.png")
                reference = np.where(
                    observed[i], ref[i], np.array([5000.0, 3.0, 0.0, 0.0, 1.0, 1.0])
                )
                corner(
                    draws[:, i],
                    f"{rid}\nGaia DR3 {int(ids[i])} · {len(draws):,} genuine six-vector draws · model-blind random development example",
                    out,
                    reference=reference,
                    observed=observed[i],
                )
                plots.append(out.name)
            fig, axes = plt.subplots(4, 3, figsize=(12, 10), layout="constrained")
            for ax, i in zip(axes.flat, selected):
                ax.hist(
                    draws[:, i, 1],
                    bins=55,
                    histtype="stepfilled",
                    alpha=0.6,
                    color="#247c9b",
                    density=True,
                )
                if observed[i, 1]:
                    ax.axvline(ref[i, 1], color="#cf6b36")
                ax.set_title(str(int(ids[i])), fontsize=9)
                ax.set(xlabel="log g [dex]", ylabel="Predictive density")
            fig.suptitle(
                rid
                + "\n12 model-blind random development stars · orange: available native reference",
                fontsize=11,
            )
            gallery = a.output / (rid + "-random-logg-gallery.png")
            fig.savefig(gallery, dpi=140)
            plt.close(fig)
            result = dict(
                status="complete",
                random_stars=int(random.sum()),
                random_clear_separated_logg_modes=int(flags[random].sum()),
                random_mode_fraction=float(flags[random].mean()),
                stress_stars=int(stress.sum()),
                stress_clear_separated_logg_modes=int(flags[stress].sum()),
                artifact_sha256=state["artifact_sha256"],
                panel_ids_sha256=state["panel_ids_sha256"],
                gallery=gallery.name,
                random_single_star_corners=plots,
            )
            result["by_native_reference_feh"] = {}
            for name, mask in [
                ("below_minus2", observed[:, 2] & (ref[:, 2] < -2)),
                ("at_least_minus2", observed[:, 2] & (ref[:, 2] >= -2)),
                ("missing_reference", ~observed[:, 2]),
            ]:
                keep = random & mask
                result["by_native_reference_feh"][name] = dict(
                    stars=int(keep.sum()), flagged=int(flags[keep].sum())
                )
            np.savez_compressed(
                a.output / (rid + "-mode-flags.npz"),
                source_id=ids,
                random_panel=random,
                stress_panel=stress,
                clear_separated_logg_modes=flags,
            )
            results[rid] = result
            save()
        if m.get("status") == "complete":
            save("complete")
            return
        time.sleep(45)
    raise TimeoutError("Upstream draw wait expired; partial QA preserved")


if __name__ == "__main__":
    main()
