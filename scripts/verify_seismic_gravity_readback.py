"""Verify native versus independent VizieR seismic references and plot saved QA."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io.votable import parse_single_table


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    root = a.directory
    manifest = json.loads(
        (root / "seismic_gravity_qa_20261106_v1-latest.json").read_text()
    )
    assert (
        sha(root / "apokasc3_vizier.xml")
        == manifest["sources"]["apokasc3_vizier"]["sha256"]
    )
    assert (
        sha(root / "apokasc3-gravity-qa-reference.parquet")
        == manifest["sources"]["apokasc3"]["reference_sha256"]
    )
    ref = pd.read_parquet(root / "apokasc3-gravity-qa-reference.parquet")
    table = parse_single_table(root / "apokasc3_vizier.xml").to_table()
    assert table["GaiaDR3"].dtype.kind in "iu"
    index = {
        int(s): i for i, s in enumerate(table["GaiaDR3"]) if not np.ma.is_masked(s)
    }
    rows = np.array([index[int(s)] for s in ref.source_id])
    for key in ["loggSeis", "e_loggSeis", "loggSpec", "Teff", "Numax"]:
        np.testing.assert_allclose(
            ref[key], np.asarray(table[key][rows]), rtol=8e-8, atol=1e-9
        )
    with np.load(root / "devtest-paired.npz") as f:
        ids = f["source_id"]
        assert ids.dtype.kind in "iu" and len(np.unique(ids)) == len(ids)
        positions = np.searchsorted(ref.source_id, ids)
        np.testing.assert_array_equal(ref.source_id.to_numpy()[positions], ids)
        y = f["logg_seismic"]
        np.testing.assert_allclose(
            y, ref.loggSeis.to_numpy()[positions], rtol=0, atol=0
        )
        common = (f["weights"] * f["component_mean"]).sum(1)
        native = f["native_logg"][:, 0]
        assert f["native_observed"][:, 0].all()
        meanbias = float(np.median(common - y))
        np.testing.assert_allclose(
            meanbias,
            manifest["roles"]["devtest"]["models"]["common"]["all"]["point"]["bias"],
        )
        plt.rcParams.update({"font.size": 11})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
        for prediction, label, color in [
            (common, "MSA common reference", "#d05d35"),
            (native, "Native APOGEE label", "#247c9b"),
        ]:
            axes[0].scatter(
                y, prediction - y, s=9, alpha=0.35, label=label, color=color
            )
            axes[1].hist(
                prediction - y,
                bins=np.linspace(-0.5, 0.7, 41),
                histtype="step",
                linewidth=1.8,
                label=label,
                color=color,
            )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set(
            xlabel="APOKASC-3 seismic log g [dex]",
            ylabel="Estimate − seismic log g [dex]",
            ylim=(-0.5, 0.7),
        )
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set(xlabel="Estimate − seismic log g [dex]", ylabel="Stars")
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle(
            f"{len(ids):,} exact-ID devtest stars · frozen three-survey model\nFine-tuning holdout; parent and survey seismic-calibration dependencies remain",
            fontsize=12,
        )
        fig.savefig(root / "apokasc-gravity-diagnosis.png", dpi=160)
        plt.close(fig)
    proof = dict(
        status="passed",
        native_vs_vizier_exact_id_and_five_columns=len(ref),
        saved_devtest_ids_and_seismic_values_verified=len(ids),
        common_bias_recomputed=meanbias,
        plot_sha256=sha(root / "apokasc-gravity-diagnosis.png"),
    )
    (root / "independent-readback-verification.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
