"""Paired fixed-terminal DESI gravity ablation with verified identical batches."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def paired_mean_ci(values, seed=2026110771):
    values = np.asarray(values, dtype="f8")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite paired observations required")
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            values[rng.integers(len(values), size=len(values))].mean()
            for _ in range(4000)
        ]
    )
    return dict(
        mean=float(values.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist()
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(status="waiting_for_terminal_readouts", training_performed=False)

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    try:
        deadline = time.monotonic() + 24 * 3600
        dirs = [
            ROOT / f"matched_desi_logg_{arm}_terminal_qa_20261107_v1"
            for arm in ["on", "off"]
        ]
        while time.monotonic() < deadline:
            states = [
                json.loads((d / "manifest.json").read_text())
                if (d / "manifest.json").exists()
                else {}
                for d in dirs
            ]
            if any(s.get("status") == "failed" for s in states):
                raise RuntimeError("Terminal readout failed")
            if all(s.get("status") == "complete" for s in states):
                break
            time.sleep(30)
        else:
            raise TimeoutError("Terminal readouts unfinished")
        hashes = [s["batch_identity_sha256"] for s in states]
        if len(set(hashes)) != 1 or any(s["terminal_update"] != 4200 for s in states):
            raise ValueError("Source/star batches or terminal update differ")
        results = {}
        for role in ["tune", "cal", "devtest"]:
            arrays = []
            for d, s in zip(dirs, states, strict=True):
                name = f"terminal-{role}-APOKASC3_loggSeis.npz"
                path = d / name
                with open(path, "rb") as stream:
                    if (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        != s["array_sha256"][name]
                    ):
                        raise ValueError("Readout array changed")
                with np.load(path, allow_pickle=False) as f:
                    g = 1
                    arrays.append(
                        dict(
                            ids=f["source_id"],
                            reference=f["reference"][:, g] * f["scale"][g]
                            + f["center"][g],
                            prediction=f["intrinsic_median"][:, g] * f["scale"][g]
                            + f["center"][g],
                            crps=f["crps"][:, g] * f["scale"][g],
                            coverage=(
                                (f["reference"][:, g] >= f["quantiles"][1, :, g])
                                & (f["reference"][:, g] <= f["quantiles"][7, :, g])
                            ).astype(float),
                        )
                    )
            on, off = arrays
            np.testing.assert_array_equal(on["ids"], off["ids"])
            np.testing.assert_array_equal(on["reference"], off["reference"])
            eon = on["prediction"] - on["reference"]
            eoff = off["prediction"] - off["reference"]
            results[role] = dict(
                stars=len(eon),
                on=dict(
                    mae=float(abs(eon).mean()),
                    bias=float(eon.mean()),
                    coverage90=float(on["coverage"].mean()),
                ),
                off=dict(
                    mae=float(abs(eoff).mean()),
                    bias=float(eoff.mean()),
                    coverage90=float(off["coverage"].mean()),
                ),
                off_minus_on_mae=paired_mean_ci(abs(eoff) - abs(eon)),
                off_minus_on_crps=paired_mean_ci(off["crps"] - on["crps"]),
                off_minus_on_coverage90=paired_mean_ci(
                    off["coverage"] - on["coverage"]
                ),
            )
            np.savez_compressed(
                a.output / f"{role}-paired.npz",
                source_id=on["ids"],
                reference=on["reference"],
                prediction_on=on["prediction"],
                prediction_off=off["prediction"],
            )
        m.update(
            status="complete",
            batch_identity_sha256=hashes[0],
            terminal_update=4200,
            results=results,
            scope="One matched seed, frozen old encoder, historical old parallax availability, identical shared-edge TUNE metric. Star bootstrap only; no training-seed uncertainty. This isolates the effect of DESI logg supervision in this recipe, not whether DESI catalogue labels are intrinsically wrong. Negative off-minus-on MAE favors dropping gravity labels. Neither split nor model selection is changed by this report.",
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
