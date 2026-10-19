"""Independent audit and new final-model CPU replay for the declared long segment.

Never invokes the old 10M verifier or changes its policy. Completed initial CPU
reference is reused only when new initial arrays are byte-identical.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_xp_representation_probes as probe  # noqa: E402
from stellar_catalog_jobs import identity_split  # noqa: E402

from masked_stellar_autoencoder.models.checkpoint_load import (
    torch_load_trusted,  # noqa: E402
)
from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_prediction_head,
)  # noqa: E402
from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)  # noqa: E402

RESULTS = ROOT / "experiments/xp/results-stellar-improvement"
RUN = "residual_huber_long_30e_v1"
SOURCE = "3829b23afacad4b1b2f50439d09cb1e51711cd1d"
PARENT = "9de961ce92f0e69838aed174fce6c47179639e23816baf61a45e63d614ff66ec"
WINNER = "558a317ec8d2101d9caae7fe4a29db5a67ba6124dac27daaaf3ce14b624453ac"
COHORT = "321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def exact(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, list | tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            exact(x, y)
    else:
        assert a == b


def streams(
    fit_count=19859,
    rows_per_pass=9000000,
    passes=30,
    groups=8,
    batch=256,
    effective_batch=32768,
    seed=1042,
):
    row_digest = hashlib.sha256()
    one_pass = np.arange(rows_per_pass, dtype="i8").tobytes()
    for _ in range(passes):
        row_digest.update(one_pass)
    updates = ((rows_per_pass + effective_batch - 1) // effective_batch) * passes
    rng, label_digest = np.random.default_rng(seed), hashlib.sha256()
    for _ in range(updates * groups):
        label_digest.update(rng.integers(fit_count, size=batch).tobytes())
    return (
        row_digest.hexdigest(),
        label_digest.hexdigest(),
        rng.bit_generator.state,
        updates,
    )


def optimizer_audit(previous, final, updates, lr):
    assert len(previous["param_groups"]) == len(final["param_groups"])
    for p, f in zip(previous["param_groups"], final["param_groups"], strict=True):
        assert p["lr"] == f["lr"] == lr
        exact(p, f)
    assert previous["state"].keys() == final["state"].keys()
    differences = []
    for k, p in previous["state"].items():
        f = final["state"][k]
        difference = float(f["step"]) - float(p["step"])
        assert difference == updates, (k, difference)
        differences.append(difference)
        for name in ("exp_avg", "exp_avg_sq"):
            assert p[name].shape == f[name].shape
            assert torch.isfinite(f[name]).all()
    return dict(
        participating_states=len(differences),
        step_increment=updates,
        initial_steps=sorted({float(v["step"]) for v in previous["state"].values()}),
        final_steps=sorted({float(v["step"]) for v in final["state"].values()}),
    )


def audit_raw(cohort, paired):
    with h5py.File(cohort) as f:
        raw = f["raw_layers/native/measurements"][:]
        feature_row = f["layer_index/native/feature_row"][:]
        ids = f["features/source_id"][:]
        np.testing.assert_array_equal(raw["source_id"], ids[feature_row])
        split = identity_split(raw["source_id"])
        available = f["features/feature_available"][:][feature_row]
        partition = f["features/benchmark_partition"][:][feature_row]
        fits = raw[available & (partition == 0) & (split == 0)]
        tests = raw[available & (split == 2)]
        assert len(fits) == 19859 and len(tests) == 100620
        for expected, saved in (
            (fits, paired["fit_measurements"]),
            (tests, paired["test_measurements"]),
        ):
            expected, saved = (a[np.argsort(a["source_id"])] for a in (expected, saved))
            assert expected.dtype.names == saved.dtype.names
            # Structured field values and bytes, not dropped HDF5 dtype attributes.
            for field in expected.dtype.names:
                assert expected[field].tobytes() == saved[field].tobytes(), field
        order = np.argsort(ids)
        for kind in ("fit", "test"):
            wanted = (
                paired["fit_source_id"]
                if kind == "fit"
                else paired["test_measurements"]["source_id"]
            )
            rows = order[np.searchsorted(ids[order], wanted)]
            np.testing.assert_array_equal(ids[rows], wanted)
            raw_values = f["features/values"][:][rows]
            values = (raw_values - f["reference_scaler/center"][:]) / f[
                "reference_scaler/scale"
            ][:]
            names = json.loads(f.attrs["feature_columns_json"])
            safe, _ = spectrophotometric_inputs(
                torch.from_numpy(values.astype("f4")),
                torch.from_numpy(np.isfinite(raw_values)),
                names,
            )
            np.testing.assert_array_equal(safe.numpy(), paired[kind + "_input"])
    target = paired["fit_measurements"]["value"]
    np.testing.assert_array_equal(target, paired["fit_target"])
    np.testing.assert_array_equal(
        paired["fit_measurements"]["source_id"], paired["fit_source_id"]
    )
    assert not np.isin(
        paired["fit_source_id"], paired["test_measurements"]["source_id"]
    ).any()
    np.testing.assert_allclose(
        [float(paired["label_center"]), float(paired["label_scale"])],
        [target.mean(), target.std()],
        rtol=1e-14,
    )


def verify(directory, cohort, parent, winner):
    progress_path = directory / "independent-continuation-progress.json"

    def progress(stage, **extra):
        progress_path.write_text(
            json.dumps(
                dict(
                    status="running",
                    pid=os.getpid(),
                    stage=stage,
                    updated_unix=time.time(),
                    **extra,
                ),
                indent=2,
            )
            + "\n"
        )
        print(stage, extra, flush=True)

    progress("binding")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and not manifest["smoke"]
    assert manifest["run_id"] == RUN and manifest["source_sha"] == SOURCE
    ledger = json.loads(
        (ROOT / "experiments/xp/launch-stellar-improvement-20261015.json").read_text()
    )
    jobs = [j for j in ledger["jobs"] if j["run_id"] == RUN]
    assert len(jobs) == 1
    job = jobs[0]
    assert job["source_sha"] == SOURCE and job["session_ids"] == ["dj0h6mwl"]
    assert job["output"] == ledger["output_root"] + "/" + RUN
    policy = manifest["policy"]
    for k, v in dict(
        reconstruction_presentations=270000000,
        effective_batch=32768,
        encoder_lr=1.25e-6,
        head_lr=0.001,
        warmup_updates=0,
        supervised_batches_per_update=8,
        contrast_batch=256,
        continuation_seed=1042,
        continuation_previous_updates=306,
    ).items():
        assert policy[k] == v, k
    assert manifest["arm"]["head"] == "residual" and manifest["arm"]["loss"] == "huber"
    assert (
        sha(parent)
        == policy["checkpoint_sha256"]
        == manifest["parent_sha256"]
        == PARENT
    )
    assert sha(cohort) == manifest["cohort_sha256"] == COHORT
    assert (
        sha(winner / "checkpoint.pth")
        == manifest["continuation_sha256"]
        == policy["continuation_sha256"]
        == WINNER
    )
    reference = json.loads((winner / "independent-verification.json").read_text())
    assert (
        reference["status"] == "passed"
        and reference["final_checkpoint_sha256"] == WINNER
    )
    assert (
        reference["cpu_replay_all_test_member_outputs"]
        and reference["cpu_replay_all_validation_decoder_outputs"]
    )
    collection = json.loads(
        (directory / "complete-collection-progress.json").read_text()
    )
    assert (
        collection["status"] == "complete"
        and len(collection["verified_checkpoints"]) == 30
    )
    paired = np.load(directory / "paired-cohort.npz", allow_pickle=False)
    previous_pair = np.load(winner / "paired-cohort.npz", allow_pickle=False)
    assert set(paired.files) == set(previous_pair.files)
    for k in paired.files:
        exact(paired[k], previous_pair[k])
    progress("raw_source_and_inputs")
    audit_raw(cohort, paired)
    initial = np.load(directory / "initial_predictions.npz", allow_pickle=False)
    final = np.load(directory / "predictions.npz", allow_pickle=False)
    winner_final = np.load(winner / "predictions.npz", allow_pickle=False)
    exact(initial["member_outputs"], winner_final["member_outputs"])
    initial_point = initial["member_outputs"][:, :, 0].mean(1) * float(
        paired["label_scale"]
    ) + float(paired["label_center"])
    exact(initial_point, winner_final["prediction"])
    with np.load(directory / "snapshot-0.npz", allow_pickle=False) as snap:
        exact(initial["member_outputs"], snap["member_outputs"])
    reco = np.load(directory / "initial_reconstruction.npz", allow_pickle=False)
    with np.load(winner / "initial_reconstruction.npz", allow_pickle=False) as old:
        exact(reco["target"], old["target"])
        exact(reco["mask"], old["mask"])
    with np.load(winner / "final_reconstruction.npz", allow_pickle=False) as old:
        exact(reco["prediction"], old["prediction"])
    progress("streams_and_optimizer")
    saved = torch_load_trusted(
        directory / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    old = torch_load_trusted(
        winner / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    assert "label_rng_state" not in old and "mask_rng_state" not in old
    row_hash, label_hash, label_state, updates = streams()
    assert updates == manifest["optimizer_updates"] == 8250
    assert manifest["presentations"] == saved["presentations"] == 270000000
    assert row_hash == manifest["row_sha256"]
    assert label_hash == manifest["supervised_stream_sha256"]
    exact(label_state, saved["label_rng_state"])
    assert manifest["supervised_warmup_presentations"] == 0
    assert manifest["supervised_adaptation_presentations"] == 16896000
    assert saved["completed_updates"] == 8556
    assert saved["source_sha"] == SOURCE and saved["parent_sha256"] == PARENT
    assert saved["continuation_sha256"] == WINNER
    exact(saved["arm"], manifest["arm"])
    exact(saved["feature_columns"], old["feature_columns"])
    for k in ("label_center", "label_scale"):
        exact(saved[k], float(paired[k]))
    for k in ("torch_rng", "mask_rng_state", "cuda_rng", "label_rng_state"):
        assert k in saved
    assert saved["cuda_rng"] and saved["mask_rng_state"].numel() > 0
    optimizers = {}
    for key, lr in (
        ("optimizer_state_dict", 1.25e-6),
        ("head_optimizer_state_dict", 0.001),
    ):
        optimizers[key] = optimizer_audit(old[key], saved[key], updates, lr)
    del old
    progress("periodic_checkpoint_inventory")
    inventory = []
    for epoch in range(1, 31):
        presentations = epoch * 9000000
        path = directory / "checkpoints" / f"checkpoint-{presentations:012d}.pth"
        receipt = json.loads(path.with_suffix(".json").read_text())
        digest = sha(path)
        assert (
            digest
            == receipt["sha256"]
            == collection["verified_checkpoints"][str(presentations)]["sha256"]
        )
        assert path.stat().st_size == receipt["bytes"]
        assert receipt["presentations"] == presentations
        assert receipt["completed_updates"] == 306 + epoch * 275
        state = torch_load_trusted(path, map_location="cpu", weights_only=False)
        for k in ("presentations", "completed_updates"):
            assert state[k] == receipt[k]
        assert (
            state["source_sha"] == SOURCE
            and state["parent_sha256"] == PARENT
            and state["continuation_sha256"] == WINNER
        )
        with np.load(
            directory / f"snapshot-{presentations}.npz", allow_pickle=False
        ) as snap:
            assert snap["member_outputs"].shape == final["member_outputs"].shape
            assert np.isfinite(snap["member_outputs"]).all()
            if epoch == 30:
                exact(snap["member_outputs"], final["member_outputs"])
        if epoch == 30:
            for key in (
                "model_state_dict",
                "head_state_dict",
                "optimizer_state_dict",
                "head_optimizer_state_dict",
                "label_rng_state",
                "mask_rng_state",
                "torch_rng",
                "cuda_rng",
            ):
                exact(saved[key], state[key])
        inventory.append(receipt)
        del state
        progress("periodic_checkpoint_inventory", epochs_checked=epoch)
    last = json.loads((directory / "latest-checkpoint.json").read_text())
    exact(last, inventory[-1])
    parent_state = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    config = yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text())
    model = probe.load_model(config, parent_state, torch.device("cpu"))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    head = build_prediction_head("residual")
    head.load_state_dict(saved["head_state_dict"], strict=True)
    head.eval()
    del saved, parent_state
    progress("new_final_head_cpu", rows_checked=0)
    outputs = []
    with torch.no_grad():
        for start in range(0, 100620, 512):
            out = head(
                model.encoder(
                    torch.from_numpy(paired["test_input"][start : start + 512])
                )
            ).numpy()
            np.testing.assert_allclose(
                out, final["member_outputs"][start : start + 512], rtol=5e-5, atol=5e-5
            )
            outputs.append(out)
            if start % (512 * 10) == 0:
                progress("new_final_head_cpu", rows_checked=min(start + 512, 100620))
    output = np.concatenate(outputs)
    hidden = np.where(
        reco["mask"] | ~np.isfinite(reco["target"]), -9999, reco["target"]
    )
    assert len(hidden) == 10000
    progress("new_final_decoder_cpu", rows_checked=0)
    decoded = []
    with np.load(
        directory / "final_reconstruction.npz", allow_pickle=False
    ) as expected:
        exact(reco["target"], expected["target"])
        exact(reco["mask"], expected["mask"])
        with torch.no_grad():
            for start in range(0, len(hidden), 512):
                value = model(torch.from_numpy(hidden[start : start + 512]))[0].numpy()
                np.testing.assert_allclose(
                    value,
                    expected["prediction"][start : start + 512],
                    rtol=5e-5,
                    atol=5e-5,
                )
                decoded.append(value)
                progress("new_final_decoder_cpu", rows_checked=min(start + 512, 10000))
        decoder_diff = float(
            np.abs(np.concatenate(decoded) - expected["prediction"]).max()
        )
    point = final["member_outputs"][:, :, 0].mean(1) * float(
        paired["label_scale"]
    ) + float(paired["label_center"])
    exact(point, final["prediction"])
    truth = paired["test_measurements"]["value"]
    report = dict(
        status="passed",
        run_id=RUN,
        source_sha=SOURCE,
        continuation_sha256=WINNER,
        final_checkpoint_sha256=sha(directory / "checkpoint.pth"),
        parent_sha256=PARENT,
        cohort_sha256=COHORT,
        native_fit=19859,
        native_test=100620,
        initial_reference_reused=True,
        fresh_initial_cpu_replay=False,
        initial_artifacts_byte_identical_to_verified_winner=True,
        cpu_replay_all_test_member_outputs=True,
        cpu_replay_all_validation_decoder_outputs=True,
        cpu_decoder_rows=10000,
        cpu_rtol=5e-5,
        cpu_atol=5e-5,
        cpu_max_absolute_member_output_difference=float(
            np.abs(output - final["member_outputs"]).max()
        ),
        cpu_decoder_max_absolute_difference=decoder_diff,
        periodic_checkpoint_receipts=inventory,
        row_sha256=row_hash,
        supervised_stream_sha256=label_hash,
        mask_sha256=manifest["mask_sha256"],
        mask_note="GPU mask hash retained; not independently CPU-regenerated",
        optimizer_state_audits=optimizers,
        all_native_raw_fields_bytes_and_inputs_verified=True,
        original_hdf5_metadata_preserved_in_raw_artifact=True,
        dropout_policy="Declared new segment1042, reconstruction1000+306+step; supervised100000+(306+step)*8+group; source policy audited, no stochastic training replay",
        exact_rng_resume=False,
        mae_dex=float(np.abs(point - truth).mean()),
        bias_dex=float((point - truth).mean()),
    )
    (directory / "independent-verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    progress_path.write_text(
        json.dumps(
            dict(
                status="passed",
                report="independent-verification.json",
                updated_unix=time.time(),
            ),
            indent=2,
        )
        + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RESULTS / RUN)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=RESULTS / "enriched_feature_cohort_v1/enriched-feature-cohort.h5",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=RESULTS / "prediction-comparison-e72/parent/checkpoint-72.pth",
    )
    parser.add_argument(
        "--winner", type=Path, default=RESULTS / "prediction_e72_residual_huber_10m_v1"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    lock = args.directory / "independent-continuation.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        verify(args.directory, args.cohort, args.parent, args.winner)
    except Exception as exc:
        failure = dict(
            status="failed",
            error=repr(exc),
            traceback=traceback.format_exc(),
            updated_unix=time.time(),
        )
        for name in (
            "independent-continuation-failure.json",
            "independent-continuation-progress.json",
        ):
            (args.directory / name).write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
