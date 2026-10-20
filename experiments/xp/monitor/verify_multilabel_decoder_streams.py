"""Independent optimizer/counter/label-stream and saved-array decoder gate.

The original benchmark validation source and CUDA training mask draws remain
explicitly pending. No training or production reconstruction helper is used.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import run_xp_representation_probes as architecture

BASE = ROOT / "experiments/xp/results-stellar-improvement"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def counters(lengths=(9000000, 1000000), batch=32768):
    seen = 0
    result = []
    for length in lengths:
        for start in range(0, length, batch):
            seen += min(batch, length - start)
            result.append(seen)
    return result


def reference_mask(shape, features):
    generator = np.random.default_rng(42)
    photo = generator.random(shape) < 0.2
    xp = [i for i, name in enumerate(features) if name.startswith(("bp_", "rp_"))]
    photo[:, xp] = False
    selected = generator.permutation(shape[0])[: shape[0] // 2]
    photo[np.ix_(selected, xp)] = True
    return photo


def verify(run):
    directory = BASE / run
    point = directory / "independent-point-source-cp.json"
    proof = json.loads(point.read_text())
    assert proof["status"] == "point_source_cp_passed_remaining_gates_pending"
    for name in ["manifest.json", "checkpoint.pth", "paired-cohort.npz"]:
        assert digest(directory / name) == proof["bindings"][name]
    m = json.loads((directory / "manifest.json").read_text())
    assert m["status"] == "complete" and not m["smoke"]
    bindings = {
        name: digest(directory / name)
        for name in [
            "manifest.json",
            "checkpoint.pth",
            "paired-cohort.npz",
            "initial_reconstruction.npz",
            "final_reconstruction.npz",
            "metrics.jsonl",
            "independent-point-source-cp.json",
        ]
    }
    bindings["verifier"] = digest(__file__)
    output = directory / "independent-decoder-stream-report.json"
    if output.exists():
        report = json.loads(output.read_text())
        assert (
            report["bindings"] == bindings
            and report["status"]
            == "decoder_arrays_streams_passed_raw_benchmark_pending"
        )
        return report
    progress = directory / "independent-decoder-stream-progress.json"
    atomic(progress, {"phase": "counters_optimizer", "bindings": bindings})
    saved = torch.load(
        directory / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    paired = np.load(directory / "paired-cohort.npz")
    fit = paired["fit_rows"]
    metrics = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    expected = counters()
    assert len(expected) == m["optimizer_updates"] == saved["completed_updates"] == 306
    assert m["presentations"] == saved["completed_presentations"] == 10000000
    np.testing.assert_array_equal(expected, [x["presentations"] for x in metrics])
    np.testing.assert_array_equal(np.arange(1, 307), [x["updates"] for x in metrics])
    row_digest = hashlib.sha256()
    for length in [9000000, 1000000]:
        row_digest.update(np.arange(length, dtype="i8").tobytes())
    assert row_digest.hexdigest() == m["row_sha256"]
    rng = np.random.default_rng(int(m["arm"]["seed"]))
    labels = hashlib.sha256()
    for _ in range(500 + 306 * 8):
        labels.update(
            rng.choice(fit, size=min(256, len(fit)), replace=len(fit) < 256)
            .astype("i8")
            .tobytes()
        )
    assert labels.hexdigest() == m["supervised_stream_sha256"]
    for key, step in [
        ("model_optimizer_state_dict", 306),
        ("head_optimizer_state_dict", 806),
    ]:
        optimizer = saved[key]
        assert optimizer["state"]
        assert {int(s["step"]) for s in optimizer["state"].values()} == {step}
        assert all(
            tuple(g["betas"]) == (0.9, 0.999) and g["eps"] == 1e-8
            for g in optimizer["param_groups"]
        )
    weight = float(m["reconstruction_weight"])
    assert m["reconstruction_optimizer_updates"] == (306 if weight else 0)
    assert m["reconstruction_training_presentations"] == (10000000 if weight else 0)
    parent = BASE / "main-checkpoints/checkpoint-86.pth"
    assert digest(parent) == m["parent_sha256"]
    disk = torch.load(parent, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text())
    model = architecture.load_model(cfg, disk, torch.device("cpu"))
    model.eval()
    names = [name for name, _ in model.named_parameters()]
    ids = saved["model_optimizer_state_dict"]["param_groups"][0]["params"]
    assert len(ids) == len(names)
    decoder_ids = [
        i
        for name, i in zip(names, ids, strict=True)
        if name.startswith(("decoder.", "reconstruction_layer."))
    ]
    assert decoder_ids
    if weight == 0:
        assert all(
            i not in saved["model_optimizer_state_dict"]["state"] for i in decoder_ids
        )
    else:
        assert all(
            i in saved["model_optimizer_state_dict"]["state"] for i in decoder_ids
        )
    initial = np.load(directory / "initial_reconstruction.npz")
    final = np.load(directory / "final_reconstruction.npz")
    target = initial["target"]
    mask = initial["mask"]
    features = disk["run_signature"]["feature_cols"]
    assert len(target) == 10000
    np.testing.assert_array_equal(mask, reference_mask(target.shape, features))
    hidden = np.where(mask | ~np.isfinite(target), np.float32(-9999), target)
    maxima = {}
    for role, expected_array in [
        ("initial", initial["prediction"]),
        ("final", final["prediction"]),
    ]:
        if role == "final":
            model.load_state_dict(saved["model_state_dict"], strict=True)
        pieces = []
        with torch.no_grad():
            for start in range(0, len(hidden), 256):
                prediction = model(torch.from_numpy(hidden[start : start + 256]))[0]
                pieces.append(prediction.numpy())
                if start % 2048 == 0:
                    atomic(
                        progress,
                        {
                            "phase": "decoder",
                            "role": role,
                            "rows_done": min(start + 256, len(hidden)),
                            "rows_total": len(hidden),
                            "bindings": bindings,
                        },
                    )
        result = np.concatenate(pieces)
        np.testing.assert_allclose(result, expected_array, rtol=5e-5, atol=5e-5)
        maxima[role] = float(np.max(np.abs(result - expected_array)))
    report = {
        "status": "decoder_arrays_streams_passed_raw_benchmark_pending",
        "bindings": bindings,
        "decoder_rows_each": 10000,
        "decoder_max_errors": maxima,
        "row_stream_verified": True,
        "supervised_label_stream_verified": True,
        "fresh_adam_steps_verified": True,
        "validation_mask_independently_regenerated": True,
        "rtol": 5e-5,
        "atol": 5e-5,
        "pending": [
            "original_benchmark_validation_source",
            "CUDA_training_mask_draws",
            "TTT_gradients_reset_calibration",
            "library_weighted_cp",
            "ensemble_and_paired_visual_reports",
        ],
    }
    atomic(output, report)
    atomic(progress, {"phase": "decoder_stream_complete", "report": str(output)})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.self_test:
        assert counters((5, 3), 4) == [4, 5, 8]
        assert counters()[274:277] == [9000000, 9032768, 9065536]
        mask = reference_mask((10, 3), ["g", "bp_1", "rp_1"])
        assert np.sum(mask[:, 1:].all(1)) == 5
        assert np.array_equal(mask[:, 1], mask[:, 2])
        print("independent counters and deterministic-mask invariants PASS")
    else:
        if not args.run:
            parser.error("--run required")
        try:
            print(json.dumps(verify(args.run), indent=2), flush=True)
        except Exception as exc:
            atomic(
                BASE / args.run / "independent-decoder-stream-failure.json",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
