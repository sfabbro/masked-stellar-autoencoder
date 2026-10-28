"""Independent per-object native MSA reconstruction TTT and recalibration.

No production refinement, mask-selection, optimizer or CP helper is called.
Only source-verified point-head checkpoints are accepted by this gate.
"""

import argparse
import hashlib
import json

import numpy as np
import torch
import yaml
from verify_multilabel_points import (
    BASE,
    ROOT,
    architecture,
    atomic,
    digest,
    order_threshold,
    reference_head,
)


def choose(ids, count, seed):
    keys = [
        int.from_bytes(
            hashlib.sha256((str(int(s)) + ":" + str(seed)).encode()).digest()[:8],
            "little",
        )
        for s in ids
    ]
    return np.argsort(np.asarray(keys, dtype="u8"), kind="stable")[
        : min(count, len(ids))
    ]


def heldout(visible, xp, source):
    valid = np.asarray([int(i) for i in xp if visible[int(i)]], dtype="i8")
    if len(valid) < 2:
        raise ValueError("Two genuine visible XP coefficients required")
    key = hashlib.sha256((str(int(source)) + ":1042:holdout").encode()).digest()
    generator = np.random.default_rng(int.from_bytes(key[:8], "little"))
    columns = valid[generator.random(len(valid)) < 0.2]
    if not len(columns):
        columns = valid[[int(generator.integers(len(valid)))]]
    if len(columns) == len(valid):
        columns = columns[:-1]
    assert 0 < len(columns) < len(valid)
    return columns


def adam_step(parameters, gradients, first, second, step, lr=1e-4):
    with torch.no_grad():
        for p, g, m, v in zip(parameters, gradients, first, second, strict=True):
            if not torch.isfinite(g).all():
                raise FloatingPointError("Nonfinite independent TTT gradient")
            m.mul_(0.9).add_(g, alpha=0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            corrected_m = m / (1 - 0.9**step)
            corrected_v = v / (1 - 0.999**step)
            p.addcdiv_(corrected_m, corrected_v.sqrt() + 1e-8, value=-lr)


def refine(model, mean_function, clean, visible, xp, source, steps=3):
    # Labels and quoted errors do not enter this function or its objective.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    selected = [
        p
        for module in model.encoder.modules()
        if isinstance(module, torch.nn.LayerNorm)
        for p in module.parameters(recurse=False)
    ]
    if not selected:
        raise ValueError("No encoder LayerNorm affine parameters")
    original = [p.detach().clone() for p in selected]
    columns = heldout(visible[0], xp, source)
    hidden = clean.clone()
    hidden[:, columns] = -9999
    try:
        for p in selected:
            p.requires_grad_(True)
        first = [torch.zeros_like(p) for p in selected]
        second = [torch.zeros_like(p) for p in selected]
        with torch.no_grad():
            anchor = mean_function(model.encoder(clean))
        gradient_norms = []
        for step in range(1, steps + 1):
            latent = model.encoder(hidden)
            reconstruction = model.reconstruction_layer(model.decoder(latent))
            if getattr(model, "heteroscedastic", False):
                reconstruction = reconstruction.chunk(2, -1)[0]
            loss = (reconstruction[:, columns] - clean[:, columns]).abs().mean()
            distance = torch.stack(
                [
                    (p - v).square().mean()
                    for p, v in zip(selected, original, strict=True)
                ]
            ).mean()
            prediction = mean_function(model.encoder(clean))
            objective = (
                loss + 0.1 * distance + 0.1 * (prediction - anchor).square().mean()
            )
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite independent TTT objective")
            gradients = torch.autograd.grad(objective, selected)
            gradient_norms.append(
                float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
            )
            adam_step(selected, gradients, first, second, step)
        with torch.no_grad():
            result = mean_function(model.encoder(clean)).detach().cpu().numpy()[0]
        return result, {
            "heldout_columns": columns.tolist(),
            "final_objective": float(objective.detach()),
            "gradient_norms": gradient_norms,
            "steps": steps,
        }
    finally:
        with torch.no_grad():
            for p, v in zip(selected, original, strict=True):
                p.copy_(v)
                assert torch.equal(p, v)
        for p in selected:
            p.requires_grad_(False)


def self_test():
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    q = p.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([q], lr=1e-4)
    first = [torch.zeros_like(p)]
    second = [torch.zeros_like(p)]
    for step, g in enumerate(
        [
            torch.tensor([2.0, -4.0]),
            torch.tensor([-0.5, 1.5]),
            torch.tensor([0.0, 1.0]),
        ],
        1,
    ):
        q.grad = g.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        adam_step([p], [g], first, second, step)
        torch.testing.assert_close(p, q, rtol=1e-7, atol=1e-7)
    visible = np.ones(4, dtype=bool)
    assert np.array_equal(
        heldout(visible, [0, 1, 2], 123), heldout(visible, [0, 1, 2], 123)
    )

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(4, 4), torch.nn.LayerNorm(4)
            )
            self.decoder = torch.nn.Linear(4, 4)
            self.reconstruction_layer = torch.nn.Identity()

    torch.manual_seed(6)
    model = Toy()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.tensor([[0.1, 0.4, -0.2, 0.7]])
    left, _ = refine(
        model, lambda z: z[:, :3], x, np.ones((1, 4), bool), [0, 1, 2], 123
    )
    right, _ = refine(
        model, lambda z: z[:, :3], x, np.ones((1, 4), bool), [0, 1, 2], 123
    )
    np.testing.assert_array_equal(left, right)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def verify(run):
    directory = BASE / run
    proof_path = directory / "independent-point-source-cp.json"
    proof = json.loads(proof_path.read_text())
    assert proof["status"] == "point_source_cp_passed_remaining_gates_pending"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["arm"]["kind"] in [
        "gaussian",
        "huber",
    ]
    files = [
        "checkpoint.pth",
        "paired-cohort.npz",
        "manifest.json",
        "ttt-predictions.npz",
        "ttt-diagnostics.json",
        "ttt-conformal.npz",
        "ttt-matched-clean-conformal.npz",
        "independent-point-source-cp.json",
        "independent-point-predictions.npz",
    ]
    bindings = {name: digest(directory / name) for name in files}
    bindings["verifier"] = digest(__file__)
    for name in ["manifest.json", "checkpoint.pth", "paired-cohort.npz"]:
        assert bindings[name] == proof["bindings"][name]
    token = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    cache = directory / "independent-ttt-cache" / token
    cache.mkdir(parents=True, exist_ok=True)
    report_path = directory / "independent-ttt-report.json"
    if report_path.exists():
        result = json.loads(report_path.read_text())
        assert (
            result["bindings"] == bindings
            and result["status"] == "ttt_replay_and_cp_passed"
        )
        return result
    progress = directory / "independent-ttt-progress.json"
    atomic(progress, {"phase": "input_selection", "bindings": bindings})
    paired = np.load(directory / "paired-cohort.npz")
    actual = np.load(directory / "ttt-predictions.npz")
    diagnostics = json.loads((directory / "ttt-diagnostics.json").read_text())
    saved = torch.load(
        directory / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    parent = BASE / "main-checkpoints/checkpoint-86.pth"
    assert digest(parent) == manifest["parent_sha256"]
    disk = torch.load(parent, map_location="cpu", weights_only=False)
    features = disk["run_signature"]["feature_cols"]
    xp = np.asarray(
        [
            i
            for i, n in enumerate(features)
            if n.startswith(("bp_", "rp_")) and n.rsplit("_", 1)[-1].isdigit()
        ]
    )
    assert len(xp) == 110
    model = architecture.load_model(
        yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text()),
        disk,
        torch.device("cpu"),
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    mean_function = lambda z: reference_head(
        saved["head_state_dict"], z, manifest["arm"]["kind"]
    )[0]
    outputs = {}
    maximum = {}
    selected = {}
    for role, count, seed in [("test", 512, 1042), ("calibration", 256, 1043)]:
        rows = paired[role + "_rows"]
        eligible = rows[paired["visible"][rows][:, xp].sum(1) >= 2]
        rows = eligible[choose(paired["source_id"][eligible], count, seed)]
        selected[role] = rows
        np.testing.assert_array_equal(rows, actual[role + "_rows"])
        np.testing.assert_array_equal(
            paired["source_id"][rows], actual[role + "_source_id"]
        )
        values = []
        errors = []
        for i, row in enumerate(rows):
            path = cache / (
                role + "-" + str(i) + "-" + str(int(paired["source_id"][row])) + ".npz"
            )
            if path.exists():
                entry = np.load(path)
                prediction = entry["prediction"]
            else:
                prediction, info = refine(
                    model,
                    mean_function,
                    torch.from_numpy(paired["input"][row : row + 1]),
                    paired["visible"][row : row + 1],
                    xp,
                    int(paired["source_id"][row]),
                )
                np.testing.assert_allclose(
                    prediction,
                    actual[role + "_standardized_prediction"][i],
                    rtol=5e-5,
                    atol=5e-5,
                )
                original = diagnostics["diagnostics"][role][i]
                assert (
                    original["heldout_count"] == len(info["heldout_columns"])
                    and original["steps"] == info["steps"] == 3
                )
                np.testing.assert_allclose(
                    info["final_objective"],
                    original["final_objective"],
                    rtol=5e-5,
                    atol=5e-5,
                )
                np.savez_compressed(
                    path,
                    prediction=prediction,
                    heldout_columns=np.asarray(info["heldout_columns"]),
                    gradient_norms=np.asarray(info["gradient_norms"]),
                )
            values.append(prediction)
            errors.append(
                float(
                    np.max(
                        np.abs(
                            prediction - actual[role + "_standardized_prediction"][i]
                        )
                    )
                )
            )
            atomic(
                progress,
                {
                    "phase": "object_replay",
                    "role": role,
                    "objects_done": i + 1,
                    "objects_total": len(rows),
                    "bindings": bindings,
                },
            )
        outputs[role] = np.stack(values)
        maximum[role] = max(errors)
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved["model_state_dict"][name]), name
    target = (
        (paired["value"][selected["calibration"]] - paired["label_center"])
        / paired["label_scale"]
    ).astype("f4")
    full = np.load(directory / "predictions.npz")
    cpu = np.load(directory / "independent-point-predictions.npz")
    for prefix, calibration_prediction, test_prediction, cpu_cal in [
        (
            "ttt-",
            actual["calibration_standardized_prediction"],
            actual["test_standardized_prediction"],
            outputs["calibration"],
        ),
        ("ttt-matched-clean-", None, None, None),
    ]:
        cp = np.load(directory / (prefix + "conformal.npz"))
        np.testing.assert_array_equal(
            cp["test_source_id"], paired["source_id"][selected["test"]]
        )
        np.testing.assert_array_equal(
            cp["calibration_source_id"], paired["source_id"][selected["calibration"]]
        )
        if calibration_prediction is None:
            indices = {int(row): i for i, row in enumerate(full["calibration_rows"])}
            positions = [indices[int(row)] for row in selected["calibration"]]
            calibration_prediction = full["calibration_mean"][positions]
            cpu_cal = cpu["calibration_mean"][positions]
            indices = {int(row): i for i, row in enumerate(full["test_rows"])}
            test_prediction = full["test_mean"][
                [indices[int(row)] for row in selected["test"]]
            ]
        for family, alpha in [("marginal", 0.1), ("simultaneous", 0.1 / 3)]:
            threshold = order_threshold(calibration_prediction, target, alpha)
            np.testing.assert_array_equal(cp[family + "_threshold"], threshold)
            for suffix, sign in [("lower", -1), ("upper", 1)]:
                np.testing.assert_array_equal(
                    cp[family + "_" + suffix], test_prediction + sign * threshold
                )
            np.testing.assert_allclose(
                order_threshold(cpu_cal, target, alpha), threshold, rtol=5e-5, atol=5e-5
            )
    report = {
        "status": "ttt_replay_and_cp_passed",
        "bindings": bindings,
        "test_objects": len(selected["test"]),
        "calibration_objects": len(selected["calibration"]),
        "test_below_minus2": int((paired["value"][selected["test"], 2] < -2).sum()),
        "max_prediction_errors": maximum,
        "per_object_exact_reset_verified": True,
        "manual_adam_gradients_replayed": True,
        "labels_or_quoted_errors_used_in_adaptation": False,
        "calibration_refined_before_recalibration": True,
        "rtol": 5e-5,
        "atol": 5e-5,
        "other_pending_gates": [
            "original_benchmark_source",
            "CUDA_training_mask_draws",
            "weighted_library_cp",
            "other_models_and_ensemble",
            "paired_visual_reports",
        ],
    }
    atomic(report_path, report)
    atomic(progress, {"phase": "complete", "report": str(report_path)})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.self_test:
        self_test()
        print("manual Adam/independent holdout/exact reset tests PASS")
    else:
        if not args.run:
            parser.error("--run required")
        try:
            print(json.dumps(verify(args.run), indent=2), flush=True)
        except Exception as exc:
            atomic(
                BASE / args.run / "independent-ttt-failure.json",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
