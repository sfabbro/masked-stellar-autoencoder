"""Independent TTA source, stateless-view and per-object gradient replay.

Only architecture/checkpoint loading is shared with production. No production
TTA operator is called. Cache entries bind inputs, policy, artifacts and verifier.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_xp_representation_probes as probe  # noqa: E402

from masked_stellar_autoencoder.models.checkpoint_load import (
    torch_load_trusted,  # noqa: E402
)
from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_prediction_head,  # noqa: E402
)


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def random_stream(source, seed, view, domain):
    payload = f"{int(source)}:{int(seed)}:{int(view)}:{domain}".encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    )


def selection(ids, count, seed):
    keys = [
        int.from_bytes(
            hashlib.sha256(f"{int(s)}:{seed}".encode()).digest()[:8], "little"
        )
        for s in ids
    ]
    return np.argsort(np.asarray(keys, dtype="u8"), kind="stable")[:count]


def split(ids):
    state = np.asarray(ids, dtype="u8") ^ np.uint64(42)
    with np.errstate(over="ignore"):
        state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        state ^= state >> np.uint64(31)
    return np.where(state % 1000 < 700, 0, np.where(state % 1000 < 850, 1, 2))


def view(values, visible, sigma, ids, xp, seed, number, noise, probability):
    result = values.copy()
    eligible = visible[:, xp] & np.isfinite(values[:, xp]) & (values[:, xp] != -9999)
    epsilon = np.asarray(
        [random_stream(s, seed, number, "noise").standard_normal(len(xp)) for s in ids],
        dtype="f4",
    )
    missing = (
        np.asarray(
            [
                random_stream(s, seed, number, "mask").random(len(xp)) < probability
                for s in ids
            ]
        )
        & eligible
    )
    result[:, xp] = np.where(
        eligible, result[:, xp] + noise * sigma * epsilon, result[:, xp]
    )
    hidden = np.zeros_like(visible)
    hidden[:, xp] = missing
    result[hidden] = -9999
    return result, hidden


def point(model, head, values):
    return head(model.encoder(values))[:, :, 0].mean(1)


def adaptation(model, head, values, visible, source, xp, policy, method):
    """Independent fixed three-step objective and explicit per-object reset."""
    for parameter in list(model.parameters()) + list(head.parameters()):
        parameter.requires_grad_(False)
    selected = (
        [
            parameter
            for module in model.encoder.modules()
            if isinstance(module, torch.nn.LayerNorm)
            for parameter in module.parameters(recurse=False)
        ]
        if method.startswith("layernorm")
        else []
    )
    saved = [parameter.detach().clone() for parameter in selected]
    try:
        for parameter in selected:
            parameter.requires_grad_(True)
        corrupted, held = view(
            values,
            visible,
            np.zeros((1, len(xp)), dtype="f4"),
            [source],
            xp,
            policy["seed"],
            0,
            0.0,
            policy["adapt_mask_probability"],
        )
        valid = [int(c) for c in xp if visible[0, c]]
        assert len(valid) >= 2
        if not held.any():
            c = valid[
                int(
                    random_stream(source, policy["seed"], 0, "fallback").integers(
                        len(valid)
                    )
                )
            ]
            held[0, c] = True
            corrupted[0, c] = -9999
        if all(held[0, c] for c in valid):
            held[0, valid[-1]] = False
            corrupted[0, valid[-1]] = values[0, valid[-1]]
        clean = torch.from_numpy(values)
        hidden = torch.from_numpy(corrupted)
        with torch.no_grad():
            baseline = point(model, head, clean)
            initial = model.encoder(hidden)
        if method == "masked_view":
            with torch.no_grad():
                return float(head(initial)[:, :, 0].mean(1)), {
                    "steps": 0,
                    "scored_features": int(held.sum()),
                }
        latent_mode = method == "latent_recon"
        latent = initial.clone().requires_grad_(latent_mode)
        optimizer = torch.optim.Adam(
            [latent] if latent_mode else selected,
            lr=0.01 if latent_mode else policy["layernorm_lr"],
        )
        for _ in range(policy["steps"]):
            optimizer.zero_grad(set_to_none=True)
            z = latent if latent_mode else model.encoder(hidden)
            decoded = model.reconstruction_layer(model.decoder(z))
            if getattr(model, "heteroscedastic", False):
                decoded = decoded.chunk(2, dim=-1)[0]
            loss = (
                (decoded - clean[:, : decoded.shape[1]])
                .abs()[torch.from_numpy(held[:, : decoded.shape[1]])]
                .mean()
            )
            if latent_mode:
                loss = loss + policy["anchor"] * (latent - initial).square().mean()
            else:
                loss = loss + policy["anchor"] * sum(
                    (p - v).square().mean()
                    for p, v in zip(selected, saved, strict=True)
                ) / len(selected)
                if method.endswith("consistency"):
                    loss = (
                        loss
                        + policy["consistency"]
                        * (point(model, head, clean) - baseline).square().mean()
                    )
            assert torch.isfinite(loss)
            loss.backward()
            assert all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in ([latent] if latent_mode else selected)
            )
            optimizer.step()
        with torch.no_grad():
            final = (
                head(latent)[:, :, 0].mean(1)
                if latent_mode
                else point(model, head, clean)
            )
        return float(final), {
            "steps": policy["steps"],
            "scored_features": int(held.sum()),
            "final_objective": float(loss.detach()),
        }
    finally:
        with torch.no_grad():
            for parameter, initial_value in zip(selected, saved, strict=True):
                parameter.copy_(initial_value)
                assert torch.equal(parameter, initial_value)
                parameter.requires_grad_(False)


def input_gate(directory, cohort, paired):
    with h5py.File(cohort) as f:
        native = f["raw_layers/native/measurements"][:]
        native_rows = f["layer_index/native/feature_row"][:]
        ids = f["features/source_id"][:]
        available = f["features/feature_available"][:][native_rows]
        partition = f["features/benchmark_partition"][:][native_rows]
        expected_test = native[available & (split(native["source_id"]) == 2)]
        candidates = np.flatnonzero(
            available & (partition == 0) & (split(native["source_id"]) == 1)
        )
        chosen = candidates[selection(native["source_id"][candidates], 256, 1042)]
        names = json.loads(f.attrs["feature_columns_json"])
        xp = np.asarray(
            [names.index(f"{arm}_{i}") for arm in ("bp", "rp") for i in range(1, 56)]
        )
        np.testing.assert_array_equal(xp, paired["xp_columns"])
        center, scale = f["reference_scaler/center"][:], f["reference_scaler/scale"][:]
        phot = (
            {"w1", "w2", "g", "bp", "rp", "j", "h", "ks", "ebv"}
            | {f"{b}_smss" for b in "uvgriz"}
            | {f"{b}_sdss" for b in "ugriz"}
            | {f"{b}_ps1" for b in "grizy"}
            | {f"{a}_{i}" for a in ("bp", "rp") for i in range(1, 56)}
        )
        allowed = np.asarray([name.casefold() in phot for name in names])
        for role, measurements in [("test", expected_test), ("tune", native[chosen])]:
            # Test producer retains its cohort order; tune preserves hash order.
            if role == "test":
                lookup = {int(s): i for i, s in enumerate(measurements["source_id"])}
                measurements = measurements[
                    [lookup[int(s)] for s in paired["test_measurements"]["source_id"]]
                ]
            for field in measurements.dtype.names:
                np.testing.assert_array_equal(
                    measurements[field], paired[role + "_measurements"][field]
                )
            order = np.argsort(ids)
            rows = order[np.searchsorted(ids[order], measurements["source_id"])]
            np.testing.assert_array_equal(ids[rows], measurements["source_id"])
            if role == "tune":
                np.testing.assert_array_equal(rows, paired["tune_feature_row"])
            sorted_rows = np.argsort(rows)
            undo = np.argsort(sorted_rows)
            raw = f["features/values"][rows[sorted_rows]][undo]
            errors = f["features/reported_feature_errors"][rows[sorted_rows]][undo]
            standardized = ((raw - center) / scale).astype("f4")
            visible = (
                np.isfinite(raw)
                & allowed
                & np.isfinite(standardized)
                & (standardized != -9999)
            )
            np.testing.assert_array_equal(visible, paired[role + "_visible"])
            np.testing.assert_array_equal(
                np.where(visible, standardized, -9999), paired[role + "_input"]
            )
            sigma = (
                errors[:, xp]
                / 10.0 ** ((8.5 - raw[:, names.index("G")]) / 2.5)[:, None]
                / scale[xp]
            ).astype("f4")
            np.testing.assert_allclose(
                sigma, paired[role + "_sigma_xp"], rtol=5e-6, atol=5e-6
            )
        np.testing.assert_array_equal(
            selection(paired["test_measurements"]["source_id"], 512, 1042),
            paired["adapt_selection"],
        )
        assert len(expected_test) == len(paired["test_measurements"]) == 100620
        assert len(np.unique(paired["test_measurements"]["source_id"])) == 100620
        assert not np.isin(
            native[chosen]["source_id"], expected_test["source_id"]
        ).any()
    return {
        "test": 100620,
        "tune": 256,
        "adaptation": 512,
        "adaptation_below_minus2": int(
            (paired["test_measurements"]["value"][paired["adapt_selection"]] < -2).sum()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and not manifest["smoke"]
    assert manifest["source_sha"] == "94929b13d2552af21261a722297b5a44f034b1ed"
    assert (
        sha(args.winner / "checkpoint.pth") == manifest["policy"]["checkpoint_sha256"]
    )
    assert (
        sha(args.cohort)
        == "321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120"
    )
    inventory = json.loads((directory / "artifact-sha256.json").read_text())
    for name, record in inventory.items():
        path = directory / name
        assert (
            path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]
        ), name
    paired = dict(np.load(directory / "paired-inputs.npz", allow_pickle=False))
    gates = input_gate(directory, args.cohort, paired)
    units = json.loads((directory / "xp-unit-audit.json").read_text())
    errors = json.loads((directory / "xp-error-source-audit.json").read_text())
    assert units["matched_stars"] == 128
    assert units["candidates"]["g_normalized"]["passed"]
    assert errors["status"] == "verified" and errors["common_stars"] == 128
    winner_report = json.loads(
        (args.winner / "independent-verification.json").read_text()
    )
    assert winner_report["status"] == "passed"
    assert (
        winner_report["final_checkpoint_sha256"]
        == manifest["policy"]["checkpoint_sha256"]
    )
    winner_inputs = np.load(args.winner / "paired-cohort.npz", allow_pickle=False)
    for key in ("test_measurements", "test_input", "label_center", "label_scale"):
        np.testing.assert_array_equal(winner_inputs[key], paired[key])
    winner_outputs = np.load(args.winner / "predictions.npz", allow_pickle=False)
    verified_baseline = winner_outputs["member_outputs"][:, :, 0].mean(1)
    test_outputs = np.load(directory / "predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(
        verified_baseline * float(paired["label_scale"])
        + float(paired["label_center"]),
        test_outputs["baseline"],
    )
    expected_policy = yaml.safe_load(
        (ROOT / "configs/test-time-screen.yaml").read_text()
    )
    assert manifest["policy"] == expected_policy
    binding = {
        "checkpoint": manifest["policy"]["checkpoint_sha256"],
        "cohort": sha(args.cohort),
        "artifacts": sha(directory / "artifact-sha256.json"),
        "manifest": sha(directory / "manifest.json"),
        "verifier": sha(__file__),
        "winner_verification": sha(args.winner / "independent-verification.json"),
        "winner_predictions": sha(args.winner / "predictions.npz"),
        "winner_inputs": sha(args.winner / "paired-cohort.npz"),
    }
    result = {
        "binding": binding,
        "source_inputs": gates,
        "methods": {},
        "status": "preflight_pass",
    }
    write(directory / "independent-tta-progress.json", result)
    if args.preflight_only:
        return
    torch.set_num_threads(4)
    disk = torch_load_trusted(
        args.winner / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    model = probe.load_model(
        yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text()),
        disk,
        torch.device("cpu"),
    ).eval()
    head = build_prediction_head("residual").eval()
    head.load_state_dict(disk["head_state_dict"], strict=True)
    for p in list(model.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    center, scale = float(paired["label_center"]), float(paired["label_scale"])
    np.testing.assert_array_equal(
        [center, scale], [disk["label_center"], disk["label_scale"]]
    )
    original = {k: v.clone() for k, v in model.state_dict().items()}
    cache = directory / "independent-tta-cache"
    cache.mkdir(exist_ok=True)
    for role in ("tune", "test"):
        saved = dict(
            np.load(
                directory
                / ("predictions.npz" if role == "test" else "tune-predictions.npz"),
                allow_pickle=False,
            )
        )
        x, visible, sigma = [
            paired[role + "_" + k] for k in ("input", "visible", "sigma_xp")
        ]
        ids = paired[role + "_measurements"]["source_id"]
        np.testing.assert_array_equal(ids, saved["source_id"])
        for label, noise, probability in [
            ("baseline", 0.0, 0.0),
            ("mask_average", 0.0, 0.1),
            ("noise_average", 1.0, 0.0),
            ("noise_mask_average", 1.0, 0.1),
        ]:
            name = role + "-" + label
            receipt = cache / (name + ".json")
            output = cache / (name + ".npy")
            if receipt.exists() and output.exists():
                record = json.loads(receipt.read_text())
                assert record["binding"] == binding and record["output_sha256"] == sha(
                    output
                )
                result["methods"][name] = record
                continue
            result["current_method"] = name
            result["status"] = "running"
            write(directory / "independent-tta-progress.json", result)
            if role == "test" and label == "baseline":
                np.save(output, verified_baseline)
                record = {
                    "binding": binding,
                    "count": len(x),
                    "status": "passed",
                    "reuse": "checksum-bound previously passed full winner CPU replay",
                    "output_sha256": sha(output),
                    "max_abs": 0.0,
                }
                write(receipt, record)
                result["methods"][name] = record
                continue
            predictions = []
            for start in range(0, len(x), 512):
                block = slice(start, start + 512)
                with torch.no_grad():
                    points = (
                        [verified_baseline[block]]
                        if role == "test"
                        else [point(model, head, torch.from_numpy(x[block])).numpy()]
                    )
                    if label != "baseline":
                        for number in range(7):
                            changed, _ = view(
                                x[block],
                                visible[block],
                                sigma[block],
                                ids[block],
                                paired["xp_columns"],
                                1042,
                                number,
                                noise,
                                probability,
                            )
                            points.append(
                                point(model, head, torch.from_numpy(changed)).numpy()
                            )
                predictions.append(
                    np.mean(np.stack(points).astype("f8"), axis=0).astype("f4")
                )
            standardized = np.concatenate(predictions)
            actual = standardized * scale + center
            np.testing.assert_allclose(actual, saved[label], rtol=5e-5, atol=5e-5)
            np.save(output, standardized)
            record = {
                "binding": binding,
                "count": len(x),
                "max_abs": float(np.max(np.abs(actual - saved[label]))),
                "output_sha256": sha(output),
                "status": "passed",
            }
            write(receipt, record)
            result["methods"][name] = record
            write(directory / "independent-tta-progress.json", result)
        indexes = paired["adapt_selection"] if role == "test" else np.arange(len(x))
        for method in (
            "masked_view",
            "latent_recon",
            "layernorm_recon",
            "layernorm_recon_consistency",
        ):
            name = role + "-" + method
            receipt, output = cache / (name + ".json"), cache / (name + ".npy")
            if receipt.exists() and output.exists():
                record = json.loads(receipt.read_text())
                assert record["binding"] == binding and record["output_sha256"] == sha(
                    output
                )
                result["methods"][name] = record
                continue
            result["current_method"] = name
            write(directory / "independent-tta-progress.json", result)
            reference = np.load(directory / (name + ".npz"), allow_pickle=False)
            diagnostics = json.loads(
                (directory / (name + "-diagnostics.json")).read_text()
            )
            np.testing.assert_array_equal(ids[indexes], reference["source_id"])
            predictions = []
            for j, i in enumerate(indexes):
                value, diagnostic = adaptation(
                    model,
                    head,
                    x[i : i + 1],
                    visible[i : i + 1],
                    ids[i],
                    paired["xp_columns"],
                    manifest["policy"],
                    method,
                )
                predictions.append(value)
                assert (
                    diagnostic["steps"] == diagnostics[j]["steps"]
                    and diagnostic["scored_features"]
                    == diagnostics[j]["scored_features"]
                )
                if "final_objective" in diagnostic:
                    np.testing.assert_allclose(
                        diagnostic["final_objective"],
                        diagnostics[j]["final_objective"],
                        rtol=5e-5,
                        atol=5e-5,
                    )
                np.testing.assert_allclose(
                    value, reference["standardized_prediction"][j], rtol=5e-5, atol=5e-5
                )
            values = np.asarray(predictions, dtype="f4")
            np.testing.assert_allclose(
                values * scale + center, saved[method + "_subset"], rtol=5e-5, atol=5e-5
            )
            for key, tensor in model.state_dict().items():
                assert torch.equal(tensor, original[key]), key
            np.save(output, values)
            record = {
                "binding": binding,
                "count": len(indexes),
                "max_abs": float(
                    np.max(np.abs(values - reference["standardized_prediction"]))
                ),
                "output_sha256": sha(output),
                "status": "passed",
            }
            write(receipt, record)
            result["methods"][name] = record
            write(directory / "independent-tta-progress.json", result)
    result["status"] = "passed"
    write(directory / "independent-tta-verification.json", result)
    write(directory / "independent-tta-progress.json", result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--directory" in sys.argv:
            failure_directory = Path(sys.argv[sys.argv.index("--directory") + 1])
            progress_path = failure_directory / "independent-tta-progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                progress.update(
                    status="failed", error=str(exc), error_type=type(exc).__name__
                )
                write(progress_path, progress)
            write(
                failure_directory / "independent-tta-failure.json",
                {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
            )
        raise
