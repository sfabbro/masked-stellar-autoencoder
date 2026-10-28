"""Independent raw-source, stream and full CPU replay for frozen screen artifacts."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from zuko.flows import NSF

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_xp_representation_probes as probe
from run_frozen_head_screen import point_head
from run_prediction_comparison import native_inputs

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.prediction_vir import VIRPredictionHead

BASE = ROOT / "experiments/xp/results-stellar-improvement"
PROGRESS = BASE / "frozen-screen-cpu-progress.json"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def prediction_errors(actual, expected):
    """Keep strict failures as failures while allowing other methods to finish."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Replay predictions must be aligned and finite")
    delta = np.abs(actual - expected)
    violations = np.flatnonzero(~np.isclose(actual, expected, rtol=5e-5, atol=5e-5))
    return float(delta.max()), violations


def run():
    torch.set_num_threads(1)
    cohort = BASE / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    parent = BASE / "prediction-comparison-e72/parent/checkpoint-72.pth"
    state = {}
    data, fit, test, safe, _, yc, ys = native_inputs(cohort)
    raw_inputs = np.concatenate((safe[fit], safe[test["used_row"]]))
    checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    model = probe.load_model(
        yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text()),
        checkpoint,
        torch.device("cpu"),
    ).eval()
    cache = BASE / "frozen-screen-independent-latents.npz"
    request = dict(
        parent_sha256=sha(parent),
        input_sha256=hashlib.sha256(raw_inputs.tobytes()).hexdigest(),
    )
    if cache.exists():
        with np.load(cache) as c:
            assert json.loads(str(c["request"])) == request
            cpu_latents = c["latent"]
    else:
        state["encoder"] = "running"
        PROGRESS.write_text(json.dumps(state, indent=2))
        cpu_latents = probe.encode(model, raw_inputs, torch.device("cpu"))
        np.savez_compressed(cache, request=json.dumps(request), latent=cpu_latents)
    state["encoder"] = {"status": "passed", "rows": len(cpu_latents)}
    parent_state = hashlib.sha256(
        b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())
    ).hexdigest()
    for runid in ["frozen_e72_vir_screen_v2", "frozen_e72_cnf_screen_v2"]:
        d = BASE / runid
        m = json.loads((d / "manifest.json").read_text())
        previous = d / "independent-verification.json"
        if previous.exists():
            done = json.loads(previous.read_text())
            if (
                done.get("status") == "passed"
                and done["artifact_sha256"] == m["artifact_sha256"]
            ):
                for name, digest in m["artifact_sha256"].items():
                    assert sha(d / name) == digest
                state[runid] = {"status": "passed", "reused_completed_report": True}
                PROGRESS.write_text(json.dumps(state, indent=2))
                continue
        p = dict(np.load(d / "paired-cohort.npz", allow_pickle=False))
        assert m["status"] == "complete" and not m["smoke"]
        assert m["source_sha"] == "6234264658c1577959c1aa494f1efc817a84ea6d"
        assert m["parent_sha256"] == request["parent_sha256"] and m[
            "cohort_sha256"
        ] == sha(cohort)
        assert m["encoder_unchanged"] and m["encoder_state_sha256"] == parent_state
        assert m["native_fit"] == 19859 and m["native_test"] == 100620
        for name, digest in m["artifact_sha256"].items():
            assert sha(d / name) == digest, name
        for key, expected in [
            ("fit_source_id", data["source_id"][fit]),
            ("fit_input", safe[fit]),
            ("test_input", safe[test["used_row"]]),
            ("fit_target", data["fit_truth"][fit]),
        ]:
            np.testing.assert_array_equal(p[key], expected)
        for key, expected in [
            ("fit_measurements", data["fit_native_measurements"]),
            ("test_measurements", test["measurements"]),
        ]:
            assert p[key].dtype.names == expected.dtype.names
            for field in expected.dtype.names:
                assert p[key][field].tobytes() == expected[field].tobytes(), field
        np.testing.assert_array_equal(p["label_center"], yc)
        np.testing.assert_array_equal(p["label_scale"], ys)
        np.testing.assert_array_equal(p["latent_center"], p["fit_latent"].mean(0))
        np.testing.assert_array_equal(
            p["latent_scale"], np.maximum(p["fit_latent"].std(0), 1e-6)
        )
        latent_diff = close(
            cpu_latents, np.concatenate((p["fit_latent"], p["test_latent"]))
        )
        fit_z = torch.tensor(
            (cpu_latents[: len(fit)] - p["latent_center"]) / p["latent_scale"]
        )
        test_z = torch.tensor(
            (cpu_latents[len(fit) :] - p["latent_center"]) / p["latent_scale"]
        )
        labels = torch.tensor(((p["fit_target"] - yc) / ys).astype("f4"))
        reports = []
        for item in m["methods"]:
            mode = item["method"]
            state[runid] = dict(method=mode, status="running")
            PROGRESS.write_text(json.dumps(state, indent=2))
            saved = torch_load_trusted(
                d / (mode + "-head.pth"), map_location="cpu", weights_only=False
            )
            rng = np.random.default_rng(42)
            digest = hashlib.sha256()
            for _ in range(3000):
                digest.update(rng.integers(19859, size=256).tobytes())
            assert saved["stream_sha256"] == item["stream_sha256"] == digest.hexdigest()
            assert item["updates"] == 3000 and item["label_presentations"] == 768000
            test_x = test_z
            if m["suite"]["kind"] == "cnf":
                if mode == "raw_mse":
                    test_x = torch.cat((test_z, torch.zeros(len(test_z), 2)), 1)
                else:
                    f = torch_load_trusted(
                        d / (mode + "-features.pth"),
                        map_location="cpu",
                        weights_only=False,
                    )
                    assert (
                        f["signal_fit_count"] == 446
                        and f["background_fit_count"] == 19413
                    )
                    torch.testing.assert_close(
                        f["pca_components"].T @ f["pca_components"],
                        torch.eye(8),
                        rtol=5e-5,
                        atol=5e-5,
                    )
                    assert float(f["state"]["cut_dex"]) == -1.0
                    close(
                        (fit_z @ f["pca_components"]).std(0, unbiased=False).numpy(),
                        f["pca_scale"].numpy(),
                    )
                    positive = np.flatnonzero(p["fit_target"] < -1)
                    negative = np.flatnonzero(p["fit_target"] >= -1)
                    sampler = np.random.default_rng(43)
                    flow_stream = hashlib.sha256()
                    for _ in range(1000):
                        flow_stream.update(
                            positive[
                                sampler.integers(len(positive), size=256)
                            ].tobytes()
                        )
                        flow_stream.update(
                            negative[
                                sampler.integers(len(negative), size=256)
                            ].tobytes()
                        )
                    assert f["sampling_sha256"] == flow_stream.hexdigest()
                    assert (
                        f["flow_updates"] == 1000
                        and f["class_presentations_per_flow"] == 256000
                    )
                    features = []
                    with torch.no_grad():
                        for key in ["signal_flow", "background_flow"]:
                            flow = NSF(
                                features=8,
                                transforms=3,
                                hidden_features=(64, 64),
                                bins=8,
                            ).eval()
                            flow.load_state_dict(
                                {
                                    k[len(key) + 1 :]: v
                                    for k, v in f["state"].items()
                                    if k.startswith(key + ".")
                                },
                                strict=True,
                            )
                            phi = []
                            for block in test_z.split(512):
                                logp = flow().log_prob(
                                    block @ f["pca_components"] / f["pca_scale"]
                                )
                                # Independent p/(1+p) reference, not helper.forward.
                                density = torch.exp(logp.clamp(-30, 30)).double()
                                phi.append((density / (1 + density)).float())
                            features.append(torch.cat(phi))
                    phi = torch.stack(features, 1)
                    phi = (phi - f["phi_center"]) / f["phi_scale"]
                    test_x = torch.cat((test_z, phi), 1)
            vir = mode in VIRPredictionHead.MODES
            head = (
                VIRPredictionHead(labels, mode=mode)
                if vir
                else point_head(test_x.shape[1])
            ).eval()
            head.load_state_dict(saved["head"], strict=True)
            output = []
            post = []
            with torch.no_grad():
                for block in test_x.split(512):
                    if vir:
                        q = head.posterior(block)
                        output.append(q.gamma.numpy())
                        post.append(
                            torch.stack((q.gamma, q.nu, q.alpha, q.beta), -1).numpy()
                        )
                    else:
                        output.append(head(block).flatten().numpy())
            final = dict(np.load(d / (mode + "-predictions.npz")))
            np.testing.assert_array_equal(final["initial_standardized"], 0)
            cpu_output = np.concatenate(output)
            np.savez_compressed(
                d / (mode + "-cpu-predictions.npz"), standardized_prediction=cpu_output
            )
            diff, violations = prediction_errors(
                cpu_output, final["final_standardized"]
            )
            posterior_diff = (
                close(np.concatenate(post), final["nig_posterior"]) if vir else None
            )
            reports.append(
                dict(
                    method=mode,
                    status="passed" if len(violations) == 0 else "failed",
                    tolerance_violations=len(violations),
                    violating_rows=violations.tolist(),
                    all100620cpu_predictions=True,
                    max_standardized_difference=diff,
                    posterior_difference=posterior_diff,
                )
            )
        report = dict(
            status="passed"
            if all(x["status"] == "passed" for x in reports)
            else "failed",
            run_id=runid,
            source_sha=m["source_sha"],
            native_fit=19859,
            native_test=100620,
            artifact_sha256=m["artifact_sha256"],
            raw_source_fields_inputs_scalers_verified=True,
            parent_cpu_rows=len(cpu_latents),
            max_latent_difference=latent_diff,
            methods=reports,
            rtol=5e-5,
            atol=5e-5,
        )
        (d / "independent-verification.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        state[runid] = {"status": report["status"], "methods": reports}
        PROGRESS.write_text(json.dumps(state, indent=2))
        print(runid, report["status"], flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        (BASE / "frozen-screen-cpu-failure.json").write_text(
            json.dumps(
                dict(status="failed", error=repr(error), tolerances_unchanged=True),
                indent=2,
            )
            + "\n"
        )
        raise
