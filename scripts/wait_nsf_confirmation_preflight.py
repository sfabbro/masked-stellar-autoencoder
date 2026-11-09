"""Admit queued confirmations only after the immutable CUDA smoke test passes."""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
RUN = "nsfconfirm_device_preflight_20261106_v1"


def validate(manifest, expected_script_sha):
    if manifest.get("status") == "failed":
        raise RuntimeError("CUDA smoke test failed: " + str(manifest.get("error")))
    if manifest.get("status") != "remediation_pilot_complete":
        return False
    if (
        manifest.get("run_id") != RUN
        or manifest.get("script_sha256") != expected_script_sha
        or manifest.get("completed_updates") != 2
        or manifest.get("finite_gradient_updates") != 2
        or manifest.get("declared_updates") != 2
        or not manifest.get("selected_checkpoint_sha256")
    ):
        raise ValueError("CUDA smoke-test completion contract differs")
    return True


def main():
    expected = hashlib.sha256(
        Path("scripts/run_nsf_confirmation.py").read_bytes()
    ).hexdigest()
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        path = ROOT / RUN / "manifest.json"
        try:
            manifest = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        if validate(manifest, expected):
            bundle = ROOT / "decision_wave_source_bundles_20261106_v1" / RUN
            hashes = json.loads((bundle / "source-hashes.json").read_text())
            for name in hashes:
                if (
                    name.endswith(".py")
                    and hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    != hashes[name]
                ):
                    raise ValueError("Smoke-tested source differs: " + name)
            print("CUDA smoke-test contract passed; starting confirmation", flush=True)
            return
        print("Waiting for immutable CUDA smoke-test completion", flush=True)
        time.sleep(30)
    raise TimeoutError("CUDA smoke test did not complete in six hours")


if __name__ == "__main__":
    main()
