"""Wait for the exact CUDA source-map training path before admitting longer runs."""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
RUN = "sixmap_device_preflight_20261107_v1"


def main():
    expected = hashlib.sha256(
        Path("scripts/run_six_survey_map_ablation.py").read_bytes()
    ).hexdigest()
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        try:
            m = json.loads((ROOT / RUN / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            m = {}
        if m.get("status") == "failed":
            raise RuntimeError(
                "Source-map CUDA smoke test failed: " + str(m.get("error"))
            )
        if m.get("status") == "complete":
            if (
                m.get("run_id") != RUN
                or m.get("script_sha256") != expected
                or m.get("completed_updates") != 14
                or m.get("finite_gradient_updates") != 14
                or not m.get("last_head_gradient_l2", 0) > 0
                or len(m.get("source_registry", [])) != 7
            ):
                raise ValueError("Source-map smoke-test completion contract differs")
            hashes = json.loads(
                (
                    ROOT
                    / "decision_wave_source_bundles_20261106_v1"
                    / RUN
                    / "source-hashes.json"
                ).read_text()
            )
            for name, sha in hashes.items():
                if (
                    name.endswith(".py")
                    and hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha
                ):
                    raise ValueError("Smoke-tested source differs: " + name)
            print(
                "Source-map CUDA smoke test passed; starting longer control", flush=True
            )
            return
        print("Waiting for source-map CUDA smoke test", flush=True)
        time.sleep(30)
    raise TimeoutError("Source-map CUDA smoke test did not complete in six hours")


if __name__ == "__main__":
    main()
