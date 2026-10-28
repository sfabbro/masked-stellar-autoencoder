"""One bounded native age-head regression experiment per immutable run ID."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from run_target_head_pilot import run_pilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=Path("configs/target-pilot-wave-20261022.yaml")
    )
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    arm = next((a for a in cfg["arms"] if a["run_id"] == args.run_id), None)
    if arm is None:
        raise ValueError("Unknown native pilot arm")
    source = Path(cfg["input"])
    manifest = Path(cfg["manifest"])
    for path, key in [(source, "input_sha256"), (manifest, "manifest_sha256")]:
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != cfg[key]:
            raise ValueError("Native prepared artifact SHA differs")
    torch.set_num_threads(4)
    result = run_pilot(
        source,
        manifest,
        args.output,
        target="age",
        steps=100,
        seed=arm["seed"],
        objective=arm["objective"],
    )
    result["run_id"] = args.run_id
    result["scope"] += (
        "; native384 objects, no shared-labelled teachers; not age-scale improvement"
    )
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
