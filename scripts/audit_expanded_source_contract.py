"""Remote metadata-only mount/request audit; no label assembly or model execution."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def audit(config, output):
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError("Run this metadata audit on CANFAR headless CPU")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id="full_label_source_contract_audit_v1",
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        native_model_execution=False,
        checkpoint_loaded=False,
        source_assembly_performed=False,
    )
    try:
        census = {}
        for name, spec in config.items():
            if isinstance(spec, dict) and "path" in spec:
                path = Path(spec["path"])
                stat = path.stat()
                if (
                    not path.is_file()
                    or stat.st_size <= 0
                    or not os.access(path, os.R_OK)
                ):
                    raise ValueError("Source is not a readable nonempty file: " + name)
                census[name] = dict(
                    path=str(path),
                    bytes=stat.st_size,
                    content_sha256_verified=False,
                    expected_sha256=spec.get("sha256"),
                )
        features = config["feature_lookup"]
        for name in ["legacy_path", "original_path", "cache_path"]:
            path = Path(features[name])
            if not path.exists() or not os.access(path, os.R_OK):
                raise ValueError("Feature source is not readable: " + name)
            census[name] = dict(path=str(path), is_directory=path.is_dir())
        original = output.parent / "full_label_evaluation_preparation_v1"
        manifest.update(
            status="expanded_source_contract_audit_complete",
            sources=census,
            output_parent_writable=os.access(output.parent, os.W_OK),
            available_project_bytes=shutil.disk_usage(output.parent).free,
            original_source_output_exists=original.exists(),
            original_source_manifest_exists=(original / "manifest.json").exists(),
            identity_scope="Mounted-file presence/size/readability only; no SHA/labels/model/ancestry certificate",
        )
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
