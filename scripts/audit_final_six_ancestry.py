"""CANFAR CPU-only source-ID audit of the completed large parent's FeH ancestry.

This is data/set arithmetic, with no model loading, forward, gradient or fitting.
It executes only the exact archived historical HDF cohort selector and ID split,
then reproduces the label-row RNG stream and checks its saved production SHA.
Cross-target FeH supervision is flagged separately from direct FeH supervision.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import yaml

ROLES = ("fit", "tune", "cal", "test")
TARGETS = ("teff", "logg", "fe_h", "alpha", "age", "distance")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def function_from_archived_source(text, name, namespace):
    """Load exactly one top-level function; never import historical modules."""
    parsed = ast.parse(text)
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise ValueError("Expected one undecorated archived function")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    environment = dict(namespace)
    exec(compile(module, "<immutable-historical-policy>", "exec"), environment)
    return environment[name]


def label_sampler_exposure(count, *, seed, warmup_updates, updates, groups, batch):
    """Exact historical NumPy label-row sampler, not an optimizer/model replay."""
    if min(count, batch, groups) < 1 or min(updates, warmup_updates) < 0:
        raise ValueError("Invalid sampler dimensions")
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    occurrences = np.zeros(count, dtype="i8")
    for _ in range(warmup_updates + updates * groups):
        rows = rng.integers(count, size=batch)
        digest.update(np.asarray(rows, dtype="<i8").tobytes())
        # Accumulate exactly, including repeated row IDs inside a sampled batch.
        np.add.at(occurrences, rows, 1)
    return occurrences, digest.hexdigest()


def ancestry_masks(prepared, exposed_ids):
    """Masks align with exact prepared source IDs, target order and role eligibility."""
    ids = np.asarray(prepared["source_id"])
    role = np.asarray(prepared["role"])
    observed = np.asarray(prepared["observed"], bool)
    eligible = observed & np.asarray(prepared["quality"], bool)
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Prepared source IDs are not unique integers")
    if role.shape != ids.shape or not np.isin(role, ROLES).all():
        raise ValueError("Prepared role shape/values differ")
    if observed.shape != (len(ids), len(TARGETS)) or eligible.shape != observed.shape:
        raise ValueError("Prepared target dimensions differ")
    if tuple(np.asarray(prepared["target_names"]).tolist()) != TARGETS:
        raise ValueError("Prepared target order differs")
    if not np.isfinite(np.asarray(prepared["value"])[eligible]).all():
        raise ValueError("Eligible source target is nonfinite")
    source_roles = np.asarray(prepared["label_source_role"])
    if source_roles.shape != observed.shape or np.any(
        eligible & (source_roles != role[:, None])
    ):
        raise ValueError("A promoted other-role target entered scoring")
    exposed = np.isin(ids, np.asarray(exposed_ids, dtype="i8"))
    scored_exposure = eligible & exposed[:, None]
    direct = np.zeros_like(eligible)
    direct[:, 2] = scored_exposure[:, 2]
    cross = scored_exposure & ~direct
    counts = {}
    for target_index, target in enumerate(TARGETS):
        counts[target] = {}
        for current_role in ROLES:
            subset = eligible[:, target_index] & (role == current_role)
            counts[target][current_role] = dict(
                eligible_count=int(subset.sum()),
                feh_aux_fit_exposed_count=int((subset & exposed).sum()),
                feh_aux_fit_unexposed_count=int((subset & ~exposed).sum()),
                exposure_kind="same_target_native_FeH_supervision"
                if target == "fe_h"
                else "cross_target_native_FeH_supervision",
            )
    return dict(
        source_id=ids,
        role=role,
        target_names=np.asarray(TARGETS),
        target_eligible=eligible,
        feh_aux_fit_exposed=exposed,
        target_feh_aux_fit_exposed=scored_exposure,
        direct_feh_target_exposed=direct,
        cross_target_feh_aux_exposed=cross,
        original_observed=observed,
        label_source_role=source_roles,
    ), counts


def audit(config, output):
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError("Run this source-ID audit on CANFAR CPU headless only")
    if output.exists():
        raise FileExistsError(output)
    archive_path = Path(config["historical_archive"])
    if sha(archive_path) != config["historical_archive_sha256"]:
        raise ValueError("Historical source archive SHA changed")
    archive = json.loads(archive_path.read_text())
    if archive["historical_commit"] != config["historical_source_commit"]:
        raise ValueError("Historical commit differs")
    texts = {}
    for path, spec in archive["files"].items():
        if hashlib.sha256(spec["text"].encode()).hexdigest() != spec["sha256"]:
            raise ValueError("Historical whole-source text hash changed")
        texts[path] = spec["text"]
    old_config = yaml.safe_load(texts["configs/residual-huber-150m.yaml"])
    if (
        old_config["checkpoint_sha256"] != config["parent86_sha256"]
        or old_config["reconstruction_presentations"] != 4_500_000_000
    ):
        raise ValueError("Historical parent or sample budget differs")
    prepared_path = Path(config["prepared"])
    cohort_path = Path(config["historical_cohort"])
    if (
        sha(prepared_path) != config["prepared_sha256"]
        or sha(cohort_path) != config["historical_cohort_sha256"]
    ):
        raise ValueError("Actual prepared/cohort source SHA changed")
    producer = json.loads(Path(config["large_manifest"]).read_text())
    receipt = json.loads(Path(config["large_final_receipt"]).read_text())
    if (
        producer["status"] != "complete"
        or producer["cohort_sha256"] != config["historical_cohort_sha256"]
        or producer["parent_sha256"] != config["parent86_sha256"]
        or producer["presentations"] != 4_500_000_000
        or producer["optimizer_updates"] != config["expected_updates"]
        or producer["replica_count"] != 1
        or producer["source_sha"] != config["historical_source_commit"]
        or receipt["sha256"] != config["large_final_checkpoint_sha256"]
        or receipt["presentations"] != 4_500_000_000
        or receipt["completed_updates"] != config["expected_updates"]
    ):
        raise ValueError("Actual completed-parent receipt/source differs")
    if (
        producer["supervised_stream_sha256"]
        != config["expected_supervised_stream_sha256"]
    ):
        raise ValueError("Actual completed-parent label-stream receipt changed")
    prepared_manifest = json.loads(Path(config["prepared_manifest"]).read_text())
    if (
        prepared_manifest["status"] != "sparse_six_target_source_assembly_complete"
        or prepared_manifest["input_sha256"] != config["prepared_sha256"]
        or prepared_manifest["source_commit"] != config["prepared_source_commit"]
    ):
        raise ValueError("Prepared v3 source/status differs")
    split = function_from_archived_source(
        texts["scripts/stellar_catalog_jobs.py"], "identity_split", {"np": np}
    )
    selector = function_from_archived_source(
        texts["scripts/run_enriched_value.py"],
        "prepare",
        {
            "np": np,
            "h5py": h5py,
            "json": json,
            "identity_split": split,
            "LAYERS": ("native", "PIGS", "Li2025_CaT_silver"),
        },
    )
    historical = selector(cohort_path)
    fit_rows = historical["native_fit"]
    fit_ids = np.asarray(historical["source_id"])[fit_rows]
    if len(fit_ids) != 19_859 or len(np.unique(fit_ids)) != len(fit_ids):
        raise ValueError("Exact historical native19859 FIT identities differ")
    if len(historical["tests"]["native"]["used_row"]) != 100_620:
        raise ValueError("Historical native100620 TEST policy differs")
    if (
        np.any(split(fit_ids) != 0)
        or np.isin(
            fit_ids, historical["tests"]["native"]["measurements"]["source_id"]
        ).any()
    ):
        raise ValueError("Historical native FIT/TEST role proof failed")
    occurrences, sampler_sha = label_sampler_exposure(
        len(fit_ids),
        seed=int(old_config["arms"][0]["experiment_seed"]),
        warmup_updates=int(old_config["warmup_updates"]),
        updates=int(receipt["completed_updates"]),
        groups=int(old_config["supervised_batches_per_update"]),
        batch=int(old_config["contrast_batch"]),
    )
    if sampler_sha != producer["supervised_stream_sha256"]:
        raise ValueError(
            "Reconstructed actual label-row stream differs from production SHA"
        )
    del historical
    with np.load(prepared_path, allow_pickle=False) as arrays:
        prepared = {
            key: arrays[key]
            for key in (
                "source_id",
                "role",
                "observed",
                "quality",
                "value",
                "target_names",
                "label_source_role",
            )
        }
    masks, counts = ancestry_masks(prepared, fit_ids[occurrences > 0])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "ancestry-masks.npz",
        **masks,
        historical_fit_source_id=fit_ids,
        historical_fit_sample_count=occurrences,
    )
    report = dict(
        status="source_id_feh_aux_ancestry_verified",
        counts=counts,
        actual_exposed_fit_count=int((occurrences > 0).sum()),
        eligible_historical_fit_count=len(fit_ids),
        minimum_fit_sample_count=int(occurrences.min()),
        maximum_fit_sample_count=int(occurrences.max()),
        label_presentations=int(occurrences.sum()),
        reconstructed_sampler_sha256=sampler_sha,
        actual_producer_sampler_sha256=producer["supervised_stream_sha256"],
        bindings=dict(
            policy=config,
            historical_archive_sha256=sha(archive_path),
            verifier_sha256=sha(Path(__file__)),
            prepared_sha256=sha(prepared_path),
            producer_manifest_sha256=sha(config["large_manifest"]),
            final_receipt_sha256=sha(config["large_final_receipt"]),
            masks_sha256=sha(output / "ancestry-masks.npz"),
        ),
        numerical_model_execution=False,
        checkpoint_loaded_or_downloaded=False,
        exposure_interpretation="Same-target FeH supervision and cross-target FeH auxiliary exposure are separate flags; neither is hidden under a universal unseen-star claim",
        limitations="Audits this large-run FeH auxiliary stream only, not parent86's earlier supervised/unsupervised ancestry or all reconstruction membership; no physical-truth or independence certification",
        direct_target_heldout_overlap_detected=any(
            counts["fe_h"][role]["feh_aux_fit_exposed_count"]
            for role in ("tune", "cal", "test")
        ),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
