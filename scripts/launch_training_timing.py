"""Launch short CANFAR timing runs of the XP-only trainer and a multi-slice probe.

Each timing run trains ``timing_updates`` updates after a 20-update warmup with
one supervised chunk size, records seconds per update in its manifest, and
exits. The probe session asks for several GPU slices and records how they are
exposed and how fast a cross-slice all-reduce is.
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE = "configs/master-label-six-nsf-fullfit-20261111-xp.yaml"
TAG = "20261112"


def timing_config(micro, updates, slices=1, backend="gloo", tag=TAG):
    cfg = yaml.safe_load((HERE.parent / BASE).read_text())
    suffix = "" if slices == 1 else f"_slices{slices}"
    suffix += "" if backend == "gloo" else f"_{backend}"
    cfg.update(
        run_id=f"timing_xp_micro{micro}{suffix}_{tag}",
        supervised_micro_batch=micro,
        timing_updates=updates,
    )
    return cfg["run_id"], yaml.safe_dump(cfg, sort_keys=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--micro", type=int, nargs="*", default=[16, 64, 256])
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--probe-gpus", type=int, default=2)
    p.add_argument(
        "--slices",
        type=int,
        default=1,
        help="GPU slices per timing run (>1 runs one rank per slice)",
    )
    p.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    p.add_argument("--tag", default=TAG, help="run-id date tag")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    from launch_master_label_training_chain import pushed_head

    jobs = []
    for micro in a.micro:
        run_id, text = timing_config(micro, a.updates, a.slices, a.backend, a.tag)
        name = f"configs/{run_id}.yaml"
        train = (
            "python -u scripts/run_expanded_six_training.py"
            f' --config {name} --output "$experiment_output"'
            " --reuse-waiting-output --retry-pre-admission-failure"
        )
        jobs.append(
            dict(
                run_id=run_id,
                files={name: text},
                gpu=a.slices,
                cpu=4 if a.slices == 1 else 2 * a.slices,
                ram=64 if a.slices == 1 else 24 * a.slices,
                command=train
                if a.slices == 1
                else f"python -u scripts/run_multi_slice.py --slices {a.slices}"
                f" --threads-per-rank 2 --backend {a.backend} -- {train}",
            )
        )
    if a.probe_gpus:
        jobs.append(
            dict(
                run_id=f"probe_multi_gpu_{a.probe_gpus}_{a.tag}",
                files={},
                gpu=a.probe_gpus,
                cpu=4,
                ram=16,
                command='python -u scripts/probe_multi_gpu.py --output "$experiment_output"',
            )
        )
    for job in jobs:
        print(dict(run_id=job["run_id"], gpu=job["gpu"], command=job["command"]))
    if a.dry_run:
        return
    branch, head = pushed_head()
    from launch_decision_wave import submit
    from launch_master_label_census import prepare_credentials

    print("credential:", prepare_credentials(), "commit:", head)
    for job in jobs:
        submit(
            job["run_id"],
            job["files"],
            job["command"],
            gpu=job["gpu"],
            cpu=job["cpu"],
            ram=job["ram"],
            branch=branch,
            commit=head,
        )


if __name__ == "__main__":
    main()
