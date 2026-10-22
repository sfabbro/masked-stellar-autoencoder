# H100 MIG slicing and workload routing on CANFAR

**Prepared:** 2026-10-22. Evidence from NVIDIA's H100 MIG profile table, CANFAR
`canfar` CLI/API, live session info, and a two-GPU probe session (`x3iio030`).

---

## 1. What we actually get

The GPU allocation on CANFAR is an **H100 NVL MIG 1g.12gb** (or **1g.10gb** on an
H100 80GB node) — confirmed in job logs:

```
NVIDIA H100 NVL MIG 1g.12gb
```

The H100 profiles that matter
([NVIDIA table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html#h100-mig-profiles)):

| Profile | Memory (94GB node) | SMs | Instances |
|---|---:|---:|---:|
| 1g.12gb | 1/8 = 12 GB | 1/7 ≈ 16 | 7 |
| 1g.24gb | 1/4 = 24 GB | 1/7 ≈ 16 | 4 |
| 2g.24gb | 2/8 = 24 GB | 2/7 ≈ 32 | 3 |
| 3g.47gb | 4/8 = 47 GB | 3/7 ≈ 48 | 2 |
| 4g.47gb | 4/8 = 47 GB | 4/7 ≈ 64 | 1 |
| 7g.94gb | full | full | 1 |

On H100 80GB the smallest is **1g.10gb**; on 94/96GB it is **1g.12gb**.
**The 1g profile is the smallest compute slice; it cannot be split further.**

### Critical measured fact: `--gpu 2` is two slices, not one bigger slice

`canfar create --gpu` is an integer count, not a MIG profile selector. A probe with
`--gpu 2` showed **two separate 1g MIG devices** in the same container:

```
MIG 0 12 0 0   15MiB / 11008MiB   16 SM
MIG 0 13 0 1   15MiB / 11008MiB   16 SM
```

So `--gpu 2` gives two 1g.10gb instances (on that node), **not** a 2g.24gb instance.
It is only useful if the runner uses both CUDA devices (DDP/sharded inference);
otherwise the second slice is wasted.

### What is not available

- No CLI request for `1g.20/24gb` (more memory, same 1 SM fraction), `2g`, `3g`,
  or `4g` profiles.
- No way to request a specific node type (80GB vs 94GB), hence 10 vs 12 GiB per slice.
- No sub-1g slice. “More efficient slicing” therefore means **correct routing and
  packing**, not creating smaller MIGs.

Platform capacity at the time of writing: **112 GPU slices total, 45 in use**;
the bottleneck is not total MIG availability, it is queueing, per-job wall time and
whether a job actually uses its slice.

---

## 2. Current routing problems

The main launcher (`scripts/launch_stellar_improvement.py`) previously gave **every
non-catalog task a GPU** (`cores:4, ram:64, gpu:1`), including CPU-only phases. The
A0 age-data audit, for example, held a 1g.12gb slice for 37 minutes while running only
pandas/h5py/JSON. The same applies to scoring, plotting, catalogue collection,
reporting, and tabular XGBoost/sklearn work.

This is the cheapest efficiency win: **CPU-only work must not consume a MIG slice.**

---

## 3. Recommended allocation

| Workload | Profile | Why |
|---|---|---|
| Catalog collection, age audits, data preparation, crossmatch | `cpu-medium` (8C/32GB) or `cpu-large` (16C/64GB) | I/O/CPU; no CUDA; avoids a slice |
| Scoring, plotting, conformal/reporting, ensembling analysis | `cpu-small/medium` | No CUDA |
| XGBoost/LightGBM/tabular screens | `cpu-medium/large` | Tree models are CPU-efficient; GPU not needed |
| MSA 10M head screens, seed sweeps | `gpu-1g` (4C/64GB) | Fits in 10–12 GB; ~14 min/arm |
| MSA long/pretrain/finetune (150M–270M presentations) | `gpu-1g-large-ram` (8C/128GB) | One 1g slice; keep GPU peak <10 GB so it can land on either node |
| MSA inference/predictions over large catalogues | `gpu-1g`, **sharded** | One slice per shard; CPU fallback for small |
| torchregress NN sweeps | `gpu-1g` | Small models |
| Track B profile-likelihood grids | `cpu-large` | Isochrone interpolation is CPU; GPU only for an emulator |
| TESS/asteroseismology preprocessing | CPU or `gpu-1g` | Measure; most light-curve work is CPU/I/O |

Defaults to aim for: **one 1g slice per normal GPU job**, no `gpu > 1` unless the
runner explicitly uses multiple devices.

---

## 4. Is packing multiple jobs into one slice worth it?

Only under specific conditions:

- Each job uses **well under** one 1g slice’s SM fraction (measure with
  `nvidia-smi --query-gpu=utilization.gpu`; the `canfar info` “GPU Usage” field is
  not a reliable utilization number).
- Combined GPU memory stays below ~8–9 GB on a 10 GB node.
- The bottleneck is session slot/queue or orchestration overhead, not total GPU
  compute.

If a job already saturates 16 SMs, packing two jobs into one slice does **not**
reduce total GPU-hours; it just serialises them and increases wall time. For the
current 10M screen (6 arms × ~14 min), running six 1g slices in parallel costs the
same total GPU-seconds as running them sequentially in one slice, but finishes ~6×
faster. Packing is a queue/wall-clock tool, not a compute-saving tool.

A safe first experiment: run two small arms in one 1g session (two processes, each
pinned to `CUDA_VISIBLE_DEVICES=0`, with memory caps), compare per-arm time and total
throughput against two separate 1g sessions. If each arm uses <50% SM and <4 GB,
packing can double throughput per slice; if not, keep one job per slice.

---

## 5. Using multiple slices for one large job

If a long training job is compute-bound, DDP across **two 1g slices** can reduce
wall time, but:

- `--gpu 2` gives two separate 1g instances; the code must use `torchrun`/DDP with
  both devices. The current MSA runners are single-device and would waste one.
- With a 34M-parameter model and batch 512, communication overhead may limit scaling
  to ~1.3–1.8×, not 2×; benchmark before committing.
- Alternatively use gradient accumulation/checkpointing to fit the 1g memory and keep
  one slice per job.

For parameter sweeps, `canfar create --replicas N` (with `REPLICA_ID`/`REPLICA_COUNT`
in the environment) is the right way to fan out embarrassingly parallel work; each
replica still gets its own 1g slice, so it does not pack.

---

## 6. What I changed

- `scripts/launch_stellar_improvement.py`: CPU-only task routing; `gpu` may be
  omitted/0; `gpu>1` is allowed only for explicit multi-device runners; A0-style
  tasks no longer request a slice.
- `configs/resource-profiles-20261022.yaml`: named profiles and task classes.
- `configs/a0-launch-plan-20261022.yaml`: now `8C/32GB`, no GPU.
- `batch_scripts/canfar_entrypoint.sh`: logs `nvidia-smi -L` and
  `name,memory.total,memory.used,utilization.gpu` at job start, so every run records
  whether it landed on 1g.10gb or 1g.12gb and how much of the slice it uses.

Next measurements: add end-of-job peak GPU memory/utilization to the manifest for a
few 10M arms, then decide whether packing or 2-slice DDP is worth implementing.

---

## 7. Bottom line

- **We cannot slice below 1g on H100.** The efficient move is to stop wasting 1g
  slices on CPU work and to size each GPU job to one slice.
- `--gpu 2` is two slices, not one bigger slice; use it only for multi-device code.
- Keep GPU peak memory below ~10 GB so jobs fit both 1g.10gb and 1g.12gb nodes.
- Measure utilization before packing; packing helps wall time/queue, not total
  GPU-hours, unless jobs underuse their slice.
- If memory-heavy, compute-light jobs become common, ask CANFAR admins whether
  `1g.20/24gb` or profile selection can be exposed; today the CLI cannot request it.
