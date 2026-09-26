# Narval / Alliance Canada batch jobs

These scripts target the [Narval](https://docs.alliancecan.ca/wiki/Narval/en) GPU cluster (Slurm, NVIDIA A100). Adjust `#SBATCH --account` to your allocation (`def-*-gpu`, `rrg-*`, etc.) and verify module names with `module spider cuda cudnn python` on the login node.

## One-time setup

1. Clone the repo under your scratch or project space and record the path as `MSA_REPO`.
2. Create a Python venv with a **CUDA build of PyTorch** that matches the loaded `cuda` module (see [Alliance PyTorch notes](https://docs.alliancecan.ca/wiki/PyTorch)):

   ```bash
   module load python/3.12 cuda cudnn   # versions per site
   bash batch_scripts/setup_venv_narval.sh "$SCRATCH/venvs/msa"
   ```

3. Copy the example configs and edit HDF5/FITS locations:

   ```bash
   cp configs/pretrain.narval.example.yaml configs/pretrain.active.yaml
   cp configs/finetune.narval.example.yaml configs/finetune.active.yaml
   # Point datafile / ft_datafile / saved_weights at your staged data.
   ```

4. In job scripts (or your shell profile), set:

   ```bash
   export SCRATCH=/scratch/$USER          # or your allocation scratch path
   export MSA_VENV=$SCRATCH/venvs/msa
   ```

Paths in YAML may use `$SCRATCH/...`; `training/config_paths.py` expands them at runtime.

## Submit jobs

From the repo root (so `slurm_logs/` is writable):

```bash
mkdir -p slurm_logs
export SCRATCH=/scratch/$USER
export MSA_VENV=$SCRATCH/venvs/msa

# Pretrain (long)
CONFIG=configs/pretrain.active.yaml sbatch batch_scripts/narval_pretrain.slurm

# Fine-tune ensemble (writes ..._seed{seed}.pth per member when finetuning.ensemble: true)
CONFIG=configs/finetune.active.yaml sbatch batch_scripts/narval_finetune.slurm

# Eval: globs ensemble members (override glob if needed)
EVAL_CKPT_GLOB="$SCRATCH/msa/runs/ft/masked_stellar_autoencoder_ft_seed*.pth" \
  CONFIG=configs/finetune.active.yaml sbatch batch_scripts/narval_eval.slurm
```

`batch_scripts/env_narval.sh` sets `PYTHONPATH`, optional venv activation, `WANDB_MODE=offline`, and creates scratch subdirectories.

## Large HDF5 staging (optional)

For faster I/O, copy the pretrain `.h5` to node-local storage at job start (add to the Slurm script after `env_narval.sh`):

```bash
cp "$SCRATCH/msa/data/sslset-realmags-full.h5" "$SLURM_TMPDIR/"
# then point data.datafile in the active pretrain YAML at $SLURM_TMPDIR/...
```

## Legacy scripts

`msa_init.slurm` and `msa_looping.slurm` are older templates. Prefer `narval_*.slurm` and the `*.narval.example.yaml` configs.

## Ensemble outputs

With `finetuning.ensemble: true`, each member is saved as
`{model_str without ext}_seed{seed}{ext}`
so runs no longer overwrite a single checkpoint file.

## CANFAR (AstroAI base image)

The CANFAR path uses `images.canfar.net/astroai/base:latest` with the repository's
Pixi lock and a CUDA 12.8 PyTorch wheel. Sync the current repository snapshot to
`/arc/projects/k-pop/software/masked-stellar-autoencoder` (or set `MSA_SOURCE`),
and make the catalogue mounts readable from the session before launching jobs.
The example defaults assume the K-pop project paths below; every input root can
be overridden with an environment variable.

After syncing the repository and logging in to CANFAR, `scripts/canfar_launch.sh`
can submit a stage by name. It uses the base image, asks for one GPU for
preflight/training stages, and accepts `MSA_CANFAR_REPO_PATH` if the remote
checkout is elsewhere:

```bash
scripts/canfar_launch.sh schema
scripts/canfar_launch.sh pretrain-pilot
```

Build the Gaia source index after the Gaia source and project catalogue mounts
are visible. This writes the index to the same project path used by preprocessing:

```bash
canfar create --name msa-source-index headless \
  images.canfar.net/astroai/base:latest -- \
  bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh source-index
```

Start with a schema report; it prints available FITS/HDF5 fields and candidate
uncertainty names without guessing a mapping:

```bash
canfar create --name msa-schema headless \
  images.canfar.net/astroai/base:latest -- \
  bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh schema
```

Set `error_cols` in both CANFAR configs from the actual catalogue schemas, then
run the bounded pilots before full training:

```bash
canfar create --name msa-pretrain-pilot --gpu 1 headless \
  images.canfar.net/astroai/base:latest -- \
  bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh pretrain-pilot

canfar create --name msa-finetune-pilot --gpu 1 headless \
  images.canfar.net/astroai/base:latest -- \
  bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh finetune-pilot
```

The preprocessing builder accepts `MSA_SOURCE_IDS_FILE`, `MSA_CATWISE_FILE`,
`MSA_GAIA_XP_DIR`, `MSA_GAIA_SOURCE_DIR`, `MSA_ADQL_MATCH_DIR`,
`MSA_PREPROCESS_DIR`, `MSA_PREPROCESS_CACHE_DIR`, `START_PART`, and `STOP_PART`.
Catalogue inputs use the original K-pop mounts. Generated outputs default to
`$MSA_OUTPUT_ROOT` (`/arc/projects/k-pop/msa_runs`): dust maps go under
`dustmaps`, while the source-ID index, partial FITS tables, and combined HDF5
file go under `preprocess`. These outputs stay on persistent project storage;
crossmatch chunks and Pixi caches stay under `$WORK`. Override the paths when
the session exposes different mounts. The XP input defaults to
`/arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_mean_spectrum` and expects
`XpContinuousMeanSpectrum_<source-id-range>.csv.gz` files; GaiaSource inputs are
HDF5 shards. Run `fetch-dustmaps` once before preprocessing.

```bash
canfar create --name msa-dustmaps headless images.canfar.net/astroai/base:latest -- \
  bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh fetch-dustmaps
```

Run `fetch-dustmaps`, then submit preprocessing in non-overlapping partition
ranges. For example, these two sessions produce portions 0–1 and 2–3; continue
with further ranges until 50 is reached, then run `combine`:

```bash
canfar create --name msa-data-00-02 headless images.canfar.net/astroai/base:latest -- \
  env START_PART=0 STOP_PART=2 bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh preprocess

canfar create --name msa-data-02-04 headless images.canfar.net/astroai/base:latest -- \
  env START_PART=2 STOP_PART=4 bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh preprocess
```

The temporary crossmatch cache includes the session hostname so concurrent
partition jobs do not overwrite each other's chunks.

Run `preprocess`, then `combine`. For a fresh pretraining build, set
`MSA_PRETRAIN_DATA` to `$MSA_PREPROCESS_DIR/pretrain_dataset_incomplete.h5` when
launching the pretrain stage. The default points at the existing catalogue file.
`START_PART` lets you continue at a later partition after checking which
`partialtable-*.fits` files completed.

After the pretraining pilot and a clean preflight, launch full pretraining and
fine-tuning with the `pretrain` and `finetune` stages. The schema and path
preflight must pass before a training stage starts.

The mounted legacy HDF5/FITS catalogues contain Gaia G/BP/RP flux uncertainties
but not the corresponding magnitude uncertainties. The CANFAR configs leave
those three `error_cols` entries null rather than mixing flux and magnitude
units; the XP coefficient errors remain mapped. Tables rebuilt with the current
preprocessing stage include `e_G`, `e_BP`, and `e_RP`, so map those columns in
both configs when switching `MSA_PRETRAIN_DATA` and `MSA_FINETUNE_DATA` to the
rebuilt tables.

```bash
canfar create --name msa-pretrain --gpu 1 headless images.canfar.net/astroai/base:latest -- \
  bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh pretrain

canfar create --name msa-finetune --gpu 1 headless images.canfar.net/astroai/base:latest -- \
  bash /arc/projects/k-pop/software/masked-stellar-autoencoder/batch_scripts/canfar_entrypoint.sh finetune
```

Pretraining resumes optimizer, scheduler, and random generator state when
`MSA_PRETRAIN_RESUME` points to a full pretraining checkpoint. `training.epochs`
is the total target epoch, so a run resuming at epoch 40 with `epochs: 100`
executes epochs 41–100. `MSA_FINETUNE_RESUME` loads saved fine-tune model and
head weights. New non-ensemble fine-tune checkpoints also resume the optimizer,
scheduler, random generator state, and total epoch target. With ensemble mode on,
`MSA_FINETUNE_RESUME` is a shared weight warm-start for each member and each
member starts a fresh optimizer; turn ensemble mode off to resume one interrupted
member. Older fine-tune checkpoints without training state also start a fresh
optimizer. Pretraining pilots use one epoch, at most two training shards and
one validation shard (512 rows per shard), and `_pilot` output paths. Fine-tuning
pilots use one epoch, bounded rows and batches, and `_pilot` output paths.

The launch command assumes CANFAR project storage is mounted at
`/arc/projects/k-pop` and Gaia XP data at `/gaia/dr3/...`; set
`CANFAR_PROJECT_ROOT`, `MSA_GAIA_XP_DIR`, or the individual data paths if this
session exposes them elsewhere. Do not install Python packages into `$HOME`.
