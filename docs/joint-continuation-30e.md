# Large joint continuation: preparation and operational contract

Status: local implementation and preparation, **not an accepted GPU run**. The
human selected 30 reconstruction epochs, QA every 3 epochs and checkpoints at
every epoch, then requested validation on the 10M sample before switching to
150M training rows. The first stage uses the immutable benchmark's 9M training /
1M validation split. A bounded three-epoch pilot covers the first scheduled QA
publication; it must also exercise interruption and resume. The production
budget is 30 × 150M = 4.5 billion reconstruction presentations. A finite smoke
loss alone does not authorize promotion: actual source associations, original XP
units, GPU resource measurements, resume, publication and declared tuning checks
must pass first. Main pretraining remains unchanged, and no accepted ID is reused.

`configs/joint-continuation-10m-pilot.yaml` records that pilot. The 30-epoch
configuration specifies 150M training rows and remains a preparation template
until the checked cache request hash and matched pilot promotion certificate are
bound to it. The runner rejects sample underfill, benchmark repetition disguised
as a larger sample, and promotion from a different source, parent or label cohort.

## What runs

`configs/joint-continuation-30e.yaml` prepares a separate continuation with the
existing encoder/decoder and a residual prediction module. Its stellar output
is a full covariance Gaussian over train-standardized **linear Kelvin Teff**,
logg and FeH. Native Gaussian measurement errors are convolved analytically;
partially missing targets use the correct observed-coordinate submatrix. No
target is imputed. Unknown native cross-target covariance and survey systematics
remain limitations, not invented zero-error truth or a universal error floor.

An optional separate log-distance point output fits raw signed parallax with
declared per-star zero points. It is **not** a joint stellar-distance flow/PDF or
a calibrated distance interval. The prepared configuration disables this output
until the Gaia quality/zero-point and correlated-teacher audits pass. It must
not silently be presented as an enabled distance model. Alpha/age are off.

The objective is a working composite: stellar observation NLL per observed
coordinate, optional parallax observation NLL per eligible row, and 0.01 times
masked-MAE reconstruction plus 0.1 relative spectral regularization. Each has
its own full optimizer-batch denominator. GALAH DR4 gravity is explicitly
parallax-adjusted, so stellar and astrometric factors cannot be described as
independent physical measurements. Actual-parent gradient balance remains a
production gate. A general nonlinear noise-convolved joint-distance flow has
not been substituted with an unjustified Gaussian approximation.

## Inputs

The first prepared core keeps the 134 existing photometry/XP observations:
110 XP coefficients, Gaia G/BP/RP, WISE W1/W2, six SkyMapper, five SDSS, five PS1,
2MASS J/H/KS. Genuine missingness is retained, all target/unknown aliases excluded,
and the 140-slot parent shape retained. RA/DEC/PM/parallax/EBV slots are withheld
in both supervised and reconstruction calls. The 136-output decoder shape remains
compatible, but only allowed observable outputs are scored. This deliberate
policy change from the six pilots needs actual-input/GPU validation.

Positions and proper motions remain preserved for matching, epoch propagation,
quality, dust and spatial diagnostics. Their exclusion is a modelling choice
to limit population/kinematic shortcuts, not a claim they have no information.
An alternative astrometric input extension requires its own tested policy.

Extinction is physically important. The current column called EBV has no retained
unit/map/error provenance and a maximum of 132.7899; silently treating it as exact
stellar E(B−V) is unsafe. Raw values remain preserved. Enabling it requires its
provenance gate; no arbitrary numerical cut was invented. A later audited dust
input can improve the model without discarding that information permanently.

## Data preparation before allocating the long GPU run

Use already acquired original GALAH/APOGEE FITS files on ARC, not new surveys.
The fixed reference union of 99,574 Hipparcos/seismic IDs and all benchmark
validation IDs are excluded from fit/tune/calibration. Native registry's prior
Li-VMPS records remain provenance but are not promoted to joint HRS labels.
New target errors and flags must be checked on actual original rows.

CPU-only plan `configs/joint-native-preparation.yaml` requests 4 cores, 32 GiB RAM
and no GPU, with a 1,800-second bound. In the repository's production XP
environment:

```bash
pixi run --manifest-path experiments/xp/pixi.toml python scripts/prepare_joint_native_labels.py --dry-run
pixi run --manifest-path experiments/xp/pixi.toml python scripts/prepare_joint_native_labels.py --record-source-pins
pixi run --manifest-path experiments/xp/pixi.toml python scripts/prepare_joint_native_labels.py --execute
```

Run the latter two on the existing ARC mounts. They refuse overwrite/reuse of
completed or incomplete source products. GALAH's earlier inventory had no
full-file hash: record the first mounted hash explicitly, without pretending
it is an independently published checksum. APOGEE's previously accepted full
hash is enforced. A second verifier compares every projected value/error/flag/
source row against the original FITS before `native_labels_verified` becomes true.

## Reliability

- One owner per output and scratch directory; no competing process can replace
  training state. Explicit `--resume` is required when a committed state exists.
- Preserve encoder/decoder/head, both Adam states, global Python/NumPy/Torch/CUDA
  RNG, label RNG, mask RNG, next row, epoch, step, scales and QA history.
- Unique immutable checkpoint payloads; file fsync, checksum and atomic latest
  pointer. A kill between writing data and pointer leaves the old state usable.
  Keep every epoch checkpoint and the latest two intermediate files; state is
  saved every 50 updates as well as every epoch. Epoch payloads are reloaded and
  every model/head tensor compared byte-exactly in production.
- Mid-epoch recovery resumes the next uncommitted row/stream. Pending boundary
  QA is replayed after interruption. Final test metrics are not appended twice.
- Fail on nonfinite losses/gradients/covariances; no silent variance clipping,
  target filling, gradient skipping or automatic new session submission.
- SIGTERM/SIGINT requests a stop after a complete optimizer step. A hard kill
  resumes the last committed state, not a half-finished optimizer update.
- Smoke mode is explicitly bounded, labels/benchmark fixtures are marked as
  such. Fixture successes cannot certify real source labels or original XP units.

## GPU/CPU plan

Provisional pilot resource request: 8 cores, 64 GiB RAM and one GPU. Cache the 9M scaled
rows in scratch (~5 GB as float32) rather than repeatedly scanning the original
121 GB HDF5. Keep model environments under scratch, never home. The effective
reconstruction batch is 32,768, with microbatches of 8,192. The supervised batch
of 2,048 is split into 256-row microbatches. Reuse parent Adam moments with fixed
encoder LR 1.25e-6 and a warmed new head with LR 1e-3.

These resources/batches must pass a real allocated-device memory/throughput
test before launch; a GPU count is not a GPU-memory guarantee on CANFAR. No
untested mixed precision/compilation or all-catalogue GPU preload is enabled.
Report measured examples/sec, CPU peak RSS and PyTorch peak allocated GPU memory.
The CSV field `gpu_memory_used_mb` contains `torch.cuda.max_memory_allocated()`
converted to MiB; it does not measure total device memory use. A blank
`gpu_utilization_pct` means utilization was not measured, rather than zero
utilization. Full-main epochs require a different cache/resource budget.

For 150M rows, the float32 feature cache alone is about84GB (140 slots), plus
integer source IDs, original shard/row indices and uniqueness checks. The large
template provisionally requests128GiB RAM; actual scratch space and memory must
be measured before allocation. Prepare the data on CPU first:

```bash
pixi run --manifest-path experiments/xp/pixi.toml python scripts/stage_joint_reconstruction.py --config configs/joint-continuation-30e.yaml
```

The sampler allocates an exact proportional count across original training
shards, selects without replacement using bounded chunks, and excludes every
original validation shard and source ID. It preserves integer IDs and original
row associations, checks uniqueness, hashes all staged arrays and publishes its
manifest last. It does not claim a full raw-catalogue checksum: original identity
uses path, bytes, modification time and schema, bound to the parent signature.
The GPU runner verifies the complete cache and matched pilot certificate rather
than scanning or staging the full catalogue while holding a GPU. No150M sample
has yet been staged or certified by these local fixtures.

## QA and local command

```bash
pixi run check-training
pixi run check-training --fetch-only
pixi run check-training --offline
```

Uses existing monitor Pixi environment, local marimo and
`vos:sfabbro/msa/joint_native_30e_v1/qa`. See [check-training.md](check-training.md)
for schema, credentials, cache and stale-data behaviour. The first real
preparation publication has header-only CSVs and status `preparing`, with an
explicit base-source-only marker; it contains no synthetic training results.

Training exports at epochs 0 (after head warmup), 3, 6, …, 30 include loss
histories; MAE, bias, RMSE and observed-label interval coverage by target, survey
and tail; actual counts; regression/residual samples; fixed-mask coefficient
and spectral validation; and resource records. All tuning predictions,
covariances, raw labels, errors and IDs are also archived in NPZ on ARC at every
QA epoch; final identity-test arrays are archived only at the end. A fixed
first-ID sample per target bounds CSV size; aggregate metrics use all eligible
role rows. Calibration is separate and not repeatedly fitted on the final test.
The earlier inspected test is not relabelled as untouched.

Publish immutable, checksummed CSV directories before the remote manifest.
The background publisher has bounded network calls and a durable retryable
outbox. VOS failure preserves local training and adds a publication warning;
it never restarts training. To retry publication independently:

```bash
pixi run -e monitor python scripts/publish_training_qa.py --output /arc/projects/k-pop/msa_runs/joint-native-30e-v1 --remote vos:sfabbro/msa/joint_native_30e_v1/qa
```

## Launch gates and commands

The runner refuses production until a policy-bound readiness report certifies
actual native labels, original 128-source XP-unit agreement, source roles,
resume, gradient balance, GPU preflight and actual VOS publication. Optional
distance and reddening require their additional gates. It also verifies source
SHA and parent/cohort/joint-sidecar SHA pins. The readiness report is not present
or implicitly true in this preparation. No launch has occurred.

Once those gates pass and the selected source/configuration is pinned:

```bash
pixi run --manifest-path experiments/xp/pixi.toml python scripts/run_joint_continuation.py --config configs/joint-continuation-30e.yaml
# Same exact source/config/data/output after interruption:
pixi run --manifest-path experiments/xp/pixi.toml python scripts/run_joint_continuation.py --config configs/joint-continuation-30e.yaml --resume
```

Do not run these production paths on a Mac. Use a pinned headless CANFAR source
checkout, existing acquired data and a unique accepted-session/output record.
Reconcile ambiguous acceptance before any retry. Main run `rrrc2ihf` is not stopped,
changed or repurposed. No claim of a bug-free trainer follows from tests; the
claim is the specific failure modes checked and the remaining gates disclosed.
