# Local training QA

The read-only marimo dashboard opens on localhost and fetches a published CSV
snapshot from the user's VOSpace. It does not start, stop or modify training.

```bash
pixi run check-training
# Select a different accepted large run:
pixi run check-training --remote vos:sfabbro/msa/RUN_ID/qa
# Validate transfer/checksums without opening a server:
pixi run check-training --fetch-only
# Never contact the service; show verified local cache with OFFLINE banner:
pixi run check-training --offline
```

Authentication uses the existing CADC proxy. `MSA_CADC_CERT` optionally selects
its certificate file; no token is put into a URI or printed. The dashboard
defaults to `vos:sfabbro/msa/joint_native_30e_v1/qa`; set `MSA_QA_REMOTE` or
`--remote` to the actual published run. A path in documentation is not evidence
that a run has been accepted or that a publication exists.

The launcher is `scripts/check_training.py`; the marimo app is
`monitor/check_training.py`; its stdlib-only transfer/contract module is
`monitor/training_qa_io.py`. The existing `monitor` Pixi environment already
contains marimo, pandas, matplotlib and vos. The root Pixi task should delegate
to that environment:

```toml
check-training = "pixi run -e monitor --no-install python scripts/check_training.py"
```

The view contains separately labelled loss histories, per-target/subset metrics
and changes from the first published evaluation, prediction-versus-reference
regression, residual-versus-reference and residual histogram plots, plus optional
GPU/memory/throughput plots. Target/split/epoch controls choose the monitoring
sample. Units, sample sizes and actual published age are displayed. No curve
silently clips extreme predictions. Error metrics and coverage have different
improvement directions; the view shows signed changes rather than declaring
every decrease an improvement.

The resource field `gpu_memory_used_mb` records PyTorch's peak allocated GPU
memory (`torch.cuda.max_memory_allocated()`), expressed in MiB. It is not total
device memory use or GPU utilization. The field name is retained for schema
compatibility. `rss_mb` likewise reports the process's peak resident memory in
MiB. A blank `gpu_utilization_pct` means utilization was not measured; the
dashboard labels it unavailable rather than drawing a fabricated zero.

## Publication protocol (schema version 1)

Each publication lives in `qa/PUBLICATION_ID/`. Upload its immutable CSVs first,
then replace `qa/manifest.json` last. Required CSVs and headers:

- `losses.csv`: epoch, step, split, loss, value.
- `metrics.csv`: epoch, split, target, subset, metric, value, unit, count.
- `predictions.csv`: epoch, split, target, source_id, truth, prediction, unit.
- Optional `resources.csv`: epoch, step, gpu_utilization_pct,
  gpu_memory_used_mb, rss_mb, examples_per_second.

Extra columns are allowed. Prediction intervals may use lower68, upper68,
lower95, upper95; source IDs remain exact integer strings. Values are finite;
unavailable metrics are omitted rather than represented as NaN or invented
zeroes. Header-only CSVs are allowed before the first evaluation. Epochs count
completed reconstruction passes; step denotes optimizer updates. Loss names
must identify their denominator/normalization in the training documentation.

`manifest.json` contains schema_version=1, run_id, publication_id,
generated_at_utc (timezone-aware UTC), status, source_sha (40 hex), parent_sha256
(64 hex), and files. Each file descriptor contains path (exactly
`PUBLICATION_ID/FILENAME`), bytes and sha256. Status is preparing, running,
complete, failed or stopped. Publication IDs/run IDs use letters, digits,
underscores and hyphens, beginning with a letter/digit; at most 96 characters.
The publisher may add metadata describing dataset roles, units and limitations.

The reader validates the schema, byte counts, SHA256 checksums and CSV types
before atomically updating the local cache pointer. A transfer failure leaves
the previous complete snapshot intact and adds a visible STALE banner. Offline
mode re-verifies all checksums. A missing or invalid first snapshot produces an
explicit unavailable notice, never synthetic training plots. Transfers are
bounded per file and snapshot size; only named CSVs under vos:sfabbro are read.

## Statistical interpretation

Published regression points are a declared monitoring sample, not necessarily
the full validation population. Aggregate CSV metrics retain their own counts.
Use tuning/monitoring roles for repeated inspection and keep a final test locked.
Agreement with noisy native labels is distinct from physical parameter accuracy.
Predictive interval coverage on observed labels does not certify deconvolved
physical-truth coverage. Successful CSV fetching does not itself establish that
the CANFAR session is healthy; status and publication age are both visible.
