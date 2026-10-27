# Remediation storage cleanup — 2026-10-26
**Final update:** the user authorized removal of the four listed batch-cache directories and all four server-side deletions completed on 2026-10-26. No archive was created. The deleted trees totaled 28,341,292,914 bytes. Prediction-array directories retained their prior URI and aggregate byte counts; this metadata check is not a full-content hash verification. Project usage fell by more than the amount deleted here while the user was also removing data, so the additional reduction cannot be attributed to this cleanup. The earlier quota-block and archive proposal below are historical and superseded; all fourteen pilots were subsequently submitted.

The fourteen supervised pilot arms are implemented and tested, including four genuinely coupled six-target NSF arms. None has been submitted. The distinct recovery audit/source/NSF-preflight jobs were accepted once and all failed with `Errno122: Disk quota exceeded` before they could supply the required admission receipts. They are preserved; there is no automatic retry or recurring monitor.

ARC project metadata reports a 20 TB quota and 20,000,005,095,420 bytes used. The launcher now checks quota before future submissions and requires 20 GB of output reserve. The complete updated test suite passed 1,090 tests with 4 skips; 13 focused checks validate the quota and launch guards.

A concrete archive proposal is available in `experiments/xp/results-stellar-improvement/remediation-wave-20261026/quota-block/archive-candidates.json`. The four `expanded_eval_*_s51_v2/batch-cache` directories total 28.34 GB. These are archive candidates, not certified redundant files. Do not discard them. Keep 83.11 GB of consolidated prediction arrays and all unique checkpoints, source labels, reports, bounded pilots and corner samples in place.

To resume: either obtain project quota/additional storage, or approve an archive destination with capacity for those exact cache trees. Before removing any original, copy and verify full content hashes, preserve manifests/source bindings and record the archive locations. No deletion or transfer has occurred. Once storage is available, preserve these failed v2 attempts and use distinct versioned recovery outputs; only actual successful prerequisite receipts may admit the fourteen training pilots.

## Exact archive list for review

Recorded sizes from the read-only inventory on 2026-10-26. Sizes below use decimal GB. These four directories are candidates for **archival outside the `/arc/projects/k-pop` quota**, followed by removal of the verified originals only after approval. Moving them elsewhere inside the same project does not free its quota. No deletion-only candidates have been certified.

- `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/expanded_eval_quantile3_s51_v2/batch-cache` — **1.209 GB** (1,209,261,006 bytes).
- `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/expanded_eval_gaussian_noise_s51_v2/batch-cache` — **9.052 GB** (9,052,200,935 bytes).
- `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/expanded_eval_nsf_noise_s51_v2/batch-cache` — **9.035 GB** (9,035,417,666 bytes).
- `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/expanded_eval_nsf_noise_mixed01_s51_v2/batch-cache` — **9.044 GB** (9,044,413,307 bytes).

Total potentially released after verified archival: **28.341 GB** (28,341,292,914 bytes). Archiving only the three density-model caches would release **27.132 GB**; the quantile cache can stay in place if preferred.

Keep in place:

- All four `prediction-arrays/` trees: **83.105 GB** total.
- Every unique selected/periodic checkpoint and its provenance, source catalogue/label assembly, and normalization.
- Reports, score tables, bounded metric pilots, corner-draw bundles, and failure/launch/hash receipts.
- The **87.005 GB** `residual-huber-e86-150m-cache-v1` reconstruction cache, which the planned reconstruction-control pilots still need.

Approval boundary: choose which exact cache trees to archive and a destination outside this quota with sufficient capacity. Build a per-file manifest, copy without altering originals, verify full-content hashes and completeness, record the archive locations/source bindings, then remove originals only if that removal was included in the user’s authorization. An archive-only approval does not authorize removal. This list authorizes no transfer, deletion, quota change, or job resubmission.
